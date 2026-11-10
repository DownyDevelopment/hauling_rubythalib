"""R&D tracking: pilihan tracker, TensorRT metadata, timing, adapter DeepSORT, benchmark & video grid."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hauling_offline.benchmark as B  # noqa: E402
import hauling_offline.cli as cli  # noqa: E402
from hauling_offline.pipeline import RunSummary  # noqa: E402
from hauling_offline.trackers import (TimingStats, deepsort_tracks_to_rows, load_model_meta,  # noqa: E402
                                      model_meta_path, patch_tracker_yaml, save_model_meta, xyxy_to_ltwh)

REQ = ["--video", "v.mp4", "--vehicle-model", "m.pt", "--config", "c.conf", "--output-dir", "out"]
BYTETRACK_YAML = "tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_buffer: 30\nmatch_thresh: 0.8\n"
BOTSORT_YAML = "tracker_type: botsort\ntrack_buffer: 30\ngmc_method: sparseOptFlow\nwith_reid: False\n"


# ----------------------------- config tracker ---------------------------- #
def test_patch_track_buffer_only_for_bytetrack():
    out = patch_tracker_yaml(BYTETRACK_YAML, 60)
    assert "track_buffer: 60" in out and "track_high_thresh: 0.25" in out and "with_reid" not in out


def test_patch_botsort_reid_on_and_off():
    assert "with_reid: True" in patch_tracker_yaml(BOTSORT_YAML, 50, with_reid=True)
    assert "with_reid: False" in patch_tracker_yaml(BOTSORT_YAML, 50, with_reid=False)
    assert "with_reid: True" in patch_tracker_yaml("tracker_type: botsort\ntrack_buffer: 30\n", 50, with_reid=True)


# ----------------------------- TensorRT metadata ------------------------- #
def test_engine_sidecar_roundtrip(tmp_path):
    engine = tmp_path / "vehicle.engine"
    engine.write_bytes(b"x")
    assert load_model_meta(engine) == {}
    save_model_meta(engine, {"imgsz": 1280, "single_cls": True, "epochs": 100}, "vehicle.pt", half=True)
    assert model_meta_path(engine).name == "vehicle.engine.hauling.json"
    assert load_model_meta(engine) == {"imgsz": 1280, "single_cls": True}


# ----------------------------- timing ------------------------------------ #
def test_timing_skips_warmup_and_splits_detect_track():
    t = TimingStats(warmup=2)
    for det, tot in [(500, 900), (400, 800), (10, 12), (10, 14), (10, 13)]:
        t.add(det, tot)
    s = t.summary()
    assert s["frames_measured"] == 3 and s["warmup_frames_skipped"] == 2
    assert s["detect_ms_mean"] == 10 and s["track_ms_mean"] == 3 and s["total_ms_mean"] == 13
    assert s["fps_detect_track"] == round(1000 / 13, 2)


# ----------------------------- DeepSORT adapter -------------------------- #
class FakeTrack:
    def __init__(self, tid, confirmed=True, since=0, box=(10, 20, 110, 220), conf=0.8):
        self.track_id, self._c, self.time_since_update, self._b, self._conf = tid, confirmed, since, box, conf

    def is_confirmed(self):
        return self._c

    def to_ltrb(self, orig=False):
        return self._b

    def get_det_conf(self):
        return self._conf

    def get_det_class(self):
        return 0


def test_deepsort_rows_only_confirmed_and_matched():
    rows = deepsort_tracks_to_rows([FakeTrack("3"), FakeTrack("4", confirmed=False), FakeTrack("5", since=2),
                                    FakeTrack("6", conf=None)])
    assert [r[0] for r in rows] == [3, 6]
    assert rows[0][1] == (10.0, 20.0, 110.0, 220.0) and rows[1][2] == 0.0
    assert xyxy_to_ltwh((10, 20, 110, 220)) == [10.0, 20.0, 100.0, 200.0]


# ----------------------------- CLI --------------------------------------- #
def test_tracker_flag_reaches_run_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run", lambda opt: seen.setdefault("opt", opt) and RunSummary())
    cli.main(REQ + ["--tracker", "deepsort"])
    assert seen["opt"].tracker == "deepsort"
    assert cli.build_parser().parse_args(REQ).tracker == "bytetrack"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(REQ + ["--tracker", "sort"])


@pytest.mark.parametrize("sub,module", [("export-engine", "export_engine"), ("benchmark-tracker", "benchmark")])
def test_rnd_subcommands_dispatch(monkeypatch, sub, module):
    import importlib
    mod = importlib.import_module(f"hauling_offline.{module}")
    got = {}
    monkeypatch.setattr(mod, "main", lambda argv: got.setdefault("argv", argv) and 0)
    cli.main([sub, "--x", "1"])
    assert got["argv"] == ["--x", "1"]


# ----------------------------- benchmark --------------------------------- #
def test_build_variants_with_engine(tmp_path):
    v = B.build_variants(["bytetrack", "deepsort"], Path("m.pt"), Path("m.engine"))
    assert [x.name for x in v] == ["bytetrack", "deepsort", "bytetrack+trt"]
    assert v[-1].model == Path("m.engine") and v[-1].tracker == "bytetrack" and "TensorRT" in v[-1].label


def test_track_metrics_fragmentation():
    rows = [{"frames_in_main": "40", "status_kendaraan": "FINISH - speed valid"},
            {"frames_in_main": "12", "status_kendaraan": "GAGAL FINISH"},
            {"frames_in_main": "0", "status_kendaraan": "DI LUAR MAIN AREA"},
            {"frames_in_main": "3", "status_kendaraan": "DETEKSI SINGKAT"}]
    m = B.track_metrics(rows, {"not_crossed_start": 1}, true_vehicles=2)
    assert m["relevant_tracks"] == 3 and m["fragmentation_ratio"] == 1.5 and m["gagal_finish"] == 1
    assert m["deteksi_singkat"] == 1 and m["not_crossed_start"] == 1 and m["mean_frames_in_main"] == 18.3


def test_grid_shape_and_tile():
    assert B.grid_shape(1) == (1, 1) and B.grid_shape(3) == (2, 2) and B.grid_shape(4) == (2, 2)
    assert B.grid_shape(5) == (3, 2)
    f = np.full((108, 192, 3), 200, np.uint8)
    out = B.tile_frames([f, None, f], ["a", "b", "c"], (96, 54), 2, 2)
    assert out.shape == (108, 192, 3) and out[30, 140].sum() == 0   # panel ke-2 (frame habis) hitam


def _video(path, n, w=192, h=108, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 7) % 255, np.uint8))
    vw.release()


def test_benchmark_end_to_end_with_fake_pipeline(tmp_path, monkeypatch):
    import hauling_offline.pipeline as P
    for name in ("v.mp4", "m.pt", "c.conf"):
        (tmp_path / name).write_bytes(b"x")

    def fake_run(opt):
        out = Path(opt.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        n = {"bytetrack": 3, "deepsort": 5}[opt.tracker]
        with (out / "tracks_report.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frames_in_main", "status_kendaraan"])
            w.writeheader()
            for _ in range(n):
                w.writerow({"frames_in_main": "20", "status_kendaraan": "FINISH - speed valid"})
        _video(out / "annotated.mp4", 6 if opt.tracker == "bytetrack" else 4)
        s = RunSummary(total_frames_read=6, unique_tracks=n, csv_events=2, speed_valid=2, processing_fps=20.0)
        s.tracker = {"detector_backend": "pt", "total_ms_mean": 30.0 if opt.tracker == "bytetrack" else 45.0,
                     "detect_ms_mean": 28.0, "track_ms_mean": 2.0, "fps_detect_track": 33.3}
        s.outputs = {"tracks_report": str(out / "tracks_report.csv"), "annotated_video": str(out / "annotated.mp4")}
        return s

    monkeypatch.setattr(P, "run", fake_run)
    monkeypatch.setattr(P, "_to_h264", lambda p: None)
    rc = B.main(["--video", str(tmp_path / "v.mp4"), "--vehicle-model", str(tmp_path / "m.pt"),
                 "--config", str(tmp_path / "c.conf"), "--output-dir", str(tmp_path / "bench"),
                 "--trackers", "bytetrack", "deepsort", "--true-vehicles", "3"])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "bench" / "benchmark_results.csv").open(encoding="utf-8")))
    assert [r["variant"] for r in rows] == ["bytetrack", "deepsort"]
    assert rows[0]["fragmentation_ratio"] == "1.0" and rows[1]["fragmentation_ratio"] == "1.67"
    assert "ByteTrack" in (tmp_path / "bench" / "benchmark_results.md").read_text(encoding="utf-8")
    cap = cv2.VideoCapture(str(tmp_path / "bench" / "comparison_grid.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
    cap.release()
    assert json.loads((tmp_path / "bench" / "benchmark_results.json").read_text())[1]["label"] == "DeepSORT"


def test_failed_variant_does_not_stop_benchmark(tmp_path, monkeypatch):
    import hauling_offline.pipeline as P
    for name in ("v.mp4", "m.pt", "c.conf"):
        (tmp_path / name).write_bytes(b"x")

    def boom(opt):
        raise ImportError("deep-sort-realtime belum terpasang")

    monkeypatch.setattr(P, "run", boom)
    rc = B.main(["--video", str(tmp_path / "v.mp4"), "--vehicle-model", str(tmp_path / "m.pt"),
                 "--config", str(tmp_path / "c.conf"), "--output-dir", str(tmp_path / "b"), "--trackers", "deepsort",
                 "--no-video"])
    assert rc == 1
    row = next(csv.DictReader((tmp_path / "b" / "benchmark_results.csv").open(encoding="utf-8")))
    assert row["status"] == "gagal" and "deep-sort-realtime" in row["error"]


def test_benchmark_passes_plate_and_ocr(tmp_path, monkeypatch):
    import hauling_offline.pipeline as P
    for name in ("v.mp4", "m.pt", "c.conf", "plate.pt"):
        (tmp_path / name).write_bytes(b"x")
    seen = []

    def fake_run(opt):
        seen.append(opt)
        s = RunSummary(csv_events=7, plate_detected=6, lambung_text_read=6)
        s.tracker = {"detector_backend": "pt"}
        return s

    monkeypatch.setattr(P, "run", fake_run)
    rc = B.main(["--video", str(tmp_path / "v.mp4"), "--vehicle-model", str(tmp_path / "m.pt"),
                 "--config", str(tmp_path / "c.conf"), "--output-dir", str(tmp_path / "b"), "--trackers", "bytetrack",
                 "--no-video", "--plate-model", str(tmp_path / "plate.pt"), "--plate-region", "0.4,1,0.25,1",
                 "--ocr-model-dir", str(tmp_path)])
    assert rc == 0
    assert seen[0].plate_model == tmp_path / "plate.pt" and seen[0].ocr_model_dir == tmp_path
    assert seen[0].plate_region == (0.4, 1.0, 0.25, 1.0)
    row = next(csv.DictReader((tmp_path / "b" / "benchmark_results.csv").open(encoding="utf-8")))
    assert row["lambung_text_read"] == "6" and row["plate_detected"] == "6"
