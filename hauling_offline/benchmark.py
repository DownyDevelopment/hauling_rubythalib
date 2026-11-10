"""R&D tracking: jalankan pipeline yang sama dengan beberapa tracker / backend detector, lalu bandingkan.

Semua varian memakai video, detector, config area, dan parameter yang SAMA; yang diganti hanya tracker
(atau backend detector .pt vs TensorRT .engine). Secara default model nomor lambung dan OCR tidak dijalankan
agar waktu yang terukur murni deteksi + tracking; aktifkan dengan --plate-model dan --ocr-model-dir untuk
mengukur pipeline lengkap (waktunya tercatat terpisah di tahap plate_detect dan ocr).

    python main.py benchmark-tracker --video video/km18_sample.mp4 \\
        --vehicle-model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt --config config/km18.conf \\
        --trackers bytetrack botsort deepsort --engine model/Hauling_Vehicle_yolov8m_SingleCls_V11.engine \\
        --true-vehicles 12 --max-frames 1800 --output-dir outputs/benchmark_01

Output:
    benchmark_results.csv / .json   metrik per varian
    benchmark_results.md            tabel ringkas (siap ditempel ke laporan)
    comparison_grid.mp4             video preview semua varian berdampingan (label + ms/frame + jumlah ID)
    <varian>/                       output pipeline per varian (events.csv, tracks_report.csv, annotated.mp4, ...)

Metrik proxy (tanpa ground truth anotasi):
    relevant_tracks      jumlah track_id yang pernah masuk main area; idealnya = jumlah kendaraan sebenarnya
    fragmentation_ratio  relevant_tracks / --true-vehicles (1.00 = ideal; > 1 = ID pecah / berganti)
    not_crossed_start    event yang track-nya "muncul" setelah garis start -> indikasi ID baru di tengah jalan
    gagal_finish         track masuk main area tapi hilang sebelum final -> indikasi track putus
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .trackers import TRACKERS

log = logging.getLogger(__name__)

LABELS = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT", "botsort-reid": "BoT-SORT + ReID", "deepsort": "DeepSORT"}
RESULT_COLUMNS = [
    "variant", "label", "tracker", "detector_backend", "status",
    "frames", "total_ms_mean", "total_ms_p95", "detect_ms_mean", "track_ms_mean", "fps_detect_track",
    "processing_fps", "gpu_peak_mem_mb",
    "unique_tracks", "relevant_tracks", "true_vehicles", "fragmentation_ratio",
    "csv_events", "speed_valid", "not_crossed_start", "gagal_finish", "deteksi_singkat",
    "mean_frames_in_main", "overspeed", "plate_detected", "lambung_text_read", "error",
]


@dataclass(frozen=True)
class Variant:
    name: str            # nama folder, mis. "bytetrack" / "bytetrack+trt"
    tracker: str
    model: Path
    label: str


def build_variants(trackers: Sequence[str], model: Path, engine: Optional[Path],
                   engine_trackers: Sequence[str] = ("bytetrack",)) -> List[Variant]:
    out = [Variant(t, t, model, LABELS[t]) for t in trackers]
    if engine is not None:
        out += [Variant(f"{t}+trt", t, engine, f"{LABELS[t]} + TensorRT") for t in engine_trackers]
    return out


# --------------------------------------------------------------------------- #
# Metrik
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def track_metrics(rows: Sequence[Dict[str, str]], speed_status_counts: Dict[str, int],
                  true_vehicles: Optional[int]) -> dict:
    """Metrik proxy stabilitas ID dari tracks_report.csv."""
    relevant = [r for r in rows if str(r.get("frames_in_main", "0")) not in ("", "0")]
    status = [r.get("status_kendaraan", "") for r in rows]
    frames_main = [int(r["frames_in_main"]) for r in relevant if str(r["frames_in_main"]).isdigit()]
    return {
        "relevant_tracks": len(relevant),
        "true_vehicles": true_vehicles,
        "fragmentation_ratio": round(len(relevant) / true_vehicles, 2) if true_vehicles else None,
        "not_crossed_start": int(speed_status_counts.get("not_crossed_start", 0)),
        "gagal_finish": status.count("GAGAL FINISH"),
        "deteksi_singkat": status.count("DETEKSI SINGKAT"),
        "mean_frames_in_main": round(sum(frames_main) / len(frames_main), 1) if frames_main else None,
    }


def result_row(variant: Variant, summary, true_vehicles: Optional[int]) -> dict:
    t = dict(summary.tracker or {})
    rows = read_rows(Path(summary.outputs.get("tracks_report", "")))
    row = {
        "variant": variant.name, "label": variant.label, "tracker": variant.tracker,
        "detector_backend": t.get("detector_backend"), "status": "ok",
        "frames": summary.total_frames_read,
        **{k: t.get(k) for k in ("total_ms_mean", "total_ms_p95", "detect_ms_mean", "track_ms_mean",
                                 "fps_detect_track", "gpu_peak_mem_mb")},
        "processing_fps": summary.processing_fps, "unique_tracks": summary.unique_tracks,
        "csv_events": summary.csv_events, "speed_valid": summary.speed_valid, "overspeed": summary.overspeed,
        "plate_detected": summary.plate_detected, "lambung_text_read": summary.lambung_text_read,
        "error": "",
    }
    row.update(track_metrics(rows, summary.speed_status_counts, true_vehicles))
    return row


def failed_row(variant: Variant, error: str) -> dict:
    row = {k: None for k in RESULT_COLUMNS}
    row.update(variant=variant.name, label=variant.label, tracker=variant.tracker, status="gagal", error=error[:300])
    return row


def write_results(out_dir: Path, results: Sequence[dict]) -> Dict[str, Path]:
    paths = {"csv": out_dir / "benchmark_results.csv", "json": out_dir / "benchmark_results.json",
             "md": out_dir / "benchmark_results.md"}
    with paths["csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in RESULT_COLUMNS})
    paths["json"].write_text(json.dumps(list(results), indent=2, ensure_ascii=False), encoding="utf-8")
    paths["md"].write_text(results_markdown(results), encoding="utf-8")
    return paths


def results_markdown(results: Sequence[dict]) -> str:
    cols = [("label", "Varian"), ("total_ms_mean", "ms/frame"), ("fps_detect_track", "FPS"),
            ("detect_ms_mean", "deteksi ms"), ("track_ms_mean", "tracking ms"), ("relevant_tracks", "ID di main"),
            ("fragmentation_ratio", "rasio fragmentasi"), ("csv_events", "event"), ("speed_valid", "speed valid"),
            ("not_crossed_start", "muncul stlh start"), ("gagal_finish", "gagal finish"),
            ("lambung_text_read", "lambung terbaca")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['label']} | gagal: {r.get('error', '')} |" + " |" * (len(cols) - 2))
            continue
        lines.append("| " + " | ".join("–" if r.get(k) is None else str(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Video preview berdampingan
# --------------------------------------------------------------------------- #
def grid_shape(n: int) -> tuple:
    cols = 1 if n <= 1 else (2 if n <= 4 else 3)
    return cols, max(1, math.ceil(n / cols))


def panel_caption(row: dict) -> str:
    if row.get("status") != "ok":
        return f"{row['label']} | gagal"
    ms = row.get("total_ms_mean")
    speed = f"{ms:.1f} ms/frame" if isinstance(ms, (int, float)) else "- ms/frame"
    return f"{row['label']} | {speed} | ID di main: {row.get('relevant_tracks')}"


def tile_frames(frames: Sequence[Optional[np.ndarray]], captions: Sequence[str], panel_wh: tuple,
                cols: int, rows: int) -> np.ndarray:
    """Susun frame per varian menjadi satu grid; panel kosong / frame habis = hitam."""
    import cv2
    pw, ph = panel_wh
    canvas = np.zeros((ph * rows, pw * cols, 3), dtype=np.uint8)
    bar = max(28, ph // 16)
    fs = bar / 40.0
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        x0, y0 = c * pw, r * ph
        if i < len(frames) and frames[i] is not None:
            canvas[y0:y0 + ph, x0:x0 + pw] = cv2.resize(frames[i], (pw, ph), interpolation=cv2.INTER_AREA)
        if i < len(captions):
            cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + bar), (20, 20, 20), -1)
            cv2.putText(canvas, captions[i], (x0 + 10, y0 + int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), max(1, int(fs * 2)), cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x0 + pw - 1, y0 + ph - 1), (90, 90, 90), 1)
    return canvas


def compose_grid(videos: Sequence[Optional[Path]], captions: Sequence[str], out_path: Path,
                 panel_width: int = 960) -> Optional[Path]:
    import cv2
    caps = [cv2.VideoCapture(str(v)) if v and Path(v).is_file() else None for v in videos]
    opened = [c for c in caps if c is not None and c.isOpened()]
    if not opened:
        return None
    fps = opened[0].get(cv2.CAP_PROP_FPS) or 25.0
    w0, h0 = opened[0].get(cv2.CAP_PROP_FRAME_WIDTH) or 1920, opened[0].get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
    panel = (int(panel_width), int(round(panel_width * h0 / w0 / 2) * 2))
    cols, rows = grid_shape(len(videos))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel[0] * cols, panel[1] * rows))
    n = 0
    try:
        while True:
            frames = []
            for c in caps:
                ok, f = (c.read() if c is not None and c.isOpened() else (False, None))
                frames.append(f if ok else None)
            if all(f is None for f in frames):
                break
            writer.write(tile_frames(frames, captions, panel, cols, rows))
            n += 1
    finally:
        writer.release()
        for c in caps:
            if c is not None:
                c.release()
    return out_path if n else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="main.py benchmark-tracker",
                                description="Bandingkan tracker (dan detector TensorRT) pada video yang sama")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--vehicle-model", required=True, type=Path, help="detector .pt (baseline)")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort", "deepsort"], choices=TRACKERS)
    p.add_argument("--engine", type=Path, default=None, help="detector TensorRT .engine (varian tambahan)")
    p.add_argument("--engine-trackers", nargs="+", default=["bytetrack"], choices=TRACKERS,
                   help="tracker yang dipasangkan dengan detector .engine")
    p.add_argument("--true-vehicles", type=int, default=None,
                   help="jumlah kendaraan sebenarnya di main area (hitung manual) -> rasio fragmentasi ID")
    p.add_argument("--max-frames", type=int, default=None, help="batasi frame, mis. 1800 = 60 s @30fps")
    p.add_argument("--config-section", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--track-buffer-sec", type=float, default=2.0)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--vehicle-classes", default=None)
    p.add_argument("--main-mode", default="centroid", choices=["centroid", "bbox"])
    p.add_argument("--plate-model", type=Path, default=None,
                   help="model nomor lambung; bila diisi, deteksi nomor lambung ikut diukur di semua varian")
    p.add_argument("--plate-region", default="0,1,0,1", help="area kabin pada crop kendaraan: y0,y1,x0,x1")
    p.add_argument("--ocr-model-dir", type=Path, default=None, help="folder model OCR nomor lambung (butuh --plate-model)")
    p.add_argument("--plate-grace-sec", type=float, default=1.0)
    p.add_argument("--no-video", dest="video_out", action="store_false", help="jangan render video (lebih cepat)")
    p.add_argument("--panel-width", type=int, default=960, help="lebar tiap panel di video grid")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .pipeline import RunOptions, _to_h264, run

    for label, path in (("video", a.video), ("vehicle model", a.vehicle_model), ("config", a.config)):
        if not path.is_file():
            print(f"[ERROR] {label} tidak ditemukan: {path}", file=sys.stderr)
            return 2
    if a.plate_model is not None and not a.plate_model.is_file():
        print(f"[ERROR] plate model tidak ditemukan: {a.plate_model}", file=sys.stderr)
        return 2
    if a.engine is not None and not a.engine.is_file():
        print(f"[ERROR] engine tidak ditemukan: {a.engine} (buat dengan `main.py export-engine`)", file=sys.stderr)
        return 2

    a.output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variants(a.trackers, a.vehicle_model, a.engine, a.engine_trackers)
    results, videos = [], []
    for v in variants:
        log.info("===== varian %s (%s, detector %s) =====", v.name, v.tracker, v.model.name)
        opt = RunOptions(
            video=a.video, vehicle_model=v.model, config=a.config, output_dir=a.output_dir / v.name,
            config_section=a.config_section, save_video=a.video_out, save_report=True, save_track_cards=False,
            conf=a.conf, imgsz=a.imgsz, device=a.device, tracker=v.tracker, track_buffer_sec=a.track_buffer_sec,
            vehicle_classes=[int(x) for x in a.vehicle_classes.split(",")] if a.vehicle_classes else None,
            main_mode=a.main_mode, max_frames=a.max_frames, h264=False,
            plate_model=a.plate_model, plate_region=tuple(float(v) for v in a.plate_region.split(",")),
            ocr_model_dir=a.ocr_model_dir, plate_grace_sec=a.plate_grace_sec,
        )
        try:
            s = run(opt)
            results.append(result_row(v, s, a.true_vehicles))
            videos.append(Path(s.outputs["annotated_video"]) if "annotated_video" in s.outputs else None)
        except Exception as e:  # noqa: BLE001 - satu varian gagal tidak menghentikan benchmark
            log.error("varian %s gagal: %s", v.name, e)
            results.append(failed_row(v, f"{type(e).__name__}: {e}"))
            videos.append(None)

    paths = write_results(a.output_dir, results)
    print("\n=== HASIL BENCHMARK ===\n" + results_markdown(results))
    for k, pth in paths.items():
        print(f"{k:>5}: {pth}")
    if a.video_out:
        grid = compose_grid(videos, [panel_caption(r) for r in results], a.output_dir / "comparison_grid.mp4",
                            a.panel_width)
        if grid:
            warn = _to_h264(grid)
            print(f" grid: {grid}" + (f"  ({warn})" if warn else ""))
    return 0 if any(r.get("status") == "ok" for r in results) else 1
