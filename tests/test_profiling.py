"""Log waktu per tahap (masukan mentor: log setiap fungsi yang berjalan, untuk analisa)."""
import csv
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hauling_offline.cli as cli  # noqa: E402
from hauling_offline.pipeline import RunSummary  # noqa: E402
from hauling_offline.profiling import FRAME_STAGES, StageProfiler, attach_file_log, detach_file_log  # noqa: E402

REQ = ["--video", "v.mp4", "--vehicle-model", "m.pt", "--config", "c.conf", "--output-dir", "out"]


def _fake_frames(prof, n=15):
    for i in range(n):
        t0 = prof.tic()
        prof.start_frame(i, i / 30, t0=t0)
        prof.add("read_frame", 1.0)
        prof.add("detect_track", 10.0 if i < 14 else 50.0)
        prof.add("detect", 9.0)
        prof.add("track", 1.0)
        with prof.stage("render_video"):
            time.sleep(0.001)
        prof.end_frame(detections=2, note="event_baru=1" if i == 3 else "")


def test_setup_stages_recorded():
    prof = StageProfiler()
    t = prof.begin_setup("load_config")
    prof.end_setup("load_config", t)
    with prof.setup("load_vehicle_model"):
        pass
    assert set(prof.setup_ms) == {"load_config", "load_vehicle_model"}


def test_stage_summary_and_files(tmp_path):
    prof = StageProfiler()
    _fake_frames(prof)
    summ = {r["stage"]: r for r in prof.stage_summary(warmup=10)}
    assert summ["read_frame"]["frames"] == 5 and summ["read_frame"]["mean_ms"] == 1.0
    assert summ["detect_track"]["max_ms"] == 50.0 and summ["detect_track"]["mean_ms"] == 18.0
    assert summ["plate_detect"]["frames_active"] == 0
    assert summ["detect"]["share_pct"] is None and summ["detect_track"]["share_pct"] > 0
    slow = prof.slowest_frames(3)
    assert len(slow) == 3 and slow[0]["total_ms"] >= slow[-1]["total_ms"]
    files = prof.write(tmp_path)
    rows = list(csv.DictReader(open(files["timing_per_frame"], encoding="utf-8")))
    assert len(rows) == 15 and set(FRAME_STAGES) <= set(rows[0])
    assert float(rows[0]["total_ms"]) >= float(rows[0]["render_video"]) > 0
    assert Path(files["timing_summary"]).is_file()


def test_run_log_has_per_frame_lines_but_console_stays_info(tmp_path, capsys):
    logging.basicConfig(level=logging.INFO)
    h = attach_file_log(tmp_path / "run.log")
    try:
        prof = StageProfiler(frame_log_every=5)
        _fake_frames(prof)
        logging.getLogger("hauling.pipeline").info("pesan dari modul lama")
    finally:
        detach_file_log(h)
    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    frame_lines = [l for l in text.splitlines() if "frame=" in l and "total=" in l]
    assert len(frame_lines) == 3                       # frame 0, 5, 10
    assert "detect_track=10.0" in frame_lines[0] and "(detect=9.0 track=1.0)" in frame_lines[0]
    assert "pesan dari modul lama" in text
    assert "frame=0 " not in capsys.readouterr().err   # DEBUG per frame tidak membanjiri konsol


def test_cli_logging_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run", lambda opt: seen.setdefault("opt", opt) and RunSummary())
    cli.main(REQ + ["--frame-log-every", "30", "--no-log-file"])
    assert seen["opt"].frame_log_every == 30 and seen["opt"].log_file is False
    assert cli.build_parser().parse_args(REQ).frame_log_every == 1
