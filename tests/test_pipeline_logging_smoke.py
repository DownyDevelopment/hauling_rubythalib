"""Smoke test pipeline penuh dengan tracker palsu: run.log, timing_per_frame.csv, timing_summary.csv terbentuk."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hauling_offline.detector as D  # noqa: E402
from hauling_offline.detector import Detection  # noqa: E402
from hauling_offline.pipeline import RunOptions, run  # noqa: E402
from hauling_offline.trackers import TimingStats  # noqa: E402

CFG = {
    "resolution_area": "1920x1080",
    "main_area_coordinates_list": "[[0,0],[1920,0],[1920,1080],[0,1080]]",
    "speed_area_start_coordinates_list": "[[0,200],[1920,200],[1920,300],[0,300]]",
    "speed_area_final_coordinates_list": "[[0,600],[1920,600],[1920,700],[0,700]]",
    "final_area_coordinates_list": "[[0,900],[1920,900],[1920,1080],[0,1080]]",
    "distance_area_speed": "47", "speed_limit": "60", "travel_direction": "South",
}


class FakeTracker:
    def __init__(self, *a, **k):
        self.names, self.train_args, self.imgsz, self.classes = {0: "vehicle"}, {"single_cls": True}, 640, None
        self.backend, self.tracker_cfg, self.timing, self.f = "pt", None, TimingStats(), 0

    def track(self, frame):
        y = 100 + 30 * self.f          # basis 1080p, video 640x360 -> skala 1/3
        self.f += 1
        self.timing.add(5.0, 6.0)
        if y > 1050:
            return []
        cy = y / 3
        return [Detection(track_id=1, bbox_xyxy=(300, cy - 20, 360, cy + 20), confidence=0.9, class_id=0)]


def test_pipeline_writes_logs_and_timing(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(40):
        vw.write(np.full((360, 640, 3), 80, np.uint8))
    vw.release()
    cfg = tmp_path / "km18.json"
    cfg.write_text(json.dumps(CFG))
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    monkeypatch.setattr(D, "VehicleTracker", FakeTracker)

    out = tmp_path / "out"
    s = run(RunOptions(video=video, vehicle_model=model, config=cfg, output_dir=out, save_video=True,
                       save_report=True, h264=False, vehicle_type_method="size", frame_log_every=1))
    assert s.csv_events == 1 and s.total_frames_read == 40
    log_text = (out / "run.log").read_text(encoding="utf-8")
    assert "[selesai] load_config" in log_text and "[selesai] load_vehicle_model" in log_text
    assert log_text.count("total=") >= 40 and "event_baru=1" in log_text
    assert "entered_start" in log_text and "WAKTU PER TAHAP" in log_text
    rows = list(csv.DictReader(open(out / "timing_per_frame.csv", encoding="utf-8")))
    assert len(rows) == 40 and float(rows[5]["detect"]) == 5.0 and float(rows[5]["track"]) == 1.0
    assert float(rows[5]["render_video"]) > 0
    assert {"load_config", "open_video", "load_vehicle_model", "build_reports"} <= set(s.timing["setup_ms"])
    stages = {r["stage"] for r in s.timing["per_frame"]}
    assert {"detect_track", "area_speed", "render_video", "total_ms"} <= stages
    assert "run_log" in s.outputs and "timing_summary" in s.outputs
