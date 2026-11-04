"""Output utama: events.csv (kontrak header WAJIB persis sama urutannya)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .speed import SPEED_DECIMALS, TS_DECIMALS
from .tracks import VehicleEvent
from .violation import evaluate_overspeed

CSV_HEADER = [
    "event_id", "source_video", "track_id", "event_timestamp_s", "event_frame_index",
    "travel_direction", "vehicle_confidence", "speed_start_timestamp_s", "speed_end_timestamp_s",
    "elapsed_seconds", "distance_meter", "speed_kmh", "speed_status", "speed_limit_kmh",
    "is_overspeed", "plate_detected", "plate_confidence", "plate_bbox_xyxy", "plate_timestamp_s",
    "plate_text", "measurement_valid", "notes",
    # task violation: ditambahkan di akhir agar urutan 22 kolom lama tidak berubah
    "violation_status", "violation_reason",
]


def fmt(value: Any, decimals: Optional[int] = None) -> str:
    """None -> '' ; bool -> 'true'/'false' ; float -> dibulatkan. Tidak pernah 'None'/'nan'."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.{decimals}f}" if decimals is not None else repr(value)
    return str(value)


def fmt_number(value: Any) -> str:
    """Angka config (47.0 -> '47', 60.5 -> '60.5')."""
    if value is None:
        return ""
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)


def event_to_row(event: VehicleEvent, event_id: str, source_video: str) -> Dict[str, str]:
    sp = event.speed
    pl = event.plate
    vi = event.violation or evaluate_overspeed(sp, event.speed_limit_kmh)
    return {
        "event_id": event_id,
        "source_video": source_video,
        "track_id": str(event.track_id),
        "event_timestamp_s": fmt(event.event_timestamp_s, TS_DECIMALS),
        "event_frame_index": str(event.event_frame_index),
        "travel_direction": event.travel_direction,
        "vehicle_confidence": fmt(float(event.vehicle_confidence), 4),
        "speed_start_timestamp_s": fmt(event.speed_start_timestamp_s, TS_DECIMALS),
        "speed_end_timestamp_s": fmt(event.speed_end_timestamp_s, TS_DECIMALS),
        "elapsed_seconds": fmt(sp.elapsed_seconds, TS_DECIMALS),
        "distance_meter": fmt_number(event.distance_meter),
        "speed_kmh": fmt(sp.speed_kmh, SPEED_DECIMALS),
        "speed_status": sp.status,
        "speed_limit_kmh": fmt_number(event.speed_limit_kmh),
        "is_overspeed": fmt(vi.is_overspeed),   # false (bukan kosong) bila speed tidak valid
        "plate_detected": fmt(pl is not None),
        "plate_confidence": fmt(pl.confidence, 4) if pl else "",
        "plate_bbox_xyxy": json.dumps(list(pl.bbox_xyxy), separators=(",", ":")) if pl else "",
        "plate_timestamp_s": fmt(pl.timestamp_s, TS_DECIMALS) if pl else "",
        "plate_text": event.plate_text or "",  # OCR nomor lambung (kosong bila OCR tidak aktif)
        "measurement_valid": fmt(sp.valid),
        "notes": event.notes or "",
        "violation_status": vi.status,
        "violation_reason": vi.reason,
    }


class EventCsvWriter:
    """Tulis baris per event (flush tiap baris). Gagal membuka file = run gagal."""

    def __init__(self, path: Path, source_video: str, id_prefix: str = "EVT"):
        self.path = Path(path)
        self.source_video = source_video
        self.id_prefix = id_prefix
        self.count = 0
        self._seen_tracks: set = set()
        self._fh = open(self.path, "w", encoding="utf-8", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=CSV_HEADER)
        self._w.writeheader()
        self._fh.flush()

    def write(self, event: VehicleEvent) -> str:
        if event.track_id in self._seen_tracks:  # pertahanan lapis kedua anti-duplikasi
            raise RuntimeError(f"Duplikasi event untuk track_id {event.track_id}")
        self._seen_tracks.add(event.track_id)
        self.count += 1
        event_id = f"{self.id_prefix}-{self.count:05d}"
        self._w.writerow(event_to_row(event, event_id, self.source_video))
        self._fh.flush()
        return event_id

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
