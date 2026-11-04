"""Renderer video anotasi: polygon area, bbox + track_id + speed, plate box, HUD."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detector import Detection
from .geometry import AreaSet
from .plate import PlateCandidate
from .speed import SpeedResult

# BGR
AREA_STYLE: Dict[str, Tuple[Tuple[int, int, int], str]] = {
    "main": ((0, 200, 0), "MAIN"),
    "speed_start": ((0, 220, 255), "SPEED START"),
    "speed_final": ((0, 140, 255), "SPEED FINAL"),
    "final": ((60, 60, 255), "FINAL"),
}
COLOR_OK = (0, 170, 0)
COLOR_OVER = (0, 0, 230)
COLOR_IDLE = (200, 200, 200)
COLOR_PLATE = (255, 0, 255)
COLOR_LV = (200, 120, 0)     # biru: Light Vehicle (sebelum speed valid)
STATUS_COLOR = {"FINISH - speed valid": (0, 150, 0), "FINISH - speed tidak valid": (0, 140, 255),
                "GAGAL FINISH": (0, 0, 200)}


def _label(img: np.ndarray, text: str, org: Tuple[int, int], color, scale: float, thick: int = 1) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(max(th + 4, org[1]))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_areas(img: np.ndarray, areas: AreaSet, alpha: float = 0.12) -> np.ndarray:
    scale = max(0.4, img.shape[0] / 1080 * 0.7)
    overlay = img.copy()
    for name, polys in areas.polygons.items():
        color, _ = AREA_STYLE[name]
        for p in polys:
            cv2.fillPoly(overlay, [p.reshape(-1, 1, 2)], color)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for name, polys in areas.polygons.items():
        color, text = AREA_STYLE[name]
        for p in polys:
            cv2.polylines(img, [p.reshape(-1, 1, 2)], True, color, max(1, int(round(scale * 3))), cv2.LINE_AA)
            x, y = p.min(axis=0)
            _label(img, text, (x, y), color, scale * 0.8)
    return img


def draw_frame(
    img: np.ndarray,
    areas: AreaSet,
    detections: Sequence[Detection],
    speeds: Dict[int, Optional[SpeedResult]],
    plates: Dict[int, Optional[PlateCandidate]],
    speed_limit_kmh: float,
    ts: float,
    frame_index: int,
    lambung_text: Optional[Dict[int, str]] = None,
    vehicle_types: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """plates = bbox nomor lambung pada frame INI saja; lambung_text = hasil OCR (voting) per track;
    vehicle_types = 'Truck' / 'Light Vehicle' per track (label bbox)."""
    lambung_text = lambung_text or {}
    vehicle_types = vehicle_types or {}
    img = draw_areas(img, areas)
    scale = max(0.4, img.shape[0] / 1080 * 0.7)
    thick = max(1, int(round(scale * 3)))

    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        sp = speeds.get(d.track_id) if d.track_id is not None else None
        kind = vehicle_types.get(d.track_id, "Vehicle") if d.track_id is not None else "Vehicle"
        kind = "Vehicle" if kind == "Unknown" else kind
        if sp is not None and sp.valid:
            color = COLOR_OVER if sp.is_overspeed else COLOR_OK
            txt = f"{kind} [{d.track_id}] ({sp.speed_kmh:.2f} km/h)" + (" OVERSPEED" if sp.is_overspeed else "")
        else:
            color = COLOR_IDLE if d.track_id is None else (COLOR_LV if kind == "Light Vehicle" else COLOR_OK)
            txt = f"{kind} [{d.track_id if d.track_id is not None else '-'}] {d.confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        _label(img, txt, (x1, y1), color, scale * 0.8)
        text = lambung_text.get(d.track_id) if d.track_id is not None else None
        if text:
            _label(img, f"No Lambung: {text}", (x1, y1 - int(32 * scale)), COLOR_PLATE, scale * 0.8)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), thick * 2, color, -1)

        pl = plates.get(d.track_id) if d.track_id is not None else None
        if pl is not None:
            px1, py1, px2, py2 = pl.bbox_xyxy
            cv2.rectangle(img, (px1, py1), (px2, py2), COLOR_PLATE, thick)
            _label(img, f"lambung {pl.confidence:.2f}", (px1, py2 + int(25 * scale)), COLOR_PLATE, scale * 0.6)

    _label(img, f"Speed Limit: {speed_limit_kmh:g} km/h", (15, int(40 * scale) + 10), (40, 40, 40), scale)
    _label(img, f"t={ts:8.3f}s  frame={frame_index}", (15, int(80 * scale) + 20), (40, 40, 40), scale * 0.8)
    return img


def draw_track_card(crop: np.ndarray, row: Dict[str, str], ts: float, width: int = 420) -> np.ndarray:
    """Foto 1 kendaraan + ringkasan status (untuk laporan per ID)."""
    h, w = crop.shape[:2]
    scale = width / max(1, w)
    img = cv2.resize(crop, (width, max(1, int(h * scale))))
    lines = [
        (f"ID {row['track_id']} | {row['vehicle_type']} | {row['travel_direction']}", (40, 40, 40)),
        (row["status_kendaraan"] + (f" | {row['speed_kmh']} km/h" if row["speed_kmh"] else ""),
         STATUS_COLOR.get(row["status_kendaraan"], (90, 90, 90))),
        ("No Lambung: " + (row["lambung_text"] or "-") + f" ({row['lambung_status']})", COLOR_PLATE),
        (f"foto t={ts:.2f}s", (90, 90, 90)),
    ]
    pad = 26
    panel = np.full((pad * len(lines) + 8, width, 3), 255, np.uint8)
    for i, (t, c) in enumerate(lines):
        cv2.putText(panel, t[:60], (6, pad * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return np.vstack([panel, img])


class SafeVideoWriter:
    """VideoWriter yang tidak menghentikan run bila gagal (cukup warning)."""

    def __init__(self, path: Path, fps: float, size: Tuple[int, int]):
        self.path = Path(path)
        self.error: Optional[str] = None
        self.frames = 0
        self._vw = None
        try:
            vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not vw.isOpened():
                raise RuntimeError("cv2.VideoWriter tidak bisa dibuka (codec mp4v)")
            self._vw = vw
        except Exception as e:  # noqa: BLE001
            self.error = f"video_writer_open_failed: {e}"

    @property
    def ok(self) -> bool:
        return self._vw is not None and self.error is None

    def write(self, frame: np.ndarray) -> None:
        if not self.ok:
            return
        try:
            self._vw.write(frame)
            self.frames += 1
        except Exception as e:  # noqa: BLE001
            self.error = f"video_writer_write_failed: {e}"

    def release(self) -> None:
        if self._vw is not None:
            self._vw.release()
