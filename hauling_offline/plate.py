"""TASK 4 - Deteksi Nomor Lambung (kode unit di atas kaca depan kabin, mis. "AEK 0078").

Di kode & CSV dipakai istilah "plate" agar sesuai kontrak header PRD (plate_detected, plate_bbox_xyxy, ...),
tetapi objek yang dideteksi adalah NOMOR LAMBUNG, bukan plat nomor polisi.
Task minimum = deteksi LOKASI. Pembacaan teks (OCR, ekstensi) ada di ocr.py dan mengisi plate_text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PlateCandidate:
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]   # relatif terhadap FRAME (bukan crop)
    timestamp_s: float
    frame_index: int


def better_plate(current: Optional[PlateCandidate], new: Optional[PlateCandidate]) -> Optional[PlateCandidate]:
    """Confidence tertinggi menang; bila sama, timestamp lebih akhir menang."""
    if new is None:
        return current
    if current is None:
        return new
    if new.confidence > current.confidence:
        return new
    if new.confidence == current.confidence and new.timestamp_s >= current.timestamp_s:
        return new
    return current


def select_best(candidates: Sequence[PlateCandidate]) -> Optional[PlateCandidate]:
    best: Optional[PlateCandidate] = None
    for c in candidates:
        best = better_plate(best, c)
    return best


def crop_with_offset(frame: np.ndarray, xyxy: Sequence[float], pad: float = 0.0):
    """Potong bbox kendaraan dari frame. Return (crop, (ox, oy)) atau (None, None) bila crop kosong."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1 = int(max(0, np.floor(x1 - pw)))
    y1 = int(max(0, np.floor(y1 - ph)))
    x2 = int(min(w, np.ceil(x2 + pw)))
    y2 = int(min(h, np.ceil(y2 + ph)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1)


def sub_region(xyxy: Sequence[float], region: Sequence[float]) -> Tuple[float, float, float, float]:
    """Ambil sub-bagian bbox kendaraan. region = (y0, y1, x0, x1) dalam fraksi 0..1 dari bbox.

    Contoh: (0.4, 1.0, 0.25, 1.0) = area kabin truk yang menghadap kamera, tempat nomor lambung di atas kaca.
    """
    y0, y1, x0, x1 = region
    if not (0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1):
        raise ValueError(f"plate region tidak valid: {region} (format y0,y1,x0,x1 dalam 0..1)")
    bx1, by1, bx2, by2 = xyxy
    w, h = bx2 - bx1, by2 - by1
    return bx1 + x0 * w, by1 + y0 * h, bx1 + x1 * w, by1 + y1 * h


def merge_parts(boxes: Sequence[Tuple[float, float, float, float, float]], gap_ratio: float = 1.0):
    """Gabungkan potongan nomor lambung (blok 'huruf' + blok 'angka') menjadi 1 bbox.

    boxes: list (x1, y1, x2, y2, conf). Anchor = box confidence tertinggi; box lain digabung bila
    sejajar vertikal (overlap tinggi >= 30%) dan jarak horizontal <= gap_ratio x tinggi anchor.
    Return (x1, y1, x2, y2, conf_anchor) atau None.
    """
    if not boxes:
        return None
    order = sorted(boxes, key=lambda b: b[4], reverse=True)
    ax1, ay1, ax2, ay2, aconf = order[0]
    merged = [ax1, ay1, ax2, ay2]
    changed = True
    rest = list(order[1:])
    while changed:
        changed = False
        for b in list(rest):
            x1, y1, x2, y2, _ = b
            h_ref = max(1e-6, min(merged[3] - merged[1], y2 - y1))
            v_overlap = max(0.0, min(merged[3], y2) - max(merged[1], y1)) / h_ref
            h_gap = max(0.0, max(merged[0], x1) - min(merged[2], x2))
            if v_overlap >= 0.3 and h_gap <= gap_ratio * (ay2 - ay1):
                merged = [min(merged[0], x1), min(merged[1], y1), max(merged[2], x2), max(merged[3], y2)]
                rest.remove(b)
                changed = True
    return merged[0], merged[1], merged[2], merged[3], aconf


def lambung_position_ok(box: Sequence[float], vehicle_xyxy: Sequence[float],
                        max_rel_y: float = 0.8, min_rel_y: float = 0.3) -> bool:
    """Nomor lambung ada di ATAS kaca depan kabin (~50-70% tinggi bbox truk yang menghadap kamera).

    Tolak kandidat di bumper (plat polisi/TNKB, rel_y > max_rel_y) dan di bak/muatan (rel_y < min_rel_y).
    0 = atas bbox, 1 = bawah bbox. Model angka_huruf mendeteksi SEMUA blok huruf/angka, termasuk TNKB.
    """
    _, vy1, _, vy2 = vehicle_xyxy
    rel = ((box[1] + box[3]) / 2 - vy1) / max(vy2 - vy1, 1e-6)
    return min_rel_y <= rel <= max_rel_y


def lambung_shape_ok(box: Sequence[float], frame_w: int, min_width_ratio: float = 24 / 1920,
                     aspect: Tuple[float, float] = (1.5, 12.0)) -> bool:
    """Tolak kandidat terlalu kecil (truk masih jauh, tak terbaca) atau bentuknya bukan pita teks."""
    w, h = box[2] - box[0], box[3] - box[1]
    if h <= 0 or w < min_width_ratio * frame_w:
        return False
    return aspect[0] <= w / h <= aspect[1]


class PlateDetector:
    """Adapter model nomor lambung (Ultralytics YOLO). Dijalankan pada crop kendaraan, bukan full frame.

    merge='best'  -> ambil 1 box confidence tertinggi (model yang langsung mendeteksi plat utuh)
    merge='union' -> gabungkan potongan plat (model 'huruf'/'angka' seperti detection_hauling_angka_huruf_V1)
    merge='auto'  -> 'union' bila nama kelas berisi huruf/angka/character, selain itu 'best'
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        region: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        merge: str = "auto",
        classes: Optional[Sequence[int]] = None,
        max_rel_y: float = 0.8,                  # buang kandidat di bumper (plat polisi)
        min_rel_y: float = 0.3,                  # buang kandidat di bak / tulisan bodi atas
        min_width_ratio: float = 24 / 1920,      # lebar minimum relatif lebar frame
        aspect: Tuple[float, float] = (1.5, 12.0),
    ):
        from ultralytics import YOLO  # import lokal agar unit test tidak butuh ultralytics
        from .detector import model_train_args
        self.model = YOLO(model_path)
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.train_args = model_train_args(self.model)
        self.conf = conf
        self.imgsz = int(imgsz or self.train_args.get("imgsz") or 640)
        self.device = device
        self.region = tuple(float(v) for v in region)
        sub_region((0, 0, 1, 1), self.region)  # validasi
        if merge == "auto":
            lowered = " ".join(str(n).lower() for n in self.names.values())
            merge = "union" if any(k in lowered for k in ("huruf", "angka", "character")) else "best"
        if merge not in ("best", "union"):
            raise ValueError("merge harus auto/best/union")
        self.merge = merge
        self.classes = list(classes) if classes is not None else None
        self.max_rel_y, self.min_rel_y = max_rel_y, min_rel_y
        self.min_width_ratio, self.aspect = min_width_ratio, aspect

    def detect_on_crops(
        self,
        frame: np.ndarray,
        vehicle_boxes: Sequence[Sequence[float]],
        timestamp_s: float,
        frame_index: int,
    ) -> List[Optional[PlateCandidate]]:
        """Satu hasil per vehicle box: kandidat plat terbaik pada crop itu (atau None)."""
        crops, offsets, idx_map = [], [], []
        for i, box in enumerate(vehicle_boxes):
            crop, off = crop_with_offset(frame, sub_region(box, self.region))
            if crop is not None:
                crops.append(crop)
                offsets.append(off)
                idx_map.append(i)

        out: List[Optional[PlateCandidate]] = [None] * len(vehicle_boxes)
        if not crops:
            return out

        kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False, classes=self.classes)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(crops, **kwargs)

        for res, (ox, oy), i in zip(results, offsets, idx_map):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            confs = res.boxes.conf.cpu().numpy()
            xyxy = res.boxes.xyxy.cpu().numpy()
            # koordinat FRAME, lalu buang kandidat di posisi bumper (plat polisi, bukan nomor lambung)
            raw = [(float(b[0]) + ox, float(b[1]) + oy, float(b[2]) + ox, float(b[3]) + oy, float(c))
                   for b, c in zip(xyxy, confs)]
            raw = [r for r in raw if lambung_position_ok(r, vehicle_boxes[i], self.max_rel_y, self.min_rel_y)]
            if not raw:
                continue
            if self.merge == "union":
                m = merge_parts(raw)
                picks = [m] if m else []
            else:
                picks = raw
            cands = [
                PlateCandidate(
                    confidence=round(c, 4),
                    bbox_xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                )
                for x1, y1, x2, y2, c in picks
                if lambung_shape_ok((x1, y1, x2, y2), frame.shape[1], self.min_width_ratio, self.aspect)
            ]
            out[i] = select_best(cands)
        return out
