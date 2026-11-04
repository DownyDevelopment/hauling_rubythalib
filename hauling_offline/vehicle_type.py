"""Pembeda jenis kendaraan: Truck vs Light Vehicle (per track, voting antar-frame).

Kenapa perlu modul terpisah: model Hauling_Vehicle_yolov8m_SingleCls_V11 dilatih dengan single_cls=True. Walau
names-nya {0: Truck, 1: Light Vehicle, 2: Motorcycle, 3: Bus, 4: Grader}, SEMUA deteksi keluar sebagai kelas 0
("Truck") - termasuk mobil LV. Jadi kelas dari model itu tidak bisa dipakai untuk membedakan.

Sumber yang didukung (method):
  - "model": kelas langsung dari model vehicle bila model-nya multi-kelas (single_cls=False), mis. model baru
             dari mentor. Otomatis dipakai bila tersedia (method="auto").
  - "coco" : model COCO umum (yolov8n.pt) dijalankan di frame, dicocokkan ke bbox track lewat IoU;
             truck/bus -> Truck, car -> Light Vehicle.
  - "size" : ukuran relatif bbox terhadap lebar main area (jalan) pada ketinggian yang sama. Dump truck mengisi
             sebagian besar lebar jalan; LV jauh lebih kecil. Tanpa model tambahan.
  - "auto" : model (bila multi-kelas) > coco + size (bila model COCO ada) > size.
Hasil akhir per track = voting berbobot semua frame di main area.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TRUCK = "Truck"
LIGHT_VEHICLE = "Light Vehicle"
VEHICLE_TYPES = (TRUCK, LIGHT_VEHICLE)

COCO_TO_TYPE = {"truck": TRUCK, "bus": TRUCK, "car": LIGHT_VEHICLE}


def map_class_name(name: str) -> Optional[str]:
    """Nama kelas model -> 'Truck' / 'Light Vehicle' / None (kelas lain, mis. motorcycle)."""
    n = str(name).strip().lower().replace("_", " ").replace("-", " ")
    if n in ("lv", "light vehicle", "light", "car", "pickup", "pick up", "mobil"):
        return LIGHT_VEHICLE
    if "light" in n:
        return LIGHT_VEHICLE
    if any(k in n for k in ("truck", "dump", "hauler", "hd", "trailer", "bus", "grader")):
        return TRUCK
    return None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def polygon_width_at(poly: np.ndarray, y: float) -> Optional[float]:
    """Lebar horizontal polygon (konveks) pada baris y; None bila y di luar polygon."""
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    xs = []
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        if y1 == y2:
            if y == y1:
                xs += [x1, x2]
            continue
        if min(y1, y2) <= y <= max(y1, y2):
            xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    if len(xs) < 2:
        return None
    return max(xs) - min(xs)


class SizeRule:
    """Truck bila lebar bbox / lebar main area (di y bawah bbox) >= threshold.

    Kalibrasi video KM-18 (V11, 1920x1080): dump truck 0.36-0.47, LV (pickup) 0.20-0.26 -> threshold 0.31.
    Bbox yang masih jauh (bawah bbox < min_y_ratio tinggi frame; tampak depan, sempit) dan bbox yang terpotong
    tepi frame hanya diberi bobot kecil, supaya keputusan didominasi frame saat kendaraan sudah dekat.
    """

    def __init__(self, main_polygon: np.ndarray, threshold: float = 0.31, min_y_ratio: float = 0.18,
                 frame_h: int = 1080, frame_w: Optional[int] = None, weak_weight: float = 0.05):
        self.poly = np.asarray(main_polygon, dtype=float).reshape(-1, 2)
        self.threshold = float(threshold)
        self.min_y = min_y_ratio * frame_h
        self.frame_h = frame_h
        self.frame_w = frame_w if frame_w is not None else frame_h * 16 / 9
        self.weak_weight = weak_weight

    def ratio(self, xyxy: Sequence[float]) -> Optional[float]:
        x1, y1, x2, y2 = xyxy
        ref_y = min(y2, float(self.poly[:, 1].max()))
        road = polygon_width_at(self.poly, ref_y)
        if not road or road <= 0:
            return None
        return (x2 - x1) / road

    def vote(self, xyxy: Sequence[float]) -> Optional[Tuple[str, float]]:
        r = self.ratio(xyxy)
        if r is None:
            return None
        kind = TRUCK if r >= self.threshold else LIGHT_VEHICLE
        x1, y1, x2, y2 = xyxy
        cut = x1 <= 5 or x2 >= self.frame_w - 5 or y2 >= self.frame_h - 5
        if y2 < self.min_y or cut:
            return kind, self.weak_weight
        # bobot makin besar bila jauh dari threshold (yakin), maks 0.6
        w = min(1.0, abs(r - self.threshold) / max(self.threshold * 0.5, 1e-6)) * 0.5 + 0.1
        return kind, round(w, 4)


class CocoTypeModel:
    """Model COCO (yolov8n.pt dsb.) untuk truck/bus vs car, dicocokkan ke bbox track via IoU."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.15, imgsz: int = 640,
                 device: Optional[str] = None, min_iou: float = 0.3):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.cls_map = {int(i): COCO_TO_TYPE[n] for i, n in names.items() if n in COCO_TO_TYPE}
        if not self.cls_map:
            raise ValueError(f"model {model_path} tidak punya kelas truck/bus/car (bukan model COCO?)")
        self.conf, self.imgsz, self.device, self.min_iou = conf, imgsz, device, min_iou
        self.file = str(model_path)

    def predict(self, frame) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        kw = dict(classes=list(self.cls_map), conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device:
            kw["device"] = self.device
        r = self.model.predict(frame, **kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return [(self.cls_map[int(k)], float(c), tuple(float(v) for v in b))
                for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy())]

    def match(self, coco_dets, boxes: Sequence[Sequence[float]]) -> List[Optional[Tuple[str, float]]]:
        out: List[Optional[Tuple[str, float]]] = []
        for b in boxes:
            best, best_iou = None, self.min_iou
            for t, c, cb in coco_dets:
                v = iou(b, cb)
                if v >= best_iou:
                    best, best_iou = (t, c), v
            out.append(best)
        return out


class VehicleTypeEstimator:
    """Gabungan sumber jenis kendaraan. vote_frame() -> {track_id: [(type, weight, source), ...]}."""

    def __init__(self, method: str = "auto", main_polygon: Optional[np.ndarray] = None, frame_h: int = 1080, frame_w: Optional[int] = None,
                 model_names: Optional[Dict[int, str]] = None, model_single_cls: bool = True,
                 coco_model: Optional[str] = None, coco_every: int = 1, size_threshold: float = 0.31,
                 device: Optional[str] = None):
        if method not in ("auto", "model", "coco", "size", "off"):
            raise ValueError(f"vehicle type method tidak dikenal: {method}")
        self.warnings: List[str] = []
        self.class_map: Dict[int, str] = {}
        if model_names and not model_single_cls:
            self.class_map = {int(i): t for i, n in model_names.items() if (t := map_class_name(n))}
        use_model = method in ("auto", "model") and bool(self.class_map)
        if method == "model" and not use_model:
            self.warnings.append("vehicle type 'model' diminta tetapi model vehicle single_cls; pakai 'size'")
        self.size = SizeRule(main_polygon, size_threshold, frame_h=frame_h, frame_w=frame_w) if main_polygon is not None else None
        self.coco: Optional[CocoTypeModel] = None
        if not use_model and method in ("auto", "coco") and coco_model:
            try:
                self.coco = CocoTypeModel(coco_model, device=device)
            except Exception as e:  # noqa: BLE001
                self.warnings.append(f"model COCO untuk jenis kendaraan tidak aktif ({e}); pakai 'size'"[:300])
        self.method = "off" if method == "off" else ("model" if use_model else
                                                    ("coco+size" if self.coco is not None else "size"))
        self.coco_every = max(1, int(coco_every))

    def vote_frame(self, frame, dets, frame_index: int) -> Dict[int, List[Tuple[str, float, str]]]:
        votes: Dict[int, List[Tuple[str, float, str]]] = {}
        dets = [d for d in dets if d.track_id is not None]
        if self.method == "off" or not dets:
            return votes
        if self.method == "model":
            for d in dets:
                t = self.class_map.get(int(d.class_id))
                if t:
                    votes.setdefault(d.track_id, []).append((t, float(d.confidence), "model"))
            return votes
        if self.coco is not None and frame_index % self.coco_every == 0:
            matched = self.coco.match(self.coco.predict(frame), [d.bbox_xyxy for d in dets])
            for d, m in zip(dets, matched):
                if m is not None:
                    votes.setdefault(d.track_id, []).append((m[0], m[1], "coco"))
        if self.size is not None:
            for d in dets:
                v = self.size.vote(d.bbox_xyxy)
                if v is not None:
                    votes.setdefault(d.track_id, []).append((v[0], v[1], "size"))
        return votes
