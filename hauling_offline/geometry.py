"""TASK 1 - Setup Area: normalisasi polygon & pengecekan titik di dalam area.

Titik kendaraan = titik tengah bounding box (konsisten untuk semua area).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import KM18Config, Polygon

AREA_ORDER = ("main", "speed_start", "speed_final", "final")


def scale_polygon(poly: Polygon, base_res: Tuple[int, int], frame_res: Tuple[int, int]) -> np.ndarray:
    """Skala proporsional dari resolution_area (w,h) ke resolusi frame (w,h). Output int32 (N,2)."""
    bw, bh = base_res
    fw, fh = frame_res
    arr = np.asarray(poly, dtype=np.float64)
    if (bw, bh) != (fw, fh):
        arr = arr * np.array([fw / bw, fh / bh], dtype=np.float64)
    return np.round(arr).astype(np.int32)


def bbox_center(xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_polygon(point: Tuple[float, float], poly: np.ndarray) -> bool:
    """True bila titik di dalam ATAU tepat di garis polygon (ray casting, tanpa dependensi cv2)."""
    x, y = float(point[0]), float(point[1])
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    inside = False
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        # titik tepat di sisi polygon -> dianggap menyentuh
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def _seg_intersect(p1, p2, q1, q2) -> bool:
    """True bila segmen p1-p2 dan q1-q2 berpotongan (termasuk menyentuh)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and \
            min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9

    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and on_seg(p1, p2, q1)) or (o2 == 0 and on_seg(p1, p2, q2)) or \
        (o3 == 0 and on_seg(q1, q2, p1)) or (o4 == 0 and on_seg(q1, q2, p2))


def segment_touches_polygon(p0: Tuple[float, float], p1: Tuple[float, float], poly: np.ndarray) -> bool:
    """True bila centroid sekarang (p1) di dalam polygon ATAU lintasan p0->p1 memotong polygon.

    Penting untuk area tipis (mis. garis start/finish KM-18 setebal ~10 px): pada fps rendah centroid bisa
    'melompati' garis di antara dua frame; lintasan antar-frame tetap terdeteksi memotong garis.
    """
    if point_in_polygon(p1, poly):
        return True
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    return any(_seg_intersect(p0, p1, pts[i], pts[(i + 1) % n]) for i in range(n))


def bbox_polygon_coverage(xyxy: Sequence[float], poly: np.ndarray) -> float:
    """Luas (bbox ∩ polygon konveks) / luas bbox."""
    import cv2
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(rect, np.asarray(poly, dtype=np.float32).reshape(-1, 2))
    return float(max(0.0, min(1.0, inter / area)))


@dataclass
class AreaSet:
    """Semua polygon area yang sudah diskalakan ke resolusi frame video."""
    polygons: Dict[str, List[np.ndarray]]
    frame_res: Tuple[int, int]

    @classmethod
    def from_config(cls, cfg: KM18Config, frame_res: Tuple[int, int]) -> "AreaSet":
        polys = {
            name: [scale_polygon(p, cfg.resolution, frame_res) for p in cfg.areas[name]]
            for name in AREA_ORDER
        }
        return cls(polygons=polys, frame_res=frame_res)

    def contains(self, area: str, point: Tuple[float, float]) -> bool:
        return any(point_in_polygon(point, p) for p in self.polygons[area])

    def touches(self, area: str, point: Tuple[float, float], prev: Optional[Tuple[float, float]] = None) -> bool:
        """contains(), ditambah: lintasan prev->point memotong area (crossing antar-frame)."""
        if prev is None:
            return self.contains(area, point)
        return any(segment_touches_polygon(prev, point, p) for p in self.polygons[area])

    def coverage(self, area: str, xyxy: Sequence[float]) -> float:
        """Fraksi luas bbox yang berada di dalam area (0..1); polygon area diasumsikan konveks."""
        return max((bbox_polygon_coverage(xyxy, p) for p in self.polygons[area]), default=0.0)

    def areas_of(self, point: Tuple[float, float]) -> Dict[str, bool]:
        return {name: self.contains(name, point) for name in AREA_ORDER}
