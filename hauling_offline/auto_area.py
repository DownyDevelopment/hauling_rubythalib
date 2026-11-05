"""TASK 1 (otomatis) - Generate config area KM-18 dari lintasan kendaraan di video.

Dipakai bila koordinat kalibrasi resmi belum tersedia. Langkah:
  1. Jalankan model vehicle + tracker pada N detik awal video, kumpulkan titik tengah bbox per track
     (titik yang sama dengan yang dipakai pipeline untuk cek area).
  2. Ambil track yang benar-benar bergerak -> arah gerak utama (vektor u) & arah travel (South/North).
  3. main_area   = trapesium "road" mengikuti tepi kiri/kanan bbox truk (perspektif) dari atas s.d. bawah frame
     speed_start = garis tipis melintang jalan pada 12% panjang lintasan
     speed_final = garis tipis pada 62% panjang lintasan
     final_area  = garis tipis pada 78% panjang lintasan (trigger 1 event per track)
     Tebal garis >= 2.5x perpindahan centroid per frame, supaya tiap kendaraan pasti "menyentuh" garis.
  4. Tulis file config dengan key yang sama persis dengan PRD; pipeline tetap MEMBACA dari config.

PENTING: distance_area_speed tidak bisa diukur dari piksel. Nilai default 47 m (referensi KM-18 di PRD)
hanya benar bila jarak nyata antara pita start & final memang 47 m. File ditandai "auto_generated": true
dan pipeline akan memberi warning.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import point_in_polygon

log = logging.getLogger("hauling.auto_area")

# (frame_index, cx, cy, w, h)
TrackPoint = Tuple[int, float, float, float, float]


@dataclass
class AutoAreaResult:
    config: dict
    direction_vector: Tuple[float, float]
    n_tracks_used: int
    coverage: Dict[str, int] = field(default_factory=dict)   # validasi: berapa track lolos start->finish->final
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Geometri murni (testable tanpa model)
# --------------------------------------------------------------------------- #
def build_areas(
    tracks: Dict[int, Sequence[TrackPoint]],
    frame_size: Tuple[int, int],
    start_frac: float = 0.12,
    finish_frac: float = 0.62,
    final_frac: float = 0.78,
    min_points: int = 8,
    min_travel_ratio: float = 0.15,
    distance_meter: float = 47.0,
    speed_limit: float = 60.0,
) -> AutoAreaResult:
    w, h = frame_size
    warnings: List[str] = []

    # 1) track yang bergerak cukup jauh
    moving = {}
    for tid, pts in tracks.items():
        if len(pts) < min_points:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        disp = arr[-1, 1:3] - arr[0, 1:3]
        if np.hypot(*disp) >= min_travel_ratio * min(w, h):
            moving[tid] = arr
    if not moving:
        raise ValueError("Tidak ada kendaraan yang bergerak cukup jauh di potongan video. "
                         "Perpanjang durasi (seconds) atau pilih bagian video yang ada truk lewat.")
    if len(moving) < 3:
        warnings.append(f"Hanya {len(moving)} track bergerak dipakai; area bisa kurang akurat (disarankan >= 3).")

    # 2) arah gerak utama: rata-rata vektor satuan perpindahan, dibobot panjang
    vecs = np.array([a[-1, 1:3] - a[0, 1:3] for a in moving.values()])
    units = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ref = units[np.argmax(np.linalg.norm(vecs, axis=1))]
    signs = np.sign(units @ ref)                      # kendaraan berlawanan arah dibalik dulu
    u = (units * signs[:, None] * np.linalg.norm(vecs, axis=1, keepdims=True)).sum(0)
    u = u / np.linalg.norm(u)
    if (signs < 0).sum() > (signs > 0).sum():        # mayoritas arah = arah travel
        u = -u
    n_opposite = int((np.sign(units @ u) < 0).sum())
    if n_opposite:
        warnings.append(f"{n_opposite} track bergerak berlawanan arah; diabaikan untuk penentuan arah.")
    direction = "South" if u[1] > 0 else "North"

    # 3) ROAD = trapesium mengikuti tepi bbox truk (perspektif: sempit di jauh, lebar di dekat)
    ax = 1 if abs(u[1]) >= abs(u[0]) else 0          # sumbu utama gerak: 1 = vertikal, 0 = horizontal
    sgn = 1.0 if u[ax] > 0 else -1.0
    if ax == 0:
        direction = "East" if u[0] > 0 else "West"
        warnings.append("Arah gerak dominan horizontal; travel_direction diisi East/West.")
    length, width = (w, h)[ax], (w, h)[1 - ax]
    P = np.concatenate(list(moving.values()))        # frame, cx, cy, w, h
    along = P[:, 1 + ax]
    half_w = P[:, 3 + (1 - ax)] / 2
    lo_e, hi_e = P[:, 2 - ax] - half_w, P[:, 2 - ax] + half_w
    edges_bins = np.linspace(along.min(), along.max(), 9)
    bc, bl, bh = [], [], []
    for b0, b1 in zip(edges_bins[:-1], edges_bins[1:]):
        m = (along >= b0) & (along <= b1)
        if m.sum() >= 3:
            bc.append(along[m].mean()); bl.append(np.percentile(lo_e[m], 5)); bh.append(np.percentile(hi_e[m], 95))
    if len(bc) >= 2:
        fl, fh = np.polyfit(bc, bl, 1), np.polyfit(bc, bh, 1)
    else:
        fl, fh = np.array([0.0, lo_e.min()]), np.array([0.0, hi_e.max()])

    def road_edges(t: float, extra: float = 0.10) -> Tuple[float, float]:
        l, r = np.polyval(fl, t), np.polyval(fh, t)
        mg = extra * (r - l)
        return float(np.clip(l - mg, 0, width - 1)), float(np.clip(r + mg, 0, width - 1))

    def xy(a_: float, c_: float) -> List[int]:
        return [int(round(c_)), int(round(a_))] if ax == 1 else [int(round(a_)), int(round(c_))]

    a0, a1 = 0.0, float(length - 1)
    road = [xy(a0, road_edges(a0)[0]), xy(a0, road_edges(a0)[1]), xy(a1, road_edges(a1)[1]), xy(a1, road_edges(a1)[0])]

    # 4) START / FINISH / FINAL = garis tipis melintang jalan; tebal >= 2.5x perpindahan centroid per frame
    s_all = along * sgn
    s_lo, s_hi = np.percentile(s_all, 3), np.percentile(s_all, 97)
    span = s_hi - s_lo
    if span < 0.2 * length:
        warnings.append("Lintasan pendek (<20% frame); garis start/finish/final berdekatan.")
    steps = np.concatenate([np.abs(np.diff(arr[:, 1 + ax])) / np.maximum(np.diff(arr[:, 0]), 1) for arr in moving.values()])
    thick = max(0.012 * length, 2.5 * float(np.percentile(steps, 95)) if len(steps) else 0.0)

    def band(frac: float) -> List[List[int]]:
        c = sgn * (s_lo + frac * span)
        l, r = road_edges(c, extra=0.15)
        return [xy(c - thick / 2, l), xy(c - thick / 2, r), xy(c + thick / 2, r), xy(c + thick / 2, l)]

    config = {
        "resolution_area": f"{w}x{h}",
        "main_area_coordinates_list": road,
        "speed_area_start_coordinates_list": band(start_frac),
        "speed_area_final_coordinates_list": band(finish_frac),
        "final_area_coordinates_list": band(final_frac),
        "distance_area_speed": distance_meter,
        "speed_limit": speed_limit,
        "travel_direction": direction,
        "auto_generated": True,
        "auto_notes": "Area dibuat otomatis dari lintasan kendaraan. distance_area_speed BELUM dikalibrasi "
                      "terhadap jarak nyata antar pita start-final; konfirmasi ke mentor sebelum hasil speed dipakai.",
    }

    # 5) validasi: berapa track (bergerak searah) melewati start -> finish -> final berurutan
    polys = {k: np.asarray(config[k], dtype=np.float64) for k in (
        "main_area_coordinates_list", "speed_area_start_coordinates_list",
        "speed_area_final_coordinates_list", "final_area_coordinates_list")}
    ok = 0
    same_dir = [a for a in moving.values() if (a[-1, 1:3] - a[0, 1:3]) @ u > 0]
    for a in same_dir:
        seq = []
        for _, cx, cy, _, _ in a:
            if not point_in_polygon((cx, cy), polys["main_area_coordinates_list"]):
                continue
            for name, key in (("start", "speed_area_start_coordinates_list"),
                              ("finish", "speed_area_final_coordinates_list"),
                              ("final", "final_area_coordinates_list")):
                if name not in seq and point_in_polygon((cx, cy), polys[key]):
                    seq.append(name)
        ok += int(seq[:3] == ["start", "finish", "final"])
    coverage = {"tracks_same_direction": len(same_dir), "tracks_start_finish_final": ok}
    if same_dir and ok < len(same_dir):
        warnings.append(f"{len(same_dir) - ok}/{len(same_dir)} track searah tidak melewati start->finish->final "
                        "(mungkin muncul/hilang di tengah frame).")

    return AutoAreaResult(config, (float(u[0]), float(u[1])), len(moving), coverage, warnings)


# --------------------------------------------------------------------------- #
# Kumpulkan lintasan dari video (butuh model)
# --------------------------------------------------------------------------- #
def collect_tracks(
    video: str | Path,
    vehicle_model: str | Path,
    seconds: float = 120.0,
    device: Optional[str] = None,
    conf: float = 0.3,
    stride: int = 1,
) -> Tuple[Dict[int, List[TrackPoint]], Tuple[int, int], float]:
    from .detector import VehicleTracker
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak bisa dibuka: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = VehicleTracker(str(vehicle_model), conf=conf, device=device, track_buffer_frames=int(2 * fps))
    tracks: Dict[int, List[TrackPoint]] = {}
    max_frames = int(seconds * fps)
    fi = -1
    while fi + 1 < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % stride:
            continue
        for d in tracker.track(frame):
            if d.track_id is None:
                continue
            x1, y1, x2, y2 = d.bbox_xyxy
            tracks.setdefault(d.track_id, []).append((fi, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    cap.release()
    return tracks, (w, h), fps


def generate_config(
    video: str | Path,
    vehicle_model: str | Path,
    out_path: str | Path,
    seconds: float = 120.0,
    distance_meter: float = 47.0,
    speed_limit: float = 60.0,
    device: Optional[str] = None,
    stride: int = 1,
) -> AutoAreaResult:
    tracks, size, fps = collect_tracks(video, vehicle_model, seconds, device, stride=stride)
    log.info("lintasan terkumpul: %d track dari %.0f detik video", len(tracks), seconds)
    res = build_areas(tracks, size, distance_meter=distance_meter, speed_limit=speed_limit)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.config, indent=2), encoding="utf-8")
    return res


def draw_preview(frame: np.ndarray, config: dict, tracks: Optional[Dict[int, Sequence[TrackPoint]]] = None,
                 alpha: float = 0.15) -> np.ndarray:
    """Outline area + isi transparan rendah (alpha) + lintasan truk."""
    style = [("main_area_coordinates_list", (255, 160, 0), "road"),
             ("speed_area_start_coordinates_list", (0, 200, 0), "start"),
             ("speed_area_final_coordinates_list", (0, 210, 255), "finish"),
             ("final_area_coordinates_list", (255, 0, 200), "final")]
    ov = frame.copy()
    for k, c, _ in style:
        cv2.fillPoly(ov, [np.asarray(config[k], np.int32).reshape(-1, 1, 2)], c)
    img = cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)
    if tracks:
        for pts in tracks.values():
            arr = np.asarray([[p[1], p[2]] for p in pts], dtype=np.int32)
            if len(arr) > 1:
                cv2.polylines(img, [arr.reshape(-1, 1, 2)], False, (200, 200, 200), 1, cv2.LINE_AA)
    for k, c, name in style:
        p = np.asarray(config[k], np.int32)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], True, c, 2, cv2.LINE_AA)
        if name != "road":
            x, y = p.min(0)
            cv2.putText(img, name, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="main.py auto-area",
                                description="Generate config area KM-18 otomatis dari lintasan kendaraan")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--vehicle-model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="path file config output (.json)")
    p.add_argument("--seconds", type=float, default=120.0, help="durasi video yang dianalisis")
    p.add_argument("--distance", type=float, default=47.0, help="distance_area_speed (meter)")
    p.add_argument("--speed-limit", type=float, default=60.0)
    p.add_argument("--stride", type=int, default=1, help="proses tiap N frame (lebih cepat)")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        res = generate_config(a.video, a.vehicle_model, a.out, a.seconds, a.distance, a.speed_limit, a.device, a.stride)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    print(f"Config tersimpan: {a.out}")
    print(f"Track dipakai: {res.n_tracks_used} | arah: {res.config['travel_direction']} | validasi: {res.coverage}")
    for wmsg in res.warnings:
        print("  - WARNING:", wmsg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
