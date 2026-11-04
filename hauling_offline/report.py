"""Laporan per ID kendaraan (tracks_report.csv): status finish / gagal finish / nomor lambung per track_id.

events.csv tetap kontrak PRD (1 baris per kendaraan yang mencapai final area, 22 kolom + 2 kolom violation). File ini melengkapinya
dengan SEMUA track, termasuk yang tidak pernah mencapai final, beserta alasannya dalam bahasa yang mudah dibaca.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .speed import SPEED_DECIMALS, TS_DECIMALS
from .tracks import TrackManager, TrackState

TRACK_HEADER = [
    "track_id", "vehicle_type", "vehicle_type_confidence", "vehicle_type_source",
    "first_seen_s", "last_seen_s", "frames_seen", "frames_in_main", "travel_direction",
    "crossed_speed_start_s", "crossed_speed_finish_s", "reached_final_s", "event_id",
    "speed_status", "speed_kmh", "is_overspeed", "violation_status",
    "status_kendaraan", "lambung_status", "lambung_text", "lambung_confidence", "lambung_frames_checked",
    "ocr_reads", "keterangan",
]

# kategori status_kendaraan
FINISH_VALID = "FINISH - speed valid"
FINISH_INVALID = "FINISH - speed tidak valid"
GAGAL_FINISH = "GAGAL FINISH"
LUAR_MAIN = "DI LUAR MAIN AREA"
DETEKSI_SINGKAT = "DETEKSI SINGKAT"

SPEED_REASON = {
    "not_crossed_start": "tidak terekam melewati speed start (muncul/masuk main area setelah garis start)",
    "not_crossed_finish": "tidak terekam melewati speed finish",
    "wrong_order": "urutan garis terbalik (finish sebelum start) - arah berlawanan",
    "started_in_finish": "pertama terlihat sudah di area speed finish",
    "invalid_elapsed_time": "selisih waktu start-finish tidak valid",
}


def _f(v: Optional[float], d: int = TS_DECIMALS) -> str:
    return "" if v is None else f"{v:.{d}f}"


def _position(manager: TrackManager, st: TrackState) -> str:
    """Posisi terakhir track yang gagal finish relatif terhadap garis start/finish/final."""
    if st.speed_start_ts is None and st.speed_end_ts is None:
        return "sebelum garis speed start"
    if st.speed_end_ts is None:
        return "di antara speed start dan speed finish"
    return "di antara speed finish dan final"


def lambung_status(st: TrackState, lambung_format: Optional[str], plate_enabled: bool) -> str:
    if st.reached_final_frame is None:
        return "tidak dicek (belum sampai final)"
    if not plate_enabled:
        return "tidak dicek (model nomor lambung tidak aktif)"
    if st.plate_frames == 0:
        return "tidak dicek (tidak ada frame di zona final)"
    if st.best_plate is None:
        return "tidak terdeteksi"
    text = st.lambung_text(lambung_format)
    if not text:
        return "terdeteksi, karakter tidak terbaca"
    if lambung_format and not re.fullmatch(lambung_format, text):
        return "terbaca sebagian"
    return "terbaca"


def classify_track(manager: TrackManager, st: TrackState, total_frames: int, fps: float,
                   min_frames: int = 5) -> Dict[str, str]:
    """status_kendaraan + keterangan untuk 1 track."""
    notes: List[str] = []
    direction = manager.direction_of(st)
    if st.reached_final_frame is not None:
        sp = st.speed(manager.distance_meter, manager.speed_limit_kmh)
        if sp.valid:
            status = FINISH_VALID
            notes.append(f"melewati start {st.speed_start_ts:.2f}s -> finish {st.speed_end_ts:.2f}s "
                         f"({sp.elapsed_seconds:.2f}s), {sp.speed_kmh:.2f} km/h"
                         + (" OVERSPEED" if sp.is_overspeed else ""))
        else:
            status = FINISH_INVALID
            notes.append(SPEED_REASON.get(sp.status, sp.status))
        notes.append(f"sampai final {st.reached_final_ts:.2f}s")
    elif st.in_main_seen:
        status = GAGAL_FINISH
        ended = st.last_seen_frame >= total_frames - max(2, int(fps * 0.2))
        where = _position(manager, st)
        if ended:
            notes.append(f"video berakhir saat kendaraan masih {where}")
        else:
            how = ("keluar main area" if not st.last_in_main
                   else "track hilang (occlusion / keluar frame / ID berganti)")
            notes.append(f"{how} pada {st.last_seen_ts:.2f}s, posisi terakhir {where}")
            if not st.last_in_main and manager.main_mode == "centroid":
                notes.append("bila truk melaju di sisi kanan jalan, coba --main-mode bbox")
    elif st.frames_seen < min_frames:
        status = DETEKSI_SINGKAT
        notes.append(f"hanya {st.frames_seen} frame (kemungkinan false positive / pecahan ID)")
    else:
        status = LUAR_MAIN
        notes.append("terdeteksi tapi tidak pernah masuk main area (lajur lain / kontraflow / parkir)")
    if direction == manager.up_direction:
        notes.append(f"arah {direction} (berlawanan arah lajur)")
    return {"status": status, "keterangan": "; ".join(notes), "direction": direction}


def build_track_rows(manager: TrackManager, event_ids: Dict[int, str], total_frames: int, fps: float,
                     plate_enabled: bool) -> List[Dict[str, str]]:
    rows = []
    for tid in sorted(manager.tracks):
        st = manager.tracks[tid]
        info = classify_track(manager, st, total_frames, fps)
        sp = st.speed(manager.distance_meter, manager.speed_limit_kmh) if st.in_main_seen else None
        text = st.lambung_text(manager.lambung_format) or ""
        lam = lambung_status(st, manager.lambung_format, plate_enabled)
        if lam == "tidak terdeteksi":
            info["keterangan"] += f"; nomor lambung tidak terdeteksi di {st.plate_frames} frame zona final"
        rows.append({
            "track_id": str(tid),
            "vehicle_type": st.vehicle_type(),
            "vehicle_type_confidence": f"{st.type_confidence():.3f}",
            "vehicle_type_source": st.type_source,
            "first_seen_s": _f(st.first_seen_ts),
            "last_seen_s": _f(st.last_seen_ts),
            "frames_seen": str(st.frames_seen),
            "frames_in_main": str(st.frames_in_main),
            "travel_direction": info["direction"],
            "crossed_speed_start_s": _f(st.speed_start_ts),
            "crossed_speed_finish_s": _f(st.speed_end_ts),
            "reached_final_s": _f(st.reached_final_ts),
            "event_id": event_ids.get(tid, ""),
            "speed_status": sp.status if sp is not None else "",
            "speed_kmh": _f(sp.speed_kmh, SPEED_DECIMALS) if sp is not None and sp.speed_kmh is not None else "",
            "is_overspeed": ("true" if sp.is_overspeed else "false") if sp is not None and sp.valid else "",
            "status_kendaraan": info["status"],
            "lambung_status": lam,
            "lambung_text": text,
            "lambung_confidence": f"{st.best_plate.confidence:.4f}" if st.best_plate is not None else "",
            "lambung_frames_checked": str(st.plate_frames),
            "ocr_reads": f"{st.ocr_reads}/{st.ocr_attempts}" if st.ocr_attempts else "",
            "keterangan": info["keterangan"],
            # terisi sejak track menyentuh speed_final (juga untuk track yang hilang sebelum final)
            "violation_status": st.violation.status if st.violation is not None else "",
        })
    return rows


def write_tracks_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACK_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out
