"""State machine per track_id — sumber kebenaran untuk area, speed, arah, plat, dan event.

Alur per frame (dipanggil oleh pipeline):
    1. observe(detections, frame_index, ts)  -> update area/arah; return track yang butuh deteksi plat
    2. add_plate(track_id, candidate)        -> simpan kandidat plat terbaik
    3. collect_events(...)                   -> event untuk track yang baru mencapai final area (maks. 1x)

Violation (overspeed) ditandai SEKALI per track saat menyentuh speed_final, lalu dicatat bersama event saat
mencapai final area. Track yang tidak pernah menyentuh speed_final dievaluasi saat event ditulis (NOT_EVALUATED).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .detector import Detection
from .geometry import AreaSet, bbox_center
from .speed import SpeedResult, compute_speed
from .violation import ViolationResult, evaluate_overspeed

if TYPE_CHECKING:  # plate.py ditulis di Task 4; import saat dipakai saja
    from .plate import PlateCandidate

log = logging.getLogger("hauling.tracks")


@dataclass
class TrackState:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    in_main_seen: bool = False
    first_main_centroid: Optional[Tuple[float, float]] = None
    last_centroid: Optional[Tuple[float, float]] = None
    last_main_frame: Optional[int] = None
    max_confidence: float = 0.0
    last_confidence: float = 0.0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    frames_seen: int = 0
    frames_in_main: int = 0
    last_any_centroid: Optional[Tuple[float, float]] = None
    last_in_main: bool = False
    # jenis kendaraan (Truck / Light Vehicle): voting antar-frame
    type_votes: Dict[str, float] = field(default_factory=dict)
    type_source: str = ""
    # speed
    speed_start_ts: Optional[float] = None
    speed_start_frame: Optional[int] = None
    speed_end_ts: Optional[float] = None
    speed_end_frame: Optional[int] = None
    started_in_finish: bool = False
    finish_before_start: bool = False
    violation: Optional[ViolationResult] = None   # ditandai saat speed_final, tidak pernah ditimpa
    # plate
    best_plate: Optional["PlateCandidate"] = None
    plate_errors: int = 0
    plate_frames: int = 0               # jumlah frame model nomor lambung dijalankan (hanya di zona final)
    # OCR nomor lambung (opsional): voting teks antar-frame
    ocr_scores: Dict[str, float] = field(default_factory=dict)
    ocr_reads: int = 0
    ocr_attempts: int = 0
    # event
    reached_final_frame: Optional[int] = None
    reached_final_ts: Optional[float] = None
    event_written: bool = False
    notes: List[str] = field(default_factory=list)

    def lambung_text(self, preferred: Optional[str] = None) -> Optional[str]:
        """Teks hasil voting. Bila `preferred` (regex format nomor lambung, mis. 3 huruf + 4 angka) diberikan,
        bacaan yang lengkap sesuai format didahulukan daripada bacaan terpotong ("EK 008", "B 3")."""
        if not self.ocr_scores:
            return None
        items = list(self.ocr_scores.items())
        if preferred:
            full = [kv for kv in items if re.fullmatch(preferred, kv[0])]
            items = full or items
        return max(items, key=lambda kv: (round(kv[1], 6), len(kv[0]), kv[0]))[0]

    def vehicle_type(self) -> str:
        if not self.type_votes:
            return "Unknown"
        return max(self.type_votes.items(), key=lambda kv: (round(kv[1], 6), kv[0]))[0]

    def type_confidence(self) -> float:
        tot = sum(self.type_votes.values())
        return round(self.type_votes.get(self.vehicle_type(), 0.0) / tot, 3) if tot else 0.0

    def speed(self, distance_meter: float, speed_limit_kmh: float) -> SpeedResult:
        return compute_speed(
            self.speed_start_ts, self.speed_end_ts, distance_meter, speed_limit_kmh,
            started_in_finish=self.started_in_finish, finish_before_start=self.finish_before_start,
        )


@dataclass
class VehicleEvent:
    """Satu baris events.csv (sebelum diformat)."""
    track_id: int
    event_timestamp_s: float
    event_frame_index: int
    travel_direction: str
    vehicle_confidence: float
    speed_start_timestamp_s: Optional[float]
    speed_end_timestamp_s: Optional[float]
    distance_meter: float
    speed_limit_kmh: float
    speed: SpeedResult
    plate: Optional["PlateCandidate"]
    notes: str = ""
    plate_text: str = ""            # hasil OCR nomor lambung (kosong bila OCR tidak aktif / tidak terbaca)
    vehicle_type: str = "Unknown"   # Truck / Light Vehicle (dicatat di notes)
    violation: Optional[ViolationResult] = None


class TrackManager:
    def __init__(
        self,
        areas: AreaSet,
        distance_meter: float,
        speed_limit_kmh: float,
        down_direction: str = "South",
        min_move_ratio: float = 0.02,
        max_gap_frames: int = 10,
        grace_frames: int = 0,
        lambung_format: Optional[str] = None,
        plate_zone: str = "final",
        main_mode: str = "centroid",
        main_min_coverage: float = 0.3,
    ):
        self.areas = areas
        self.distance_meter = distance_meter
        self.speed_limit_kmh = speed_limit_kmh
        self.down_direction = down_direction
        self.up_direction = {"South": "North", "North": "South"}.get(down_direction, "Unknown")
        self.min_move_px = max(1.0, min_move_ratio * areas.frame_res[1])
        self.tracks: Dict[int, TrackState] = {}
        self.detections_without_id = 0
        self.max_gap_frames = max_gap_frames   # crossing antar-frame hanya dihitung bila jeda observasi <= ini
        # Event ditulis grace_frames SETELAH menyentuh final (timestamp event tetap saat final), supaya nomor
        # lambung masih bisa dibaca saat truk paling dekat ke kamera. 0 = langsung ditulis.
        self.grace_frames = max(0, int(grace_frames))
        self.lambung_format = lambung_format
        # "final" (default, sesuai arahan mentor): nomor lambung + OCR hanya dijalankan mulai kendaraan menyentuh
        # final area s.d. akhir grace window. "main" = perilaku lama (sepanjang main area).
        if plate_zone not in ("final", "main"):
            raise ValueError(f"plate_zone harus 'final' atau 'main', bukan {plate_zone!r}")
        self.plate_zone = plate_zone
        # Keanggotaan main area: "centroid" (default PRD) atau "bbox" (fraksi luas bbox di main >= main_min_coverage).
        # "bbox" menolong truk yang melaju di sisi kanan jalan: centroid-nya sedikit keluar garis main area
        # sehingga garis speed finish / final tidak pernah terekam.
        if main_mode not in ("centroid", "bbox"):
            raise ValueError(f"main_mode harus 'centroid' atau 'bbox', bukan {main_mode!r}")
        self.main_mode, self.main_min_coverage = main_mode, float(main_min_coverage)

    def in_main(self, det: Detection) -> bool:
        if self.main_mode == "bbox":
            return self.areas.coverage("main", det.bbox_xyxy) >= self.main_min_coverage
        return self.areas.contains("main", bbox_center(det.bbox_xyxy))

    # ------------------------------------------------------------------ #
    def direction_of(self, st: TrackState) -> str:
        """Arah sederhana dari perubahan centroid (sumbu y frame)."""
        if st.first_main_centroid is None or st.last_centroid is None:
            return "Unknown"
        dy = st.last_centroid[1] - st.first_main_centroid[1]
        if abs(dy) < self.min_move_px:
            return "Unknown"
        return self.down_direction if dy > 0 else self.up_direction

    # ------------------------------------------------------------------ #
    def observe(self, detections: Sequence[Detection], frame_index: int, ts: float) -> List[Detection]:
        """Update state semua track. Return deteksi yang perlu dicek nomor lambungnya:
        plate_zone="final" -> hanya track yang sudah menyentuh final area dan event-nya belum ditulis (grace window);
        plate_zone="main"  -> semua track di main area yang event-nya belum ditulis."""
        need_plate: List[Detection] = []
        for det in detections:
            if det.track_id is None:
                self.detections_without_id += 1
                continue
            tid = det.track_id
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(tid, frame_index, frame_index, first_seen_ts=ts)
            st.last_seen_frame, st.last_seen_ts = frame_index, ts
            st.frames_seen += 1
            st.last_confidence = det.confidence
            st.max_confidence = max(st.max_confidence, det.confidence)
            c = bbox_center(det.bbox_xyxy)
            st.last_any_centroid = c
            if st.event_written:
                continue
            st.last_in_main = self.in_main(det)
            if not st.last_in_main:
                # sudah lewat final tapi keluar main area saat grace window -> nomor lambung tetap boleh dibaca
                if self.plate_zone == "final" and st.reached_final_frame is not None:
                    need_plate.append(det)
                continue  # hanya kendaraan di main area yang diproses
            st.frames_in_main += 1

            prev = st.last_centroid if (st.last_main_frame is not None and
                                        frame_index - st.last_main_frame <= self.max_gap_frames) else None
            st.last_centroid, st.last_main_frame = c, frame_index
            # centroid di dalam area ATAU lintasan dari frame sebelumnya memotong area (area tipis/fps rendah)
            in_start = self.areas.touches("speed_start", c, prev)
            in_finish = self.areas.touches("speed_final", c, prev)
            in_final = self.areas.touches("final", c, prev)

            if not st.in_main_seen:
                st.in_main_seen = True
                st.first_main_centroid = c
                if in_finish and not in_start:
                    st.started_in_finish = True
                    st.notes.append("first_seen_in_speed_final")
                    log.info("track %s first seen inside speed-final area (f=%d)", tid, frame_index)

            if in_start and st.speed_start_ts is None:
                st.speed_start_ts, st.speed_start_frame = ts, frame_index
                if st.speed_end_ts is not None:
                    st.finish_before_start = True
                log.info("track %s entered_start t=%.3fs f=%d", tid, ts, frame_index)

            if in_finish and st.speed_end_ts is None:
                st.speed_end_ts, st.speed_end_frame = ts, frame_index
                if st.speed_start_ts is None:
                    st.finish_before_start = True
                log.info("track %s entered_finish t=%.3fs f=%d", tid, ts, frame_index)
                self._mark_violation(st)

            if in_final and st.reached_final_frame is None:
                st.reached_final_frame, st.reached_final_ts = frame_index, ts
                log.info("track %s reached_final t=%.3fs f=%d -> mulai deteksi nomor lambung", tid, ts, frame_index)

            if self.plate_zone == "main" or st.reached_final_frame is not None:
                need_plate.append(det)
        return need_plate

    def add_plate(self, track_id: int, cand: Optional["PlateCandidate"]) -> None:
        from .plate import better_plate
        st = self.tracks.get(track_id)
        if st is not None and not st.event_written:
            st.plate_frames += 1
            st.best_plate = better_plate(st.best_plate, cand)

    def add_type_vote(self, track_id: int, vehicle_type: Optional[str], weight: float = 1.0, source: str = "") -> None:
        """1 vote jenis kendaraan ('Truck' / 'Light Vehicle') untuk track; None diabaikan."""
        st = self.tracks.get(track_id)
        if st is None or not vehicle_type or weight <= 0:
            return
        st.type_votes[vehicle_type] = st.type_votes.get(vehicle_type, 0.0) + float(weight)
        if source and source not in st.type_source.split("+"):
            st.type_source = f"{st.type_source}+{source}" if st.type_source else source

    def add_ocr(self, track_id: int, text: Optional[str], conf: float = 1.0) -> None:
        """Tambahkan 1 vote hasil baca nomor lambung (None = percobaan gagal)."""
        st = self.tracks.get(track_id)
        if st is None or st.event_written:
            return
        st.ocr_attempts += 1
        if text:
            st.ocr_scores[text] = st.ocr_scores.get(text, 0.0) + conf
            st.ocr_reads += 1

    def mark_plate_error(self, track_id: int) -> None:
        st = self.tracks.get(track_id)
        if st is not None:
            st.plate_errors += 1

    def collect_events(self, frame_index: Optional[int] = None, flush: bool = False) -> List[VehicleEvent]:
        """Event untuk track yang sudah mencapai final area dan belum pernah ditulis (anti-duplikasi).

        Dengan grace_frames > 0, event dirilis setelah frame_index - final_frame >= grace_frames, atau saat flush
        (akhir video). event_timestamp_s / event_frame_index tetap = saat menyentuh final area.
        """
        events: List[VehicleEvent] = []
        for st in self.tracks.values():
            if st.event_written or st.reached_final_frame is None:
                continue
            if (self.grace_frames and not flush and frame_index is not None
                    and frame_index - st.reached_final_frame < self.grace_frames):
                continue
            st.event_written = True
            sp = st.speed(self.distance_meter, self.speed_limit_kmh)
            self._mark_violation(st, sp)
            notes = [f"vehicle_type={st.vehicle_type()}"] + list(st.notes)
            if st.plate_errors:
                notes.append(f"plate_detector_errors={st.plate_errors}")
            if st.ocr_attempts:
                notes.append(f"ocr_reads={st.ocr_reads}/{st.ocr_attempts}")
            events.append(VehicleEvent(
                track_id=st.track_id,
                event_timestamp_s=st.reached_final_ts,
                event_frame_index=st.reached_final_frame,
                travel_direction=self.direction_of(st),
                vehicle_confidence=st.max_confidence,
                speed_start_timestamp_s=st.speed_start_ts,
                speed_end_timestamp_s=st.speed_end_ts,
                distance_meter=self.distance_meter,
                speed_limit_kmh=self.speed_limit_kmh,
                speed=sp,
                plate=st.best_plate,
                notes=";".join(notes),
                plate_text=st.lambung_text(self.lambung_format) or "",
                vehicle_type=st.vehicle_type(),
                violation=st.violation,
            ))
            log.info("track %s event_written t=%.3fs status=%s speed=%s violation=%s", st.track_id,
                     st.reached_final_ts, sp.status, sp.speed_kmh, st.violation.status)
        return events

    def _mark_violation(self, st: TrackState, sp: Optional[SpeedResult] = None) -> None:
        """Satu violation per track: hanya diisi bila belum ada, tidak pernah ditimpa."""
        if st.violation is not None:
            return
        sp = sp or st.speed(self.distance_meter, self.speed_limit_kmh)
        st.violation = evaluate_overspeed(sp, self.speed_limit_kmh)
        log.info("track %s violation=%s speed=%s limit=%s", st.track_id, st.violation.status, sp.speed_kmh,
                 self.speed_limit_kmh)

    # ------------------------------------------------------------------ #
    def lost_before_final(self) -> List[int]:
        """Track yang masuk main area tetapi tidak pernah mencapai final area (tidak ditulis paksa)."""
        return sorted(t for t, s in self.tracks.items() if s.in_main_seen and not s.event_written)

    def live_speed(self, track_id: int) -> Optional[SpeedResult]:
        st = self.tracks.get(track_id)
        if st is None:
            return None
        return st.speed(self.distance_meter, self.speed_limit_kmh)
