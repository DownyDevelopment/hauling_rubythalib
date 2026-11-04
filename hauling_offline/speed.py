"""TASK 3 - Logic Speed Detection: kalkulator speed murni (tanpa I/O).

speed_kmh = (distance_meter / elapsed_seconds) x 3.6
Timestamp WAJIB berasal dari clock video (frame_index / fps), bukan time.time().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Status yang mungkin muncul di kolom speed_status
STATUS_VALID = "valid"
STATUS_NOT_CROSSED_START = "not_crossed_start"
STATUS_NOT_CROSSED_FINISH = "not_crossed_finish"
STATUS_WRONG_ORDER = "wrong_order"
STATUS_STARTED_IN_FINISH = "started_in_finish"
STATUS_INVALID_ELAPSED = "invalid_elapsed_time"

TS_DECIMALS = 6      # presisi timestamp & elapsed di CSV
SPEED_DECIMALS = 2   # presisi speed_kmh (sesuai acceptance criteria)


def video_timestamp(frame_index: int, fps: float) -> float:
    """Clock video: frame_index / fps (deterministik untuk input yang sama)."""
    if fps <= 0:
        raise ValueError("fps harus > 0")
    return frame_index / fps


@dataclass(frozen=True)
class SpeedResult:
    status: str
    elapsed_seconds: Optional[float]
    speed_kmh: Optional[float]
    is_overspeed: Optional[bool]

    @property
    def valid(self) -> bool:
        return self.status == STATUS_VALID


def compute_speed(
    start_ts: Optional[float],
    end_ts: Optional[float],
    distance_meter: float,
    speed_limit_kmh: float,
    started_in_finish: bool = False,
    finish_before_start: bool = False,
) -> SpeedResult:
    """Validasi urutan crossing lalu hitung speed. Speed None bila tidak valid."""
    if start_ts is None:
        return SpeedResult(STATUS_NOT_CROSSED_START, None, None, None)
    if started_in_finish:
        return SpeedResult(STATUS_STARTED_IN_FINISH, None, None, None)
    if end_ts is None:
        return SpeedResult(STATUS_NOT_CROSSED_FINISH, None, None, None)
    if finish_before_start or end_ts < start_ts:
        return SpeedResult(STATUS_WRONG_ORDER, None, None, None)

    elapsed = round(end_ts - start_ts, TS_DECIMALS)
    if elapsed <= 0:
        return SpeedResult(STATUS_INVALID_ELAPSED, None, None, None)

    speed = round((distance_meter / elapsed) * 3.6, SPEED_DECIMALS)
    return SpeedResult(STATUS_VALID, elapsed, speed, speed > speed_limit_kmh)
