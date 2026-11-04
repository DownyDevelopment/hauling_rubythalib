"""Violation Overspeed: aturan pelanggaran yang dievaluasi dari hasil pengukuran speed.

    IF measurement_valid is false   -> is_overspeed=false, violation_status=NOT_EVALUATED
    ELSE IF speed_kmh > limit       -> is_overspeed=true,  violation_status=OVERSPEED,
                                       violation_reason="Speed exceeds configured limit"
    ELSE                            -> is_overspeed=false, violation_status=NORMAL

Batas = speed_limit di config (KM-18: 60 km/jam). Tidak ada toleransi tambahan; perubahan toleransi hanya lewat
parameter config baru + persetujuan eksplisit. Dibandingkan dengan speed_kmh yang sudah dibulatkan (nilai di CSV).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .speed import SpeedResult

NOT_EVALUATED = "NOT_EVALUATED"
NORMAL = "NORMAL"
OVERSPEED = "OVERSPEED"
OVERSPEED_REASON = "Speed exceeds configured limit"


@dataclass(frozen=True)
class ViolationResult:
    status: str
    is_overspeed: bool
    reason: str = ""


def evaluate_overspeed(speed: Optional[SpeedResult], speed_limit_kmh: float) -> ViolationResult:
    if speed is None or not speed.valid or speed.speed_kmh is None:
        return ViolationResult(NOT_EVALUATED, False)
    if speed.speed_kmh > speed_limit_kmh:          # tepat sama dengan batas = bukan pelanggaran
        return ViolationResult(OVERSPEED, True, OVERSPEED_REASON)
    return ViolationResult(NORMAL, False)
