"""Unit test violation overspeed (aturan + integrasi state machine, events.csv, tracks_report)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hauling_offline.config import config_from_dict  # noqa: E402
from hauling_offline.csv_writer import CSV_HEADER, EventCsvWriter  # noqa: E402
from hauling_offline.detector import Detection  # noqa: E402
from hauling_offline.geometry import AreaSet  # noqa: E402
from hauling_offline.report import build_track_rows  # noqa: E402
from hauling_offline.speed import SpeedResult, compute_speed, video_timestamp  # noqa: E402
from hauling_offline.tracks import TrackManager  # noqa: E402
from hauling_offline.violation import (  # noqa: E402
    NORMAL, NOT_EVALUATED, OVERSPEED, OVERSPEED_REASON, evaluate_overspeed)

LIMIT = 60.0
BASE = {
    "resolution_area": "1920x1080",
    "main_area_coordinates_list": "[[0,0],[1920,0],[1920,1080],[0,1080]]",
    "speed_area_start_coordinates_list": "[[0,200],[1920,200],[1920,300],[0,300]]",
    "speed_area_final_coordinates_list": "[[0,600],[1920,600],[1920,700],[0,700]]",
    "final_area_coordinates_list": "[[0,900],[1920,900],[1920,1080],[0,1080]]",
    "distance_area_speed": "47",
    "speed_limit": "60",
    "travel_direction": "South",
}


def _manager():
    cfg = config_from_dict(BASE)
    return TrackManager(AreaSet.from_config(cfg, (1920, 1080)), cfg.distance_meter, cfg.speed_limit_kmh)


def _det(tid, cy):
    return Detection(track_id=tid, bbox_xyxy=(900, cy - 40, 1000, cy + 40), confidence=0.9, class_id=0)


def _drive(m, ys, fps=10.0, tid=1):
    events = []
    for i, y in enumerate(ys):
        m.observe([_det(tid, y)], i, video_timestamp(i, fps))
        events += m.collect_events()
    return events


# ----------------------------- aturan ----------------------------------- #
def _valid(speed):
    return SpeedResult("valid", 2.0, speed, None)


def test_rule_above_equal_below_limit():
    over = evaluate_overspeed(_valid(60.01), LIMIT)
    assert (over.status, over.is_overspeed, over.reason) == (OVERSPEED, True, OVERSPEED_REASON)
    exact = evaluate_overspeed(_valid(60.0), LIMIT)
    assert (exact.status, exact.is_overspeed, exact.reason) == (NORMAL, False, "")
    assert evaluate_overspeed(_valid(45.5), LIMIT).status == NORMAL


def test_rule_invalid_measurement_not_evaluated():
    for status in ("not_crossed_start", "not_crossed_finish", "wrong_order", "started_in_finish",
                   "invalid_elapsed_time"):
        v = evaluate_overspeed(SpeedResult(status, None, None, None), LIMIT)
        assert v.status == NOT_EVALUATED and v.is_overspeed is False and v.reason == ""
    assert evaluate_overspeed(None, LIMIT).status == NOT_EVALUATED


def test_rule_uses_rounded_speed():
    sp = compute_speed(0.0, 2.82, 47, LIMIT)          # 47 / 2.82 * 3.6 = 60.0 setelah dibulatkan
    assert sp.speed_kmh == 60.0 and evaluate_overspeed(sp, LIMIT).status == NORMAL


# ----------------------------- state machine ---------------------------- #
def test_marked_at_speed_final_then_recorded_at_final():
    m = _manager()
    ys = [100, 250, 400, 650, 800, 950]                # start f1, finish f3, final f5
    _drive(m, ys[:4])                                  # berhenti tepat setelah menyentuh speed_final
    st = m.tracks[1]
    assert st.violation is not None and st.violation.status == OVERSPEED and not st.event_written
    marked = st.violation
    ev = []
    for i in range(4, len(ys)):
        m.observe([_det(1, ys[i])], i, video_timestamp(i, 10.0))
        ev += m.collect_events()
    assert len(ev) == 1 and ev[0].violation is marked   # objek yang sama: tidak dievaluasi ulang
    assert m.collect_events() == []                    # satu event (dan satu violation) per track


def test_normal_speed_track():
    m = _manager()
    ev = _drive(m, list(range(100, 1001, 10)))         # start f10, finish f50 -> 4.0 s -> 42.3 km/jam
    assert len(ev) == 1 and ev[0].speed.speed_kmh == 42.3 and ev[0].violation.status == NORMAL


def test_no_start_is_not_evaluated():
    ev = _drive(_manager(), [500, 650, 800, 950])
    assert ev[0].speed.status == "not_crossed_start" and ev[0].violation.status == NOT_EVALUATED


def test_never_touches_speed_final_evaluated_at_final():
    cfg = dict(BASE, speed_area_final_coordinates_list="[[0,600],[100,600],[100,700],[0,700]]")  # di luar lintasan
    c = config_from_dict(cfg)
    m = TrackManager(AreaSet.from_config(c, (1920, 1080)), 47, 60)
    ev = _drive(m, [100, 250, 400, 650, 800, 950])
    assert ev[0].speed.status == "not_crossed_finish" and ev[0].violation.status == NOT_EVALUATED


# ----------------------------- output ----------------------------------- #
def test_events_csv_violation_columns(tmp_path):
    m = _manager()
    for i, y in enumerate([250, 650, 950]):                       # track 1: overspeed
        m.observe([_det(1, y), _det(2, 650 + i * 150)], i, video_timestamp(i, 10.0))  # track 2: tanpa start
    evs = m.collect_events()
    path = tmp_path / "events.csv"
    with EventCsvWriter(path, "sample.mp4", "sample") as w:
        for e in evs:
            w.write(e)
    raw = path.read_text(encoding="utf-8")
    assert CSV_HEADER[-2:] == ["violation_status", "violation_reason"] and len(CSV_HEADER) == 24
    assert raw.splitlines()[0] == ",".join(CSV_HEADER)
    rows = {r["track_id"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert rows["1"]["is_overspeed"] == "true" and rows["1"]["violation_status"] == OVERSPEED
    assert rows["1"]["violation_reason"] == OVERSPEED_REASON
    assert rows["2"]["measurement_valid"] == "false"
    assert rows["2"]["is_overspeed"] == "false"                    # bukan kosong
    assert rows["2"]["violation_status"] == NOT_EVALUATED and rows["2"]["violation_reason"] == ""


def test_tracks_report_shows_violation_even_if_lost_before_final():
    m = _manager()
    assert _drive(m, [100, 250, 400, 650, 800]) == []             # overspeed, hilang sebelum final
    rows = build_track_rows(m, {}, total_frames=100, fps=10.0, plate_enabled=False)
    assert rows[0]["violation_status"] == OVERSPEED and rows[0]["status_kendaraan"] == "GAGAL FINISH"
