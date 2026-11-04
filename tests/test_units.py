"""Unit test sintetis: config, polygon, state machine, speed, plate selector, CSV writer."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hauling_offline.config import ConfigError, config_from_dict, load_config, parse_polygons  # noqa: E402
from hauling_offline.csv_writer import CSV_HEADER, EventCsvWriter  # noqa: E402
from hauling_offline.detector import Detection  # noqa: E402
from hauling_offline.geometry import AreaSet, point_in_polygon, scale_polygon  # noqa: E402
from hauling_offline.plate import PlateCandidate, select_best  # noqa: E402
from hauling_offline.speed import compute_speed, video_timestamp  # noqa: E402
from hauling_offline.tracks import TrackManager  # noqa: E402

# Area sintetis 1920x1080, kendaraan bergerak dari atas ke bawah (South)
BASE = {
    "resolution_area": "1920x1080",
    "main_area_coordinates_list": "[[0,0],[1920,0],[1920,1080],[0,1080]]",
    "speed_area_start_coordinates_list": "[[0,200],[1920,200],[1920,300],[0,300]]",
    "speed_area_final_coordinates_list": "[[0,600],[1920,600],[1920,700],[0,700]]",
    "final_area_coordinates_list": "[[0,900],[1920,900],[1920,1080],[0,1080]]",
    "distance_area_speed": "47",
    "speed_limit": "60",
    "travel_direction": "South",
    "rtsp_url": "rtsp://user:secret@10.0.0.1/stream",  # harus diabaikan
}


# ----------------------------- config ----------------------------------- #
def test_config_reads_km18_values():
    cfg = config_from_dict(BASE)
    assert cfg.resolution == (1920, 1080)
    assert cfg.distance_meter == 47.0
    assert cfg.speed_limit_kmh == 60.0
    assert cfg.travel_direction == "South"
    assert len(cfg.areas["main"]) == 1 and len(cfg.areas["main"][0]) == 4
    assert "secret" not in json.dumps(cfg.safe_summary())


def test_config_missing_required_key():
    bad = dict(BASE)
    del bad["distance_area_speed"]
    with pytest.raises(ConfigError, match="distance_area_speed"):
        config_from_dict(bad)


@pytest.mark.parametrize("value,n_polys", [
    ("[[0,0],[10,0],[10,10]]", 1),
    ([[0, 0], [10, 0], [10, 10]], 1),
    ("[[[0,0],[10,0],[10,10]],[[20,20],[30,20],[30,30]]]", 2),
    ("[(0,0),(10,0),(10,10)]", 1),
    ("[0,0,10,0,10,10]", 1),
])
def test_parse_polygons_formats(value, n_polys):
    assert len(parse_polygons(value, "k")) == n_polys


@pytest.mark.parametrize("value", ["[[0,0],[1,1]]", "[[0,'a'],[1,1],[2,2]]", "__import__('os')", "[]"])
def test_parse_polygons_rejects_invalid(value):
    with pytest.raises(ConfigError):
        parse_polygons(value, "k")


def test_load_config_ini_and_json(tmp_path):
    ini = tmp_path / "km18.ini"
    ini.write_text("[KM18]\n" + "\n".join(f"{k} = {v}" for k, v in BASE.items()), encoding="utf-8")
    js = tmp_path / "km18.json"
    js.write_text(json.dumps({"cameras": {"KM18": BASE}}), encoding="utf-8")
    assert load_config(ini).distance_meter == 47.0
    assert load_config(js).speed_limit_kmh == 60.0


# ----------------------------- polygon ---------------------------------- #
def test_polygon_scaler_same_and_other_resolution():
    poly = [(0, 0), (1920, 0), (960, 1080)]
    same = scale_polygon(poly, (1920, 1080), (1920, 1080))
    assert same.tolist() == [[0, 0], [1920, 0], [960, 1080]]
    half = scale_polygon(poly, (1920, 1080), (1280, 720))
    assert half.tolist() == [[0, 0], [1280, 0], [640, 720]]


def test_point_in_polygon_inside_edge_outside():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert point_in_polygon((5, 5), sq)
    assert point_in_polygon((10, 5), sq)       # tepat di garis = menyentuh
    assert not point_in_polygon((15, 5), sq)


# ----------------------------- speed ------------------------------------ #
def test_speed_formula_and_rounding():
    r = compute_speed(1.0, 4.0, 47, 60)
    assert r.valid and r.elapsed_seconds == 3.0 and r.speed_kmh == round(47 / 3 * 3.6, 2) == 56.4
    assert r.is_overspeed is False
    fast = compute_speed(0.0, 2.0, 47, 60)
    assert fast.speed_kmh == 84.6 and fast.is_overspeed is True


@pytest.mark.parametrize("start,end,kw,status", [
    (None, None, {}, "not_crossed_start"),
    (None, 3.0, {}, "not_crossed_start"),
    (1.0, None, {}, "not_crossed_finish"),
    (3.0, 1.0, {}, "wrong_order"),
    (1.0, 3.0, {"finish_before_start": True}, "wrong_order"),
    (1.0, 3.0, {"started_in_finish": True}, "started_in_finish"),
    (2.0, 2.0, {}, "invalid_elapsed_time"),
])
def test_speed_invalid_status(start, end, kw, status):
    r = compute_speed(start, end, 47, 60, **kw)
    assert r.status == status and r.speed_kmh is None and r.elapsed_seconds is None and not r.valid


def test_video_timestamp_deterministic():
    assert video_timestamp(75, 25.0) == 3.0
    with pytest.raises(ValueError):
        video_timestamp(1, 0)


# ----------------------------- state machine ---------------------------- #
def _manager():
    cfg = config_from_dict(BASE)
    return TrackManager(AreaSet.from_config(cfg, (1920, 1080)), cfg.distance_meter, cfg.speed_limit_kmh)


def _det(tid, cy, conf=0.9):
    return Detection(track_id=tid, bbox_xyxy=(900, cy - 40, 1000, cy + 40), confidence=conf, class_id=0)


def _run(manager, path_y, fps=10.0, tid=1):
    events = []
    for i, y in enumerate(path_y):
        manager.observe([_det(tid, y)], i, video_timestamp(i, fps))
        events += manager.collect_events()
    return events


def test_start_finish_final_one_event():
    m = _manager()
    ys = [100, 250, 400, 650, 800, 950, 1000, 1050, 1060]   # start f1, finish f3, final f5..
    ev = _run(m, ys)
    assert len(ev) == 1
    e = ev[0]
    assert e.speed.valid and e.speed_start_timestamp_s == 0.1 and e.speed_end_timestamp_s == 0.3
    assert e.speed.speed_kmh == round(47 / 0.2 * 3.6, 2)
    assert e.event_frame_index == 5 and e.travel_direction == "South"


def test_finish_without_start_invalid():
    ev = _run(_manager(), [500, 650, 800, 950])
    assert len(ev) == 1 and ev[0].speed.status == "not_crossed_start" and ev[0].speed.speed_kmh is None


def test_reverse_order_invalid():
    m = _manager()
    ev = _run(m, [800, 650, 400, 250, 100])  # bergerak ke atas: finish dulu baru start, tidak ke final
    assert ev == [] and m.tracks[1].finish_before_start
    assert m.direction_of(m.tracks[1]) == "North"
    assert m.live_speed(1).status == "wrong_order"


def test_started_in_finish_invalid():
    ev = _run(_manager(), [650, 800, 950])
    assert ev[0].speed.status == "not_crossed_start"


def test_detection_without_track_id_ignored():
    m = _manager()
    m.observe([Detection(None, (900, 910, 1000, 990), 0.9, 0)], 0, 0.0)
    assert m.collect_events() == [] and m.detections_without_id == 1


def test_lost_before_final():
    m = _manager()
    _run(m, [100, 250, 400])
    assert m.lost_before_final() == [1]


def test_outside_main_area_ignored():
    cfg = dict(BASE, main_area_coordinates_list="[[0,0],[500,0],[500,1080],[0,1080]]")
    c = config_from_dict(cfg)
    m = TrackManager(AreaSet.from_config(c, (1920, 1080)), 47, 60)
    assert _run(m, [250, 650, 950]) == []          # bbox center x=950 di luar main area


# ----------------------------- plate ------------------------------------ #
def test_plate_selector_highest_conf_and_tie_later():
    a = PlateCandidate(0.5, (1, 1, 2, 2), 1.0, 10)
    b = PlateCandidate(0.8, (3, 3, 4, 4), 2.0, 20)
    c = PlateCandidate(0.8, (5, 5, 6, 6), 3.0, 30)
    assert select_best([a, b]) is b
    assert select_best([c, b, a]) is c              # sama conf -> timestamp lebih akhir
    assert select_best([]) is None


def test_event_without_plate_still_written():
    m = _manager()
    ev = _run(m, [250, 650, 950])
    assert len(ev) == 1 and ev[0].plate is None


# ----------------------------- CSV -------------------------------------- #
def test_csv_contract(tmp_path):
    m = _manager()
    for i, y in enumerate([250, 650, 950]):
        m.observe([_det(1, y), _det(2, y + 1)], i, video_timestamp(i, 10.0))
        if i == 1:
            m.add_plate(1, PlateCandidate(0.77, (910, 600, 960, 620), 0.1, 1))
    evs = m.collect_events()
    path = tmp_path / "events.csv"
    with EventCsvWriter(path, "sample.mp4", "sample") as w:
        ids = [w.write(e) for e in evs]
        with pytest.raises(RuntimeError):
            w.write(evs[0])  # duplikasi ditolak

    raw = path.read_text(encoding="utf-8")
    assert raw.splitlines()[0] == ",".join(CSV_HEADER)
    assert "None" not in raw and "nan" not in raw
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2 and len(set(ids)) == 2
    r1 = next(r for r in rows if r["track_id"] == "1")
    r2 = next(r for r in rows if r["track_id"] == "2")
    assert r1["plate_detected"] == "true" and r1["plate_bbox_xyxy"] == "[910,600,960,620]"
    assert r1["plate_text"] == "" and r1["measurement_valid"] == "true" and r1["speed_kmh"] == "1692.00"
    assert r1["is_overspeed"] == "true" and r1["distance_meter"] == "47" and r1["speed_limit_kmh"] == "60"
    assert r2["plate_detected"] == "false" and r2["plate_confidence"] == "" and r2["plate_bbox_xyxy"] == ""


# ----------------------------- model adapters (pure helpers) ------------ #
from hauling_offline.detector import auto_vehicle_classes  # noqa: E402
from hauling_offline.plate import merge_parts, sub_region  # noqa: E402

HAULING_NAMES = {0: "Truck", 1: "Light Vehicle", 2: "Motorcycle", 3: "Bus", 4: "Grader"}


def test_auto_classes_single_cls_not_filtered():
    # Model Hauling_Vehicle_*_SingleCls: names 5 kelas tapi semua deteksi keluar sebagai kelas 0
    assert auto_vehicle_classes(HAULING_NAMES, {"single_cls": True}) is None
    assert auto_vehicle_classes({0: "Vehicle", 1: "Person"}, {}) == [0]
    coco = {i: f"c{i}" for i in range(80)}
    coco.update({2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    assert auto_vehicle_classes(coco, {}) == [2, 3, 5, 7]


def test_merge_parts_union_huruf_angka():
    huruf = (100, 50, 140, 70, 0.9)
    angka = (145, 52, 200, 71, 0.8)
    far = (400, 300, 420, 320, 0.5)          # jauh -> tidak digabung
    assert merge_parts([angka, far, huruf]) == (100, 50, 200, 71, 0.9)
    assert merge_parts([]) is None


def test_sub_region():
    assert sub_region((0, 0, 100, 200), (0.5, 1.0, 0.0, 1.0)) == (0, 100, 100, 200)
    with pytest.raises(ValueError):
        sub_region((0, 0, 1, 1), (0.8, 0.2, 0, 1))


# ----------------------------- OCR nomor lambung (ekstensi) ------------- #
from hauling_offline.ocr import clean_chars  # noqa: E402


def test_clean_chars_drops_narrow_and_low_conf_edges():
    # (x1, x2, label, cls_conf) — kasus nyata dari blok angka "0078" + karakter palsu di tepi
    chars = [(0, 3, "0", 0.40), (2, 11, "0", 1.0), (9, 18, "0", 1.0), (16, 24, "7", 1.0), (22, 31, "8", 0.99), (30, 33, "1", 0.66)]
    assert "".join(c[2] for c in clean_chars(chars)) == "0078"
    huruf = [(0, 8, "A", 0.48), (9, 17, "A", 1.0), (17, 25, "E", 1.0), (24, 33, "K", 1.0)]
    assert "".join(c[2] for c in clean_chars(huruf)) == "AEK"
    assert clean_chars([]) == []


def test_ocr_votes_to_csv(tmp_path):
    m = _manager()
    for i, y in enumerate([250, 650, 950]):
        m.observe([_det(1, y)], i, video_timestamp(i, 10.0))
        if i < 2:
            m.add_plate(1, PlateCandidate(0.8, (910, 600, 960, 620), i / 10, i))
            m.add_ocr(1, "AEK 0078", 0.99)
        if i == 2:
            m.add_ocr(1, "AEK 0076", 0.6)   # 1 bacaan salah kalah voting
            m.add_ocr(1, None)
    ev = m.collect_events()
    assert ev[0].plate_text == "AEK 0078" and "ocr_reads=3/4" in ev[0].notes
    path = tmp_path / "e.csv"
    with EventCsvWriter(path, "v.mp4") as w:
        w.write(ev[0])
    row = next(csv.DictReader(path.open(encoding="utf-8")))
    assert row["plate_text"] == "AEK 0078"


def test_config_content_sniffing_and_empty(tmp_path):
    # file bernama .json tapi isinya KEY=VALUE -> tetap terbaca
    f = tmp_path / "km18.json"
    f.write_text("\n".join(f"{k}={v}" for k, v in BASE.items()), encoding="utf-8")
    assert load_config(f).distance_meter == 47.0
    # .json berisi INI
    f.write_text("[KM18]\n" + "\n".join(f"{k} = {v}" for k, v in BASE.items()), encoding="utf-8")
    assert load_config(f).speed_limit_kmh == 60.0
    # file kosong -> pesan jelas
    f.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="kosong"):
        load_config(f)
    # isi tanpa key KM-18
    f.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ConfigError, match="key KM-18"):
        load_config(f)


# ----------------------------- auto area (Task 1 otomatis) -------------- #
from hauling_offline.auto_area import build_areas  # noqa: E402


def _synthetic_tracks(n=4, down=True):
    tracks = {}
    for t in range(n):
        pts = []
        for i in range(60):
            y = 50 + i * 16 if down else 1010 - i * 16
            x = 900 + t * 40 + (i * 2 if down else -i * 2)
            pts.append((i, float(x), float(y), 260.0, 300.0))
        tracks[t + 1] = pts
    return tracks


def test_auto_area_south_and_order():
    res = build_areas(_synthetic_tracks(), (1920, 1080))
    cfg = config_from_dict(res.config)          # harus lolos validasi loader PRD
    assert cfg.travel_direction == "South" and cfg.auto_generated
    assert cfg.distance_meter == 47.0 and cfg.speed_limit_kmh == 60.0
    ys = {k: np.mean([p[1] for p in cfg.areas[k][0]]) for k in ("speed_start", "speed_final", "final")}
    assert ys["speed_start"] < ys["speed_final"] < ys["final"]
    assert res.coverage["tracks_start_finish_final"] == res.coverage["tracks_same_direction"] == 4


def test_auto_area_north_and_state_machine():
    res = build_areas(_synthetic_tracks(down=False), (1920, 1080))
    cfg = config_from_dict(res.config)
    assert cfg.travel_direction == "North"
    m = TrackManager(AreaSet.from_config(cfg, (1920, 1080)), 47, 60, down_direction="South")
    evs = []
    for i, (fi, x, y, w, h) in enumerate(_synthetic_tracks(1, down=False)[1]):
        m.observe([Detection(1, (x - w / 2, y - h / 2, x + w / 2, y + h / 2), 0.9, 0)], fi, fi / 25)
        evs += m.collect_events()
    assert len(evs) == 1 and evs[0].speed.valid


def test_auto_area_no_moving_tracks():
    parked = {1: [(i, 500.0, 500.0, 100.0, 100.0) for i in range(30)]}
    with pytest.raises(ValueError, match="bergerak"):
        build_areas(parked, (1920, 1080))


def test_auto_area_road_trapezoid_perspective():
    # truk makin lebar saat mendekati kamera (bawah) -> road harus melebar ke bawah
    tracks = {t + 1: [(i, 900.0 + t * 30 + i * 3, 40.0 + i * 16, 100.0 + i * 5, 120.0 + i * 5) for i in range(60)]
              for t in range(4)}
    cfg = build_areas(tracks, (1920, 1080)).config
    road = np.array(cfg["main_area_coordinates_list"])
    top_w, bottom_w = road[1, 0] - road[0, 0], road[2, 0] - road[3, 0]
    assert road[0, 1] == 0 and road[2, 1] == 1079 and bottom_w > top_w
    for k in ("speed_area_start_coordinates_list", "speed_area_final_coordinates_list", "final_area_coordinates_list"):
        band = np.array(cfg[k])
        assert band[:, 1].max() - band[:, 1].min() < 60          # garis tipis


# ----------------------------- config resmi KM-18 (.conf) & crossing ----- #
KM18_CONF = """[CONFIG]
resolution = 1920,1080
travel_direction = South
conf_model = 0.25
iou_model = 0.7
imgsz_model = 640
max_det_model = 100
distance_area_speed = [47]
speed_limit = 60
main_area_coordinates_list = [[842, 3], [1010, 3], [1728, 1080], [490, 1080]]
speed_area_start_coordinates_list = [[769, 101],[1213, 101],[1213, 110],[769,110]]
speed_area_final_coordinates_list = [[640, 479],[1518, 479],[1518, 489],[640, 489]]
final_area_coordinates_list = [[609, 553],[1565, 553],[1565, 563],[609, 563]]
resolution_area = 1920,1080

[DATABASE]
db_password = secret
"""


def test_official_conf_format(tmp_path):
    f = tmp_path / "_track_km18_01_V2.conf"
    f.write_text(KM18_CONF, encoding="utf-8")
    cfg = load_config(f)
    assert cfg.distance_meter == 47.0 and cfg.speed_limit_kmh == 60.0 and cfg.resolution == (1920, 1080)
    assert cfg.model_params == {"conf": 0.25, "iou": 0.7, "imgsz": 640.0, "max_det": 100.0}
    assert "secret" not in json.dumps(cfg.safe_summary())


def test_thin_line_crossing_between_frames(tmp_path):
    """Garis start/finish KM-18 cuma ~10 px: truk 30 px/frame tidak pernah TEPAT di dalam garis,
    tapi lintasan antar-frame memotong garis -> tetap tercatat (speed valid)."""
    f = tmp_path / "km18.conf"
    f.write_text(KM18_CONF, encoding="utf-8")
    cfg = load_config(f)
    m = TrackManager(AreaSet.from_config(cfg, (1920, 1080)), cfg.distance_meter, cfg.speed_limit_kmh)
    ys = [20 + 30 * i for i in range(25)]           # 20, 50, 80, 110? -> cek tidak ada y di 101..110 kecuali kebetulan
    ys = [y for y in ys if not (101 <= y <= 110 or 479 <= y <= 489 or 553 <= y <= 563)]
    evs = []
    for i, y in enumerate(ys):
        m.observe([_det(1, y)], i, video_timestamp(i, 15.0)); evs += m.collect_events()
    assert len(evs) == 1 and evs[0].speed.valid


# ----------------------------- filter kandidat nomor lambung ------------ #
from hauling_offline.plate import lambung_position_ok, lambung_shape_ok  # noqa: E402


def test_lambung_filters_reject_bumper_plate_and_tiny():
    truck = (1099, 163, 1405, 576)                       # bbox truk (frame PRD)
    lambung = (1286, 396, 1345, 417)                     # "AEK 0078" di atas kaca (rel y ~0.59)
    tnkb = (1220, 545, 1290, 565)                        # plat polisi di bumper (rel y ~0.95)
    body = (1150, 180, 1250, 200)                        # tulisan di bak (rel y ~0.06)
    assert lambung_position_ok(lambung, truck) and not lambung_position_ok(tnkb, truck)
    assert not lambung_position_ok(body, truck)
    assert lambung_shape_ok(lambung, 1920)
    assert not lambung_shape_ok((10, 10, 20, 15), 1920)  # terlalu kecil (truk masih jauh)
    assert not lambung_shape_ok((0, 0, 40, 60), 1920)    # bentuk tegak, bukan pita teks


# ----------------------------- grace window & voting format ------------- #
def test_grace_window_delays_event_but_keeps_final_timestamp():
    cfg = config_from_dict(BASE)
    m = TrackManager(AreaSet.from_config(cfg, (1920, 1080)), 47, 60, grace_frames=3)
    ys = [250, 650, 950, 980, 1000, 1010, 1020]
    evs = []
    for i, y in enumerate(ys):
        m.observe([_det(1, y)], i, video_timestamp(i, 10.0))
        evs += [(i, e) for e in m.collect_events(i)]
    assert len(evs) == 1
    released_at, ev = evs[0]
    assert ev.event_frame_index == 2 and released_at == 5   # tercatat saat final (f2), ditulis 3 frame kemudian
    m2 = TrackManager(AreaSet.from_config(cfg, (1920, 1080)), 47, 60, grace_frames=100)
    for i, y in enumerate([250, 650, 950]):
        m2.observe([_det(1, y)], i, video_timestamp(i, 10.0))
        assert m2.collect_events(i) == []
    assert len(m2.collect_events(flush=True)) == 1           # akhir video -> tetap ditulis


def test_lambung_text_prefers_full_format():
    m = _manager()
    m.observe([_det(1, 250)], 0, 0.0)
    for text, conf in [("EK 008", 0.99), ("EK 0080", 0.98), ("EK 008", 0.97), ("AEK 0083", 0.9)]:
        m.add_ocr(1, text, conf)
    st = m.tracks[1]
    assert st.lambung_text() == "EK 008"                      # tanpa format: skor terbesar
    assert st.lambung_text(r"[A-Z]{3} \d{4}") == "AEK 0083"   # format lengkap didahulukan


# ----------------------------- feedback mentor --------------------------- #
from hauling_offline.report import (FINISH_INVALID, FINISH_VALID, GAGAL_FINISH, LUAR_MAIN,  # noqa: E402
                                    build_track_rows, lambung_status)
from hauling_offline.vehicle_type import (LIGHT_VEHICLE, TRUCK, SizeRule, VehicleTypeEstimator,  # noqa: E402
                                          map_class_name, polygon_width_at)


def test_plate_zone_final_only_from_final_area():
    m = TrackManager(_manager().areas, 47, 60, grace_frames=3, plate_zone="final")
    need = [m.observe([_det(1, y)], i, video_timestamp(i, 10.0)) for i, y in enumerate([100, 250, 650, 800, 950])]
    assert [len(n) for n in need] == [0, 0, 0, 0, 1]        # baru dicek saat menyentuh final (frame 4)
    assert len(m.observe([_det(1, 1000)], 5, 0.5)) == 1      # tetap dicek selama grace window
    m.collect_events(frame_index=7)                           # grace habis -> event ditulis
    assert m.observe([_det(1, 1050)], 8, 0.8) == []           # sesudah event: tidak dicek lagi


def test_plate_zone_main_legacy():
    m = TrackManager(_manager().areas, 47, 60, plate_zone="main")
    assert len(m.observe([_det(1, 250)], 0, 0.0)) == 1
    with pytest.raises(ValueError):
        TrackManager(_manager().areas, 47, 60, plate_zone="semua")


def test_plate_zone_final_continues_outside_main_during_grace():
    cfg = dict(BASE, main_area_coordinates_list="[[0,0],[1920,0],[1920,960],[0,960]]")
    c = config_from_dict(cfg)
    m = TrackManager(AreaSet.from_config(c, (1920, 1080)), 47, 60, grace_frames=5)
    m.observe([_det(1, 920)], 0, 0.0)                         # final + main
    assert len(m.observe([_det(1, 1040)], 1, 0.1)) == 1       # keluar main, masih grace -> tetap dicek


def test_map_class_names_and_polygon_width():
    assert map_class_name("Light Vehicle") == LIGHT_VEHICLE and map_class_name("LV") == LIGHT_VEHICLE
    assert map_class_name("Truck") == TRUCK and map_class_name("Dump_Truck") == TRUCK
    assert map_class_name("Motorcycle") is None
    trap = np.array([[842, 3], [1010, 3], [1728, 1080], [490, 1080]])
    assert polygon_width_at(trap, 1080) == pytest.approx(1238)
    assert polygon_width_at(trap, 3) == pytest.approx(168)
    assert polygon_width_at(trap, 2000) is None


def test_size_rule_truck_vs_lv():
    rule = SizeRule(np.array([[842, 3], [1010, 3], [1728, 1080], [490, 1080]]), threshold=0.42)
    road = polygon_width_at(rule.poly, 600)
    assert rule.vote((800, 300, 800 + 0.7 * road, 600))[0] == TRUCK
    assert rule.vote((900, 450, 900 + 0.25 * road, 600))[0] == LIGHT_VEHICLE


def test_type_votes_and_multiclass_model():
    m = _manager()
    m.observe([_det(1, 250)], 0, 0.0)
    for t, w in [(TRUCK, 0.3), (LIGHT_VEHICLE, 0.9), (TRUCK, 0.2)]:
        m.add_type_vote(1, t, w, "size")
    assert m.tracks[1].vehicle_type() == LIGHT_VEHICLE and m.tracks[1].type_source == "size"
    est = VehicleTypeEstimator("auto", model_names={0: "Truck", 1: "Light Vehicle", 2: "Motorcycle"},
                               model_single_cls=False)
    assert est.method == "model"
    votes = est.vote_frame(None, [Detection(5, (0, 0, 10, 10), 0.8, 1)], 0)
    assert votes[5][0][0] == LIGHT_VEHICLE
    single = VehicleTypeEstimator("auto", main_polygon=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
                                  model_names={0: "Truck", 1: "Light Vehicle"}, model_single_cls=True)
    assert single.method == "size"                         # V11 single_cls -> tidak pakai kelas model


def test_tracks_report_categories():
    m = TrackManager(_manager().areas, 47, 60, lambung_format=r"[A-Z]{3} \d{4}")
    frames = {1: [250, 650, 950], 2: [500, 650, 950], 3: [100, 250, 400]}
    for i in range(3):
        m.observe([_det(t, ys[i]) for t, ys in frames.items()], i, video_timestamp(i, 10.0))
    m.observe([Detection(4, (10, -500, 20, -400), 0.9, 0)] * 1, 3, 0.3)   # di luar main area (di atas frame)
    for _ in range(5):
        m.observe([Detection(4, (10, -500, 20, -400), 0.9, 0)], 4, 0.4)
    m.add_ocr(1, "AEK 0083", 0.9)
    m.tracks[1].plate_frames = 3
    m.tracks[1].best_plate = PlateCandidate(0.7, (1, 1, 5, 5), 0.2, 2)
    m.tracks[2].plate_frames = 4
    m.collect_events()
    rows = {r["track_id"]: r for r in build_track_rows(m, {1: "v-00001"}, total_frames=100, fps=10.0,
                                                       plate_enabled=True)}
    assert rows["1"]["status_kendaraan"] == FINISH_VALID and rows["1"]["lambung_status"] == "terbaca"
    assert rows["1"]["event_id"] == "v-00001" and rows["1"]["lambung_text"] == "AEK 0083"
    assert rows["2"]["status_kendaraan"] == FINISH_INVALID and rows["2"]["lambung_status"] == "tidak terdeteksi"
    assert rows["3"]["status_kendaraan"] == GAGAL_FINISH and "hilang" in rows["3"]["keterangan"]
    assert rows["3"]["lambung_status"].startswith("tidak dicek")
    assert rows["4"]["status_kendaraan"] == LUAR_MAIN
    st = m.tracks[1]
    st.ocr_scores = {"EK 008": 2.0}
    assert lambung_status(st, r"[A-Z]{3} \d{4}", True) == "terbaca sebagian"


def test_main_mode_bbox_keeps_truck_on_road_edge():
    cfg = dict(BASE, main_area_coordinates_list="[[0,0],[1000,0],[1000,1080],[0,1080]]")
    areas = AreaSet.from_config(config_from_dict(cfg), (1920, 1080))
    det = Detection(1, (900, 910, 1300, 990), 0.9, 0)          # centroid x=1100 di luar, 25% luas bbox di main
    assert not TrackManager(areas, 47, 60).in_main(det)
    assert TrackManager(areas, 47, 60, main_mode="bbox", main_min_coverage=0.2).in_main(det)
    assert not TrackManager(areas, 47, 60, main_mode="bbox", main_min_coverage=0.3).in_main(det)
    assert areas.coverage("main", (900, 910, 1300, 990)) == pytest.approx(0.25)
