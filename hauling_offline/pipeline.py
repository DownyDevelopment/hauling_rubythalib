"""Orkestrasi pipeline offline: video -> detect+track -> area/speed -> plate -> event CSV + video + summary.

Tidak ada koneksi database / RTSP / API / notifikasi di modul ini.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .config import KM18Config, load_config
from .csv_writer import EventCsvWriter
from .geometry import AreaSet
from .renderer import SafeVideoWriter, draw_frame, draw_track_card
from .report import build_track_rows, count_by, write_tracks_csv
from .speed import video_timestamp
from .tracks import TrackManager

log = logging.getLogger("hauling.pipeline")



class PipelineError(RuntimeError):
    """Error yang dapat ditindaklanjuti (model/video/config tidak terbaca, CSV gagal ditulis)."""


@dataclass
class RunOptions:
    video: Path
    vehicle_model: Path
    config: Path
    output_dir: Path
    plate_model: Optional[Path] = None
    config_section: Optional[str] = None
    save_video: bool = True                    # --render: annotated.mp4
    save_report: bool = True                   # --report: tracks_report.csv (+ foto per track)
    conf: Optional[float] = None
    iou: Optional[float] = None
    imgsz: Optional[int] = None
    max_det: Optional[int] = None
    plate_conf: float = 0.15
    plate_imgsz: Optional[int] = None          # None -> imgsz training model plate
    plate_region: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    plate_merge: str = "auto"
    vehicle_classes: Optional[List[int]] = None
    ocr_model_dir: Optional[Path] = None       # aktifkan OCR nomor lambung (ekstensi plate_text)
    ocr_block_order: str = "position"
    ocr_min_plate_conf: float = 0.0            # kandidat sudah disaring posisi/ukuran; voting yang memilih
    plate_grace_sec: float = 1.0               # baca nomor lambung s.d. 1 detik setelah truk menyentuh final
    lambung_format: Optional[str] = r"[A-Z]{3} \d{4}"   # format nomor lambung lengkap, mis. "AEK 0083"
    ocr_max_reads: int = 30                    # maks. pembacaan per track (hemat waktu)
    ocr_min_width_ratio: float = 60 / 1920     # OCR hanya bila kotak nomor lambung cukup besar (truk sudah dekat)
    plate_every: int = 1
    main_mode: str = "centroid"                # centroid (PRD) / bbox (fraksi luas bbox di main area)
    main_min_coverage: float = 0.3
    plate_zone: str = "final"                  # "final" = nomor lambung + OCR hanya mulai final area (+ grace)
    vehicle_type_method: str = "auto"          # auto / model / coco / size / off  (Truck vs Light Vehicle)
    type_model: Optional[Path] = None          # model COCO (mis. yolov8n.pt) untuk jenis kendaraan, opsional
    type_every: int = 1                        # jalankan model COCO tiap N frame
    type_size_threshold: float = 0.31          # rule ukuran: lebar bbox / lebar main area >= ini -> Truck
    save_track_cards: bool = True              # simpan foto per track_id (tracks/track_<id>.jpg)
    device: Optional[str] = None
    track_buffer_sec: float = 2.0
    down_direction: str = "South"
    max_frames: Optional[int] = None
    h264: bool = True


@dataclass
class RunSummary:
    video_name: str = ""
    fps: float = 0.0
    video_resolution: List[int] = field(default_factory=list)
    total_frames_reported: int = 0
    total_frames_read: int = 0
    total_detections: int = 0
    detections_without_track_id: int = 0
    unique_tracks: int = 0
    csv_events: int = 0
    speed_valid: int = 0
    overspeed: int = 0
    plate_detected: int = 0
    lambung_text_read: int = 0
    tracks_lost_before_final: int = 0
    vehicle_types: dict = field(default_factory=dict)
    tracks_by_status: dict = field(default_factory=dict)
    lambung_by_status: dict = field(default_factory=dict)
    speed_status_counts: dict = field(default_factory=dict)
    violation_counts: dict = field(default_factory=lambda: {"OVERSPEED": 0, "NORMAL": 0, "NOT_EVALUATED": 0})
    processing_seconds: float = 0.0
    config: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _check_inputs(opt: RunOptions) -> None:
    for label, p in (("video", opt.video), ("vehicle model", opt.vehicle_model), ("config", opt.config)):
        if not Path(p).is_file():
            raise PipelineError(f"File {label} tidak ditemukan: {p}")
    if opt.plate_model is not None and not Path(opt.plate_model).is_file():
        raise PipelineError(f"File plate model tidak ditemukan: {opt.plate_model}")


def _to_h264(src: Path) -> Optional[str]:
    """Konversi mp4v -> H.264 agar bisa diputar di browser/Colab. Gagal = warning saja."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return "ffmpeg tidak tersedia; annotated.mp4 tetap codec mp4v"
    tmp = src.with_name(src.stem + "_h264.mp4")
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-preset", "veryfast", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        return f"konversi H.264 gagal: {r.stderr.strip()[:200]}"
    tmp.replace(src)
    return None


def run(opt: RunOptions) -> RunSummary:
    t0 = time.perf_counter()  # HANYA untuk durasi proses, tidak dipakai untuk speed
    summary = RunSummary()
    _check_inputs(opt)
    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Task 1: config & area ----------
    try:
        cfg: KM18Config = load_config(opt.config, opt.config_section)
    except Exception as e:
        raise PipelineError(f"Config gagal dibaca: {e}") from e
    summary.config = cfg.safe_summary()
    if cfg.auto_generated:
        summary.warnings.append("config area hasil auto-generate (auto_area.py): distance_area_speed belum "
                                "terkalibrasi terhadap jarak nyata; speed_kmh bersifat indikatif")

    cap = cv2.VideoCapture(str(opt.video))
    if not cap.isOpened():
        raise PipelineError(f"Video tidak dapat dibuka oleh OpenCV: {opt.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or w <= 0 or h <= 0:
        cap.release()
        raise PipelineError(f"Metadata video tidak valid (fps={fps}, res={w}x{h}): {opt.video}")
    summary.video_name = Path(opt.video).name
    summary.fps = round(fps, 4)
    summary.video_resolution = [w, h]
    summary.total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if (w, h) != cfg.resolution:
        summary.warnings.append(f"polygon diskalakan dari {cfg.resolution[0]}x{cfg.resolution[1]} ke {w}x{h}")
    areas = AreaSet.from_config(cfg, (w, h))

    # ---------- Task 2: detector + tracker ----------
    mp = cfg.model_params
    conf = opt.conf if opt.conf is not None else mp.get("conf", 0.25)
    iou = opt.iou if opt.iou is not None else mp.get("iou", 0.5)
    imgsz = opt.imgsz if opt.imgsz is not None else (int(mp["imgsz"]) if "imgsz" in mp else None)
    max_det = opt.max_det if opt.max_det is not None else int(mp.get("max_det", 100))
    track_buffer = max(1, int(round(opt.track_buffer_sec * fps)))
    try:
        from .detector import VehicleTracker
        tracker = VehicleTracker(str(opt.vehicle_model), conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                 device=opt.device, track_buffer_frames=track_buffer,
                                 classes=opt.vehicle_classes)
    except Exception as e:
        cap.release()
        raise PipelineError(f"Model vehicle gagal dimuat ({opt.vehicle_model}): {e}") from e
    summary.models["vehicle"] = {"file": Path(opt.vehicle_model).name, "conf": conf, "iou": iou,
                                 "imgsz": tracker.imgsz, "max_det": max_det, "classes": tracker.classes,
                                 "single_cls": bool(tracker.train_args.get("single_cls")),
                                 "track_buffer_frames": track_buffer}

    # ---------- Task 4: plate detector (opsional, gagal != stop) ----------
    plate_det = None
    if opt.plate_model is not None:
        try:
            from .plate import PlateDetector
            plate_det = PlateDetector(str(opt.plate_model), conf=opt.plate_conf, imgsz=opt.plate_imgsz,
                                      device=opt.device, region=opt.plate_region, merge=opt.plate_merge)
            summary.models["plate"] = {"file": Path(opt.plate_model).name, "conf": opt.plate_conf,
                                       "imgsz": plate_det.imgsz, "region_y0y1x0x1": list(plate_det.region),
                                       "merge": plate_det.merge, "names": plate_det.names,
                                       "every_n_frames": opt.plate_every}
        except Exception as e:  # noqa: BLE001
            summary.warnings.append(f"plate model gagal dimuat, plate_detected=false untuk semua event: {e}")
    else:
        summary.warnings.append("--plate-model tidak diberikan; plate_detected=false untuk semua event")

    # ---------- Ekstensi: OCR nomor lambung (opsional, gagal != stop) ----------
    ocr = None
    if opt.ocr_model_dir is not None:
        if plate_det is None:
            summary.warnings.append("OCR dilewati: butuh --plate-model (bbox nomor lambung) yang aktif")
        else:
            try:
                from .ocr import LambungOCR, find_ocr_models
                ocr = LambungOCR(find_ocr_models(opt.ocr_model_dir), device=opt.device,
                                 block_order=opt.ocr_block_order)
                summary.models["ocr"] = {"files": ocr.files, "block_order": opt.ocr_block_order,
                                         "min_plate_conf": opt.ocr_min_plate_conf, "max_reads_per_track": opt.ocr_max_reads}
            except Exception as e:  # noqa: BLE001
                summary.warnings.append(f"OCR nomor lambung tidak aktif: {e}"[:300])

    manager = TrackManager(areas, cfg.distance_meter, cfg.speed_limit_kmh, down_direction=opt.down_direction,
                           grace_frames=int(round(opt.plate_grace_sec * fps)) if plate_det is not None else 0,
                           lambung_format=opt.lambung_format, plate_zone=opt.plate_zone,
                           main_mode=opt.main_mode, main_min_coverage=opt.main_min_coverage)
    summary.config["main_mode"] = opt.main_mode
    if opt.main_mode == "bbox":
        summary.config["main_min_coverage"] = opt.main_min_coverage
    if plate_det is not None:
        summary.models["plate"]["zone"] = opt.plate_zone
        summary.models["plate"]["grace_sec"] = opt.plate_grace_sec

    # ---------- Jenis kendaraan: Truck vs Light Vehicle ----------
    from .vehicle_type import VehicleTypeEstimator
    vtype = VehicleTypeEstimator(
        opt.vehicle_type_method, main_polygon=areas.polygons["main"][0], frame_h=h, frame_w=w,
        model_names=tracker.names, model_single_cls=bool(tracker.train_args.get("single_cls")),
        coco_model=str(opt.type_model) if opt.type_model else None, coco_every=opt.type_every,
        size_threshold=opt.type_size_threshold, device=opt.device)
    summary.warnings.extend(vtype.warnings)
    summary.models["vehicle_type"] = {"method": vtype.method, "size_threshold": opt.type_size_threshold,
                                      "coco_model": Path(vtype.coco.file).name if vtype.coco else None}
    cards = {}   # track_id -> (area bbox, crop) foto terbesar selama di main area

    # ---------- Output ----------
    csv_path = out_dir / "events.csv"
    try:
        writer = EventCsvWriter(csv_path, source_video=Path(opt.video).name,
                                id_prefix=f"{Path(opt.video).stem}")
    except OSError as e:
        cap.release()
        raise PipelineError(f"events.csv tidak bisa dibuat: {e}") from e

    video_path = out_dir / "annotated.mp4"
    vw = SafeVideoWriter(video_path, fps, (w, h)) if opt.save_video else None

    event_ids = {}

    def write_events(events) -> None:
        for ev in events:
            event_ids[ev.track_id] = writer.write(ev)  # error di sini = kegagalan run (CSV = kontrak utama)
            summary.csv_events += 1
            summary.speed_status_counts[ev.speed.status] = summary.speed_status_counts.get(ev.speed.status, 0) + 1
            summary.speed_valid += int(ev.speed.valid)
            summary.overspeed += int(ev.violation.is_overspeed)
            summary.violation_counts[ev.violation.status] = summary.violation_counts.get(ev.violation.status, 0) + 1
            summary.plate_detected += int(ev.plate is not None)
            summary.lambung_text_read += int(bool(ev.plate_text))

    plate_err_logged = False
    type_err_logged = False
    ocr_err_logged = False
    frame_index = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if opt.max_frames is not None and frame_index >= opt.max_frames:
                frame_index -= 1
                break
            ts = video_timestamp(frame_index, fps)   # clock video, BUKAN time.time()

            try:
                dets = tracker.track(frame)
            except Exception as e:  # noqa: BLE001 - frame bermasalah tidak menghentikan run
                summary.errors.append(f"frame {frame_index}: tracker error {e}"[:300])
                dets = []
            summary.total_detections += len(dets)

            need_plate = manager.observe(dets, frame_index, ts)
            frame_plates = {}  # kandidat plat di frame ini (untuk digambar)

            in_main = [d for d in dets if d.track_id is not None and manager.in_main(d)]
            try:
                for tid, vs in vtype.vote_frame(frame, in_main, frame_index).items():
                    for t, wgt, src in vs:
                        manager.add_type_vote(tid, t, wgt, src)
            except Exception as e:  # noqa: BLE001
                if not type_err_logged:
                    summary.warnings.append(f"vehicle type error (frame {frame_index}): {e}"[:300])
                    type_err_logged = True
            if opt.save_report and opt.save_track_cards:
                for d in in_main:
                    x1, y1, x2, y2 = (int(round(v)) for v in d.bbox_xyxy)
                    a = (x2 - x1) * (y2 - y1)
                    st = manager.tracks.get(d.track_id)
                    if a > cards.get(d.track_id, (0, None))[0] and st is not None and not st.event_written:
                        cards[d.track_id] = (a, frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy(), ts)

            if plate_det is not None and need_plate and frame_index % max(1, opt.plate_every) == 0:
                try:
                    cands = plate_det.detect_on_crops(frame, [d.bbox_xyxy for d in need_plate], ts, frame_index)
                    for d, c in zip(need_plate, cands):
                        manager.add_plate(d.track_id, c)
                        if c is not None:
                            frame_plates[d.track_id] = c
                except Exception as e:  # noqa: BLE001
                    for d in need_plate:
                        manager.mark_plate_error(d.track_id)
                    if not plate_err_logged:
                        summary.warnings.append(f"plate detector error (frame {frame_index}): {e}"[:300])
                        plate_err_logged = True

            if ocr is not None:
                for tid, c in frame_plates.items():
                    st = manager.tracks.get(tid)
                    if st is None or c.confidence < opt.ocr_min_plate_conf or st.ocr_attempts >= opt.ocr_max_reads:
                        continue
                    if (c.bbox_xyxy[2] - c.bbox_xyxy[0]) < opt.ocr_min_width_ratio * w:
                        continue  # masih terlalu kecil untuk dibaca; jangan habiskan jatah baca
                    try:
                        res = ocr.read(frame, c.bbox_xyxy)
                        manager.add_ocr(tid, res.text if res else None, res.confidence if res else 0.0)
                        if res and st.ocr_reads == 1:
                            log.info("track %s lambung_read '%s' (f=%d)", tid, res.text, frame_index)
                    except Exception as e:  # noqa: BLE001
                        manager.add_ocr(tid, None)
                        if not ocr_err_logged:
                            summary.warnings.append(f"OCR error (frame {frame_index}): {e}"[:300])
                            ocr_err_logged = True

            write_events(manager.collect_events(frame_index))

            if vw is not None and vw.ok:
                speeds = {d.track_id: manager.live_speed(d.track_id) for d in dets if d.track_id is not None}
                texts = {d.track_id: manager.tracks[d.track_id].lambung_text(opt.lambung_format) for d in dets
                         if d.track_id is not None and d.track_id in manager.tracks}
                types = {d.track_id: manager.tracks[d.track_id].vehicle_type() for d in dets
                         if d.track_id is not None and d.track_id in manager.tracks}
                vw.write(draw_frame(frame, areas, dets, speeds, frame_plates, cfg.speed_limit_kmh, ts,
                                    frame_index, {k: v for k, v in texts.items() if v}, types))

            if frame_index % 500 == 0:
                log.info("frame %d (t=%.1fs) events=%d", frame_index, ts, summary.csv_events)
        write_events(manager.collect_events(flush=True))   # event yang masih dalam masa grace saat video habis
    finally:
        cap.release()
        writer.close()
        if vw is not None:
            vw.release()

    summary.total_frames_read = frame_index + 1
    summary.detections_without_track_id = manager.detections_without_id
    summary.unique_tracks = len(manager.tracks)
    lost = manager.lost_before_final()
    summary.tracks_lost_before_final = len(lost)
    if lost:
        summary.warnings.append(f"{len(lost)} track masuk main area tapi hilang sebelum final area: {lost[:20]}")

    summary.outputs["events_csv"] = str(csv_path)

    # ---------- Laporan per ID kendaraan ----------
    rows = build_track_rows(manager, event_ids, summary.total_frames_read, fps, plate_enabled=plate_det is not None)
    if opt.save_report:
        tracks_path = out_dir / "tracks_report.csv"
        write_tracks_csv(tracks_path, rows)
        summary.outputs["tracks_report"] = str(tracks_path)
    relevant = [r for r in rows if r["frames_in_main"] != "0"]
    summary.vehicle_types = count_by(relevant, "vehicle_type")
    summary.tracks_by_status = count_by(rows, "status_kendaraan")
    summary.lambung_by_status = count_by([r for r in rows if r["reached_final_s"]], "lambung_status")
    if opt.save_report and opt.save_track_cards and cards:
        card_dir = out_dir / "tracks"
        card_dir.mkdir(exist_ok=True)
        by_id = {r["track_id"]: r for r in rows}
        for tid, (_, crop, cts) in cards.items():
            if crop is None or crop.size == 0:
                continue
            try:
                cv2.imwrite(str(card_dir / f"track_{tid:03d}.jpg"), draw_track_card(crop, by_id[str(tid)], cts),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            except Exception as e:  # noqa: BLE001
                summary.warnings.append(f"foto track {tid} gagal disimpan: {e}"[:200])
        summary.outputs["track_cards"] = str(card_dir)
    if vw is not None:
        if vw.error:
            summary.warnings.append(vw.error)
        elif vw.frames > 0:
            if opt.h264:
                warn = _to_h264(video_path)
                if warn:
                    summary.warnings.append(warn)
            summary.outputs["annotated_video"] = str(video_path)

    summary.processing_seconds = round(time.perf_counter() - t0, 2)
    summary_path = out_dir / "run_summary.json"
    summary.outputs["run_summary"] = str(summary_path)
    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
