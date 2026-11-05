"""Parser argumen & orkestrasi CLI. Dijalankan lewat main.py di root repo (entry point utama).

Contoh:
    python main.py --video video/km18_sample.mp4 --vehicle-model model/vehicle.pt \
        --plate-model model/plate.pt --config config/km18.conf --output-dir outputs/run_01 --render --report

    python main.py auto-area --video video/km18_sample.mp4 --vehicle-model model/vehicle.pt --out config/km18_auto.json

Output default : events.csv + run_summary.json
--render       : + annotated.mp4
--report       : + tracks_report.csv + foto per track (tracks/track_<id>.jpg)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .pipeline import PipelineError, RunOptions, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Pipeline inference hauling offline (KM-18). Subcommand lain: 'main.py auto-area --help'.")
    p.add_argument("--video", required=True, type=Path, help="path video input")
    p.add_argument("--vehicle-model", required=True, type=Path, help="path model vehicle (.pt/.onnx)")
    p.add_argument("--config", required=True, type=Path, help="path config KM-18 (.json/.yaml/.ini/.env)")
    p.add_argument("--output-dir", required=True, type=Path, help="folder output")
    p.add_argument("--plate-model", "--lambung-model", dest="plate_model", type=Path, default=None,
                   help="path model nomor lambung (kolom CSV plate_*), opsional")
    p.add_argument("--config-section", default=None, help="section/kamera di config bila berisi banyak kamera")
    out = p.add_argument_group("output tambahan (default: hanya events.csv + run_summary.json)")
    out.add_argument("--render", "--save-video", dest="render", action=argparse.BooleanOptionalAction,
                     default=False, help="simpan video anotasi annotated.mp4 (--save-video = alias lama)")
    out.add_argument("--report", action=argparse.BooleanOptionalAction, default=False,
                     help="simpan laporan per ID tracks_report.csv + foto per track di tracks/")
    p.add_argument("--conf", type=float, default=None, help="override confidence vehicle")
    p.add_argument("--iou", type=float, default=None, help="override IoU NMS vehicle")
    p.add_argument("--imgsz", type=int, default=None, help="override ukuran input vehicle")
    p.add_argument("--max-det", type=int, default=None, help="override batas deteksi per frame")
    p.add_argument("--plate-conf", type=float, default=0.15, help="confidence minimum kandidat nomor lambung")
    p.add_argument("--plate-grace-sec", type=float, default=1.0,
                   help="lanjut membaca nomor lambung s.d. N detik setelah truk menyentuh final (event tetap 1x)")
    p.add_argument("--lambung-format", default=r"[A-Z]{3} \d{4}",
                   help="regex format nomor lambung lengkap; bacaan yang cocok didahulukan saat voting ('' = nonaktif)")
    p.add_argument("--plate-imgsz", type=int, default=None, help="default: imgsz training model plate")
    p.add_argument("--plate-region", default="0,1,0,1",
                   help="bagian bbox kendaraan (kabin/atas kaca) yang dikirim ke model nomor lambung: y0,y1,x0,x1")
    p.add_argument("--plate-merge", default="auto", choices=["auto", "best", "union"],
                   help="union = gabung blok huruf+angka jadi 1 bbox nomor lambung")
    p.add_argument("--vehicle-classes", default=None,
                   help="filter kelas vehicle, mis. '0,1' (default otomatis; model single_cls tidak difilter)")
    p.add_argument("--ocr-model-dir", type=Path, default=None,
                   help="folder model OCR nomor lambung (character_*_detection, classification_hauling_*); "
                        "bila diisi, plate_text terisi & anotasi 'No Lambung: ...'")
    p.add_argument("--ocr-block-order", default="position", choices=["position", "class"],
                   help="position = blok kiri huruf, kanan angka (default; label kelas model tertukar)")
    p.add_argument("--ocr-max-reads", type=int, default=30, help="maks. pembacaan OCR per track")
    p.add_argument("--plate-every", type=int, default=1, help="jalankan plate model tiap N frame")
    p.add_argument("--plate-zone", default="final", choices=["final", "main"],
                   help="final = nomor lambung + OCR hanya mulai kendaraan menyentuh final area s.d. grace window "
                        "(default); main = sepanjang main area (perilaku lama)")
    p.add_argument("--main-mode", default="centroid", choices=["centroid", "bbox"],
                   help="keanggotaan main area: centroid (default) atau bbox (fraksi luas bbox >= --main-min-coverage); "
                        "bbox menolong truk di sisi kanan jalan yang centroid-nya keluar garis main area")
    p.add_argument("--main-min-coverage", type=float, default=0.3, help="dipakai bila --main-mode bbox")
    p.add_argument("--vehicle-type", default="auto", choices=["auto", "model", "coco", "size", "off"],
                   help="cara membedakan Truck vs Light Vehicle (model V11 single_cls tidak bisa membedakan)")
    p.add_argument("--type-model", type=Path, default=None,
                   help="model COCO (mis. yolov8n.pt) untuk Truck vs Light Vehicle; kosong = rule ukuran saja")
    p.add_argument("--type-every", type=int, default=1, help="jalankan model COCO tiap N frame")
    p.add_argument("--type-size-threshold", type=float, default=0.31,
                   help="rule ukuran: lebar bbox / lebar main area >= nilai ini -> Truck")
    out.add_argument("--no-track-cards", dest="track_cards", action="store_false",
                     help="bersama --report: tulis tracks_report.csv saja, tanpa foto per track")
    p.add_argument("--device", default=None, help="cpu / 0 / cuda:0")
    p.add_argument("--track-buffer-sec", type=float, default=2.0, help="toleransi occlusion tracker (detik)")
    p.add_argument("--down-direction", default="South", choices=["South", "North"],
                   help="arah yang dianggap 'turun' di frame (y bertambah)")
    p.add_argument("--max-frames", type=int, default=None, help="batasi jumlah frame (debug)")
    p.add_argument("--no-h264", dest="h264", action="store_false", help="jangan konversi video ke H.264")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "auto-area":            # Task 1 otomatis: generate config area dari video
        from .auto_area import main as auto_area_main
        return auto_area_main(argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    opt = RunOptions(
        video=args.video, vehicle_model=args.vehicle_model, config=args.config, output_dir=args.output_dir,
        plate_model=args.plate_model, config_section=args.config_section, save_video=args.render, save_report=args.report,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, max_det=args.max_det, plate_conf=args.plate_conf,
        plate_imgsz=args.plate_imgsz, plate_every=args.plate_every, device=args.device,
        plate_region=tuple(float(v) for v in args.plate_region.split(",")), plate_merge=args.plate_merge,
        vehicle_classes=[int(v) for v in args.vehicle_classes.split(",")] if args.vehicle_classes else None,
        plate_grace_sec=args.plate_grace_sec, lambung_format=args.lambung_format or None,
        ocr_model_dir=args.ocr_model_dir, ocr_block_order=args.ocr_block_order, ocr_max_reads=args.ocr_max_reads,
        track_buffer_sec=args.track_buffer_sec, down_direction=args.down_direction,
        max_frames=args.max_frames, h264=args.h264, plate_zone=args.plate_zone,
        main_mode=args.main_mode, main_min_coverage=args.main_min_coverage,
        vehicle_type_method=args.vehicle_type, type_model=args.type_model, type_every=args.type_every,
        type_size_threshold=args.type_size_threshold, save_track_cards=args.track_cards,
    )
    try:
        s = run(opt)
    except PipelineError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    print("\n=== OUTPUT ===")
    for k, v in s.outputs.items():
        print(f"{k:>16}: {v}")
    print("\n=== RUN SUMMARY ===")
    print(json.dumps({k: getattr(s, k) for k in (
        "video_name", "fps", "video_resolution", "total_frames_read", "total_detections", "unique_tracks",
        "csv_events", "speed_valid", "overspeed", "violation_counts", "plate_detected", "lambung_text_read", "tracks_lost_before_final",
        "vehicle_types", "tracks_by_status", "lambung_by_status", "processing_seconds")}, indent=2))
    tr = s.outputs.get("tracks_report")
    if tr:
        import csv
        print("\n=== LAPORAN PER ID (tracks_report.csv) ===")
        with open(tr, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["frames_in_main"] == "0":
                    continue
                print(f"ID {r['track_id']:>3} | {r['vehicle_type']:<13} | {r['status_kendaraan']:<26} | "
                      f"{(r['speed_kmh'] or '-'):>6} km/h | lambung: {(r['lambung_text'] or '-'):<9} "
                      f"({r['lambung_status']}) | {r['keterangan']}")
    if s.warnings:
        print("Warnings:", *s.warnings, sep="\n  - ")
    if s.errors:
        print(f"Errors ({len(s.errors)}):", *s.errors[:10], sep="\n  - ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
