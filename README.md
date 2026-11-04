# Hauling Offline Pipeline (KM-18)

Video file -> vehicle detection + ByteTrack -> area/speed state machine -> plate detection on vehicle crop
-> `events.csv` + `annotated.mp4` + `run_summary.json`. Tanpa DB / RTSP / API / notifikasi.

## Install
    pip install ultralytics lap pytest

## Area otomatis (bila config kalibrasi resmi belum ada)
    python -m hauling_offline.auto_area --video km18.mp4 \
        --vehicle-model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt --out config/km18_auto.json --seconds 120
Area (main/start/final speed/final) dibuat dari lintasan truk; key config sama persis dengan PRD.
distance_area_speed=47 & speed_limit=60 = nilai referensi PRD; jarak belum terkalibrasi -> speed indikatif.

## Run
    python -m hauling_offline.cli --video km18.mp4 \
        --vehicle-model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt \
        --plate-model model/detection_hauling_angka_huruf_V1.pt --plate-region 0.4,1,0.25,1 \
        --ocr-model-dir model \
        --config km18.json --output-dir outputs/run_01 [--config-section KM18] [--device 0] [--no-save-video]

## Pembaruan feedback mentor
- Nomor lambung + OCR hanya dijalankan di zona FINAL: mulai kendaraan menyentuh final area s.d. `--plate-grace-sec`
  (default 1 s). `--plate-zone main` = perilaku lama.
- Truck vs Light Vehicle per track (`--vehicle-type`, default auto): V11 single_cls tidak bisa membedakan, jadi dipakai
  rule ukuran (lebar bbox / lebar main area >= 0.31 -> Truck), opsional model COCO (`--type-model yolov8n.pt`), dan
  otomatis memakai kelas model bila nanti ada model vehicle multi-kelas. Label video: "Truck [id]" / "Light Vehicle [id]";
  events.csv tetap 22 kolom (jenis dicatat di notes `vehicle_type=...`).
- Laporan per ID: `tracks_report.csv` (semua track: FINISH valid / FINISH speed tidak valid / GAGAL FINISH / di luar
  main area + alasan, status nomor lambung) dan foto per track di `tracks/track_<id>.jpg`.
- Opsi `--main-mode bbox --main-min-coverage 0.3`: keanggotaan main area dari luas bbox, untuk truk yang melaju di sisi
  kanan jalan (centroid keluar garis main area).

## Violation Overspeed
Modul `violation.py`: speed tidak valid -> `NOT_EVALUATED`; `speed_kmh > speed_limit` -> `OVERSPEED`
(reason "Speed exceeds configured limit"); selain itu (termasuk tepat 60) -> `NORMAL`. Ditandai sekali per track saat
menyentuh speed_final, dicatat bersama event saat mencapai final area. events.csv mendapat 2 kolom di akhir:
`violation_status`, `violation_reason`; `is_overspeed` kini `false` (bukan kosong) bila speed tidak valid.
Summary: `violation_counts`. tracks_report.csv: kolom `violation_status`. Tanpa toleransi tambahan.

## Model
- Vehicle: Hauling_Vehicle_yolov8m_SingleCls_V11.pt (single_cls -> semua output kelas 0, jangan filter kelas)
- Nomor lambung (di atas kaca depan, mis. AEK 0078): detection_hauling_angka_huruf_V1.pt (blok huruf + angka digabung jadi 1 bbox, region kabin truk).
  Kolom CSV tetap plate_* sesuai kontrak PRD; isinya = nomor lambung. CLI: --plate-model atau --lambung-model.
- OCR nomor lambung (ekstensi, opsional --ocr-model-dir): character_huruf/angka_detection_*_V2 + classification_hauling_huruf/angka_*_v9
  -> plate_text = "AEK 0078" & anotasi "No Lambung: AEK 0078". Catatan: label kelas angka_huruf_V1 tertukar -> blok kiri=huruf, kanan=angka.
- Tidak dipakai (out of scope): Classification_truck_mitra, No_Lambung_TruckHD (khusus truk HD)

## Test
    python -m pytest -q tests

## Modul
| File | Isi |
|---|---|
| auto_area.py | Task 1 otomatis - generate area KM-18 dari lintasan kendaraan |
| config.py | Task 1 - baca config KM-18 tanpa eval, validasi polygon |
| geometry.py | Task 1 - skala polygon & point-in-polygon (titik tengah bbox) |
| detector.py | Task 2 - YOLO + ByteTrack adapter |
| speed.py | Task 3 - kalkulator speed & status invalid |
| violation.py | violation overspeed (NOT_EVALUATED / NORMAL / OVERSPEED) |
| tracks.py | Task 3 - state machine per track (start -> finish -> final, anti-duplikasi) |
| plate.py | Task 4 - deteksi nomor lambung pada crop kabin + pemilih kandidat terbaik |
| ocr.py | ekstensi - text recognition nomor lambung (3 tahap + voting) |
| csv_writer.py | kontrak events.csv |
| vehicle_type.py | Truck vs Light Vehicle (rule ukuran / COCO / kelas model) |
| report.py | laporan per ID kendaraan (tracks_report.csv) |
| renderer.py | video anotasi + foto per track |
| pipeline.py / cli.py | orkestrasi & entry point |
