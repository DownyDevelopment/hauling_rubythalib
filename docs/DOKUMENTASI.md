# Dokumentasi Pipeline Hauling Offline KM-18

Pipeline ini memproses video CCTV jalan hauling KM-18 secara offline. Kendaraan dideteksi dan dilacak, kecepatannya dihitung di antara dua garis area, pelanggaran overspeed dievaluasi, lalu nomor lambung dideteksi di zona final. Hasil utamanya `events.csv`. Pipeline tidak memakai database, RTSP, API, maupun notifikasi.

```
video ──> Task 2: deteksi + tracking ──> Task 1: cek area ──> Task 3: speed ──> Task 5: violation
                                                                 │
                                    Task 4: nomor lambung (zona final) ──> events.csv (+ video, laporan)
```

Isi dokumen: [Struktur repo](#struktur-repo) · [Instalasi](#instalasi) · [Menjalankan](#menjalankan) ·
[Task 1](#task-1--setup-area) · [Task 2](#task-2--deteksi-kendaraan) · [Task 3](#task-3--logic-speed-detection) ·
[Task 4](#task-4--deteksi-nomor-lambung) · [Task 5](#task-5--violation-overspeed) · [Output](#output) ·
[Testing](#testing) · [Riwayat perubahan](#riwayat-perubahan)

---

## Struktur repo

```
hauling_rubythalib/
├── main.py                  # entry point utama (satu-satunya file yang dieksekusi)
├── requirements.txt         # dependency (pip install -r)
├── pyproject.toml           # metadata paket + dependency (pip install -e .)
├── docs/DOKUMENTASI.md      # dokumen ini
├── hauling_offline/         # seluruh kode program, dipakai sebagai modul
│   ├── config.py            # Task 1  - baca config KM-18
│   ├── geometry.py          # Task 1  - polygon, skala, cek titik/crossing
│   ├── auto_area.py         # Task 1  - generate area otomatis dari lintasan (opsional)
│   ├── detector.py          # Task 2  - YOLO + ByteTrack
│   ├── vehicle_type.py      # Task 2  - Truck vs Light Vehicle
│   ├── speed.py             # Task 3  - rumus & status speed
│   ├── tracks.py            # Task 3  - state machine per track_id
│   ├── plate.py             # Task 4  - deteksi lokasi nomor lambung
│   ├── ocr.py               # Task 4  - baca teks nomor lambung (opsional)
│   ├── violation.py         # Task 5  - aturan overspeed
│   ├── csv_writer.py        # output  - events.csv
│   ├── report.py            # output  - tracks_report.csv (--report)
│   ├── renderer.py          # output  - annotated.mp4 & foto per track (--render / --report)
│   ├── pipeline.py          # orkestrasi seluruh task
│   └── cli.py               # parser argumen yang dipanggil main.py
└── tests/                   # unit test (pytest)
```

Video, model (`.pt`), config kamera (`.conf`), dan folder output tidak masuk repo (lihat `.gitignore`).

## Instalasi

Butuh Python 3.9 atau lebih baru.

```bash
pip install -r requirements.txt
# atau, sebagai paket (sekaligus menyediakan perintah `hauling`):
pip install -e ".[dev]"
```

`ffmpeg` opsional, dipakai untuk mengubah `annotated.mp4` ke H.264 agar bisa diputar di browser. Di Colab sudah tersedia.

## Menjalankan

Semua eksekusi lewat `main.py` dari root repo.

```bash
python main.py \
  --video video/km18_sample.mp4 \
  --vehicle-model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt \
  --plate-model model/detection_hauling_angka_huruf_V1.pt --plate-region 0.4,1,0.25,1 \
  --ocr-model-dir model \
  --config config/_track_km18_01_V2.conf \
  --output-dir outputs/run_01 \
  --render --report
```

| Flag | Output yang ditambahkan |
| --- | --- |
| (tanpa flag) | `events.csv`, `run_summary.json` |
| `--render` | `annotated.mp4` (video anotasi area, bbox, speed, label OVERSPEED, nomor lambung) |
| `--report` | `tracks_report.csv` dan foto per track di `tracks/track_<id>.jpg` |
| `--report --no-track-cards` | `tracks_report.csv` saja, tanpa foto |

`--save-video` / `--no-save-video` masih diterima sebagai alias lama dari `--render` / `--no-render`.

Daftar lengkap argumen: `python main.py --help`. Di Colab, jalankan dari folder hasil ekstraksi:

```python
%cd /content/hauling
!python main.py --video "{VIDEO}" --vehicle-model "{VEHICLE_MODEL}" --config "{CONFIG}" --output-dir "{OUTPUT_DIR}" --render --report
```

---

## Task 1 — Setup Area

**Tujuan.** Membaca area KM-18 dari config, menskalakannya ke resolusi video, lalu menentukan apakah kendaraan berada di (atau melintasi) area tertentu.

**Modul.** `config.py`, `geometry.py`, `auto_area.py`.

**Key config yang dibaca** (nama sesuai PRD). Key lain di file config, seperti password, URL RTSP, atau token, tidak pernah dibaca, dicetak, maupun disimpan.

| Key | Isi |
| --- | --- |
| `resolution_area` | Resolusi basis koordinat, mis. `1920x1080` |
| `main_area_coordinates_list` | Polygon area jalan yang dipantau |
| `speed_area_start_coordinates_list` | Garis/area awal pengukuran speed |
| `speed_area_final_coordinates_list` | Garis/area akhir pengukuran speed |
| `final_area_coordinates_list` | Area tempat event kendaraan dicatat |
| `distance_area_speed` | Jarak start–finish dalam meter (KM-18: 47) |
| `speed_limit` | Batas kecepatan km/jam (KM-18: 60) |
| `travel_direction` | Arah lajur yang dipantau, mis. `South` |

**Cara kerja.**
- Format config yang didukung: `.json`, `.yaml/.yml`, `.ini/.cfg/.conf`, `.env/.txt`. Nilai polygon boleh berupa list atau string JSON, tanpa `eval`. Kalau satu file berisi beberapa kamera, pilih dengan `--config-section`.
- Polygon diskalakan secara proporsional dari `resolution_area` ke resolusi frame video.
- Posisi kendaraan diwakili titik tengah bbox. Selain dicek "titik di dalam polygon", lintasan antar-frame juga dicek terhadap polygon. Dengan begitu, garis start/finish yang tipis tetap tersentuh walaupun kendaraan bergerak cepat.
- Keanggotaan main area: `--main-mode centroid` (default, sesuai PRD) atau `--main-mode bbox` (fraksi luas bbox ≥ `--main-min-coverage`). Mode `bbox` berguna untuk truk di sisi kanan jalan yang titik tengahnya keluar garis main area.

**Area otomatis (opsional).** Kalau koordinat kalibrasi resmi belum ada, area bisa dibuat dari lintasan kendaraan di video:

```bash
python main.py auto-area --video video/km18_sample.mp4 \
  --vehicle-model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt --out config/km18_auto.json --seconds 120
```

Hasilnya memakai key yang sama dengan PRD. Jarak pada config otomatis belum terkalibrasi, jadi speed-nya hanya indikatif.

## Task 2 — Deteksi Kendaraan

**Tujuan.** Mendeteksi kendaraan di setiap frame dan memberinya `track_id` yang konsisten selama kendaraan terlihat.

**Modul.** `detector.py`, `vehicle_type.py`.

**Cara kerja.**
- Model YOLO (Ultralytics) dengan ByteTrack. Path video, model, dan config dicek keberadaannya sebelum run dimulai.
- `--track-buffer-sec` (default 2,0 s) menentukan toleransi occlusion sebelum track dianggap hilang.
- Deteksi tanpa `track_id` tetap dihitung di summary, tapi tidak pernah membuat event.
- **Truck vs Light Vehicle.** Model V11 dilatih `single_cls`, jadi semua deteksi keluar sebagai satu kelas. Jenis kendaraan ditentukan per track lewat voting antar-frame (`--vehicle-type`, default `auto`):
  - `size`: lebar bbox / lebar main area ≥ 0,31 berarti Truck (kalibrasi KM-18: dump truck 0,36–0,47, pickup 0,20–0,26).
  - `coco`: model COCO umum (`--type-model yolov8n.pt`).
  - `model`: kelas dari model vehicle bila nanti tersedia model multi-kelas.

**Parameter.** `--conf`, `--iou`, `--imgsz`, `--max-det` (override nilai di config), `--vehicle-classes`, `--device`.

## Task 3 — Logic Speed Detection

**Tujuan.** Menghitung kecepatan setiap kendaraan dari waktu tempuh antara area speed start dan speed final.

**Modul.** `speed.py` (rumus), `tracks.py` (state machine per `track_id`).

**Rumus.** Timestamp diambil dari clock video, bukan jam sistem, sehingga hasil untuk input yang sama selalu identik.

```
timestamp_s = frame_index / fps
speed_kmh   = (distance_area_speed / (t_finish - t_start)) × 3.6     # dibulatkan 2 desimal
```

**Alur per track.**
1. Masuk main area: track menjadi relevan.
2. Pertama kali menyentuh speed start: `speed_start_timestamp_s` dicatat.
3. Pertama kali menyentuh speed final: `speed_end_timestamp_s` dicatat, speed dihitung, dan violation ditandai (Task 5).
4. Pertama kali menyentuh final area: event ditulis **sekali** ke `events.csv`.

**Status speed.** Hanya `valid` yang mengisi `speed_kmh`. Status lain mengosongkan kolom speed dan mengisi `measurement_valid` = false.

| `speed_status` | Arti |
| --- | --- |
| `valid` | Melewati start lalu finish dengan selisih waktu > 0 |
| `not_crossed_start` | Tidak terekam melewati speed start (muncul setelah garis start) |
| `not_crossed_finish` | Tidak terekam melewati speed finish |
| `wrong_order` | Finish tersentuh sebelum start (arah berlawanan) |
| `started_in_finish` | Pertama terlihat sudah berada di area speed finish |
| `invalid_elapsed_time` | Selisih waktu start–finish ≤ 0 |

**Arah.** Ditentukan dari perubahan posisi vertikal. `--down-direction` (default `South`) adalah label arah ketika kendaraan bergerak ke bawah frame.

## Task 4 — Deteksi Nomor Lambung

**Tujuan.** Mendeteksi lokasi nomor lambung (kode unit di atas kaca depan kabin, mis. "AEK 0083"), lalu membacanya bila OCR diaktifkan. Kolom CSV tetap bernama `plate_*` sesuai kontrak PRD, tetapi yang dideteksi adalah nomor lambung, bukan plat polisi.

**Modul.** `plate.py` (lokasi), `ocr.py` (teks, opsional).

**Cara kerja.**
- Model nomor lambung dijalankan pada crop kendaraan, tepatnya bagian kabin yang ditentukan `--plate-region y0,y1,x0,x1` (contoh KM-18: `0.4,1,0.25,1`).
- `--plate-zone final` (default): deteksi hanya berjalan sejak kendaraan menyentuh final area sampai `--plate-grace-sec` detik sesudahnya (default 1,0 s). Event tetap hanya ditulis sekali. `--plate-zone main` = perilaku lama, sepanjang main area.
- Kandidat terbaik per track = confidence tertinggi. Bila sama, dipilih yang timestamp-nya lebih akhir.
- OCR (aktif bila `--ocr-model-dir` diisi): blok huruf dan angka dipisah, setiap karakter dideteksi lalu diklasifikasi. Hasil antar-frame di-voting, dengan prioritas pada bacaan yang cocok `--lambung-format` (default `[A-Z]{3} \d{4}`). Hasilnya mengisi `plate_text`.
- Error pada model nomor lambung hanya dicatat sebagai warning. Event kendaraan tetap ditulis dengan `plate_detected` = false.

**Parameter lain.** `--plate-conf`, `--plate-imgsz`, `--plate-merge`, `--plate-every`, `--ocr-block-order`, `--ocr-max-reads`.

## Task 5 — Violation Overspeed

**Tujuan.** Menandai kendaraan yang melanggar batas kecepatan.

**Modul.** `violation.py`, dipanggil dari `tracks.py`.

| Kondisi | `is_overspeed` | `violation_status` | `violation_reason` |
| --- | --- | --- | --- |
| `measurement_valid` = false | false | `NOT_EVALUATED` | (kosong) |
| `speed_kmh` > `speed_limit` | true | `OVERSPEED` | Speed exceeds configured limit |
| Selain itu, termasuk tepat 60 km/jam | false | `NORMAL` | (kosong) |

**Aturan penandaan.**
- Violation ditandai saat track menyentuh speed final, lalu dicatat bersama event kendaraan saat mencapai final area.
- Satu violation per `track_id`; setelah ditandai, nilainya tidak pernah ditimpa.
- Track yang tidak pernah menyentuh speed final dievaluasi saat event ditulis, dan hasilnya `NOT_EVALUATED`.
- Batas diambil langsung dari `speed_limit` di config. Tidak ada toleransi tambahan; perubahan toleransi harus lewat parameter config baru dan persetujuan eksplisit.
- Speed yang dibandingkan adalah nilai yang sudah dibulatkan 2 desimal, sama dengan yang tertulis di CSV.

---

## Output

| File | Kapan dibuat | Isi |
| --- | --- | --- |
| `events.csv` | selalu | 1 baris per kendaraan yang mencapai final area (24 kolom) |
| `run_summary.json` | selalu | Ringkasan run: frame, deteksi, track, speed valid, `overspeed`, `violation_counts`, warning, error |
| `annotated.mp4` | `--render` | Video anotasi |
| `tracks_report.csv` | `--report` | Semua track, termasuk yang tidak mencapai final, beserta alasannya |
| `tracks/track_<id>.jpg` | `--report` | Foto kendaraan per track dengan ringkasan statusnya |

**Kolom `events.csv`** (urutan tetap). Kolom 1–22 adalah kontrak PRD, kolom 23–24 ditambahkan untuk Task 5.

```
event_id, source_video, track_id, event_timestamp_s, event_frame_index, travel_direction,
vehicle_confidence, speed_start_timestamp_s, speed_end_timestamp_s, elapsed_seconds, distance_meter,
speed_kmh, speed_status, speed_limit_kmh, is_overspeed, plate_detected, plate_confidence,
plate_bbox_xyxy, plate_timestamp_s, plate_text, measurement_valid, notes,
violation_status, violation_reason
```

Nilai yang belum tersedia ditulis kosong (bukan `0`, `None`, atau `nan`). Boolean ditulis `true`/`false`. `notes` berisi informasi tambahan seperti `vehicle_type=Truck`.

**`status_kendaraan` di `tracks_report.csv`.**

| Status | Arti |
| --- | --- |
| `FINISH - speed valid` | Mencapai final dan speed terukur |
| `FINISH - speed tidak valid` | Mencapai final, tetapi speed tidak terukur (alasannya di `keterangan`) |
| `GAGAL FINISH` | Masuk main area tetapi hilang atau keluar sebelum final |
| `DI LUAR MAIN AREA` | Terdeteksi, tetapi tidak pernah masuk main area |
| `DETEKSI SINGKAT` | Kurang dari 5 frame (kemungkinan false positive) |

## Testing

```bash
python -m pytest -q        # 71 test
```

| File | Cakupan |
| --- | --- |
| `tests/test_units.py` | Config, geometri, state machine, speed, nomor lambung, CSV, laporan per ID |
| `tests/test_violation.py` | Aturan overspeed (tepat 60, di atas/bawah batas, status tidak valid), penandaan di speed final, satu violation per track |
| `tests/test_cli.py` | `main.py` sebagai entry point, flag `--render`/`--report`, alias lama, subcommand `auto-area` |

Test tidak membutuhkan model maupun video, sehingga bisa dijalankan di mana saja.

## Riwayat perubahan

| Versi | Perubahan |
| --- | --- |
| 0.3.0 | `main.py` sebagai entry point; flag `--render` dan `--report` (default: hanya `events.csv` + `run_summary.json`); subcommand `main.py auto-area`; `requirements.txt` dan `pyproject.toml`; dokumentasi per task ini |
| 0.2.0 | Task 5 violation overspeed: `violation.py`, kolom `violation_status`/`violation_reason`, `violation_counts` di summary, label OVERSPEED di video |
| 0.1.0 | Task 1–4: setup area, deteksi + tracking kendaraan, logic speed, deteksi dan OCR nomor lambung, laporan per ID, jenis kendaraan |
