# Hauling Offline Pipeline (KM-18)

Pipeline offline untuk video CCTV jalan hauling KM-18: deteksi + tracking kendaraan, perhitungan speed di antara
area start–finish, violation overspeed, dan deteksi nomor lambung. Output utama `events.csv`.
Tanpa database / RTSP / API / notifikasi.

## Instalasi
```bash
pip install -r requirements.txt        # PyTorch versi CUDA dipasang dulu; lihat docs: Setup lokal
```

## Menjalankan
`main.py` adalah satu-satunya file yang dieksekusi; seluruh kode program ada di package `hauling_offline/`.

```bash
python main.py --video video/km18_sample.mp4 \
  --vehicle-model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt \
  --plate-model model/detection_hauling_angka_huruf_V1.pt --plate-region 0.4,1,0.25,1 --ocr-model-dir model \
  --config config/_track_km18_01_V2.conf --output-dir outputs/run_01 \
  --render --report
```

| Flag | Output tambahan |
| --- | --- |
| (tanpa flag) | `events.csv`, `run_summary.json` |
| `--render` | `annotated.mp4` |
| `--report` | `tracks_report.csv` + foto per track `tracks/track_<id>.jpg` |

Setiap run juga menulis `run.log` (waktu setiap tahap per frame), `timing_per_frame.csv`, dan `timing_summary.csv` untuk analisa bottleneck.

Tracker: `--tracker bytetrack|botsort|botsort-reid|deepsort` (default bytetrack). Detector TensorRT: `.engine` di `--vehicle-model`.

Subcommand:
- `python main.py auto-area --help`: area otomatis bila kalibrasi resmi belum ada
- `python main.py export-engine --help`: export detector ke TensorRT
- `python main.py benchmark-tracker --help`: bandingkan tracker + video preview grid

Semua argumen: `python main.py --help`.

## Testing
```bash
python -m pytest -q
```

## Dokumentasi
Setup lokal Windows, penjelasan per task (Setup Area, Deteksi Kendaraan, Logic Speed, Nomor Lambung, Violation
Overspeed, R&D Tracking), format output,
dan riwayat perubahan: **[docs/DOKUMENTASI.md](docs/DOKUMENTASI.md)**.
