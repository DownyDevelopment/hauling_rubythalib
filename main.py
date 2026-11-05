"""Entry point utama pipeline Hauling KM-18.

Semua kode program ada di package hauling_offline/ (modul). Jalankan dari root repo:

    python main.py --video <video> --vehicle-model <pt> --config <conf> --output-dir <folder> [--render] [--report]
    python main.py auto-area --video <video> --vehicle-model <pt> --out <config.json>
    python main.py --help

Dokumentasi per task: docs/DOKUMENTASI.md
"""
import sys

from hauling_offline.cli import main

if __name__ == "__main__":
    sys.exit(main())
