"""Log waktu setiap fungsi/tahap pipeline, untuk analisa bottleneck.

Dua jenis catatan:
  - tahap SETUP (sekali jalan): load config, buka video, load model, tulis laporan, konversi video, ...
    -> dicatat ke log (INFO) dan ke run_summary.json["timing"]["setup_ms"]
  - tahap PER FRAME: baca frame, deteksi+tracking, area/speed, jenis kendaraan, nomor lambung, OCR, event, render, ...
    -> satu baris log per frame (DEBUG, masuk run.log), timing_per_frame.csv (1 baris per frame),
       timing_summary.csv (rata-rata, p95, maks, porsi % per tahap)

Waktu di sini adalah waktu PROSES (time.perf_counter). Speed kendaraan tetap memakai clock video.
"""
from __future__ import annotations

import csv
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Urutan kolom tahap per frame (tahap yang tidak aktif bernilai 0).
FRAME_STAGES = [
    "read_frame", "detect_track", "area_speed", "vehicle_type", "track_card",
    "plate_detect", "ocr", "events_csv", "render_video",
]
SUB_STAGES = ["detect", "track"]   # pecahan detect_track dari tracker (model vs algoritma tracking)


def _pct(values: List[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else 0.0


class StageProfiler:
    def __init__(self, frame_log_every: int = 1) -> None:
        self.frame_log_every = max(0, int(frame_log_every))
        self.setup_ms: Dict[str, float] = {}
        self.rows: List[dict] = []
        self._cur: Optional[dict] = None
        self._frame_t0 = 0.0

    # ------------------------------------------------------------- setup
    @contextmanager
    def setup(self, name: str):
        t = time.perf_counter()
        log.info("[mulai] %s", name)
        try:
            yield
        finally:
            ms = (time.perf_counter() - t) * 1000.0
            self.setup_ms[name] = round(self.setup_ms.get(name, 0.0) + ms, 2)
            log.info("[selesai] %s: %.1f ms", name, ms)

    def begin_setup(self, name: str) -> float:
        log.info("[mulai] %s", name)
        return time.perf_counter()

    def end_setup(self, name: str, t: float) -> None:
        ms = (time.perf_counter() - t) * 1000.0
        self.setup_ms[name] = round(self.setup_ms.get(name, 0.0) + ms, 2)
        log.info("[selesai] %s: %.1f ms", name, ms)

    # ------------------------------------------------------------- per frame
    @staticmethod
    def tic() -> float:
        return time.perf_counter()

    def toc(self, name: str, t: float) -> None:
        self.add(name, (time.perf_counter() - t) * 1000.0)

    def start_frame(self, frame_index: int, ts: float, t0: Optional[float] = None) -> None:
        self._cur = {"frame_index": frame_index, "timestamp_s": round(ts, 3), "detections": 0,
                     **{s: 0.0 for s in FRAME_STAGES}, **{s: 0.0 for s in SUB_STAGES}}
        self._frame_t0 = t0 if t0 is not None else time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            if self._cur is not None:
                self._cur[name] = self._cur.get(name, 0.0) + (time.perf_counter() - t) * 1000.0

    def add(self, name: str, ms: float) -> None:
        if self._cur is not None and ms is not None:
            self._cur[name] = self._cur.get(name, 0.0) + float(ms)

    def end_frame(self, detections: int = 0, note: str = "") -> None:
        if self._cur is None:
            return
        row = self._cur
        row["detections"] = detections
        row["total_ms"] = (time.perf_counter() - self._frame_t0) * 1000.0
        for k in FRAME_STAGES + SUB_STAGES + ["total_ms"]:
            row[k] = round(row[k], 2)
        self.rows.append(row)
        if self.frame_log_every and row["frame_index"] % self.frame_log_every == 0:
            parts = " ".join(f"{s}={row[s]:.1f}" for s in FRAME_STAGES if row[s] > 0)
            log.debug("frame=%d t=%.2fs det=%d total=%.1fms | %s (detect=%.1f track=%.1f)%s",
                      row["frame_index"], row["timestamp_s"], detections, row["total_ms"], parts,
                      row["detect"], row["track"], f" | {note}" if note else "")
        self._cur = None

    # ------------------------------------------------------------- ringkasan
    def stage_summary(self, warmup: int = 10) -> List[dict]:
        rows = self.rows[warmup:] if len(self.rows) > warmup else self.rows
        if not rows:
            return []
        total_all = sum(r["total_ms"] for r in rows) or 1.0
        out = []
        for s in FRAME_STAGES + SUB_STAGES + ["total_ms"]:
            vals = [r[s] for r in rows]
            active = [v for v in vals if v > 0]
            out.append({
                "stage": s,
                "frames": len(rows),
                "frames_active": len(active),
                "mean_ms": round(sum(vals) / len(vals), 2),
                "mean_ms_when_active": round(sum(active) / len(active), 2) if active else 0.0,
                "p95_ms": round(_pct(vals, 0.95), 2),
                "max_ms": round(max(vals), 2),
                "total_s": round(sum(vals) / 1000.0, 2),
                "share_pct": (round(100.0 * sum(vals) / total_all, 1)
                              if s not in SUB_STAGES and s != "total_ms" else None),
            })
        return out

    def slowest_frames(self, n: int = 5) -> List[dict]:
        return [{"frame_index": r["frame_index"], "timestamp_s": r["timestamp_s"], "total_ms": r["total_ms"],
                 "detections": r["detections"]}
                for r in sorted(self.rows, key=lambda r: r["total_ms"], reverse=True)[:n]]

    def write(self, out_dir: Path) -> Dict[str, str]:
        out_dir = Path(out_dir)
        files = {}
        if self.rows:
            per_frame = out_dir / "timing_per_frame.csv"
            cols = ["frame_index", "timestamp_s", "detections"] + FRAME_STAGES + SUB_STAGES + ["total_ms"]
            with per_frame.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)
            files["timing_per_frame"] = str(per_frame)
        summary = self.stage_summary()
        if summary:
            p = out_dir / "timing_summary.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader()
                w.writerows(summary)
            files["timing_summary"] = str(p)
        return files

    def log_summary(self) -> None:
        summ = self.stage_summary()
        if not summ:
            return
        log.info("=== WAKTU PER TAHAP (rata-rata per frame, tanpa 10 frame pemanasan) ===")
        for r in summ:
            if r["stage"] in SUB_STAGES:
                continue
            share = f"{r['share_pct']:5.1f}%" if r["share_pct"] is not None else "      "
            log.info("%-14s mean=%8.2f ms  p95=%8.2f ms  max=%8.2f ms  %s", r["stage"], r["mean_ms"], r["p95_ms"],
                     r["max_ms"], share)


LOGGER_NAMES = ("hauling", "hauling_offline")   # modul lama memakai "hauling.*", modul baru "hauling_offline.*"


def attach_file_log(path: Path, logger_names=LOGGER_NAMES) -> logging.Handler:
    """Simpan semua log paket (termasuk DEBUG per frame) ke file. Konsol tetap mengikuti --log-level."""
    h = logging.FileHandler(path, mode="w", encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    for name in logger_names:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(h)
    _quiet_console()
    return h


def _quiet_console() -> None:
    """Handler konsol tanpa level eksplisit akan ikut mencetak DEBUG; batasi ke INFO."""
    for h in logging.getLogger().handlers:
        if h.level == logging.NOTSET:
            h.setLevel(logging.INFO)


def detach_file_log(h: logging.Handler, logger_names=LOGGER_NAMES) -> None:
    for name in logger_names:
        logging.getLogger(name).removeHandler(h)
    h.close()
