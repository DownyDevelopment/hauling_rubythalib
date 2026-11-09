"""TASK 2 / R&D tracking: pilihan algoritma tracker + pengukur waktu.

Tracker yang didukung (semua menghasilkan Detection ber-track_id dengan format yang sama, jadi Task 1/3/4/5
tidak berubah apa pun tracker-nya):

    bytetrack     Ultralytics ByteTrack  - Kalman + IoU, asosiasi 2 tahap (deteksi conf rendah ikut dipakai)
    botsort       Ultralytics BoT-SORT   - ByteTrack + kompensasi gerak kamera (GMC), tanpa ReID
    botsort-reid  Ultralytics BoT-SORT   - sama + fitur tampilan ReID (butuh Ultralytics versi yang mendukung)
    deepsort      deep-sort-realtime     - Kalman + IoU + fitur tampilan (embedder MobileNet), 1 tahap

TensorRT BUKAN tracker: ia mempercepat model DETEKSI. Pakai `main.py export-engine` lalu berikan file .engine
ke --vehicle-model; tracker mana pun bisa dipasangkan dengan detector .engine.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

TRACKERS = ("bytetrack", "botsort", "botsort-reid", "deepsort")
ULTRALYTICS_TRACKERS = {"bytetrack": "bytetrack", "botsort": "botsort", "botsort-reid": "botsort"}
MODEL_META_SUFFIX = ".hauling.json"   # sidecar metadata untuk model hasil export (.engine/.onnx)


# --------------------------------------------------------------------------- #
# Config tracker Ultralytics
# --------------------------------------------------------------------------- #
def patch_tracker_yaml(text: str, track_buffer_frames: int, with_reid: Optional[bool] = None) -> str:
    """Ubah track_buffer (toleransi occlusion) dan, untuk BoT-SORT, with_reid. Baris lain dibiarkan apa adanya."""
    out, seen_reid = [], False
    for line in text.splitlines():
        key = line.strip().split(":", 1)[0]
        if key == "track_buffer":
            line = f"track_buffer: {int(track_buffer_frames)}"
        elif key == "with_reid" and with_reid is not None:
            line = f"with_reid: {'True' if with_reid else 'False'}"
            seen_reid = True
        out.append(line)
    if with_reid and not seen_reid:
        out.append("with_reid: True")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Metadata model hasil export (TensorRT .engine tidak menyimpan train_args checkpoint)
# --------------------------------------------------------------------------- #
def model_meta_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(p.name + MODEL_META_SUFFIX)


def load_model_meta(model_path: str | Path) -> dict:
    """train_args (imgsz, single_cls, ...) dari sidecar `<model>.hauling.json` bila ada."""
    meta = model_meta_path(model_path)
    if not meta.is_file():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data.get("train_args") or {})


def save_model_meta(model_path: str | Path, train_args: dict, source: str | Path, **extra) -> Path:
    keep = {k: train_args[k] for k in ("imgsz", "single_cls", "task") if k in train_args}
    meta = model_meta_path(model_path)
    meta.write_text(json.dumps({"source": Path(source).name, "train_args": keep, **extra}, indent=2),
                    encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# Pengukur waktu per frame (deteksi vs tracking)
# --------------------------------------------------------------------------- #
@dataclass
class TimingStats:
    """Waktu per frame dalam ms. Frame pemanasan (load model, inisialisasi CUDA) tidak dihitung."""
    warmup: int = 10
    frames: int = 0
    detect_ms: List[float] = field(default_factory=list)
    track_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)

    last_detect_ms: float = 0.0
    last_track_ms: float = 0.0

    def add(self, detect_ms: float, total_ms: float) -> None:
        detect_ms = min(detect_ms, total_ms)
        self.last_detect_ms, self.last_track_ms = detect_ms, total_ms - detect_ms
        self.frames += 1
        if self.frames <= self.warmup:
            return
        detect_ms = min(detect_ms, total_ms)
        self.detect_ms.append(detect_ms)
        self.track_ms.append(total_ms - detect_ms)
        self.total_ms.append(total_ms)

    @staticmethod
    def _mean(v: Sequence[float]) -> Optional[float]:
        return round(sum(v) / len(v), 2) if v else None

    @staticmethod
    def _pct(v: Sequence[float], q: float) -> Optional[float]:
        if not v:
            return None
        s = sorted(v)
        return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 2)

    def summary(self) -> dict:
        total = self._mean(self.total_ms)
        return {
            "frames_measured": len(self.total_ms),
            "warmup_frames_skipped": min(self.frames, self.warmup),
            "detect_ms_mean": self._mean(self.detect_ms),
            "track_ms_mean": self._mean(self.track_ms),
            "total_ms_mean": total,
            "total_ms_p95": self._pct(self.total_ms, 0.95),
            "fps_detect_track": round(1000.0 / total, 2) if total else None,
        }


class Stopwatch:
    def __init__(self) -> None:
        self.t = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t) * 1000.0


# --------------------------------------------------------------------------- #
# DeepSORT (deep-sort-realtime)
# --------------------------------------------------------------------------- #
def xyxy_to_ltwh(b: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def deepsort_tracks_to_rows(tracks: Iterable) -> List[tuple]:
    """Track DeepSORT -> (track_id, bbox_xyxy, conf, cls). Hanya track yang confirmed DAN cocok dengan
    deteksi di frame ini (time_since_update == 0), supaya posisi = posisi deteksi nyata, bukan prediksi Kalman."""
    rows = []
    for t in tracks:
        if not t.is_confirmed() or getattr(t, "time_since_update", 1) != 0:
            continue
        try:
            box = t.to_ltrb(orig=True)
        except TypeError:          # versi lama tanpa argumen orig
            box = t.to_ltrb()
        if box is None:
            continue
        conf = t.get_det_conf() if hasattr(t, "get_det_conf") else None
        cls = t.get_det_class() if hasattr(t, "get_det_class") else 0
        rows.append((int(t.track_id), tuple(float(v) for v in box), float(conf if conf is not None else 0.0),
                     int(cls if cls is not None else 0)))
    return rows


class DeepSortAdapter:
    """Bungkus deep_sort_realtime.DeepSort. max_age = toleransi occlusion (frame), n_init = frame konfirmasi."""

    def __init__(self, max_age: int, n_init: int = 3, max_cosine_distance: float = 0.2, use_gpu: bool = True):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:  # pragma: no cover - tergantung environment
            raise ImportError("DeepSORT butuh paket deep-sort-realtime: pip install deep-sort-realtime") from e
        self.tracker = DeepSort(max_age=int(max_age), n_init=int(n_init), max_cosine_distance=max_cosine_distance,
                                embedder="mobilenet", half=use_gpu, bgr=True, embedder_gpu=use_gpu)

    def update(self, frame, xyxy, conf, cls) -> List[tuple]:
        raw = [(xyxy_to_ltwh(b), float(c), int(k)) for b, c, k in zip(xyxy, conf, cls)]
        tracks = self.tracker.update_tracks(raw, frame=frame)
        return deepsort_tracks_to_rows(tracks)
