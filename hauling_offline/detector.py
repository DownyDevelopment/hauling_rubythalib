"""TASK 2 - Deteksi Kendaraan: adapter detector + multi-object tracker.

Detector: Ultralytics YOLO (.pt) atau hasil export TensorRT (.engine) / ONNX.
Tracker : bytetrack (default) / botsort / botsort-reid / deepsort  -> lihat trackers.py.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .trackers import (TRACKERS, ULTRALYTICS_TRACKERS, DeepSortAdapter, Stopwatch, TimingStats,
                       load_model_meta, patch_tracker_yaml)

# Cegah ultralytics melakukan sync/telemetry/download apa pun saat inference.
os.environ.setdefault("YOLO_OFFLINE", "1")

COCO_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class Detection:
    track_id: Optional[int]          # None = belum punya track_id -> tidak boleh membuat event
    bbox_xyxy: tuple                 # (x1, y1, x2, y2) float, koordinat frame
    confidence: float
    class_id: int
    class_name: str = "Vehicle"      # output task ini selalu kategori "Vehicle"


def model_train_args(yolo_model, model_path: Optional[str] = None) -> dict:
    """train_args checkpoint Ultralytics (imgsz, single_cls, dll). Model hasil export (.engine/.onnx) tidak
    menyimpannya, jadi dibaca dari sidecar `<model>.hauling.json` yang ditulis `main.py export-engine`."""
    ckpt = getattr(yolo_model, "ckpt", None) or {}
    args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
    if not args and model_path:
        args = load_model_meta(model_path)
    return dict(args or {})


def auto_vehicle_classes(names: dict, train_args: Optional[dict] = None) -> Optional[List[int]]:
    """Tentukan filter kelas untuk model vehicle.

    - Model dilatih single_cls (mis. Hauling_Vehicle_yolov8m_SingleCls_*): SEMUA deteksi keluar sebagai
      kelas 0, walau names berisi 5 nama -> jangan difilter (None).
    - Ada kelas bernama persis 'vehicle' -> pakai kelas itu.
    - Model COCO -> car/motorcycle/bus/truck.
    - Lainnya -> semua kelas.
    """
    if (train_args or {}).get("single_cls"):
        return None
    lower = {int(i): str(n).strip().lower() for i, n in names.items()}
    exact = [i for i, n in lower.items() if n == "vehicle"]
    if exact:
        return exact
    coco = [i for i, n in lower.items() if n in COCO_VEHICLE_NAMES]
    if len(coco) == len(COCO_VEHICLE_NAMES) and len(lower) >= 80:
        return coco
    return None


def write_tracker_yaml(track_buffer_frames: int, out_dir: Path, tracker: str = "bytetrack") -> Path:
    """Salin config tracker bawaan Ultralytics, ubah track_buffer (toleransi occlusion) dan with_reid."""
    import ultralytics
    base_name = ULTRALYTICS_TRACKERS[tracker]
    base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{base_name}.yaml"
    with_reid = (tracker == "botsort-reid") if base_name == "botsort" else None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tracker}_hauling.yaml"
    out.write_text(patch_tracker_yaml(base.read_text(encoding="utf-8"), track_buffer_frames, with_reid),
                   encoding="utf-8")
    return out


class VehicleTracker:
    """Jalankan model vehicle + tracking per frame. Hasilnya list Detection ber-track_id.

    Waktu per frame dicatat di self.timing (deteksi vs tracking) untuk R&D perbandingan tracker.
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: Optional[int] = None,       # None -> imgsz saat training model
        max_det: int = 100,
        device: Optional[str] = None,
        classes: Optional[Sequence[int]] = None,
        track_buffer_frames: int = 60,
        tracker: str = "bytetrack",
    ):
        if tracker not in TRACKERS:
            raise ValueError(f"tracker tidak dikenal: {tracker} (pilihan: {', '.join(TRACKERS)})")
        from ultralytics import YOLO
        self.tracker_name = tracker
        self.backend = Path(model_path).suffix.lower().lstrip(".") or "pt"   # pt / engine / onnx
        self.model = YOLO(model_path, task="detect")
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.train_args = model_train_args(self.model, model_path)
        if imgsz is None:
            imgsz = self.train_args.get("imgsz") or 640
        self.conf, self.iou, self.imgsz, self.max_det, self.device = conf, iou, int(imgsz), int(max_det), device
        self.classes = list(classes) if classes is not None else auto_vehicle_classes(self.names, self.train_args)
        self.timing = TimingStats()
        self.deepsort: Optional[DeepSortAdapter] = None
        self.tracker_cfg: Optional[str] = None
        if tracker == "deepsort":
            self.deepsort = DeepSortAdapter(max_age=track_buffer_frames, use_gpu=_use_gpu(device))
        else:
            tmp = Path(tempfile.mkdtemp(prefix="hauling_trk_"))
            self.tracker_cfg = str(write_tracker_yaml(track_buffer_frames, tmp, tracker))

    def _kwargs(self) -> dict:
        kwargs = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, max_det=self.max_det,
                      classes=self.classes, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        return kwargs

    def track(self, frame: np.ndarray) -> List[Detection]:
        sw = Stopwatch()
        if self.deepsort is not None:
            res = self.model.predict(frame, **self._kwargs())[0]
            detect_ms = sw.ms()
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                rows = self.deepsort.update(frame, [], [], [])
            else:
                rows = self.deepsort.update(frame, boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                            boxes.cls.cpu().numpy().astype(int))
            self.timing.add(detect_ms, sw.ms())
            return [Detection(track_id=t, bbox_xyxy=b, confidence=round(c, 4), class_id=k) for t, b, c, k in rows]

        res = self.model.track(frame, persist=True, tracker=self.tracker_cfg, **self._kwargs())[0]
        total_ms = sw.ms()
        # res.speed = waktu preprocess + inference + postprocess (NMS) model; sisanya = tracker
        detect_ms = sum(v for v in (res.speed or {}).values() if v is not None)
        self.timing.add(detect_ms, total_ms)
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        return [
            Detection(
                track_id=(int(t) if t is not None else None),
                bbox_xyxy=tuple(float(v) for v in b),
                confidence=round(float(c), 4),
                class_id=int(k),
            )
            for b, c, k, t in zip(xyxy, conf, cls, ids)
        ]


def _use_gpu(device: Optional[str]) -> bool:
    if device is not None and str(device).lower() == "cpu":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False
