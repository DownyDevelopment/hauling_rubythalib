"""TASK 2 - Deteksi Kendaraan: adapter detector + multi-object tracker (Ultralytics YOLO + ByteTrack)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

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


def model_train_args(yolo_model) -> dict:
    """train_args yang tersimpan di checkpoint Ultralytics (imgsz, single_cls, dll)."""
    ckpt = getattr(yolo_model, "ckpt", None) or {}
    args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
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
    """Buat file tracker custom dengan track_buffer sesuai fps (toleransi occlusion pendek)."""
    import ultralytics
    base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{tracker}.yaml"
    lines = []
    for line in base.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("track_buffer:"):
            line = f"track_buffer: {int(track_buffer_frames)}"
        lines.append(line)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tracker}_hauling.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


class VehicleTracker:
    """Jalankan model vehicle + tracking per frame. Hasilnya list Detection ber-track_id."""

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
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.train_args = model_train_args(self.model)
        if imgsz is None:
            imgsz = self.train_args.get("imgsz") or 640
        self.conf, self.iou, self.imgsz, self.max_det, self.device = conf, iou, int(imgsz), int(max_det), device
        self.classes = list(classes) if classes is not None else auto_vehicle_classes(self.names, self.train_args)
        tmp = Path(tempfile.mkdtemp(prefix="hauling_trk_"))
        self.tracker_cfg = str(write_tracker_yaml(track_buffer_frames, tmp, tracker))

    def track(self, frame: np.ndarray) -> List[Detection]:
        kwargs = dict(
            persist=True, tracker=self.tracker_cfg, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, max_det=self.max_det, classes=self.classes, verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        res = self.model.track(frame, **kwargs)[0]
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
