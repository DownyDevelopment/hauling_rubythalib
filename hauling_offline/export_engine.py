"""R&D: export model detector ke TensorRT (.engine) untuk mempercepat inference.

TensorRT adalah inference engine NVIDIA, BUKAN algoritma tracking. Engine yang dihasilkan bisa dipakai bersama
tracker mana pun:  python main.py --vehicle-model model/<nama>.engine --tracker bytetrack ...

Syarat: GPU NVIDIA + driver, PyTorch versi CUDA, dan paket `tensorrt` (pip install tensorrt).
Engine terikat ke GPU, versi TensorRT, dan imgsz saat export -> export ulang di setiap mesin.

    python main.py export-engine --model model/Hauling_Vehicle_yolov8m_SingleCls_V11.pt --half
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .trackers import save_model_meta

log = logging.getLogger(__name__)


def export(model: Path, imgsz=None, half: bool = True, device: str = "0", workspace=None) -> Path:
    from ultralytics import YOLO

    from .detector import model_train_args

    yolo = YOLO(str(model))
    train_args = model_train_args(yolo)
    imgsz = int(imgsz or train_args.get("imgsz") or 640)   # sama dengan imgsz yang dipakai pipeline untuk .pt
    kwargs = dict(format="engine", imgsz=imgsz, half=half, device=device, dynamic=False, simplify=True)
    if workspace:
        kwargs["workspace"] = workspace
    out = Path(yolo.export(**kwargs))
    meta = save_model_meta(out, {**train_args, "imgsz": imgsz}, model, half=half, export_device=str(device))
    log.info("engine: %s (imgsz=%s, half=%s) | metadata: %s", out, imgsz, half, meta)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="main.py export-engine", description="Export detector .pt ke TensorRT .engine")
    p.add_argument("--model", required=True, type=Path, help="model .pt sumber")
    p.add_argument("--imgsz", type=int, default=None, help="default: imgsz training model")
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="FP16 (default: ya)")
    p.add_argument("--device", default="0", help="GPU index (TensorRT wajib GPU NVIDIA)")
    p.add_argument("--workspace", type=float, default=None, help="workspace TensorRT (GB), opsional")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not a.model.is_file():
        print(f"[ERROR] model tidak ditemukan: {a.model}", file=sys.stderr)
        return 2
    try:
        out = export(a.model, a.imgsz, a.half, a.device, a.workspace)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] export TensorRT gagal: {e}\n"
              "Cek: GPU NVIDIA terdeteksi (nvidia-smi), torch versi CUDA, dan `pip install tensorrt`.", file=sys.stderr)
        return 1
    print(f"Engine siap: {out}")
    return 0
