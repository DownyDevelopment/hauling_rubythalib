"""EKSTENSI Task 4 - Text recognition nomor lambung (mis. "AEK 0078").

PRD menempatkan OCR sebagai task lanjutan (kolom plate_text). Modul ini OPSIONAL: aktif hanya bila
--ocr-model-dir diberikan. Rantai model (semua dari folder model/):

    bbox nomor lambung (dari Task 4)
      -> detection_hauling_angka_huruf_V1        : pisahkan blok huruf ("AEK") & blok angka ("0078")
      -> character_huruf/angka_detection_*_V2    : posisi tiap karakter di dalam blok   (imgsz 224)
      -> classification_hauling_huruf/angka_*_v9 : baca tiap karakter (A-Z / 0-9)         (imgsz 96)
      -> "AEK 0078"

Catatan temuan: pada checkpoint detection_hauling_angka_huruf_V1 nama kelas tampak TERTUKAR
(blok huruf berlabel 'angka' dan sebaliknya). Karena itu default block_order='position':
blok kiri = huruf, blok kanan = angka (format nomor lambung "HHH 0000").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plate import crop_with_offset

# (x1, x2, label, cls_conf)
Char = Tuple[float, float, str, float]

OCR_MODEL_PATTERNS = {
    "block": "detection_hauling_angka_huruf*.pt",
    "huruf_det": "character_huruf_detection*.pt",
    "angka_det": "character_angka_detection*.pt",
    "huruf_cls": "classification_hauling_huruf*.pt",
    "angka_cls": "classification_hauling_angka*.pt",
}


@dataclass(frozen=True)
class OcrResult:
    text: str          # "AEK 0078"
    huruf: str
    angka: str
    confidence: float  # rata-rata confidence klasifikasi karakter


def clean_chars(chars: Sequence[Char], min_width_ratio: float = 0.6, min_cls_conf: float = 0.6) -> List[Char]:
    """Buang karakter palsu di tepi blok: terlalu sempit (< ratio x median lebar) atau confidence klasifikasi rendah.

    Urutan hasil = kiri ke kanan.
    """
    if not chars:
        return []
    widths = [c[1] - c[0] for c in chars]
    med = float(np.median(widths))
    kept = [c for c, w in zip(chars, widths) if w >= min_width_ratio * med and c[3] >= min_cls_conf]
    return sorted(kept, key=lambda c: c[0])


def find_ocr_models(model_dir: str | Path) -> Dict[str, Path]:
    d = Path(model_dir)
    found: Dict[str, Path] = {}
    missing = []
    for key, pat in OCR_MODEL_PATTERNS.items():
        hits = sorted(d.glob(pat))
        if hits:
            found[key] = hits[-1]  # versi terakhir secara nama
        else:
            missing.append(pat)
    if missing:
        raise FileNotFoundError(f"Model OCR tidak ditemukan di {d}: {missing}")
    return found


class LambungOCR:
    def __init__(
        self,
        models: Dict[str, Path],
        device: Optional[str] = None,
        block_order: str = "position",
        pad: float = 0.15,
        block_pad: float = 0.1,
        block_conf: float = 0.25,
        char_conf: float = 0.25,
        min_cls_conf: float = 0.6,
    ):
        from ultralytics import YOLO
        self.m = {k: YOLO(str(v)) for k, v in models.items()}
        self.files = {k: Path(v).name for k, v in models.items()}
        if block_order not in ("position", "class"):
            raise ValueError("block_order harus 'position' atau 'class'")
        self.block_order, self.pad, self.block_pad = block_order, pad, block_pad
        self.block_conf, self.char_conf, self.min_cls_conf = block_conf, char_conf, min_cls_conf
        self.device = device

    def _kw(self, **kw):
        if self.device:
            kw["device"] = self.device
        kw["verbose"] = False
        return kw

    def _blocks(self, crop: np.ndarray) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
        m = self.m["block"]
        r = m.predict(crop, **self._kw(imgsz=640, conf=self.block_conf))[0]
        if r.boxes is None or len(r.boxes) < 2:
            return None
        boxes = [([float(v) for v in b.xyxy[0]], float(b.conf), str(m.names[int(b.cls)]).lower()) for b in r.boxes]
        top2 = sorted(boxes, key=lambda b: -b[1])[:2]
        if self.block_order == "position":
            left, right = sorted(top2, key=lambda b: b[0][0])
            return {"huruf": tuple(map(int, left[0])), "angka": tuple(map(int, right[0]))}
        out = {name: tuple(map(int, xyxy)) for xyxy, _, name in top2}
        return out if {"huruf", "angka"} <= set(out) else None

    def _read_block(self, block: np.ndarray, kind: str) -> List[Char]:
        det, cls = self.m[f"{kind}_det"], self.m[f"{kind}_cls"]
        r = det.predict(block, **self._kw(imgsz=224, conf=self.char_conf, iou=0.5))[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = sorted(([int(v) for v in b.xyxy[0]] for b in r.boxes), key=lambda b: b[0])
        crops = [block[max(0, y1):y2, max(0, x1):x2] for x1, y1, x2, y2 in boxes]
        keep = [(b, c) for b, c in zip(boxes, crops) if c.size > 0]
        if not keep:
            return []
        res = cls.predict([c for _, c in keep], **self._kw(imgsz=96))
        chars: List[Char] = []
        for (b, _), p in zip(keep, res):
            chars.append((b[0], b[2], str(cls.names[int(p.probs.top1)]), float(p.probs.top1conf)))
        return clean_chars(chars, min_cls_conf=self.min_cls_conf)

    def read(self, frame: np.ndarray, lambung_xyxy: Sequence[float]) -> Optional[OcrResult]:
        """Baca nomor lambung dari bbox (koordinat frame). None bila tidak terbaca lengkap."""
        crop, _ = crop_with_offset(frame, lambung_xyxy, pad=self.pad)
        if crop is None:
            return None
        blocks = self._blocks(crop)
        if not blocks:
            return None
        parts: Dict[str, List[Char]] = {}
        for kind in ("huruf", "angka"):
            blk, _ = crop_with_offset(crop, blocks[kind], pad=self.block_pad)
            if blk is None:
                return None
            parts[kind] = self._read_block(blk, kind)
        huruf = "".join(c[2] for c in parts["huruf"])
        angka = "".join(c[2] for c in parts["angka"])
        if not huruf or not angka:
            return None
        confs = [c[3] for c in parts["huruf"] + parts["angka"]]
        return OcrResult(f"{huruf} {angka}", huruf, angka, round(float(np.mean(confs)), 4))
