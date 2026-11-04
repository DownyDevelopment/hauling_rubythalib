"""TASK 1 - Setup Area: pembaca konfigurasi KM-18.

Hanya membaca parameter yang relevan untuk inference offline (area, jarak,
speed limit, parameter model). Kunci lain (password, RTSP URL, token, endpoint)
TIDAK pernah dibaca, dicetak, ataupun disimpan.

Format config yang didukung: .json, .yaml/.yml, .ini/.cfg/.conf, .env/.txt
(baris KEY=VALUE). Nilai polygon boleh berupa list asli atau string JSON.
Tidak ada eval / literal_eval.
"""
from __future__ import annotations

import configparser
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

REQUIRED_KEYS = (
    "resolution_area",
    "main_area_coordinates_list",
    "speed_area_start_coordinates_list",
    "speed_area_final_coordinates_list",
    "final_area_coordinates_list",
    "distance_area_speed",
    "speed_limit",
)

AREA_KEYS = {
    "main": "main_area_coordinates_list",
    "speed_start": "speed_area_start_coordinates_list",
    "speed_final": "speed_area_final_coordinates_list",
    "final": "final_area_coordinates_list",
}

# Alias parameter model yang *mungkin* ada di config (opsional).
OPTIONAL_MODEL_KEYS = {
    "conf": ("conf_model", "confidence", "conf", "conf_threshold", "confidence_threshold", "vehicle_conf", "model_conf"),
    "iou": ("iou_model", "iou", "iou_threshold", "model_iou", "vehicle_iou"),
    "imgsz": ("imgsz_model", "imgsz", "img_size", "image_size", "input_size", "model_imgsz"),
    "max_det": ("max_det_model", "max_det", "max_detection", "max_detections"),
}


class ConfigError(ValueError):
    """Config tidak dapat dibaca / tidak valid. Pesan selalu bisa ditindaklanjuti."""


@dataclass
class KM18Config:
    resolution: Tuple[int, int]                 # (width, height) basis polygon
    areas: Dict[str, List[Polygon]]             # main/speed_start/speed_final/final -> list polygon
    distance_meter: float
    speed_limit_kmh: float
    travel_direction: str = "Unknown"
    model_params: Dict[str, float] = field(default_factory=dict)
    source_path: str = ""
    auto_generated: bool = False                # True bila area dibuat oleh auto_area.py (belum terkalibrasi)

    def safe_summary(self) -> Dict[str, Any]:
        """Ringkasan aman untuk log/summary (tanpa kredensial)."""
        return {
            "config_file": Path(self.source_path).name,
            "resolution_area": list(self.resolution),
            "distance_area_speed": self.distance_meter,
            "speed_limit": self.speed_limit_kmh,
            "travel_direction": self.travel_direction,
            "n_polygons": {k: len(v) for k, v in self.areas.items()},
            "model_params": self.model_params,
            "auto_generated": self.auto_generated,
        }


# --------------------------------------------------------------------------- #
# Loader mentah
# --------------------------------------------------------------------------- #
def _parse_json(text: str, section: Optional[str], name: str) -> Dict[str, Any]:
    return _flatten_find(json.loads(text), section)


def _parse_yaml(text: str, section: Optional[str], name: str) -> Dict[str, Any]:
    import yaml  # type: ignore  (PyYAML sudah ada di Colab / ikut ultralytics)
    return _flatten_find(yaml.safe_load(text), section)


def _parse_ini(text: str, section: Optional[str], name: str) -> Dict[str, Any]:
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str  # pertahankan huruf besar/kecil
    cp.read_string(text)
    if section:
        sections = [section]
    else:  # pilih otomatis section yang berisi key KM-18 (mis. [CONFIG]); section DB/kredensial tidak ikut dibaca
        sections = [s_ for s_ in cp.sections() if any(_norm(k) in REQUIRED_KEYS for k in cp[s_])] or cp.sections()
    merged: Dict[str, Any] = dict(cp.defaults())
    for s in sections:
        if s not in cp:
            raise ConfigError(f"Section '{s}' tidak ada di {name}. Tersedia: {cp.sections()}")
        merged.update(dict(cp[s]))
    return merged


def _parse_kv(text: str, section: Optional[str], name: str) -> Dict[str, Any]:
    return _parse_key_value(text)


_PARSERS = {"json": _parse_json, "yaml": _parse_yaml, "ini": _parse_ini, "key=value": _parse_kv}
_BY_SUFFIX = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".ini": "ini", ".cfg": "ini", ".conf": "ini"}


def _load_raw(path: Path, section: Optional[str]) -> Dict[str, Any]:
    """Baca config. Format ditebak dari ekstensi dulu, lalu dicoba format lain bila isi file tidak cocok
    (mis. file bernama .json tapi isinya KEY=VALUE / INI)."""
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ConfigError(f"File config kosong (0 isi): {path}. Isi dulu file-nya atau arahkan CONFIG ke file config KM-18 yang asli.")

    first = _BY_SUFFIX.get(path.suffix.lower(), "key=value")
    order = [first] + [f for f in _PARSERS if f != first]
    errors = []
    for fmt in order:
        try:
            raw = _PARSERS[fmt](text, section, path.name)
        except ConfigError:
            raise
        except Exception as e:  # noqa: BLE001 - format salah, coba format berikutnya
            errors.append(f"{fmt}: {type(e).__name__}")
            continue
        if isinstance(raw, dict) and any(_norm(k) in REQUIRED_KEYS for k in raw):
            return raw
        errors.append(f"{fmt}: tidak ada key KM-18")
    raise ConfigError(
        f"Config {path.name} tidak bisa dibaca sebagai JSON/YAML/INI/KEY=VALUE yang berisi key KM-18 "
        f"({'; '.join(errors)}). Pastikan file berisi key seperti 'main_area_coordinates_list'."
    )


def _parse_key_value(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        out[key.strip()] = val
    return out


def _flatten_find(data: Any, section: Optional[str]) -> Dict[str, Any]:
    """Cari dict yang memuat key wajib (config bisa nested per kamera)."""
    if section is not None:
        if not isinstance(data, dict) or section not in data:
            raise ConfigError(f"Section/kamera '{section}' tidak ditemukan di config.")
        data = data[section]

    def has_keys(d: Any) -> bool:
        return isinstance(d, dict) and any(_norm(k) in REQUIRED_KEYS for k in d.keys())

    stack = [data]
    while stack:
        node = stack.pop(0)
        if has_keys(node):
            return dict(node)
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return data if isinstance(data, dict) else {}


def _norm(key: str) -> str:
    return str(key).strip().lower()


# --------------------------------------------------------------------------- #
# Parser nilai
# --------------------------------------------------------------------------- #
def _to_structure(value: Any, key: str) -> Any:
    """String -> struktur Python via JSON (tanpa eval). Tuple '(x, y)' diubah ke list."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    s = s.replace("(", "[").replace(")", "]").replace("'", '"')
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Nilai '{key}' bukan JSON yang valid: {e}") from e


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_point(p: Any, key: str) -> Point:
    if isinstance(p, dict):
        if "x" in p and "y" in p:
            p = [p["x"], p["y"]]
        else:
            raise ConfigError(f"Titik di '{key}' harus [x, y] atau {{x, y}}: {p}")
    if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_num(v) for v in p)):
        raise ConfigError(f"Titik di '{key}' harus pasangan angka [x, y], didapat: {p!r}")
    return float(p[0]), float(p[1])


def parse_polygons(value: Any, key: str = "polygon") -> List[Polygon]:
    """Terima satu polygon atau list polygon -> selalu list polygon.

    Bentuk yang diterima:
      [[x,y],[x,y],[x,y]]                 -> 1 polygon
      [[[x,y],...], [[x,y],...]]          -> beberapa polygon
      [x1,y1,x2,y2,x3,y3]                 -> 1 polygon (flat)
    """
    data = _to_structure(value, key)
    if not isinstance(data, list) or not data:
        raise ConfigError(f"'{key}' harus list koordinat, didapat: {type(data).__name__}")

    if all(_is_num(v) for v in data):  # flat list
        if len(data) % 2:
            raise ConfigError(f"'{key}' flat list harus berjumlah genap (x,y).")
        raw_polys = [[data[i:i + 2] for i in range(0, len(data), 2)]]
    elif all(isinstance(p, (list, tuple, dict)) and (isinstance(p, dict) or (len(p) == 2 and all(_is_num(v) for v in p))) for p in data):
        raw_polys = [data]
    else:
        raw_polys = data

    polys: List[Polygon] = []
    for i, poly in enumerate(raw_polys):
        if isinstance(poly, list) and poly and all(_is_num(v) for v in poly):
            poly = [poly[j:j + 2] for j in range(0, len(poly), 2)]
        if not isinstance(poly, list):
            raise ConfigError(f"'{key}' polygon #{i} tidak valid.")
        pts = [_as_point(p, key) for p in poly]
        if len(pts) < 3:
            raise ConfigError(f"'{key}' polygon #{i} hanya {len(pts)} titik (minimal 3).")
        polys.append(pts)
    return polys


def parse_resolution(value: Any) -> Tuple[int, int]:
    if isinstance(value, dict):
        w = value.get("width") or value.get("w")
        h = value.get("height") or value.get("h")
        nums = [w, h]
    elif isinstance(value, (list, tuple)):
        nums = list(value)
    else:
        nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    try:
        w, h = int(float(nums[0])), int(float(nums[1]))
    except (IndexError, TypeError, ValueError) as e:
        raise ConfigError(f"resolution_area tidak valid: {value!r} (contoh: 1920x1080)") from e
    if w <= 0 or h <= 0:
        raise ConfigError(f"resolution_area harus positif: {value!r}")
    return w, h


def parse_number(value: Any, key: str) -> float:
    data = value
    if isinstance(value, str):
        m = re.findall(r"-?\d+(?:\.\d+)?", value)
        if not m:
            raise ConfigError(f"'{key}' harus angka, didapat: {value!r}")
        data = [float(x) for x in m]
    if isinstance(data, (list, tuple)):
        if not data:
            raise ConfigError(f"'{key}' kosong.")
        data = data[0]  # KM-18 saat ini: satu jarak di antara dua area speed
    if not _is_num(data):
        raise ConfigError(f"'{key}' harus angka, didapat: {value!r}")
    return float(data)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_config(path: str | Path, section: Optional[str] = None) -> KM18Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"File config tidak ditemukan: {path}")
    raw = _load_raw(path, section)
    return config_from_dict(raw, source_path=str(path))


def config_from_dict(raw: Dict[str, Any], source_path: str = "") -> KM18Config:
    norm = {_norm(k): v for k, v in raw.items()}
    missing = [k for k in REQUIRED_KEYS if k not in norm or norm[k] in (None, "")]
    if missing:
        raise ConfigError("Key wajib tidak ada di config KM-18: " + ", ".join(missing))

    distance = parse_number(norm["distance_area_speed"], "distance_area_speed")
    if distance <= 0:
        raise ConfigError("distance_area_speed harus > 0.")
    speed_limit = parse_number(norm["speed_limit"], "speed_limit")

    areas = {name: parse_polygons(norm[key], key) for name, key in AREA_KEYS.items()}

    direction = str(norm.get("travel_direction", "Unknown")).strip().strip('"').capitalize() or "Unknown"

    model_params: Dict[str, float] = {}
    for target, aliases in OPTIONAL_MODEL_KEYS.items():
        for a in aliases:
            if a in norm and norm[a] not in (None, ""):
                try:
                    model_params[target] = parse_number(norm[a], a)
                except ConfigError:
                    pass
                break

    return KM18Config(
        resolution=parse_resolution(norm["resolution_area"]),
        areas=areas,
        distance_meter=distance,
        speed_limit_kmh=speed_limit,
        travel_direction=direction,
        model_params=model_params,
        source_path=source_path,
        auto_generated=str(norm.get("auto_generated", "")).strip().lower() in ("true", "1", "yes"),
    )
