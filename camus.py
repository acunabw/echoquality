from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


QUALITY_ALIASES = {
    "good": "Good",
    "buena": "Good",
    "medium": "Medium",
    "moderate": "Medium",
    "media": "Medium",
    "poor": "Poor",
    "bad": "Poor",
    "mala": "Poor",
    "pobre": "Poor",
}


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_quality(value: str) -> str:
    raw = _strip_accents(str(value)).lower().strip()
    raw = re.sub(r"[^a-z]+", " ", raw).strip()
    for token in raw.split():
        if token in QUALITY_ALIASES:
            return QUALITY_ALIASES[token]
    raise ValueError(f"Categoria de calidad desconocida: {value!r}")


def parse_cfg(path: str | Path) -> dict[str, str]:
    p = Path(path)
    data: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        match = re.match(r"^([^:=]+?)\s*[:=]\s*(.*?)\s*$", line)
        if match:
            key = re.sub(r"\s+", "", match.group(1)).lower()
            data[key] = match.group(2).strip()
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            data[re.sub(r"\s+", "", parts[0]).lower()] = parts[1].strip()
    return data


def _quality_from_cfg(cfg: dict[str, str]) -> str:
    for key, value in cfg.items():
        compact = re.sub(r"[^a-z]", "", key.lower())
        if compact in {"imagequality", "quality", "imgquality"}:
            return normalize_quality(value)
    raise KeyError("No se encontro ImageQuality en el archivo CFG")


def _view_from_name(name: str) -> str:
    up = name.upper()
    if "2CH" in up:
        return "2CH"
    if "4CH" in up:
        return "4CH"
    raise ValueError(f"No se pudo inferir vista de {name}")


def discover_cfg_files(root: str | Path) -> list[Path]:
    base = Path(root)
    found: set[Path] = set()
    for pattern in ("**/*Info*2CH*.cfg", "**/*Info*4CH*.cfg", "**/*info*2ch*.cfg", "**/*info*4ch*.cfg"):
        found.update(base.glob(pattern))
    return sorted(found)


def _find_image(patient_dir: Path, patient_id: str, view: str, phase: str, gt: bool = False) -> Path | None:
    gt_token = "_gt" if gt else ""
    candidates: list[Path] = []
    for ext in (".mhd", ".nii.gz", ".nii", ".png", ".tif", ".npy"):
        patterns = [
            f"{patient_id}_{view}_{phase}{gt_token}{ext}",
            f"*{view}*{phase}*{gt_token}*{ext}",
        ]
        for pattern in patterns:
            candidates.extend(patient_dir.glob(pattern))
    # Evita seleccionar GT cuando se busca imagen y viceversa.
    filtered = [p for p in candidates if ("_gt" in p.stem.lower()) == gt]
    return sorted(set(filtered))[0] if filtered else None


def build_manifest(camus_root: str | Path, phases: Iterable[str] = ("ED", "ES")) -> pd.DataFrame:
    """Construye un manifiesto por paciente, vista y fase a partir de CAMUS."""

    rows: list[dict[str, object]] = []
    cfg_files = discover_cfg_files(camus_root)
    if not cfg_files:
        raise FileNotFoundError("No se localizaron Info_2CH.cfg/Info_4CH.cfg")
    for cfg_path in cfg_files:
        patient_dir = cfg_path.parent
        patient_id = patient_dir.name
        view = _view_from_name(cfg_path.name)
        quality = _quality_from_cfg(parse_cfg(cfg_path))
        for phase in phases:
            image_path = _find_image(patient_dir, patient_id, view, phase, gt=False)
            mask_path = _find_image(patient_dir, patient_id, view, phase, gt=True)
            if image_path is None:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "view": view,
                    "phase": phase,
                    "quality": quality,
                    "apt_target": int(quality in {"Good", "Medium"}),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else "",
                    "cfg_path": str(cfg_path),
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("Se encontraron CFG, pero no imagenes ED/ES compatibles")
    return manifest.sort_values(["patient_id", "view", "phase"]).reset_index(drop=True)
