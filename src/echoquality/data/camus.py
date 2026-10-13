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
    return "".join(character for character in unicodedata.normalize("NFKD", text) if not unicodedata.combining(character))


def normalize_quality(value: str) -> str:
    raw = re.sub(r"[^a-z]+", " ", _strip_accents(str(value)).lower()).strip()
    for token in raw.split():
        if token in QUALITY_ALIASES:
            return QUALITY_ALIASES[token]
    raise ValueError(f"Unknown CAMUS quality category: {value!r}")


def parse_cfg(path: str | Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        match = re.match(r"^([^:=]+?)\s*[:=]\s*(.*?)\s*$", line)
        if match:
            cfg[re.sub(r"\s+", "", match.group(1)).lower()] = match.group(2).strip()
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            cfg[re.sub(r"\s+", "", parts[0]).lower()] = parts[1].strip()
    return cfg


def _quality_from_cfg(cfg: dict[str, str]) -> str:
    for key, value in cfg.items():
        if re.sub(r"[^a-z]", "", key.lower()) in {"imagequality", "quality", "imgquality"}:
            return normalize_quality(value)
    raise KeyError("ImageQuality was not found in CAMUS CFG")


def _view_from_name(name: str) -> str:
    upper = name.upper()
    if "2CH" in upper:
        return "2CH"
    if "4CH" in upper:
        return "4CH"
    raise ValueError(f"Could not infer CAMUS view from {name}")


def discover_cfg_files(root: str | Path) -> list[Path]:
    base = Path(root)
    found: set[Path] = set()
    for pattern in ("**/*Info*2CH*.cfg", "**/*Info*4CH*.cfg", "**/*info*2ch*.cfg", "**/*info*4ch*.cfg"):
        found.update(base.glob(pattern))
    return sorted(found)


def _find_image(patient_dir: Path, patient_id: str, view: str, phase: str, *, ground_truth: bool) -> Path | None:
    gt_token = "_gt" if ground_truth else ""
    extensions = (".mhd", ".nii.gz", ".nii", ".png", ".tif", ".tiff", ".npy")
    candidates: set[Path] = set()
    files = [path for path in patient_dir.iterdir() if path.is_file()]
    for extension in extensions:
        exact = patient_dir / f"{patient_id}_{view}_{phase}{gt_token}{extension}"
        if exact.exists():
            candidates.add(exact)
        for file_path in files:
            lower = file_path.name.lower()
            if view.lower() not in lower or phase.lower() not in lower or not lower.endswith(extension):
                continue
            has_gt = "_gt" in lower
            if has_gt == ground_truth:
                candidates.add(file_path)
    return sorted(candidates)[0] if candidates else None


def build_manifest(camus_root: str | Path, phases: Iterable[str] = ("ED", "ES")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cfg_files = discover_cfg_files(camus_root)
    if not cfg_files:
        raise FileNotFoundError("No CAMUS Info_2CH.cfg or Info_4CH.cfg files were found")
    for cfg_path in cfg_files:
        patient_dir = cfg_path.parent
        patient_id = patient_dir.name
        view = _view_from_name(cfg_path.name)
        quality = _quality_from_cfg(parse_cfg(cfg_path))
        for phase in phases:
            image_path = _find_image(patient_dir, patient_id, view, phase, ground_truth=False)
            mask_path = _find_image(patient_dir, patient_id, view, phase, ground_truth=True)
            if image_path is None:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "view": view,
                    "phase": phase,
                    "quality": quality,
                    "apt_target": int(quality in {"Good", "Medium"}),
                    "not_apt": int(quality == "Poor"),
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()) if mask_path else "",
                    "cfg_path": str(cfg_path.resolve()),
                    "source_dataset": "CAMUS",
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("CFG files were found, but no compatible ED/ES images were located")
    return manifest.sort_values(["patient_id", "view", "phase"]).reset_index(drop=True)
