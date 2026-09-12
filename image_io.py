from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_META_DTYPES: dict[str, np.dtype[Any]] = {
    "MET_UCHAR": np.dtype("uint8"),
    "MET_CHAR": np.dtype("int8"),
    "MET_USHORT": np.dtype("uint16"),
    "MET_SHORT": np.dtype("int16"),
    "MET_UINT": np.dtype("uint32"),
    "MET_INT": np.dtype("int32"),
    "MET_FLOAT": np.dtype("float32"),
    "MET_DOUBLE": np.dtype("float64"),
}


def _as_2d(array: np.ndarray) -> np.ndarray:
    """Convierte una matriz 2D/3D a una imagen 2D.

    Para una secuencia 3D se selecciona el corte central. En el pipeline real se
    recomienda indicar explicitamente ED o ES en el manifiesto de datos.
    """

    x = np.asarray(array)
    x = np.squeeze(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # Preferimos el eje con menor dimension como eje temporal si es evidente.
        axis = int(np.argmin(x.shape))
        return np.take(x, x.shape[axis] // 2, axis=axis)
    raise ValueError(f"Se esperaba una imagen 2D o 3D; se recibio {x.shape}")


def _parse_mhd_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    return header


def load_mhd(path: str | Path) -> np.ndarray:
    """Lee archivos MetaImage MHD/RAW usados por CAMUS sin requerir SimpleITK."""

    p = Path(path)
    header = _parse_mhd_header(p)
    if "DimSize" not in header or "ElementType" not in header or "ElementDataFile" not in header:
        raise ValueError(f"Encabezado MHD incompleto: {p}")
    dims = tuple(int(v) for v in header["DimSize"].split())
    element_type = header["ElementType"].upper()
    if element_type not in _META_DTYPES:
        raise ValueError(f"ElementType no soportado: {element_type}")
    dtype = _META_DTYPES[element_type]
    msb = header.get("BinaryDataByteOrderMSB", "False").lower() in {"true", "1"}
    dtype = dtype.newbyteorder(">" if msb else "<")
    raw_name = header["ElementDataFile"]
    if raw_name.upper() == "LOCAL":
        raise ValueError("ElementDataFile=LOCAL no esta soportado por el lector ligero")
    raw_path = p.parent / raw_name
    data = np.fromfile(raw_path, dtype=dtype)
    expected = int(np.prod(dims))
    if data.size != expected:
        raise ValueError(f"Tamano RAW inesperado: {data.size}; esperado {expected}")
    # En MetaImage, x es el indice mas rapido; numpy requiere invertir DimSize.
    return data.reshape(tuple(reversed(dims)))


def load_image(path_or_array: str | Path | np.ndarray) -> np.ndarray:
    """Carga PNG/JPG/TIFF/NPY/MHD/NIfTI y devuelve float64 2D."""

    if isinstance(path_or_array, np.ndarray):
        return _as_2d(path_or_array).astype(np.float64, copy=False)
    path = Path(path_or_array)
    name = path.name.lower()
    if name.endswith(".mhd"):
        return _as_2d(load_mhd(path)).astype(np.float64, copy=False)
    if name.endswith(".npy"):
        return _as_2d(np.load(path)).astype(np.float64, copy=False)
    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("Instale nibabel para leer NIfTI") from exc
        return _as_2d(np.asarray(nib.load(str(path)).get_fdata())).astype(np.float64)
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def load_mask(path_or_array: str | Path | np.ndarray) -> np.ndarray:
    return np.rint(load_image(path_or_array)).astype(np.int16)


def robust_scale(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Escala robustamente a [0, 1] usando percentiles."""

    x = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    values = x[finite]
    lo, hi = np.percentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        out = np.zeros_like(x)
        out[finite] = 0.5
        return out
    out = (x - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def save_grayscale(path: str | Path, image: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = robust_scale(image)
    Image.fromarray(np.round(x * 255).astype(np.uint8), mode="L").save(p)
    return p


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(p)
    return p
