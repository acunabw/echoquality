from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from echoquality.types import ImageMetadata, LoadedImage

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
    x = np.asarray(array)
    x = np.squeeze(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        axis = int(np.argmin(x.shape))
        return np.take(x, x.shape[axis] // 2, axis=axis)
    raise ValueError(f"Expected a 2-D image or a 3-D stack; received shape {x.shape}")


def _bit_depth(dtype: np.dtype[Any]) -> int | None:
    return int(dtype.itemsize * 8) if np.issubdtype(dtype, np.integer) else None


def _metadata(array: np.ndarray, path: Path | None, spacing_yx: tuple[float, float] | None = None) -> ImageMetadata:
    finite = np.asarray(array)[np.isfinite(array)]
    return ImageMetadata(
        path=str(path) if path else None,
        original_shape=tuple(int(v) for v in np.asarray(array).shape),
        dtype=str(np.asarray(array).dtype),
        intensity_min=float(finite.min()) if finite.size else float("nan"),
        intensity_max=float(finite.max()) if finite.size else float("nan"),
        spacing_yx=spacing_yx,
        bit_depth=_bit_depth(np.asarray(array).dtype),
    )


def _parse_mhd_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    return header


def _load_mhd(path: Path) -> tuple[np.ndarray, tuple[float, float] | None]:
    header = _parse_mhd_header(path)
    required = {"DimSize", "ElementType", "ElementDataFile"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Incomplete MHD header {path}; missing {sorted(missing)}")
    dims = tuple(int(value) for value in header["DimSize"].split())
    element_type = header["ElementType"].upper()
    if element_type not in _META_DTYPES:
        raise ValueError(f"Unsupported MHD ElementType: {element_type}")
    dtype = _META_DTYPES[element_type]
    msb = header.get("BinaryDataByteOrderMSB", "False").lower() in {"true", "1"}
    dtype = dtype.newbyteorder(">" if msb else "<")
    raw_name = header["ElementDataFile"]
    if raw_name.upper() == "LOCAL":
        raise ValueError("ElementDataFile=LOCAL is not supported by the lightweight reader")
    data = np.fromfile(path.parent / raw_name, dtype=dtype)
    expected = int(np.prod(dims))
    if data.size != expected:
        raise ValueError(f"Unexpected RAW size {data.size}; expected {expected}")
    array = data.reshape(tuple(reversed(dims)))
    spacing = None
    spacing_raw = header.get("ElementSpacing") or header.get("ElementSize")
    if spacing_raw:
        values = tuple(float(value) for value in spacing_raw.split())
        if len(values) >= 2:
            spacing = (values[1], values[0])
    return array, spacing


def load_image_with_metadata(path_or_array: str | Path | np.ndarray) -> LoadedImage:
    if isinstance(path_or_array, np.ndarray):
        raw = _as_2d(path_or_array)
        return LoadedImage(raw.astype(np.float64, copy=False), _metadata(raw, None))

    path = Path(path_or_array)
    if not path.exists():
        raise FileNotFoundError(path)
    name = path.name.lower()
    spacing: tuple[float, float] | None = None
    if name.endswith(".mhd"):
        raw, spacing = _load_mhd(path)
    elif name.endswith(".npy"):
        raw = np.load(path)
    elif name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("Install nibabel to read NIfTI files") from exc
        image = nib.load(str(path))
        raw = np.asanyarray(image.dataobj)
        zooms = tuple(float(value) for value in image.header.get_zooms())
        if len(zooms) >= 2:
            spacing = (zooms[1], zooms[0])
    else:
        with Image.open(path) as image:
            raw = np.asarray(image.convert("L"))
    raw2d = _as_2d(raw)
    return LoadedImage(raw2d.astype(np.float64, copy=False), _metadata(raw2d, path, spacing))


def load_image(path_or_array: str | Path | np.ndarray) -> np.ndarray:
    return load_image_with_metadata(path_or_array).array


def load_mask(path_or_array: str | Path | np.ndarray) -> np.ndarray:
    return np.rint(load_image(path_or_array)).astype(np.int16)


def robust_scale(
    image: np.ndarray,
    *,
    support: np.ndarray | None = None,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """Scale an image to [0, 1] using percentiles from a declared support.

    This is a computational normalization, not a physical dynamic-range estimate.
    The original image and metadata must be retained for traceability.
    """

    x = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(x)
    valid = finite if support is None else finite & np.asarray(support, dtype=bool)
    if not valid.any():
        return np.zeros_like(x, dtype=np.float64)
    values = x[valid]
    lo, hi = np.percentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        output = np.zeros_like(x, dtype=np.float64)
        output[valid] = 0.5
        return output
    output = (x - lo) / (hi - lo)
    output[~finite] = 0.0
    return np.clip(output, 0.0, 1.0)


def save_grayscale(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scaled = robust_scale(image)
    Image.fromarray(np.round(scaled * 255).astype(np.uint8), mode="L").save(output)
    return output


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(output)
    return output
