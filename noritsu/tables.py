"""Decoded Noritsu CorrectServ tables with explicit group/type selection.

Only numeric fields whose layout has been verified against ``Read_Pcb_Para``
and the image kernels are exposed. Real tables can be selected with
``NORITSU_TABLES``; otherwise the checked-in numeric extraction is used.
"""

from __future__ import annotations

import json
import os
import hashlib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


_HERE = Path(__file__).resolve().parent
_VENDOR_JSON = _HERE / "tables" / "vendor_nega_135.json"

DEFAULT_GROUP = 0
DEFAULT_TYPE = 1
DEFAULT_VARIANT = "00000001"  # compatibility name; Nega main-quality 1
_GENERIC_SIZE = 0x6006
_COMMON_SIZE = 0x567A
_CALC_SIZE = 0xFF20
_NORMAL_OUTPUT_PATTERN = 5


def tables_dir() -> Optional[str]:
    """Return a configured real ``CorrectServ`` directory, if present."""

    return os.environ.get("NORITSU_TABLES")


def _normalize_group(group: int | str) -> int:
    if isinstance(group, str):
        group = int(group.removeprefix("IQGrp"))
    group = int(group)
    if group not in (0, 1):
        raise ValueError("quality group must be 0 or 1")
    return group


def _normalize_type(quality_type: int | str) -> int:
    quality_type = int(quality_type)
    if quality_type not in (0, 1):
        raise ValueError(
            "Nega quality type must be 0 or 1; ImgQltMrg.xml does not map Nega to types 2-6"
        )
    return quality_type


def _variant_key(group: int | str, quality_type: int | str) -> str:
    return f"g{_normalize_group(group)}_t{_normalize_type(quality_type)}"


def _configured_root_key() -> Optional[str]:
    configured = tables_dir()
    if not configured:
        return None
    return str(Path(configured).expanduser().resolve())


def _real_variant_dir(
    group: int, quality_type: int, root_key: Optional[str] = None
) -> Optional[Path]:
    if root_key is None:
        return None
    return (
        Path(root_key)
        / f"IQGrp{group:03d}"
        / f"{quality_type:08d}"
        / "PCB_PARA"
    )


@lru_cache(maxsize=1)
def _shipped_document() -> dict:
    with _VENDOR_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def available_variants() -> list[str]:
    """Return supported ``group:type`` selections."""

    return [
        "%d:%d" % (group, quality_type) for group in (0, 1) for quality_type in (0, 1)
    ]


def _load_film_lut_cached(
    group: int | str = DEFAULT_GROUP,
    quality_type: int | str = DEFAULT_TYPE,
    root_key: Optional[str] = None,
) -> np.ndarray:
    """Return the exact 3x4096 Nega film LUT as RGB ``uint16``.

    The generic file is a six-byte header followed by R, G and B arrays. The
    RD5X row kernel applies them to the corresponding BGR input slots in
    reverse slot order; :func:`noritsu.vendor.apply_film_lut` performs that
    channel reorder explicitly.
    """

    group = _normalize_group(group)
    quality_type = _normalize_type(quality_type)
    directory = _real_variant_dir(group, quality_type, root_key)
    if directory is not None:
        path = directory / "NEGA_para.dat"
        if path.is_file():
            data = path.read_bytes()
            if len(data) < _GENERIC_SIZE or data[:6] != b"\x01\x01\x00\x60\x00\x00":
                raise ValueError(f"{path}: invalid NEGA generic table")
            return (
                np.frombuffer(data, dtype="<u2", count=3 * 4096, offset=6)
                .reshape(3, 4096)
                .copy()
            )

    # All four installer-backed Nega selections have the same generic LUT.
    # Still validate the requested key so unsupported selections cannot silently
    # fall back to it.
    _shipped_document()["variants"][_variant_key(group, quality_type)]
    values = _shipped_document()["film_lut_rgb_u16"]
    return np.asarray(values, dtype=np.uint16)


def load_film_lut(
    group: int | str = DEFAULT_GROUP,
    quality_type: int | str = DEFAULT_TYPE,
) -> np.ndarray:
    """Return an independent copy of the selected exact Nega FilmLUT."""

    return _load_film_lut_cached(
        group, quality_type, _configured_root_key()
    ).copy()


def _load_common_control_points_cached(
    group: int | str = DEFAULT_GROUP,
    quality_type: int | str = DEFAULT_TYPE,
    root_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return CommonCalc's three verified ``(x, y)`` 20-point pairs.

    ``x`` is ``uint32`` in the 0..65535 input domain. ``y`` is signed 64-bit
    because the vendor tables intentionally overshoot 65535 before later
    anchoring and clipping. The first 20 words are X—not Y as the original
    implementation claimed.
    """

    group = _normalize_group(group)
    quality_type = _normalize_type(quality_type)
    directory = _real_variant_dir(group, quality_type, root_key)
    if directory is not None:
        path = directory / "CommonCalcPara.Dat"
        if path.is_file():
            data = path.read_bytes()
            if len(data) < _COMMON_SIZE:
                raise ValueError(f"{path}: truncated CommonCalcPara.Dat")
            words = np.frombuffer(data, dtype="<u4", count=120).reshape(6, 20)
            return words[0::2].copy(), words[1::2].astype(np.int64)

    points = _shipped_document()["variants"][_variant_key(group, quality_type)][
        "common_gamma_control_points"
    ]
    x = np.asarray([item["x_u16"] for item in points], dtype=np.uint32)
    y = np.asarray([item["y_i32"] for item in points], dtype=np.int64)
    return x, y


def load_common_control_points(
    group: int | str = DEFAULT_GROUP,
    quality_type: int | str = DEFAULT_TYPE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return independent copies of the selected CommonCalc controls."""

    x, y = _load_common_control_points_cached(
        group, quality_type, _configured_root_key()
    )
    return x.copy(), y.copy()


def source_provenance(
    group: int | str = DEFAULT_GROUP,
    quality_type: int | str = DEFAULT_TYPE,
) -> dict:
    """Return sizes and SHA-256 hashes recorded for the source files."""

    group = _normalize_group(group)
    quality_type = _normalize_type(quality_type)
    directory = _real_variant_dir(group, quality_type, _configured_root_key())
    if directory is not None:
        sizes = {
            "NEGA_para.dat": _GENERIC_SIZE,
            "CommonCalcPara.Dat": _COMMON_SIZE,
            "NEGA_135_COL_calc.dat": _CALC_SIZE,
        }
        if all((directory / name).is_file() for name in sizes):
            result = {}
            for name, consumed_size in sizes.items():
                data = (directory / name).read_bytes()
                result[name] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "file_size": len(data),
                    "consumed_size": consumed_size,
                    "path": str((directory / name).resolve()),
                }
            return result
    key = _variant_key(group, quality_type)
    return deepcopy(_shipped_document()["variants"][key]["source"])


def load_normal_output_selectors(
    group: int | str = DEFAULT_GROUP,
    quality_type: int | str = DEFAULT_TYPE,
) -> dict[str, int]:
    """Return ExecPattern 5's 33 base selector bytes for normal film output.

    ``CreateOutputForFilm`` chooses pattern 5. ``FUN_100047f0`` reads source
    row ``r`` at ``30*r + 5`` and maps the destination across ``0x10..0x1f``
    then ``0x21..0x31`` (there is no destination selector ``0x20``).
    Later builders may rewrite these base values from frame/controller state.
    """

    group = _normalize_group(group)
    quality_type = _normalize_type(quality_type)
    directory = _real_variant_dir(group, quality_type, _configured_root_key())
    if directory is not None:
        path = directory / "NEGA_135_COL_calc.dat"
        if path.is_file():
            data = path.read_bytes()
            if len(data) < _CALC_SIZE:
                raise ValueError(f"{path}: truncated NEGA_135_COL_calc.dat")
            return {
                (f"0x{0x10 + row:02x}" if row < 16 else f"0x{0x11 + row:02x}"): data[
                    30 * row + _NORMAL_OUTPUT_PATTERN
                ]
                for row in range(33)
            }
    key = _variant_key(group, quality_type)
    return dict(_shipped_document()["variants"][key]["normal_output_selectors"])


def load_base_tone_curve(variant: str = DEFAULT_VARIANT) -> list[float]:
    """Compatibility wrapper returning channel-0 CommonCalc Y controls.

    Unlike the old loader, this reads the second 20-word block and therefore
    no longer mistakes X coordinates for outputs. Prefer
    :func:`load_common_control_points` when constructing a real curve because
    its X coordinates are non-uniform.
    """

    quality_type = _normalize_type(variant)
    _, y = load_common_control_points(DEFAULT_GROUP, quality_type)
    return np.clip(y[0] / 65535.0, 0.0, 1.0).tolist()


def tone_curve_lut(n: int = 4096, variant: str = DEFAULT_VARIANT) -> list[float]:
    """Compatibility LUT using the recovered non-uniform X/Y coordinates."""

    if n < 2:
        raise ValueError("tone LUT size must be at least two")
    quality_type = _normalize_type(variant)
    x, y = load_common_control_points(DEFAULT_GROUP, quality_type)
    domain = np.linspace(0.0, 65535.0, n)
    return np.clip(np.interp(domain, x[0], y[0]) / 65535.0, 0.0, 1.0).tolist()
