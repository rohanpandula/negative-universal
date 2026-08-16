"""Decode the verified subset of an EZ Controller ICCS ``Original/*.prm`` file.

The captured file is a flat correction-file (CF) serialization, not the frame
XML used elsewhere by GFunction.  This parser intentionally exposes only nodes
whose offsets and consumers were confirmed in the matching structure parsers.
The installed loader rejects the original 2017 producer stamp, while a
disposable stamp/checksum compatibility copy has been verified to execute the
2015 builder without changing the source capture.  This decoder never mutates
either form.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any


SUPPORTED_SIZES = (0x2C098, 0x2C8E0, 0x504E8)
AUDITED_LAYOUT_SIZE = 0x504E8
INSTALLED_LOADER_VERSION = "18:41:09 on Jul 29 2015"


def _i32_node(data: bytes, offset: int) -> dict[str, int]:
    count, value = struct.unpack_from("<ii", data, offset)
    return {"count": count, "value": value}


def _f64_node(data: bytes, offset: int) -> dict[str, float | int]:
    count = struct.unpack_from("<i", data, offset)[0]
    value = struct.unpack_from("<d", data, offset + 8)[0]
    return {"count": count, "value": value}


def _utf16z(data: bytes, offset: int, maximum_code_units: int = 128) -> str:
    end = offset
    limit = min(len(data), offset + maximum_code_units * 2)
    while end + 1 < limit and data[end : end + 2] != b"\0\0":
        end += 2
    return data[offset:end].decode("utf-16-le")


def _signed_byte_sum(data: bytes) -> int:
    return sum(value if value < 128 else value - 256 for value in data)


def decode_prm(path: str | Path) -> dict[str, Any]:
    """Validate and decode the audited ICCS fields from *path*."""

    source = Path(path).resolve()
    data = source.read_bytes()
    if len(data) not in SUPPORTED_SIZES:
        raise ValueError(
            f"{source}: unsupported ICCS correction-file size 0x{len(data):x}"
        )
    if len(data) != AUDITED_LAYOUT_SIZE:
        raise ValueError(
            f"{source}: size is loader-known but its field layout was not audited"
        )

    stored_checksum = struct.unpack_from("<i", data, 0)[0]
    computed_checksum = _signed_byte_sum(data[4:])
    declared_size = struct.unpack_from("<I", data, 0xA0)[0]
    version = data[4:64].split(b"\0", 1)[0].decode("ascii")
    root_count = struct.unpack_from("<i", data, 0xB8)[0]
    root_name = _utf16z(data, 0xBC)

    adjustment_offsets = {
        "automatic_contrast_overall": 0xE38,
        "automatic_contrast_shadow": 0xE40,
        "automatic_contrast_highlight": 0xE48,
        "sharpness": 0xE50,
        "saturation_cf_delta": 0xE58,
        "grain": 0xE60,
        "moire": 0xE68,
        "automatic_contrast_2_dodge": 0xE70,
        "point_mode": 0xE78,
        "slope_r": 0xFE0,
        "slope_b": 0xFE8,
    }
    frame_switch_offsets = {
        "red_eye": 0xE80,
        "soft_filter": 0xE88,
        "cross_filter": 0xE90,
        "spot_filter": 0xE98,
        "merge": 0xEA0,
        "lens_aberration": 0xEA8,
        "distortion": 0xEB0,
        "trapezoid": 0xEB8,
        "algorithm_type": 0xEC0,
        "sgs": 0xEC8,
        "automatic_sharpness_mode": 0xED0,
        "ccd_h_value": 0xED8,
        "jpeg_highlight": 0xEE0,
    }
    correction_offsets = {
        "fixed_color": 0xB90,
        "face_find": 0xB98,
        "automatic_red_eye": 0xBA0,
        "minus_limit_position": 0xBA8,
        "plus_limit_position": 0xBB0,
        "minus_limit_ratio": 0xBB8,
        "plus_limit_ratio": 0xBC0,
        "minus_ratio": 0xBC8,
        "plus_ratio": 0xBD0,
        "standard_density": 0xBD8,
        "standard_color": 0xBE0,
        "scanner_density": 0xBE8,
        "scanner_color": 0xBF0,
        "red_eye_ratio": 0xBF8,
        "tuning_level": 0xC00,
        "cf": 0xC08,
    }
    adjustments = {
        name: _i32_node(data, offset) for name, offset in adjustment_offsets.items()
    }
    saturation_delta = adjustments["saturation_cf_delta"]["value"]
    adjustments["saturation_m00_value"] = {
        "count": adjustments["saturation_cf_delta"]["count"],
        "value": (
            saturation_delta + 100
            if saturation_delta >= 0
            else (saturation_delta + 200) // 2
        ),
    }

    result = {
        "source": {
            "path": str(source),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "validation": {
            "stored_signed_byte_checksum": stored_checksum,
            "computed_signed_byte_checksum": computed_checksum,
            "checksum_matches": stored_checksum == computed_checksum,
            "declared_size": declared_size,
            "size_matches": declared_size == len(data),
            "root_count": root_count,
            "root_name": root_name,
        },
        "producer_version": version,
        "installed_loader_version": INSTALLED_LOADER_VERSION,
        "installed_loader_compatible": version == INSTALLED_LOADER_VERSION,
        "installed_loader_error_if_replayed": (
            None if version == INSTALLED_LOADER_VERSION else "0x83d"
        ),
        "scanner_charge": _i32_node(data, 0x1480),
        "scanner_gain_rgb": [
            struct.unpack_from("<d", data, offset)[0]
            for offset in (0x4AF66, 0x4AF6E, 0x4AF76)
        ],
        "final_image_filter_rotation_degrees": _f64_node(data, 0x1770),
        "adjustments": adjustments,
        "total_balance": [_i32_node(data, offset) for offset in (0xEE8, 0xEF0, 0xEF8)],
        "frame_switches": {
            name: _i32_node(data, offset)
            for name, offset in frame_switch_offsets.items()
        },
        "correction_switches": {
            name: _i32_node(data, offset) for name, offset in correction_offsets.items()
        },
        "cms": {
            "switch": _i32_node(data, 0xFF0),
            "data_flag": _i32_node(data, 0xFF8),
        },
        "pmcf_name": _utf16z(data, 0xC2C),
        "limitations": [
            "original four Y/M/C/D key values are not independently recoverable",
            "independent GUI/source rotations are collapsed into the final rotation",
            "CMS profile payload and NKC destination type are outside this decoded "
            "PRM subset; CmsSW itself is decoded from FrameInf/CmsInf",
            "2017 ToneHold behavior requires the matching legacy GFunction/frame XML",
            "the installed 2015 ImgCorrectDLL rejects this unmodified 2017 producer "
            "stamp; a disposable stamp/checksum compatibility copy was independently "
            "verified through the native SpecialPcb builder",
        ],
        "scanner_gain_provenance": {
            "serialized_parent": "tagSCN_ORD_OUT at correction-file offset 0x4aee6",
            "rgb_double_offsets": ["0x4af66", "0x4af6e", "0x4af76"],
            "consumer": "SpecialPcb sub_10023710 / sub_10024430",
            "formula": "round(clamp(gain / uint8(ScnCrg) * 4096, 0, 65535))",
        },
    }
    if not result["validation"]["checksum_matches"]:
        raise ValueError(f"{source}: signed-byte checksum mismatch")
    if not result["validation"]["size_matches"]:
        raise ValueError(f"{source}: declared size does not match file length")
    if root_count != 1 or root_name != "NKC-ICCS":
        raise ValueError(f"{source}: expected one NKC-ICCS root")
    return result


__all__ = ["decode_prm", "INSTALLED_LOADER_VERSION"]
