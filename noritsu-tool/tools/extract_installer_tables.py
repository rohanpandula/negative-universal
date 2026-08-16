#!/usr/bin/env python3
"""Extract the decoded Noritsu table subset used by the reimplementation.

The output contains numbers and hashes only, never vendor binaries. It is
deterministic so the checked-in artifact can be regenerated from an EZ
Controller ``CorrectServ`` directory and reviewed as ordinary JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


GROUPS = (0, 1)
TYPES = (0, 1)
GENERIC_SIZE = 0x6006
COMMON_SIZE = 0x567A
CALC_SIZE = 0xFF20
NORMAL_OUTPUT_PATTERN = 5


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _variant(root: Path, group: int, quality_type: int) -> Path:
    return root / f"IQGrp{group:03d}" / f"{quality_type:08d}" / "PCB_PARA"


def extract(root: Path) -> dict:
    document = {
        "schema_version": 1,
        "description": (
            "Decoded numeric subsets of EZ Controller 6.5 CorrectServ tables; "
            "no vendor binary payloads are distributed."
        ),
        "selection": {
            "film": "NEGA",
            "format": "135_COL",
            "groups": list(GROUPS),
            "quality_types": list(TYPES),
            "source_mapping": "ImgQltMrg.xml maps Nega main-quality 0/1 to type 0/1",
        },
        "film_lut_rgb_u16": None,
        "variants": {},
    }

    for group in GROUPS:
        for quality_type in TYPES:
            directory = _variant(root, group, quality_type)
            generic_path = directory / "NEGA_para.dat"
            common_path = directory / "CommonCalcPara.Dat"
            calc_path = directory / "NEGA_135_COL_calc.dat"
            generic = generic_path.read_bytes()
            common = common_path.read_bytes()
            calc = calc_path.read_bytes()
            if len(generic) < GENERIC_SIZE:
                raise ValueError(
                    f"{generic_path}: expected at least {GENERIC_SIZE} bytes"
                )
            if len(common) < COMMON_SIZE:
                raise ValueError(
                    f"{common_path}: expected at least {COMMON_SIZE} bytes"
                )
            if len(calc) < CALC_SIZE:
                raise ValueError(f"{calc_path}: expected at least {CALC_SIZE} bytes")
            if generic[:6] != b"\x01\x01\x00\x60\x00\x00":
                raise ValueError(f"{generic_path}: unexpected generic-table header")

            film_values = struct.unpack_from("<12288H", generic, 6)
            common_values = struct.unpack_from("<120I", common, 0)
            key = f"g{group}_t{quality_type}"
            extracted_film_lut = [
                list(film_values[0:4096]),
                list(film_values[4096:8192]),
                list(film_values[8192:12288]),
            ]
            if document["film_lut_rgb_u16"] is None:
                document["film_lut_rgb_u16"] = extracted_film_lut
            elif document["film_lut_rgb_u16"] != extracted_film_lut:
                raise ValueError(
                    "selected Nega variants contain different generic film LUTs; "
                    "extend the artifact schema before extracting them"
                )
            document["variants"][key] = {
                "group": group,
                "quality_type": quality_type,
                "source": {
                    "NEGA_para.dat": {
                        "sha256": _sha256(generic),
                        "file_size": len(generic),
                        "consumed_size": GENERIC_SIZE,
                    },
                    "CommonCalcPara.Dat": {
                        "sha256": _sha256(common),
                        "file_size": len(common),
                        "consumed_size": COMMON_SIZE,
                    },
                    "NEGA_135_COL_calc.dat": {
                        "sha256": _sha256(calc),
                        "file_size": len(calc),
                        "consumed_size": CALC_SIZE,
                    },
                },
                "normal_output_selectors": {
                    (
                        f"0x{0x10 + row:02x}" if row < 16 else f"0x{0x11 + row:02x}"
                    ): calc[30 * row + NORMAL_OUTPUT_PATTERN]
                    for row in range(33)
                },
                "common_gamma_control_points": [
                    {
                        "x_u16": list(common_values[channel * 40 : channel * 40 + 20]),
                        "y_i32": list(
                            common_values[channel * 40 + 20 : channel * 40 + 40]
                        ),
                    }
                    for channel in range(3)
                ],
            }
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("correctserv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    document = extract(args.correctserv.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
