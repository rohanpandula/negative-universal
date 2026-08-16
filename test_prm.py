#!/usr/bin/env python3
"""Tests for the verified ICCS correction-file decoder."""

import os
import struct
import tempfile
import unittest
from pathlib import Path

from noritsu.prm import AUDITED_LAYOUT_SIZE, decode_prm


REAL_PRM = (
    Path(os.environ["NORITSU_PORTRA400_DIR"]) / "raw/1/Original/1.prm"
    if os.environ.get("NORITSU_PORTRA400_DIR")
    else None
)


class PrmDecoderTests(unittest.TestCase):
    def test_validates_synthetic_layout_and_version_wall(self):
        data = bytearray(AUDITED_LAYOUT_SIZE)
        version = b"13:14:42 on May  9 2017"
        data[4 : 4 + len(version)] = version
        struct.pack_into("<I", data, 0xA0, len(data))
        struct.pack_into("<i", data, 0xB8, 1)
        data[0xBC : 0xBC + len("NKC-ICCS".encode("utf-16-le"))] = "NKC-ICCS".encode(
            "utf-16-le"
        )
        struct.pack_into("<ii", data, 0xE58, 1, 0)
        checksum = sum(value if value < 128 else value - 256 for value in data[4:])
        struct.pack_into("<i", data, 0, checksum)

        with tempfile.TemporaryDirectory(prefix="noritsu_prm_test_") as temporary:
            path = Path(temporary) / "capture.prm"
            path.write_bytes(data)
            decoded = decode_prm(path)
        self.assertTrue(decoded["validation"]["checksum_matches"])
        self.assertEqual(decoded["validation"]["root_name"], "NKC-ICCS")
        self.assertEqual(decoded["adjustments"]["saturation_m00_value"]["value"], 100)
        self.assertFalse(decoded["installed_loader_compatible"])
        self.assertEqual(decoded["installed_loader_error_if_replayed"], "0x83d")
        self.assertEqual(decoded["cms"]["switch"], {"count": 0, "value": 0})

    @unittest.skipUnless(
        REAL_PRM is not None and REAL_PRM.is_file(),
        "owner correction-file corpus is absent (set NORITSU_PORTRA400_DIR)",
    )
    def test_decodes_fixed_portra400_capture(self):
        decoded = decode_prm(REAL_PRM)
        self.assertEqual(
            decoded["source"]["sha256"],
            "098c33a4f11f4f338ffcb9c9c784543546fc13741b47eab1009543b757f288b0",
        )
        self.assertEqual(decoded["scanner_charge"]["value"], 1)
        self.assertEqual(decoded["scanner_gain_rgb"], [1.0, 1.0, 1.0])
        self.assertEqual(
            decoded["adjustments"]["automatic_contrast_overall"]["value"], 0
        )
        self.assertEqual(decoded["adjustments"]["saturation_m00_value"]["value"], 100)
        self.assertEqual(decoded["total_balance"][0]["value"], 0)
        self.assertEqual(decoded["frame_switches"]["red_eye"]["value"], 0)
        self.assertEqual(decoded["frame_switches"]["merge"]["value"], 0)
        self.assertEqual(decoded["correction_switches"]["face_find"]["value"], 1)
        self.assertEqual(decoded["cms"]["switch"], {"count": 1, "value": 0})
        self.assertEqual(decoded["cms"]["data_flag"], {"count": 1, "value": 0})
        self.assertEqual(decoded["pmcf_name"], "OFF")


if __name__ == "__main__":
    unittest.main()
