#!/usr/bin/env python3
"""Opt-in full-resolution regression for the canonical Portra 400 pair.

Run explicitly with::

    NORITSU_RUN_REAL_PAIR=1 python -m unittest test_real_pair

The ordinary test suite only evaluates the inexpensive skip conditions below.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tifffile

from noritsu import audit


_RUN_REAL_PAIR = os.environ.get("NORITSU_RUN_REAL_PAIR") == "1"
_CANONICAL_INPUTS = (
    audit.PORTRA400_RAW,
    audit.PORTRA400_REFERENCE,
    audit.PORTRA400_PRM,
    audit.PORTRA400_FINAL_PCB,
    audit.PORTRA400_NATIVE_REFERENCE,
    audit.PORTRA400_HUE_LATTICE,
    audit.PORTRA400_GRAIN_CURVE,
    audit.PORTRA400_IMAGE_DESCRIPTOR,
    audit.PORTRA400_GAMMA_U8_ORACLE,
    audit.PORTRA400_DEPTH_CAPTURE_LOG,
    audit.PORTRA400_TRZ,
    audit.PORTRA400_SETTINGS,
)
_CANONICAL_INPUTS_EXIST = all(path.is_file() for path in _CANONICAL_INPUTS)

_BOUNDARY_PIXEL_SHA256 = (
    "0f5babb019761f0bfdab5af2ef66d9cbfd405f032cce5045fa4f077ee5add791"
)
_EXTERNAL_UNQUANTIZED_PIXEL_SHA256 = (
    "c59ed5e8065dc243d92b64757341873acf773bbacee4bf8370592a63ce6e48d0"
)
_EXTERNAL_QUANTIZED_PIXEL_SHA256 = (
    "47e6fd751f7d86ae04f5024783eb4947069fbe97258af5151c0762bd6ed6a2c0"
)
_TRZ_SHA256 = "d6cc296da52c1dd5a6091d9f7183efca5d41650d6f03eb1e9502c05f479bd526"
_SETTINGS_SHA256 = "d1c3a42cf44ba1886fe275489f1d9c0f019ec4b19d37df2bb129b6a121fa6522"


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(path: Path, chunk_rows: int = 256) -> str:
    pixels = tifffile.memmap(str(path), mode="r")
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise AssertionError(f"expected an RGB uint16 TIFF, got {pixels.shape} {pixels.dtype}")
    digest = hashlib.sha256()
    for start in range(0, pixels.shape[0], chunk_rows):
        chunk = np.asarray(pixels[start : start + chunk_rows], dtype="<u2")
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


@unittest.skipUnless(
    _RUN_REAL_PAIR and _CANONICAL_INPUTS_EXIST,
    "set NORITSU_RUN_REAL_PAIR=1 and provide the canonical Portra 400 evidence bundle",
)
class CanonicalRealPairRegressionTests(unittest.TestCase):
    def test_portra400_frame1_full_audit_matches_validated_outputs(self) -> None:
        self.assertEqual(_file_sha256(audit.PORTRA400_RAW), audit.PORTRA400_RAW_SHA256)
        self.assertEqual(
            _file_sha256(audit.PORTRA400_REFERENCE),
            audit.PORTRA400_REFERENCE_SHA256,
        )
        self.assertEqual(_file_sha256(audit.PORTRA400_PRM), audit.PORTRA400_PRM_SHA256)
        for path, expected in (
            (audit.PORTRA400_FINAL_PCB, audit.PORTRA400_FINAL_PCB_SHA256),
            (
                audit.PORTRA400_NATIVE_REFERENCE,
                audit.PORTRA400_NATIVE_REFERENCE_SHA256,
            ),
            (audit.PORTRA400_HUE_LATTICE, audit.PORTRA400_HUE_LATTICE_SHA256),
            (audit.PORTRA400_GRAIN_CURVE, audit.PORTRA400_GRAIN_CURVE_SHA256),
            (
                audit.PORTRA400_IMAGE_DESCRIPTOR,
                audit.PORTRA400_IMAGE_DESCRIPTOR_SHA256,
            ),
            (
                audit.PORTRA400_GAMMA_U8_ORACLE,
                audit.PORTRA400_GAMMA_U8_ORACLE_SHA256,
            ),
            (
                audit.PORTRA400_DEPTH_CAPTURE_LOG,
                audit.PORTRA400_DEPTH_CAPTURE_LOG_SHA256,
            ),
            (audit.PORTRA400_TRZ, _TRZ_SHA256),
            (audit.PORTRA400_SETTINGS, _SETTINGS_SHA256),
        ):
            self.assertEqual(_file_sha256(path), expected)

        with tempfile.TemporaryDirectory(prefix="noritsu_real_pair_") as temporary:
            output = Path(temporary) / "portra400-frame1"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = audit.main(
                    [
                        "--preset",
                        "portra400-frame1",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(return_code, 0, stdout.getvalue())

            with (output / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            with (output / "recovered_boundary_metrics.json").open(
                encoding="utf-8"
            ) as handle:
                boundary_record = json.load(handle)
            with (output / "recovery_report.json").open(encoding="utf-8") as handle:
                recovery = json.load(handle)

            provenance = manifest["run_provenance"]
            self.assertEqual(
                provenance["input"]["sha256"], audit.PORTRA400_RAW_SHA256
            )
            self.assertEqual(
                provenance["machine_reference"]["sha256"],
                audit.PORTRA400_REFERENCE_SHA256,
            )
            self.assertEqual(
                provenance["captured_job"]["prm"]["sha256"],
                audit.PORTRA400_PRM_SHA256,
            )
            captured = provenance["captured_job"]
            self.assertEqual(
                captured["native_final_pcb"]["sha256"],
                audit.PORTRA400_FINAL_PCB_SHA256,
            )
            self.assertEqual(
                provenance["native_engine_reference"]["sha256"],
                audit.PORTRA400_NATIVE_REFERENCE_SHA256,
            )
            native_com = captured["native_com_para"]
            self.assertEqual(
                native_com["hue_saturation_lattice"]["sha256"],
                audit.PORTRA400_HUE_LATTICE_SHA256,
            )
            self.assertEqual(
                native_com["grain_variance_curve"]["sha256"],
                audit.PORTRA400_GRAIN_CURVE_SHA256,
            )
            self.assertEqual(
                native_com["image_descriptor"]["sha256"],
                audit.PORTRA400_IMAGE_DESCRIPTOR_SHA256,
            )
            self.assertEqual(
                native_com["gamma_u8_lut_oracle"]["sha256"],
                audit.PORTRA400_GAMMA_U8_ORACLE_SHA256,
            )
            self.assertEqual(
                native_com["depth_capture_log"]["sha256"],
                audit.PORTRA400_DEPTH_CAPTURE_LOG_SHA256,
            )
            self.assertEqual(native_com["decoded_image_descriptor"]["dst_type"], 0)
            self.assertEqual(
                native_com["hue_lattice_statistics"],
                {
                    "shape": [9, 33, 33, 3],
                    "minimum": -485,
                    "maximum": 681,
                    "nonzero_values": 19339,
                },
            )

            selectors = captured["decoded_native_final_pcb"]["effective_selectors"]
            for selector, expected in {
                "0x12": 1,
                "0x18": 0,
                "0x1b": 2,
                "0x1e": 2,
                "0x1f": 0,
                "0x21": 1,
                "0x22": 1,
                "0x23": 1,
                "0x26": 1,
                "0x27": 0,
            }.items():
                self.assertEqual(selectors[selector], expected)

            final_path = output / "final.tif"
            boundary_path = output / "recovered_boundary.tif"
            external_unquantized_path = output / "external_fit_unquantized.tif"
            external_quantized_path = output / "external_fit.tif"
            native_boundary_path = output / "installer_color_boundary_native.tif"

            self.assertTrue(os.path.samefile(final_path, boundary_path))
            self.assertTrue(native_boundary_path.is_file())
            cms = next(
                stage
                for stage in manifest["stages"]
                if stage["name"] == "CMS / NKC-ICCS"
            )
            self.assertTrue(
                os.path.samefile(
                    native_boundary_path,
                    output / cms["artifacts"]["checkpoint_tiff"],
                )
            )
            self.assertEqual(cms["status"], "disabled_by_captured_prm")
            self.assertEqual(cms["params"]["captured_cms_switch"], 0)
            boundary_hash = _pixel_sha256(boundary_path)
            external_unquantized_hash = _pixel_sha256(external_unquantized_path)
            external_quantized_hash = _pixel_sha256(external_quantized_path)
            self.assertEqual(boundary_hash, _BOUNDARY_PIXEL_SHA256)
            self.assertEqual(_pixel_sha256(native_boundary_path), boundary_hash)
            self.assertEqual(
                external_unquantized_hash, _EXTERNAL_UNQUANTIZED_PIXEL_SHA256
            )
            self.assertEqual(external_quantized_hash, _EXTERNAL_QUANTIZED_PIXEL_SHA256)
            self.assertNotEqual(boundary_hash, external_unquantized_hash)
            self.assertNotEqual(boundary_hash, external_quantized_hash)

            stages = {stage["name"]: stage for stage in manifest["stages"]}
            for name in (
                "RD5X CCD noise clear S0",
                "RD5X trimming and Scaling S0",
                "RGB to YCC S1",
                "Film sharp",
                "YCC correction",
                "Hue / saturation correction S0",
                "YCC to RGB with fused gamma",
            ):
                self.assertTrue(stages[name]["applied"], name)
            scaling = stages["RD5X trimming and Scaling S0"]
            self.assertEqual(scaling["params"]["scaling_selector"], 2)
            self.assertEqual(scaling["params"]["sampling"], 2)
            gamma = stages["YCC to RGB with fused gamma"]
            self.assertEqual(gamma["params"]["dst_type"], 0)
            self.assertEqual(
                gamma["params"]["gamma_luts_bgr_sha256"],
                audit.PORTRA400_GAMMA_U8_ORACLE_SHA256,
            )

            boundary = boundary_record["comparison"]
            self.assertEqual(boundary["status"], "ok")
            self.assertAlmostEqual(
                boundary["overall"]["mae"], 0.06708299052282021, delta=1e-12
            )
            self.assertAlmostEqual(
                boundary["overall"]["rmse"], 0.0975748935994854, delta=1e-12
            )
            self.assertAlmostEqual(
                boundary["luma_ncc"], 0.9455270482413177, delta=1e-12
            )

            native = recovery["recovered_boundary"]["native_engine_comparison"]
            self.assertEqual(native["overall"]["mae"], 0.0)
            self.assertEqual(native["overall"]["rmse"], 0.0)
            self.assertEqual(native["changed_pixel_count"], 0)
            self.assertEqual(native["luma_ncc"], 1.0)
            self.assertEqual(
                recovery["remaining_gap_boundary"]["local_software_conformance"],
                [],
            )
            self.assertEqual(
                recovery[
                    "selected_stages_blocked_by_external_legacy_controller_state"
                ],
                [],
            )

            external = manifest["stages"][-1]["difference_from_reference"]
            self.assertEqual(external["status"], "ok")
            self.assertAlmostEqual(
                external["overall"]["mae"], 0.0668002165746828, delta=1e-12
            )
            self.assertAlmostEqual(
                external["overall"]["rmse"], 0.09564640486826892, delta=1e-12
            )
            self.assertAlmostEqual(
                external["luma_ncc"], 0.945930351642374, delta=1e-12
            )


if __name__ == "__main__":
    unittest.main()
