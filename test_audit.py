#!/usr/bin/env python3
"""Integration tests for the evidence-labelled stage audit runner."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from noritsu import pcb, tables, vendor
from noritsu.audit import AuditConfig, _pre_rotation_crop_and_scale, run_audit
from noritsu.prm import AUDITED_LAYOUT_SIZE


class AuditRunnerTests(unittest.TestCase):
    @staticmethod
    def _write_prm_with_unavailable_neutral_values(path: Path) -> None:
        data = bytearray(AUDITED_LAYOUT_SIZE)
        version = b"13:14:42 on May  9 2017"
        data[4 : 4 + len(version)] = version
        struct.pack_into("<I", data, 0xA0, len(data))
        struct.pack_into("<i", data, 0xB8, 1)
        root = "NKC-ICCS".encode("utf-16-le")
        data[0xBC : 0xBC + len(root)] = root
        # Mode 0 is known, but rotation, hue/saturation, and TotalBal nodes retain
        # count=0/value=0. A zero payload without a present-node count is unknown.
        struct.pack_into("<ii", data, 0xBD8, 1, 1)
        struct.pack_into("<ii", data, 0xBE0, 1, 0)
        checksum = sum(value if value < 128 else value - 256 for value in data[4:])
        struct.pack_into("<i", data, 0, checksum)
        path.write_bytes(data)

    @staticmethod
    def _write_prm_with_explicitly_disabled_job(path: Path) -> None:
        """Write the smallest audited PRM whose optional job switches are known-off."""

        data = bytearray(AUDITED_LAYOUT_SIZE)
        version = b"13:14:42 on May  9 2017"
        data[4 : 4 + len(version)] = version
        struct.pack_into("<I", data, 0xA0, len(data))
        struct.pack_into("<i", data, 0xB8, 1)
        root = "NKC-ICCS".encode("utf-16-le")
        data[0xBC : 0xBC + len(root)] = root

        # The two correction switches select mode 0.  All frame-dependent
        # optional branches used by the audit are present and explicitly off.
        struct.pack_into("<ii", data, 0xBD8, 1, 1)
        struct.pack_into("<ii", data, 0xBE0, 1, 0)
        for offset in (0xE80, 0xE88, 0xE90, 0xE98, 0xEA0, 0xEA8, 0xEB0, 0xEB8):
            struct.pack_into("<ii", data, offset, 1, 0)
        struct.pack_into("<i", data, 0x1770, 1)  # f64 value at +8 is already 0.0
        struct.pack_into("<ii", data, 0xFF0, 1, 0)

        checksum = sum(value if value < 128 else value - 256 for value in data[4:])
        struct.pack_into("<i", data, 0, checksum)
        path.write_bytes(data)

    @staticmethod
    def _write_geometry_only_final_pcb(path: Path) -> None:
        """Write a valid native PCB with S0 geometry and post-film stages disabled."""

        data = bytearray(pcb.FINAL_PCB_SIZE)
        # The synthetic RAW is 5x4 in storage and becomes a 4x5 native raster
        # after the configured one-quarter-turn counter-clockwise rotation.
        struct.pack_into("<4I", data, 0, 5, 4, 2, 3)
        data[0x10] = 1  # CCD gain dispatcher
        data[0x11] = 1  # Nega FilmLUT dispatcher
        data[0x1A] = 1  # trimming
        data[0x1B] = 2  # Selector 2 dispatches FmScalingS0_Do.
        data[0x1C] = 1  # captured zero-degree image-filter rotation
        struct.pack_into("<4i", data, pcb.TRIMMING_OFFSET, 1, 1, 3, 2)
        struct.pack_into("<4I", data, pcb.SCALING_OFFSET, 3, 2, 2, 3)
        data[pcb.SCALING_OFFSET + 0x10] = 1  # no box-prefilter reduction
        data[pcb.SCALING_OFFSET + 0x11] = 1  # one sample per output pixel
        path.write_bytes(data)

    def test_same_frame_pipeline_emits_every_checkpoint_and_final_metrics(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_audit_test_") as temporary:
            root = Path(temporary)
            raw_path = root / "FULL000000010000.RAW"
            reference_path = root / "machine.tif"
            output = root / "audit"

            raw = (np.arange(8 * 6 * 3, dtype=np.uint16).reshape(8, 6, 3) * 23) % 4096
            raw.astype("<u2").tofile(raw_path)
            provisional = AuditConfig(
                raw=raw_path,
                reference=reference_path,
                output_dir=output,
                raw_width=6,
                raw_height=8,
                rotation_k=1,
                rotated_crop_y=1,
                rotated_crop_x=1,
                crop_height=4,
                crop_width=6,
                output_height=4,
                output_width=6,
                fit_sample_step=1,
            )
            working = vendor.apply_film_lut(raw, tables.load_film_lut(0, 1))
            working, _ = _pre_rotation_crop_and_scale(working, provisional)
            working = np.ascontiguousarray(np.rot90(working, 1))
            target = vendor.quantize_oct_to_hex(working)
            tifffile.imwrite(reference_path, target, photometric="rgb", metadata=None)

            manifest = run_audit(provisional)
            names = [stage["name"] for stage in manifest["stages"]]
            self.assertIn("RD5X Nega FilmLUT base-table sub-kernel", names)
            self.assertIn("RGB to YCC S1", names)
            self.assertIn("CMS / NKC-ICCS", names)
            self.assertIn(
                "External legacy-reference residual fit (not a Noritsu stage)", names
            )
            self.assertGreaterEqual(len(manifest["stages"]), 30)
            for stage in manifest["stages"]:
                checkpoint = output / stage["artifacts"]["checkpoint_tiff"]
                preview = output / stage["artifacts"]["preview_png"]
                self.assertTrue(checkpoint.is_file(), stage["name"])
                self.assertTrue(preview.is_file(), stage["name"])

            final = manifest["stages"][-1]
            self.assertEqual(final["difference_from_reference"]["status"], "ok")
            self.assertEqual(final["difference_from_reference"]["overall"]["mae"], 0.0)
            self.assertEqual(tifffile.imread(output / "final.tif").shape, (4, 6, 3))
            self.assertTrue((output / "recovered_boundary.tif").is_file())
            self.assertTrue(
                (output / "installer_color_boundary_native.tif").is_file()
            )
            self.assertTrue((output / "external_fit.tif").is_file())
            self.assertTrue((output / "external_fit_unquantized.tif").is_file())
            self.assertFalse(
                any(
                    stage["status"]
                    in ("disabled_by_captured_prm", "applied_exact_identity")
                    for stage in manifest["stages"]
                )
            )
            for filename in (
                "config.json",
                "manifest.json",
                "metrics.csv",
                "contact_sheet.png",
                "recovery_report.json",
                "installer_color_boundary_native_metrics.json",
                "recovered_boundary_metrics.json",
                "report.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)

    def test_absent_prm_nodes_are_not_interpreted_as_zero_controls(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_audit_counts_") as temporary:
            root = Path(temporary)
            raw_path = root / "tiny.RAW"
            prm_path = root / "capture.prm"
            output = root / "audit"
            np.zeros((4, 4, 3), dtype="<u2").tofile(raw_path)
            self._write_prm_with_unavailable_neutral_values(prm_path)

            manifest = run_audit(
                AuditConfig(
                    raw=raw_path,
                    reference=None,
                    output_dir=output,
                    raw_width=4,
                    raw_height=4,
                    rotation_k=0,
                    rotated_crop_y=0,
                    rotated_crop_x=0,
                    crop_height=4,
                    crop_width=4,
                    output_height=4,
                    output_width=4,
                    fit_external_reference=False,
                    prm=prm_path,
                )
            )
            statuses = {stage["name"]: stage["status"] for stage in manifest["stages"]}
            self.assertEqual(
                statuses["RD5X image-filter rotation"],
                "blocked_dynamic_frame_rewrite",
            )
            self.assertEqual(
                statuses["Hue / saturation correction S0"],
                "blocked_missing_native_runtime_parameters",
            )
            self.assertEqual(
                statuses["CMS / NKC-ICCS"],
                "blocked_external_legacy_controller_state",
            )
            cms = next(
                stage
                for stage in manifest["stages"]
                if stage["name"] == "CMS / NKC-ICCS"
            )
            self.assertEqual(
                cms["difference_from_previous"]["status"], "not_available"
            )

    def test_synthetic_final_pcb_drives_native_geometry_and_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_audit_pcb_") as temporary:
            root = Path(temporary)
            raw_path = root / "tiny.RAW"
            prm_path = root / "capture.prm"
            pcb_path = root / "final-pcb.bin"
            output = root / "audit"

            raw_bgr = (
                np.arange(5 * 4 * 3, dtype=np.uint16).reshape(5, 4, 3) * 61
            ) % 4096
            raw_bgr.astype("<u2").tofile(raw_path)
            self._write_prm_with_explicitly_disabled_job(prm_path)
            self._write_geometry_only_final_pcb(pcb_path)

            # Deliberately contradictory fallback crop/output dimensions prove
            # that the supplied native PCB, rather than AuditConfig registration,
            # determines the executed geometry.
            manifest = run_audit(
                AuditConfig(
                    raw=raw_path,
                    reference=None,
                    output_dir=output,
                    raw_width=4,
                    raw_height=5,
                    rotation_k=1,
                    rotated_crop_y=0,
                    rotated_crop_x=0,
                    crop_height=1,
                    crop_width=1,
                    output_height=1,
                    output_width=1,
                    fit_external_reference=False,
                    prm=prm_path,
                    final_pcb=pcb_path,
                )
            )

            stages = {stage["name"]: stage for stage in manifest["stages"]}
            self.assertEqual(
                stages["FULL storage orientation to RD5X source raster"]["status"],
                "applied_explicit_capture_geometry",
            )
            self.assertEqual(
                stages["RD5X trimming and Scaling S0"]["status"],
                "applied_exact_native_final_pcb",
            )
            self.assertNotIn("Scaling / trimming to reference raster", stages)
            self.assertEqual(
                stages["RD5X image-filter rotation"]["status"],
                "applied_exact_identity_native_final_pcb",
            )

            for stage_name in (
                "RD5X CCD noise clear S0",
                "RGB to YCC S1",
                "Automatic contrast S0",
                "Film sharp",
                "YCC correction",
                "Hue / saturation correction S0",
                "YCC to RGB with fused gamma",
            ):
                self.assertIn(
                    stages[stage_name]["status"],
                    {
                        "not_selected_by_effective_selector",
                        "disabled_by_native_final_pcb",
                    },
                    stage_name,
                )

            expected = vendor.apply_film_lut(
                np.ascontiguousarray(np.rot90(raw_bgr, 1)),
                tables.load_film_lut(0, 1),
            )
            expected = vendor.apply_scaling_s0_u16(
                expected,
                output_width=2,
                output_height=3,
                pre_scale=1,
                sampling=1,
                trim=(1, 1, 3, 2),
            )
            final = tifffile.imread(output / "final.tif")
            np.testing.assert_array_equal(final, expected)
            self.assertEqual(final.shape, (3, 2, 3))

            with (output / "recovered_boundary_metrics.json").open(
                encoding="utf-8"
            ) as handle:
                recovered = json.load(handle)
            native = recovered["native_installer_boundary"]
            self.assertEqual(native["stage_index"], recovered["stage_index"])
            boundary_stage = manifest["stages"][recovered["stage_index"]]
            self.assertEqual(boundary_stage["name"], recovered["stage_name"])
            self.assertEqual(native["pixel_hash"], boundary_stage["pixel_hash"])
            for filename in (
                "recovered_boundary.tif",
                "installer_color_boundary_native.tif",
            ):
                np.testing.assert_array_equal(tifffile.imread(output / filename), final)
            self.assertFalse((output / "external_fit.tif").exists())


if __name__ == "__main__":
    unittest.main()
