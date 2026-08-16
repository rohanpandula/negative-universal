#!/usr/bin/env python3
"""Focused tests for deterministic, incremental stage tracing."""

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from noritsu.trace import StageTrace, compare_u16_rgb, write_comparison_artifacts


def _gradient(height=5, width=7):
    return np.linspace(0.0, 1.0, height * width * 3, dtype=np.float32).reshape(
        height, width, 3
    )


class StageTraceTests(unittest.TestCase):
    def test_streams_checkpoints_metrics_and_visual_reports(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_trace_test_") as temporary:
            output = Path(temporary)
            original = _gradient()
            changed = original.copy()
            changed[0, 0, 0] = 1.0

            trace = StageTrace(
                output,
                reference=original,
                reference_name="synthetic target",
                preview_size=(32, 24),
                diff_amplification=4.0,
            )
            first = trace.add(
                "scanner input",
                original,
                provenance={"source": "unit test", "ordinal": np.int64(1)},
                confidence="exact",
                params={"enabled": True},
            )
            second = trace.add("one-pixel change", changed, confidence=0.9)
            skipped = trace.skip(
                "disabled stage",
                changed,
                reason="installer setting is off",
                provenance="installer configuration",
            )
            manifest = trace.finalize()

            self.assertEqual(
                [stage["name"] for stage in manifest["stages"]],
                [
                    "scanner input",
                    "one-pixel change",
                    "disabled stage",
                ],
            )
            self.assertEqual(
                first["difference_from_previous"]["status"], "not_available"
            )
            self.assertEqual(first["difference_from_reference"]["overall"]["mae"], 0.0)
            self.assertEqual(
                first["difference_from_reference"]["overall"]["psnr_status"],
                "identical",
            )
            self.assertGreater(
                second["difference_from_previous"]["overall"]["mae"], 0.0
            )
            self.assertEqual(skipped["status"], "skipped")
            self.assertFalse(skipped["applied"])
            self.assertEqual(skipped["skipped_reason"], "installer setting is off")
            self.assertEqual(
                skipped["difference_from_previous"]["overall"]["rmse"], 0.0
            )
            self.assertEqual(
                skipped["artifacts"]["checkpoint_storage"], "hardlink_to_previous"
            )

            checkpoint = output / first["artifacts"]["checkpoint_tiff"]
            quantized = tifffile.imread(str(checkpoint))
            expected = np.floor(original * 65535.0 + 0.5).astype(np.uint16)
            np.testing.assert_array_equal(quantized, expected)
            expected_hash = hashlib.sha256(expected.astype("<u2").tobytes()).hexdigest()
            self.assertEqual(first["pixel_hash"]["value"], expected_hash)
            skipped_checkpoint = output / skipped["artifacts"]["checkpoint_tiff"]
            second_checkpoint = output / second["artifacts"]["checkpoint_tiff"]
            self.assertEqual(
                (os.stat(second_checkpoint).st_dev, os.stat(second_checkpoint).st_ino),
                (
                    os.stat(skipped_checkpoint).st_dev,
                    os.stat(skipped_checkpoint).st_ino,
                ),
            )

            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "contact_sheet.png").is_file())
            self.assertTrue(
                (output / second["artifacts"]["previous_diff_rgb_png"]).is_file()
            )
            self.assertTrue(
                (output / second["artifacts"]["previous_diff_heatmap_png"]).is_file()
            )
            with (output / "manifest.json").open(encoding="utf-8") as handle:
                on_disk = json.load(handle)
            self.assertEqual(on_disk, manifest)
            with (output / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["previous_status"], "ok")
            self.assertEqual(rows[2]["previous_overall_psnr_status"], "identical")

            # Disk-backed mode retains paths, records, and previews—not complete frames.
            self.assertIsNone(trace._previous_q)
            self.assertIsNone(trace._reference_q)

    def test_shape_mismatch_is_an_explicit_result(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_trace_shape_") as temporary:
            reference = _gradient(4, 6)
            trace = StageTrace(temporary, reference=reference, preview_size=(16, 16))
            first = trace.add("different from reference", _gradient(3, 6))
            second = trace.add("different from previous", _gradient(3, 5))
            trace.finalize()

            self.assertEqual(
                first["difference_from_reference"]["status"], "shape_mismatch"
            )
            self.assertEqual(
                first["difference_from_reference"]["current_shape"], [3, 6, 3]
            )
            self.assertEqual(
                first["difference_from_reference"]["other_shape"], [4, 6, 3]
            )
            self.assertEqual(
                second["difference_from_previous"]["status"], "shape_mismatch"
            )
            self.assertNotIn("previous_diff_rgb_png", second["artifacts"])

    def test_manifest_and_reports_are_deterministic(self):
        image = _gradient(6, 8)
        changed = np.clip(image * 0.9 + 0.03, 0.0, 1.0)
        outputs = []
        with tempfile.TemporaryDirectory(prefix="noritsu_trace_a_") as first_dir:
            with tempfile.TemporaryDirectory(prefix="noritsu_trace_b_") as second_dir:
                for directory in (first_dir, second_dir):
                    trace = StageTrace(
                        directory,
                        reference=image,
                        preview_size=(20, 15),
                        diff_amplification=3.0,
                    )
                    trace.add("input", image, params={"b": 2, "a": 1})
                    trace.add("render", changed, provenance=["binary", "table"])
                    trace.finalize()
                    root = Path(directory)
                    outputs.append(
                        (
                            (root / "manifest.json").read_bytes(),
                            (root / "metrics.csv").read_bytes(),
                            (root / "contact_sheet.png").read_bytes(),
                        )
                    )
        self.assertEqual(outputs[0], outputs[1])

    def test_comparison_reports_zero_variance_without_nan(self):
        black = np.zeros((2, 3, 3), dtype=np.uint16)
        white = np.full((2, 3, 3), 65535, dtype=np.uint16)
        comparison = compare_u16_rgb(black, white)
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["overall"]["mae"], 1.0)
        self.assertIsNone(comparison["luma_ncc"])
        self.assertEqual(comparison["luma_ncc_status"], "zero_variance")

    def test_skipped_stage_requires_a_reason(self):
        trace = StageTrace()
        with self.assertRaises(ValueError):
            trace.add("ambiguous skip", _gradient(), applied=False)

    def test_cross_domain_comparison_is_not_reported_as_a_pixel_difference(self):
        first = np.zeros((2, 3, 3), dtype=np.uint16)
        second = np.full((2, 3, 3), 4095, dtype=np.uint16)
        trace = StageTrace()
        trace.add(
            "scanner",
            first,
            domain="scanner_u12",
            encoding="u12_in_u16",
            expected_range=(0, 4095),
        )
        record = trace.add(
            "film LUT",
            second,
            domain="rd5x_rgb",
            encoding="u16",
            expected_range=(0, 65535),
        )
        self.assertEqual(record["difference_from_previous"]["status"], "not_available")
        self.assertIn("domain", record["difference_from_previous"]["reason"])

    def test_u12_comparisons_use_the_declared_native_range(self):
        black = np.zeros((1, 1, 3), dtype=np.uint16)
        white = np.full((1, 1, 3), 4095, dtype=np.uint16)
        comparison = compare_u16_rgb(black, white, value_range=(0, 4095))
        self.assertEqual(comparison["overall"]["mae"], 1.0)
        self.assertEqual(comparison["normalization_range"], [0.0, 4095.0])

    def test_unchanged_stage_reuses_artifacts_and_exact_metrics(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_trace_alias_") as temporary:
            trace = StageTrace(temporary, preview_size=(16, 16))
            first = trace.add("input", _gradient())
            alias = trace.unchanged(
                "disabled filter",
                reason="selector is zero",
                status="disabled_by_job",
            )
            trace.finalize()
            root = Path(temporary)
            first_path = root / first["artifacts"]["checkpoint_tiff"]
            alias_path = root / alias["artifacts"]["checkpoint_tiff"]
            self.assertEqual(
                (os.stat(first_path).st_dev, os.stat(first_path).st_ino),
                (os.stat(alias_path).st_dev, os.stat(alias_path).st_ino),
            )
            self.assertEqual(alias["difference_from_previous"]["overall"]["mae"], 0.0)
            self.assertEqual(
                alias["difference_from_previous"]["changed_pixel_count"], 0
            )
            self.assertEqual(
                alias["difference_from_reference"], first["difference_from_reference"]
            )

    def test_blocked_stage_carries_pixels_but_reports_unknown_effect(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_trace_blocked_") as temporary:
            trace = StageTrace(temporary, preview_size=(16, 16))
            first = trace.add("input", _gradient())
            blocked = trace.unchanged(
                "runtime-built filter",
                reason="final controller parameter block was not captured",
                status="blocked_external_legacy_controller_state",
            )
            trace.finalize()
            root = Path(temporary)
            first_path = root / first["artifacts"]["checkpoint_tiff"]
            blocked_path = root / blocked["artifacts"]["checkpoint_tiff"]
            self.assertEqual(
                (os.stat(first_path).st_dev, os.stat(first_path).st_ino),
                (os.stat(blocked_path).st_dev, os.stat(blocked_path).st_ino),
            )
            self.assertEqual(
                blocked["difference_from_previous"]["status"], "not_available"
            )
            self.assertIn(
                "not executed", blocked["difference_from_previous"]["reason"]
            )
            self.assertNotIn("previous_diff_rgb_png", blocked["artifacts"])

    def test_float_native_range_maps_once_without_silent_clipping(self):
        values = np.array([[[0.0, 1.5, 3.0]]], dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="noritsu_trace_float_") as temporary:
            trace = StageTrace(temporary, preview_size=(8, 8))
            record = trace.add(
                "density-like domain",
                values,
                domain="density",
                expected_range=(0.0, 3.0),
            )
            trace.finalize()
            checkpoint = tifffile.imread(
                str(Path(temporary) / record["artifacts"]["checkpoint_tiff"])
            )
            self.assertEqual(checkpoint.tolist(), [[[0, 32768, 65535]]])
            self.assertEqual(record["checkpoint_value_range"], [0.0, 65535.0])
            self.assertEqual(
                record["native_statistics"]["channels"][2]["above_expected_count"],
                0,
            )

    def test_public_comparison_rejects_invalid_frames(self):
        valid = np.zeros((1, 1, 3), dtype=np.uint16)
        with self.assertRaises(ValueError):
            compare_u16_rgb(np.zeros((0, 1, 3), dtype=np.uint16), valid)
        with self.assertRaises(ValueError):
            compare_u16_rgb(np.zeros((1, 1), dtype=np.uint16), valid)
        with self.assertRaises(TypeError):
            compare_u16_rgb(valid.astype(np.uint32), valid)

    def test_non_rgb_checkpoint_is_rejected_instead_of_mislabeled(self):
        trace = StageTrace()
        with self.assertRaisesRegex(ValueError, "must be RGB"):
            trace.add("stored BGR", _gradient(), channel_order="BGR")

    def test_standalone_comparison_writes_numerical_and_visual_artifacts(self):
        current = np.zeros((3, 4, 3), dtype=np.uint16)
        other = np.full((3, 4, 3), 4095, dtype=np.uint16)
        with tempfile.TemporaryDirectory(prefix="noritsu_compare_") as temporary:
            result = write_comparison_artifacts(
                current,
                other,
                temporary,
                prefix="recovered boundary",
                preview_size=(16, 16),
                value_range=(0, 4095),
            )
            self.assertEqual(result["metrics"]["overall"]["mae"], 1.0)
            for relative in result["artifacts"].values():
                self.assertTrue((Path(temporary) / relative).is_file())


if __name__ == "__main__":
    unittest.main()
