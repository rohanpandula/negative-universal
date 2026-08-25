from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

from negative_universal.calibration import (
    _channel_statistics,
    _validate_camera_white_levels,
    calibrate_clear_base,
)
from negative_universal.cli import (
    NegativeUniversalError,
    PgmImage,
    sha256_file,
    write_pgm,
)


class ClearBaseCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.exposures = {"R": 1000, "G": 2000, "B": 3000}
        self.powers = {"R": 0.25, "G": 0.5, "B": 0.75}

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _capture(
        self,
        name: str,
        kind: str,
        values: dict[str, list[int]],
        *,
        width: int = 2,
        max_value: int = 255,
        sensor_layout: str = "mono",
        settings_sha256: str = "a" * 64,
    ) -> Path:
        directory = self.root / name
        directory.mkdir()
        frames = {}
        for channel in ("R", "G", "B"):
            pixels = values[channel]
            image = PgmImage(
                width,
                len(pixels) // width,
                max_value,
                array("H", pixels),
            )
            path = directory / f"{channel}.pgm"
            write_pgm(path, image)
            frames[channel] = {
                "file": path.name,
                "width": image.width,
                "height": image.height,
                "max_value": image.max_value,
                "sha256": sha256_file(path),
                "commanded_power": 0.0 if kind == "dark" else self.powers[channel],
                "commanded_exposure_us": self.exposures[channel],
            }
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "kind": kind,
                    "channel_order": ["R", "G", "B"],
                    "rig": "/fake/rig",
                    "rig_metadata": {
                        "configuration_id": "fake-rig-v1",
                        "sensor_layout": sensor_layout,
                    },
                    "settings_sha256": settings_sha256,
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
        return directory

    def test_mono_recommends_exposure_only_and_preserves_sources(self) -> None:
        dark = self._capture("dark", "dark", {c: [10, 10] for c in "RGB"})
        flat = self._capture(
            "flat",
            "flat",
            {"R": [110, 110], "G": [180, 180], "B": [220, 220]},
        )
        clear = self._capture(
            "clear", "film", {"R": [70, 70], "G": [100, 100], "B": [50, 50]}
        )
        source_hashes = {
            path: sha256_file(path)
            for directory in (clear, dark, flat)
            for path in directory.iterdir()
        }
        output = self.root / "calibration"

        calibrate_clear_base(clear, dark, flat, (0, 0, 2, 1), 0.3, 0.8, output)

        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(
            list(path.name for path in output.iterdir()), ["manifest.json"]
        )
        self.assertEqual(
            {path: sha256_file(path) for path in source_hashes}, source_hashes
        )
        self.assertAlmostEqual(
            manifest["acquisition"]["R"]["ideal_exposure_time_multiplier"],
            1.225,
        )
        self.assertEqual(
            manifest["acquisition"]["R"]["recommendation"],
            {"commanded_power": 0.25, "commanded_exposure_us": 1225},
        )
        self.assertEqual(
            manifest["acquisition"]["G"]["recommendation"]["commanded_exposure_us"],
            1633,
        )
        self.assertEqual(manifest["acquisition"]["B"]["limited_by"], "flat")
        self.assertEqual(
            manifest["acquisition"]["B"]["recommendation"]["commanded_exposure_us"],
            2800,
        )
        self.assertAlmostEqual(
            manifest["diagnostics"]["R"]["mono"]["transmittance_median"],
            0.6,
        )
        self.assertEqual(manifest["pgm_max_value"], 255)
        self.assertEqual(
            manifest["channel_saturation_levels"], {channel: 255 for channel in "RGB"}
        )
        self.assertNotIn("white_level", manifest)
        self.assertNotIn("white_balance", manifest)
        self.assertNotIn("awb", manifest)
        self.assertNotIn("dose_multiplier", manifest["acquisition"]["R"])

    def test_bayer_uses_matching_meter_sites_and_reports_every_class(self) -> None:
        dark = self._capture(
            "dark", "dark", {c: [10, 10, 10, 10] for c in "RGB"}, sensor_layout="rggb"
        )
        flat = self._capture(
            "flat",
            "flat",
            {c: [110, 110, 110, 110] for c in "RGB"},
            sensor_layout="rggb",
        )
        clear = self._capture(
            "clear",
            "film",
            {
                "R": [80, 20, 30, 40],
                "G": [20, 60, 80, 30],
                "B": [20, 30, 40, 90],
            },
            sensor_layout="rggb",
        )
        output = self.root / "calibration"

        calibrate_clear_base(clear, dark, flat, (0, 0, 2, 2), 0.3, 0.8, output)

        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["acquisition"]["R"]["meter_sites"], ["R"])
        self.assertEqual(manifest["acquisition"]["G"]["meter_sites"], ["G1", "G2"])
        self.assertEqual(manifest["acquisition"]["B"]["meter_sites"], ["B"])
        self.assertEqual(set(manifest["diagnostics"]["R"]), {"R", "G1", "G2", "B"})
        self.assertAlmostEqual(
            manifest["acquisition"]["G"]["base_fraction_median"], 60 / 245
        )

    def test_bayer_flat_cap_uses_the_highest_site_p99(self) -> None:
        dark_values = {channel: [0] * 400 for channel in "RGB"}
        clear_values = {channel: [50] * 400 for channel in "RGB"}
        flat_plane = [100] * 400
        for index in (0, 2, 4, 6):
            flat_plane[index] = 250
        flat_values = {channel: list(flat_plane) for channel in "RGB"}
        dark = self._capture(
            "dark", "dark", dark_values, width=20, sensor_layout="rggb"
        )
        flat = self._capture(
            "flat", "flat", flat_values, width=20, sensor_layout="rggb"
        )
        clear = self._capture(
            "clear", "film", clear_values, width=20, sensor_layout="rggb"
        )
        output = self.root / "calibration"

        calibrate_clear_base(clear, dark, flat, (0, 0, 20, 20), 0.2, 0.8, output)

        acquisition = json.loads((output / "manifest.json").read_text())["acquisition"][
            "R"
        ]
        self.assertEqual(acquisition["flat_ceiling_sites"], ["R"])
        self.assertAlmostEqual(acquisition["flat_fraction_max_site_p99"], 250 / 255)
        self.assertAlmostEqual(
            acquisition["ideal_exposure_time_multiplier"], 0.8 / (250 / 255)
        )

    def test_uses_equal_receipt_saturation_below_the_pgm_maximum(self) -> None:
        base = PgmImage(1, 1, 16383, array("H", [7811]))
        dark = PgmImage(1, 1, 16383, array("H", [311]))
        flat = PgmImage(1, 1, 16383, array("H", [12311]))
        acquisition, diagnostics = _channel_statistics(
            channel="R",
            clear_base=base,
            dark=dark,
            flat=flat,
            roi=(0, 0, 1, 1),
            sensor_layout="mono",
            target_base_fraction=0.3,
            flat_ceiling_fraction=0.8,
            saturation_level=15311,
            frame={
                "commanded_power": 0.25,
                "commanded_exposure_us": 1000,
                "native_source": {"applied_exposure_us": 1010, "pwm_code": 64},
            },
        )
        self.assertEqual(
            acquisition["recommendation_basis"],
            {"exposure_us": 1010, "source": "applied_exposure_us"},
        )
        self.assertAlmostEqual(acquisition["base_fraction_median"], 7500 / 15000)
        self.assertAlmostEqual(acquisition["flat_fraction_max_site_p99"], 12000 / 15000)
        self.assertEqual(acquisition["recommendation"]["commanded_exposure_us"], 606)
        self.assertEqual(diagnostics["mono"]["flat_at_or_above_saturation"], 0)

        def receipt_capture(name: str, levels: list[int]):
            directory = self.root / name
            directory.mkdir()
            receipt = directory / "receipt.json"
            receipt.write_text(
                json.dumps({"raster": {"camera_white_levels": levels}}),
                encoding="utf-8",
            )
            manifest = {
                "frames": {
                    channel: {"native_source": {"receipt_file": receipt.name}}
                    for channel in "RGB"
                }
            }
            return directory, manifest, {channel: base for channel in "RGB"}

        clear_capture = receipt_capture("native-clear", [15311] * 4)
        dark_capture = receipt_capture("native-dark", [15311] * 4)
        flat_capture = receipt_capture("native-flat", [15311] * 4)
        self.assertEqual(
            _validate_camera_white_levels((clear_capture, dark_capture, flat_capture)),
            {channel: 15311 for channel in "RGB"},
        )

        for name, levels in (
            ("native-unequal", [15310, 15311, 15311, 15311]),
            ("native-too-high", [16384] * 4),
        ):
            with self.subTest(levels=levels):
                with self.assertRaisesRegex(
                    NegativeUniversalError, "camera white levels"
                ):
                    _validate_camera_white_levels((receipt_capture(name, levels),))

        mismatched_flat = receipt_capture("native-mismatch", [15310] * 4)
        with self.assertRaisesRegex(NegativeUniversalError, "does not match"):
            _validate_camera_white_levels(
                (clear_capture, dark_capture, mismatched_flat)
            )

    def test_rejects_unmatched_or_invalid_inputs_without_output(self) -> None:
        dark = self._capture("dark", "dark", {c: [10, 10] for c in "RGB"})
        flat = self._capture("flat", "flat", {c: [110, 110] for c in "RGB"})
        clear = self._capture("clear", "film", {c: [60, 60] for c in "RGB"})
        output = self.root / "calibration"
        flat_manifest_path = flat / "manifest.json"
        flat_manifest = json.loads(flat_manifest_path.read_text())
        flat_manifest["settings_sha256"] = "b" * 64
        flat_manifest_path.write_text(json.dumps(flat_manifest), encoding="utf-8")

        with self.assertRaisesRegex(NegativeUniversalError, "settings do not match"):
            calibrate_clear_base(clear, dark, flat, (0, 0, 2, 1), 0.3, 0.8, output)
        self.assertFalse(output.exists())

        flat_manifest["settings_sha256"] = "a" * 64
        flat_manifest_path.write_text(json.dumps(flat_manifest), encoding="utf-8")
        for roi, target, ceiling in (
            ((0, 0, 3, 1), 0.3, 0.8),
            ((0, 0, 2, 1), 0.0, 0.8),
            ((0, 0, 2, 1), 0.9, 0.8),
        ):
            with self.subTest(roi=roi, target=target, ceiling=ceiling):
                with self.assertRaises(NegativeUniversalError):
                    calibrate_clear_base(
                        clear, dark, flat, roi, target, ceiling, output
                    )
                self.assertFalse(output.exists())

        for channel in "RGB":
            frame = flat / f"{channel}.pgm"
            write_pgm(frame, PgmImage(2, 1, 255, array("H", [10, 10])))
            flat_manifest["frames"][channel]["sha256"] = sha256_file(frame)
        flat_manifest_path.write_text(json.dumps(flat_manifest), encoding="utf-8")
        with self.assertRaisesRegex(NegativeUniversalError, "flat must exceed dark"):
            calibrate_clear_base(clear, dark, flat, (0, 0, 2, 1), 0.3, 0.8, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
