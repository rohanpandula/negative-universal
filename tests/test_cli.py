from __future__ import annotations

import json
import stat
import sys
import tempfile
import time
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

from negative_universal.cli import (
    NegativeUniversalError,
    PgmImage,
    _capture_receipt_source,
    capture_triplet,
    normalize_triplets,
    read_pgm,
    sha256_file,
    write_pgm,
)


class PgmTests(unittest.TestCase):
    def test_round_trips_8_and_16_bit_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = (
                PgmImage(2, 2, 255, array("H", [0, 1, 127, 255])),
                PgmImage(2, 2, 65535, array("H", [0, 1, 256, 65535])),
            )
            for index, image in enumerate(images):
                path = root / f"{index}.pgm"
                write_pgm(path, image)
                self.assertEqual(read_pgm(path), image)
                if image.max_value == 65535:
                    self.assertTrue(
                        path.read_bytes().endswith(b"\x00\x00\x00\x01\x01\x00\xff\xff")
                    )

    def test_reads_comments_in_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "commented.pgm"
            path.write_bytes(b"P5\n# generated\n2 1\n255\n\x01\x02")
            self.assertEqual(list(read_pgm(path).pixels), [1, 2])

    def test_reads_crlf_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "crlf.pgm"
            path.write_bytes(b"P5\r\n1 1\r\n255\r\n\x0a")
            self.assertEqual(list(read_pgm(path).pixels), [10])


class CaptureAndNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.settings = self.root / "settings.json"
        self.settings.write_text(
            json.dumps(
                {
                    "_meta": {
                        "configuration_id": "fake-rig-v1",
                        "camera": "fake-camera",
                        "gain": 1.0,
                    },
                    "R": {"power": 0.25, "exposure_us": 1000},
                    "G": {"power": 0.5, "exposure_us": 2000},
                    "B": {"power": 0.75, "exposure_us": 3000},
                }
            ),
            encoding="utf-8",
        )
        self.log = self.root / "rig.log"
        self.rig = self.root / "fake-rig"
        self.rig.write_text(
            f"""#!{sys.executable}
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--kind')
parser.add_argument('--channel')
parser.add_argument('--power')
parser.add_argument('--exposure-us')
parser.add_argument('--output')
parser.add_argument('--receipt')
args = parser.parse_args()
with Path({str(self.log)!r}).open('a') as log:
    if args.command == 'off':
        log.write('OFF\\n')
    else:
        log.write(f'{{args.channel}} {{args.kind}} {{args.power}} {{args.exposure_us}}\\n')
if args.command == 'off':
    raise SystemExit(0)
values = {{'dark': 10, 'flat': 110, 'film': 60}}
Path(args.output).write_bytes(b'P5\\n2 1\\n255\\n' + bytes([values[args.kind]] * 2))
if args.receipt:
    import json
    native = Path(args.output).with_suffix('.NEF')
    native.write_bytes(b'fake-native-' + args.channel.encode())
    power = float(args.power)
    receipt = {{
        'schema': 1,
        'native_file': native.name,
        'decoder': {{
            'name': 'rawpy', 'version': '0.27.0',
            'libraw_version': '0.22.1',
            'operation': 'raw_image_visible',
        }},
        'raster': {{
            'raw_width': 2, 'raw_height': 1,
            'crop_left': 0, 'crop_top': 0,
            'width': 2, 'height': 1,
            'sensor_layout': 'rggb',
            'black_levels': [0, 0, 0, 0],
            'white_level': 255,
            'camera_white_levels': [255, 255, 255, 255],
            'libraw_flip': 0,
            'orientation_applied': False,
        }},
        'capture': {{
            'kind': args.kind, 'channel': args.channel,
            'commanded_power': power,
            'commanded_exposure_us': int(args.exposure_us),
            'applied_exposure_us': int(args.exposure_us),
            'camera_port': 'fake-camera-port',
            'gphoto2_version': 'fake',
        }},
        'camera': {{
            'model': 'Fake Camera', 'serial_number': 'FAKE123',
            'iso': '100', 'image_format': 'NEF (Raw)',
            'image_format_config': '/main/capturesettings/imagequality',
            'exposure_mode': 'M', 'capture_target': 'Memory card',
            'gphoto_driver': None,
        }},
        'scanlight': {{
            'port': 'fake-scanlight-port',
            'hardware_id': 1, 'firmware_id': 2,
            'pwm_code': int(power * 255 + 0.5),
            'settle_ms': 250.0,
        }},
    }}
    Path(args.receipt).write_text(json.dumps(receipt))
""",
            encoding="utf-8",
        )
        self.rig.chmod(self.rig.stat().st_mode | stat.S_IXUSR)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _capture(self, kind: str) -> Path:
        output = self.root / kind
        capture_triplet(
            kind=kind,
            settings_path=self.settings,
            rig=self.rig,
            output=output,
        )
        return output

    def _replace_frame(
        self,
        directory: Path,
        channel: str,
        pixels: list[int],
        *,
        width: int = 2,
        max_value: int = 255,
    ) -> None:
        frame_path = directory / f"{channel}.pgm"
        height, remainder = divmod(len(pixels), width)
        self.assertEqual(remainder, 0)
        image = PgmImage(width, height, max_value, array("H", pixels))
        write_pgm(frame_path, image)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = manifest["frames"][channel]
        record["width"] = image.width
        record["height"] = image.height
        record["max_value"] = image.max_value
        record["sha256"] = sha256_file(frame_path)
        if record.get("native_source") is not None:
            receipt_path = directory / f"{channel}.receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["raster"].update(
                {
                    "raw_width": image.width,
                    "raw_height": image.height,
                    "width": image.width,
                    "height": image.height,
                    "white_level": image.max_value,
                    "camera_white_levels": [image.max_value] * 4,
                }
            )
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            record["native_source"] = _capture_receipt_source(
                directory=directory,
                channel=channel,
                kind=manifest["kind"],
                image=image,
                expected_layout=manifest["rig_metadata"]["sensor_layout"],
                commanded_power=record["commanded_power"],
                commanded_exposure_us=record["commanded_exposure_us"],
            )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_capture_and_normalize(self) -> None:
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")
        normalized = self.root / "normalized"

        normalize_triplets(film=film, dark=dark, flat=flat, output=normalized)

        for channel in ("R", "G", "B"):
            image = read_pgm(normalized / f"{channel}.pgm")
            self.assertEqual(list(image.pixels), [32768, 32768])
        film_manifest = json.loads((film / "manifest.json").read_text())
        self.assertEqual(film_manifest["rig_metadata"]["camera"], "fake-camera")
        lines = self.log.read_text().splitlines()
        captures = [line for line in lines if line != "OFF"]
        self.assertEqual([line.split()[0] for line in captures], list("RGB") * 3)
        self.assertEqual([line.split()[2] for line in captures[:3]], ["0", "0", "0"])
        self.assertEqual(lines.count("OFF"), 12)

    def test_reports_low_and_high_clipping(self) -> None:
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")
        for channel in ("R", "G", "B"):
            self._replace_frame(film, channel, [0, 200])
        normalized = self.root / "normalized"

        normalize_triplets(film=film, dark=dark, flat=flat, output=normalized)

        self.assertEqual(list(read_pgm(normalized / "R.pgm").pixels), [0, 65535])
        manifest = json.loads((normalized / "manifest.json").read_text())
        self.assertEqual(manifest["frames"]["R"]["clipped_low"], 1)
        self.assertEqual(manifest["frames"]["R"]["clipped_high"], 1)

    def test_bayer_layout_and_phase_survive_normalization(self) -> None:
        settings = json.loads(self.settings.read_text())
        settings["_meta"]["sensor_layout"] = "RGGB"
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")

        for channel in ("R", "G", "B"):
            self._replace_frame(dark, channel, [10, 10, 10, 10], max_value=65535)
            self._replace_frame(flat, channel, [110, 110, 110, 110], max_value=65535)
            self._replace_frame(film, channel, [20, 35, 60, 85], max_value=65535)

        normalized = self.root / "normalized-bayer"
        normalize_triplets(film=film, dark=dark, flat=flat, output=normalized)

        capture_manifest = json.loads((film / "manifest.json").read_text())
        normalized_manifest = json.loads((normalized / "manifest.json").read_text())
        self.assertEqual(capture_manifest["schema"], 2)
        self.assertEqual(capture_manifest["rig_metadata"]["sensor_layout"], "rggb")
        self.assertEqual(normalized_manifest["sensor_layout"], "rggb")
        for channel in ("R", "G", "B"):
            source = capture_manifest["frames"][channel]["native_source"]
            self.assertEqual(source["file"], f"{channel}.NEF")
            self.assertEqual(source["sha256"], sha256_file(film / f"{channel}.NEF"))
            self.assertEqual(
                source["receipt_sha256"],
                sha256_file(film / f"{channel}.receipt.json"),
            )
            image = read_pgm(normalized / f"{channel}.pgm")
            self.assertEqual((image.width, image.height), (2, 2))
            self.assertEqual(list(image.pixels), [6554, 16384, 32768, 49151])

    def test_missing_sensor_layout_defaults_to_mono(self) -> None:
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")
        manifest = json.loads((film / "manifest.json").read_text())
        self.assertEqual(manifest["rig_metadata"]["sensor_layout"], "mono")

        for directory in (dark, flat, film):
            manifest_path = directory / "manifest.json"
            legacy_manifest = json.loads(manifest_path.read_text())
            legacy_manifest["schema"] = 1
            legacy_manifest["rig_metadata"].pop("sensor_layout")
            for record in legacy_manifest["frames"].values():
                record.pop("native_source")
            manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

        normalized = self.root / "normalized-legacy"
        normalize_triplets(film=film, dark=dark, flat=flat, output=normalized)
        normalized_manifest = json.loads((normalized / "manifest.json").read_text())
        self.assertEqual(normalized_manifest["sensor_layout"], "mono")

    def test_rejects_invalid_or_mismatched_sensor_layout(self) -> None:
        settings = json.loads(self.settings.read_text())
        settings["_meta"]["sensor_layout"] = "xtrans"
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        with self.assertRaisesRegex(NegativeUniversalError, "sensor_layout"):
            self._capture("film")

        settings["_meta"]["sensor_layout"] = None
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        with self.assertRaisesRegex(NegativeUniversalError, "sensor_layout"):
            self._capture("film")

        settings["_meta"]["sensor_layout"] = "rggb"
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")
        flat_manifest_path = flat / "manifest.json"
        flat_manifest = json.loads(flat_manifest_path.read_text())
        flat_manifest["rig_metadata"]["sensor_layout"] = "bggr"
        flat_manifest_path.write_text(json.dumps(flat_manifest), encoding="utf-8")
        with self.assertRaisesRegex(NegativeUniversalError, "sensor layout"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=self.root / "bad-layout",
            )

    def test_rejects_tampered_native_raw_or_receipt(self) -> None:
        settings = json.loads(self.settings.read_text())
        settings["_meta"]["sensor_layout"] = "rggb"
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")

        (film / "R.NEF").write_bytes(b"tampered")
        with self.assertRaisesRegex(NegativeUniversalError, "native source metadata"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=self.root / "tampered-native",
            )

        manifest_path = film / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        (film / "R.NEF").write_bytes(b"fake-native-R")
        receipt_path = film / "R.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["raster"]["crop_left"] = 9
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(NegativeUniversalError, "visible crop"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=self.root / "tampered-receipt",
            )

    def test_rejects_mixed_native_decode_configuration(self) -> None:
        settings = json.loads(self.settings.read_text())
        settings["_meta"]["sensor_layout"] = "rggb"
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")

        receipt_path = flat / "G.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["decoder"]["version"] = "different"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        manifest_path = flat / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        frame = manifest["frames"]["G"]
        frame["native_source"] = _capture_receipt_source(
            directory=flat,
            channel="G",
            kind="flat",
            image=read_pgm(flat / "G.pgm"),
            expected_layout="rggb",
            commanded_power=frame["commanded_power"],
            commanded_exposure_us=frame["commanded_exposure_us"],
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(NegativeUniversalError, "decode configuration"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=self.root / "mixed-decoder",
            )

    def test_rejects_mixed_camera_identity(self) -> None:
        settings = json.loads(self.settings.read_text())
        settings["_meta"]["sensor_layout"] = "rggb"
        self.settings.write_text(json.dumps(settings), encoding="utf-8")
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")

        receipt_path = flat / "G.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["camera"]["serial_number"] = "OTHER456"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        manifest_path = flat / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        frame = manifest["frames"]["G"]
        frame["native_source"] = _capture_receipt_source(
            directory=flat,
            channel="G",
            kind="flat",
            image=read_pgm(flat / "G.pgm"),
            expected_layout="rggb",
            commanded_power=frame["commanded_power"],
            commanded_exposure_us=frame["commanded_exposure_us"],
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(NegativeUniversalError, "device configuration"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=self.root / "mixed-camera",
            )

    def test_refuses_nonempty_destination(self) -> None:
        destination = self.root / "existing"
        destination.mkdir()
        (destination / "keep.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(NegativeUniversalError, "refusing to overwrite"):
            capture_triplet(
                kind="film",
                settings_path=self.settings,
                rig=self.rig,
                output=destination,
            )

    def test_rejects_invalid_flat_denominator(self) -> None:
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")
        for channel in ("R", "G", "B"):
            self._replace_frame(flat, channel, [10, 10])

        normalized = self.root / "normalized"
        with self.assertRaisesRegex(NegativeUniversalError, "flat must exceed dark"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=normalized,
            )
        self.assertFalse(normalized.exists())

    def test_rejects_tampered_manifest_metadata(self) -> None:
        dark = self._capture("dark")
        flat = self._capture("flat")
        film = self._capture("film")
        manifest_path = film / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["frames"]["R"]["width"] = 99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(NegativeUniversalError, "metadata mismatch"):
            normalize_triplets(
                film=film,
                dark=dark,
                flat=flat,
                output=self.root / "normalized",
            )

    def test_reports_rig_failure(self) -> None:
        failing_rig = self.root / "failing-rig"
        failing_rig.write_text(
            f"""#!{sys.executable}
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--kind')
parser.add_argument('--channel')
parser.add_argument('--power')
parser.add_argument('--exposure-us')
parser.add_argument('--output')
args = parser.parse_args()
if args.command == 'off':
    raise SystemExit(0)
if args.channel == 'G':
    raise SystemExit(7)
Path(args.output).write_bytes(b'P5\\n1 1\\n255\\n\\x01')
""",
            encoding="utf-8",
        )
        failing_rig.chmod(failing_rig.stat().st_mode | stat.S_IXUSR)
        output = self.root / "failed"
        with self.assertRaisesRegex(NegativeUniversalError, "rig failed"):
            capture_triplet(
                kind="film",
                settings_path=self.settings,
                rig=failing_rig,
                output=output,
            )
        self.assertFalse(output.exists())

    def test_times_out_and_cleans_staging_output(self) -> None:
        slow_rig = self.root / "slow-rig"
        orphan_marker = self.root / "orphan-finished"
        slow_rig.write_text(
            f"""#!{sys.executable}
import subprocess
import sys
import time
if len(sys.argv) > 1 and sys.argv[1] == 'off':
    raise SystemExit(0)
subprocess.Popen([
    sys.executable,
    '-c',
    "import pathlib, time; time.sleep(0.2); pathlib.Path({str(orphan_marker)!r}).write_text('alive')",
])
time.sleep(1)
""",
            encoding="utf-8",
        )
        slow_rig.chmod(slow_rig.stat().st_mode | stat.S_IXUSR)
        output = self.root / "timed-out"
        with self.assertRaisesRegex(NegativeUniversalError, "timed out"):
            capture_triplet(
                kind="film",
                settings_path=self.settings,
                rig=slow_rig,
                output=output,
                timeout_seconds=0.05,
            )
        time.sleep(0.3)
        self.assertFalse(output.exists())
        self.assertFalse(orphan_marker.exists())

    def test_keyboard_interrupt_still_attempts_lights_off(self) -> None:
        output = self.root / "interrupted"
        effects = [None, KeyboardInterrupt(), None]
        with patch("negative_universal.cli._run_rig", side_effect=effects) as run_rig:
            with self.assertRaises(KeyboardInterrupt):
                capture_triplet(
                    kind="film",
                    settings_path=self.settings,
                    rig=self.rig,
                    output=output,
                )
        self.assertEqual(run_rig.call_count, 3)
        self.assertEqual(
            run_rig.call_args_list[-1].kwargs["action"],
            "turning all lights off",
        )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
