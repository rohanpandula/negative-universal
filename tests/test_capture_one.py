from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - optional dependency gate
    raise unittest.SkipTest("hardware adapter tests require numpy") from error

from negative_universal.capture_one import (
    OFF_PACKET,
    PROBE_PACKET,
    CaptureOneError,
    _Config,
    _camera_state,
    _color_packet,
    _configure_shutter,
    _probe,
    capture_frame,
)
from negative_universal.cli import read_pgm


class _FakeSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.incoming = bytearray()
        self.closed = False

    def write(self, packet: bytes) -> int:
        self.writes.append(bytes(packet))
        if packet == PROBE_PACKET:
            self.incoming.extend(bytes.fromhex("fe01040000a000"))
            self.incoming.extend(bytes.fromhex("fe030400010002"))
        return len(packet)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.incoming.clear()

    def read(self, size: int) -> bytes:
        result = bytes(self.incoming[:size])
        del self.incoming[:size]
        return result

    def close(self) -> None:
        self.closed = True


class _RawType:
    Flat = object()


class _FakeRaw:
    raw_type = _RawType.Flat
    raw_image_visible = np.array([[64, 256], [1024, 4095]], dtype=np.uint16)
    raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
    color_desc = b"RGBG"
    white_level = 4095
    black_level_per_channel = [64, 64, 64, 64]
    camera_white_level_per_channel = [4095, 4095, 4095, 4095]
    sizes = SimpleNamespace(
        raw_width=4,
        raw_height=4,
        left_margin=1,
        top_margin=1,
        flip=0,
    )

    def __enter__(self) -> _FakeRaw:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def raw_color(self, row: int, column: int) -> int:
        return int(self.raw_pattern[row % 2, column % 2])


def _fake_rawpy() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="0.27.0-test",
        libraw_version=(0, 22, 1),
        RawType=_RawType,
        imread=lambda _path: _FakeRaw(),
    )


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _environment() -> dict[str, str]:
    return {
        "NEGATIVE_UNIVERSAL_SCANLIGHT_PORT": "/dev/fake-scanlight",
        "NEGATIVE_UNIVERSAL_SCANLIGHT_HW_ID": "1",
        "NEGATIVE_UNIVERSAL_GPHOTO_PORT": "usb:001,002",
        "NEGATIVE_UNIVERSAL_SETTLE_MS": "0",
    }


class ProtocolTests(unittest.TestCase):
    def test_exact_one_hot_packets_and_rounding(self) -> None:
        self.assertEqual(
            _color_packet("R", 0.5), (bytes.fromhex("fe0006800000000000"), 128)
        )
        self.assertEqual(
            _color_packet("G", 1.0), (bytes.fromhex("fe000600ff00000000"), 255)
        )
        self.assertEqual(
            _color_packet("B", 1 / 255), (bytes.fromhex("fe0006000001000000"), 1)
        )
        self.assertEqual(_color_packet("R", 0.0)[0], OFF_PACKET)

    def test_probe_ignores_telemetry_and_decodes_actual_wire_order(self) -> None:
        serial = _FakeSerial()
        self.assertEqual(_probe(serial), (2, 1))
        self.assertEqual(serial.writes, [PROBE_PACKET])


class ShutterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _Config(
            scanlight_port="/dev/fake",
            gphoto_port="usb:1,2",
            gphoto_camera=None,
            gphoto2="gphoto2",
            settle_seconds=0.0,
            expected_hardware_id=1,
        )

    def test_sets_and_reverifies_a_writable_shutter_choice(self) -> None:
        before = """Readonly: 0
Current: 1/60
Choice: 15 1/125
Choice: 16 1/100
Choice: 17 bulb
"""
        after = """Readonly: 0
Current: 1/100
Choice: 15 1/125
Choice: 16 1/100
"""
        results = [_completed(before), _completed(), _completed(after)]
        with patch(
            "negative_universal.capture_one._run_gphoto", side_effect=results
        ) as runner:
            self.assertEqual(_configure_shutter(self.config, 10_000), 10_000)
        self.assertIn(
            "/main/capturesettings/shutterspeed=16",
            runner.call_args_list[1].args[0],
        )

    def test_rejects_a_mismatched_read_only_shutter(self) -> None:
        readonly = "Readonly: 1\nCurrent: 0.0005s\nChoice: 0 0.0005s\n"
        with patch(
            "negative_universal.capture_one._run_gphoto",
            return_value=_completed(readonly),
        ):
            with self.assertRaisesRegex(CaptureOneError, "read-only shutter"):
                _configure_shutter(self.config, 1_000)


class CaptureTests(unittest.TestCase):
    def _gphoto(
        self,
        *,
        fail_capture: bool = False,
        image_format: str = "NEF (Raw)",
        capture_target_available: bool = True,
    ):
        shutter = "Readonly: 1\nCurrent: 1/100\nChoice: 0 1/100\n"
        capture_target = "Internal RAM"
        configs = {
            "/main/status/cameramodel": "Readonly: 1\nCurrent: Z f\n",
            "/main/status/serialnumber": "Readonly: 1\nCurrent: ZF123\n",
            "/main/imgsettings/iso": "Readonly: 0\nCurrent: 100\n",
            "/main/capturesettings/imagequality": (
                f"Readonly: 0\nCurrent: {image_format}\n"
            ),
            "/main/capturesettings/expprogram": "Readonly: 1\nCurrent: M\n",
        }
        if capture_target_available:
            configs["/main/settings/capturetarget"] = ""

        def run(command: list[str], *, timeout_seconds: float):
            nonlocal capture_target
            del timeout_seconds
            if "--version" in command:
                return _completed("gphoto2 2.5.32\n")
            if "--list-config" in command:
                return _completed("\n".join(configs) + "\n")
            if "--get-config" in command:
                path = command[command.index("--get-config") + 1]
                if path.endswith("shutterspeed"):
                    return _completed(shutter)
                if path.endswith("capturetarget"):
                    return _completed(
                        "Readonly: 0\n"
                        f"Current: {capture_target}\n"
                        "Choice: 0 Internal RAM\n"
                        "Choice: 1 Memory card\n"
                    )
                return _completed(configs[path])
            if "--set-config-index" in command:
                setting = command[command.index("--set-config-index") + 1]
                self.assertEqual(setting, "/main/settings/capturetarget=1")
                capture_target = "Memory card"
                return _completed()
            if "--capture-image-and-download" in command:
                if fail_capture:
                    raise CaptureOneError("simulated camera failure")
                self.assertIn("--keep", command)
                self.assertNotIn("--no-keep", command)
                pattern = Path(command[command.index("--filename") + 1])
                Path(str(pattern).replace("%C", "NEF")).write_bytes(b"native raw")
                return _completed()
            self.fail(f"unexpected gphoto command: {command}")

        return run

    def test_captures_retains_native_and_augments_decoder_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "R.pgm"
            serial = _FakeSerial()
            with (
                patch(
                    "negative_universal.capture_one._open_scanlight",
                    return_value=serial,
                ),
                patch(
                    "negative_universal.capture_one._run_gphoto",
                    side_effect=self._gphoto(),
                ),
                patch("negative_universal.capture_one.time.sleep"),
                patch.dict(sys.modules, {"rawpy": _fake_rawpy()}),
            ):
                capture_frame(
                    kind="film",
                    channel="R",
                    power=0.5,
                    exposure_us=10_000,
                    output=output,
                    environment=_environment(),
                )

            self.assertEqual((root / "R.NEF").read_bytes(), b"native raw")
            self.assertEqual(list(read_pgm(output).pixels), [64, 256, 1024, 4095])
            receipt = json.loads((root / "R.receipt.json").read_text())
            self.assertEqual(
                receipt["capture"],
                {
                    "kind": "film",
                    "channel": "R",
                    "commanded_power": 0.5,
                    "commanded_exposure_us": 10_000,
                    "applied_exposure_us": 10_000,
                    "camera_port": "usb:001,002",
                    "gphoto2_version": "2.5.32",
                },
            )
            self.assertEqual(
                receipt["camera"],
                {
                    "model": "Z f",
                    "serial_number": "ZF123",
                    "iso": "100",
                    "image_format": "NEF (Raw)",
                    "image_format_config": "/main/capturesettings/imagequality",
                    "exposure_mode": "M",
                    "capture_target": "Memory card",
                    "gphoto_driver": None,
                },
            )
            self.assertEqual(
                receipt["scanlight"],
                {
                    "port": "/dev/fake-scanlight",
                    "hardware_id": 1,
                    "firmware_id": 2,
                    "pwm_code": 128,
                    "settle_ms": 0.0,
                },
            )
            self.assertEqual(
                serial.writes,
                [
                    OFF_PACKET,
                    PROBE_PACKET,
                    bytes.fromhex("fe0006800000000000"),
                    OFF_PACKET,
                ],
            )
            self.assertTrue(serial.closed)

    def test_camera_failure_still_sends_off_and_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "B.pgm"
            serial = _FakeSerial()
            with (
                patch(
                    "negative_universal.capture_one._open_scanlight",
                    return_value=serial,
                ),
                patch(
                    "negative_universal.capture_one._run_gphoto",
                    side_effect=self._gphoto(fail_capture=True),
                ),
                patch("negative_universal.capture_one.time.sleep"),
            ):
                with self.assertRaisesRegex(
                    CaptureOneError, "simulated camera failure"
                ):
                    capture_frame(
                        kind="film",
                        channel="B",
                        power=0.25,
                        exposure_us=10_000,
                        output=output,
                        environment=_environment(),
                    )

            self.assertEqual(serial.writes[-1], OFF_PACKET)
            self.assertTrue(serial.closed)
            self.assertFalse(output.exists())
            self.assertFalse((root / "B.NEF").exists())
            self.assertFalse((root / "B.receipt.json").exists())

    def test_rejects_non_raw_camera_state_before_enabling_light(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            serial = _FakeSerial()
            with (
                patch(
                    "negative_universal.capture_one._open_scanlight",
                    return_value=serial,
                ),
                patch(
                    "negative_universal.capture_one._run_gphoto",
                    side_effect=self._gphoto(image_format="NEF+Fine"),
                ),
                patch("negative_universal.capture_one.time.sleep"),
            ):
                with self.assertRaisesRegex(CaptureOneError, "RAW-only"):
                    capture_frame(
                        kind="film",
                        channel="R",
                        power=0.5,
                        exposure_us=10_000,
                        output=root / "R.pgm",
                        environment=_environment(),
                    )

            self.assertEqual(serial.writes, [OFF_PACKET, PROBE_PACKET, OFF_PACKET])

    def test_camera_state_permits_unavailable_capture_target(self) -> None:
        config = _Config(
            scanlight_port="/dev/fake",
            gphoto_port="usb:1,2",
            gphoto_camera="Named Driver",
            gphoto2="gphoto2",
            settle_seconds=0.0,
            expected_hardware_id=1,
        )
        with patch(
            "negative_universal.capture_one._run_gphoto",
            side_effect=self._gphoto(capture_target_available=False),
        ):
            state = _camera_state(config)
        self.assertIsNone(state["capture_target"])
        self.assertEqual(state["gphoto_driver"], "Named Driver")

    def test_post_decode_race_does_not_delete_a_foreign_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "R.pgm"
            serial = _FakeSerial()

            def raced_decode(_native: Path, destination: Path, receipt: Path) -> None:
                destination.write_bytes(b"decoder output")
                receipt.write_text("{}", encoding="utf-8")
                foreign = root / ".foreign.pgm"
                foreign.write_bytes(b"foreign output")
                foreign.replace(destination)

            with (
                patch(
                    "negative_universal.capture_one._open_scanlight",
                    return_value=serial,
                ),
                patch(
                    "negative_universal.capture_one._run_gphoto",
                    side_effect=self._gphoto(),
                ),
                patch("negative_universal.capture_one.time.sleep"),
                patch(
                    "negative_universal.raw_decode.decode_raw",
                    side_effect=raced_decode,
                ),
                patch(
                    "negative_universal.capture_one._augment_receipt",
                    side_effect=RuntimeError("simulated augmentation failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "simulated augmentation failure"
                ):
                    capture_frame(
                        kind="film",
                        channel="R",
                        power=0.5,
                        exposure_us=10_000,
                        output=output,
                        environment=_environment(),
                    )

            self.assertEqual(output.read_bytes(), b"foreign output")


if __name__ == "__main__":
    unittest.main()
