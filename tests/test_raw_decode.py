from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - optional dependency gate
    raise unittest.SkipTest("raw decoder tests require numpy") from error

from negative_universal.cli import read_pgm
import negative_universal.raw_decode as raw_decode
from negative_universal.raw_decode import RawDecodeError, decode_raw


class _RawType:
    Flat = object()
    Stack = object()


class _FakeRaw:
    def __init__(
        self,
        image: np.ndarray,
        *,
        pattern: np.ndarray | None = None,
        raw_type: object = _RawType.Flat,
        white_level: int = 4095,
    ) -> None:
        self.raw_type = raw_type
        self.raw_image_visible = image
        self.raw_pattern = (
            np.array([[0, 1], [3, 2]], dtype=np.uint8) if pattern is None else pattern
        )
        self.color_desc = b"RGBG"
        self.white_level = white_level
        self.black_level_per_channel = [64, 65, 66, 67]
        self.camera_white_level_per_channel = [4000, 4001, 4002, 4003]
        height, width = image.shape
        self.sizes = SimpleNamespace(
            raw_width=width + 3,
            raw_height=height + 2,
            left_margin=1,
            top_margin=1,
            flip=5,
        )

    def __enter__(self) -> _FakeRaw:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def raw_color(self, row: int, column: int) -> int:
        return int(self.raw_pattern[row % 2, column % 2])


def _fake_rawpy(raw: _FakeRaw) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="9.8.7",
        libraw_version=(1, 2, 3),
        RawType=_RawType,
        imread=lambda _path: raw,
    )


class RawDecodeTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, Path]:
        native = root / "R.NEF"
        native.write_bytes(b"fake native RAW")
        return native, root / "R.pgm", root / "R.receipt.json"

    def test_preserves_codes_and_writes_exact_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            native, output, receipt = self._paths(root)
            image = np.array([[64, 256], [1024, 4095]], dtype=np.uint16)
            raw = _FakeRaw(image)

            with patch.dict(sys.modules, {"rawpy": _fake_rawpy(raw)}):
                decode_raw(native, output, receipt)

            decoded = read_pgm(output)
            self.assertEqual(decoded.max_value, 4095)
            self.assertEqual(list(decoded.pixels), [64, 256, 1024, 4095])
            self.assertEqual(native.read_bytes(), b"fake native RAW")
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8")),
                {
                    "schema": 1,
                    "native_file": "R.NEF",
                    "decoder": {
                        "name": "rawpy",
                        "version": "9.8.7",
                        "libraw_version": "1.2.3",
                        "operation": "raw_image_visible",
                    },
                    "raster": {
                        "raw_width": 5,
                        "raw_height": 4,
                        "crop_left": 1,
                        "crop_top": 1,
                        "width": 2,
                        "height": 2,
                        "sensor_layout": "bggr",
                        "black_levels": [64, 65, 66, 67],
                        "white_level": 4095,
                        "camera_white_levels": [4000, 4001, 4002, 4003],
                        "libraw_flip": 5,
                        "orientation_applied": False,
                    },
                },
            )

    def test_accepts_a_mono_plane_without_a_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            native, output, receipt = self._paths(root)
            raw = _FakeRaw(np.array([[1, 2]], dtype=np.uint16))
            raw.raw_pattern = None
            raw.camera_white_level_per_channel = None

            with patch.dict(sys.modules, {"rawpy": _fake_rawpy(raw)}):
                decode_raw(native, output, receipt)

            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["raster"]["sensor_layout"], "mono")
            self.assertIsNone(payload["raster"]["camera_white_levels"])

    def test_rejects_ambiguous_or_out_of_scale_raw(self) -> None:
        cases = (
            (
                _FakeRaw(
                    np.ones((2, 2), dtype=np.uint16),
                    pattern=np.zeros((6, 6), dtype=np.uint8),
                ),
                "only mono or 2x2 Bayer",
            ),
            (
                _FakeRaw(
                    np.array([[4096]], dtype=np.uint16),
                    white_level=4095,
                ),
                "exceeds camera white level",
            ),
            (
                _FakeRaw(
                    np.ones((2, 2), dtype=np.uint16),
                    raw_type=_RawType.Stack,
                ),
                "only flat RAW",
            ),
        )
        for raw, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    native, output, receipt = self._paths(root)
                    with patch.dict(sys.modules, {"rawpy": _fake_rawpy(raw)}):
                        with self.assertRaisesRegex(RawDecodeError, message):
                            decode_raw(native, output, receipt)
                    self.assertFalse(output.exists())
                    self.assertFalse(receipt.exists())

    def test_refuses_to_overwrite_either_output(self) -> None:
        for occupied in ("R.pgm", "R.receipt.json"):
            with self.subTest(occupied=occupied):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    native, output, receipt = self._paths(root)
                    existing = root / occupied
                    existing.write_bytes(b"keep")
                    raw = _FakeRaw(np.array([[1]], dtype=np.uint16))

                    with patch.dict(sys.modules, {"rawpy": _fake_rawpy(raw)}):
                        with self.assertRaisesRegex(
                            RawDecodeError, "refusing to overwrite"
                        ):
                            decode_raw(native, output, receipt)

                    self.assertEqual(existing.read_bytes(), b"keep")
                    other = receipt if existing == output else output
                    self.assertFalse(other.exists())

    def test_interrupted_publish_exposes_only_a_complete_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            temporary = root / ".R.pgm.complete"
            destination = root / "R.pgm"
            payload = b"complete raster payload"
            temporary.write_bytes(payload)
            real_replace = os.replace

            def replace_then_interrupt(source: Path, target: Path) -> None:
                real_replace(source, target)
                raise KeyboardInterrupt("simulated interruption after rename")

            with patch.object(
                raw_decode.os, "replace", side_effect=replace_then_interrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    raw_decode._publish(temporary, destination)

            self.assertEqual(destination.read_bytes(), payload)

    def test_publish_syncs_source_before_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            temporary = root / ".R.pgm.complete"
            destination = root / "R.pgm"
            temporary.write_bytes(b"complete raster payload")
            source_identity = (temporary.stat().st_dev, temporary.stat().st_ino)
            events: list[str] = []
            real_replace = os.replace

            def record_fsync(descriptor: int) -> None:
                status = os.fstat(descriptor)
                self.assertEqual((status.st_dev, status.st_ino), source_identity)
                events.append("source fsync")

            def record_replace(source: Path, target: Path) -> None:
                self.assertEqual(events, ["source fsync"])
                real_replace(source, target)
                events.append("replace")

            def record_directory_fsync(_directory: Path) -> None:
                events.append("directory fsync")

            with (
                patch.object(raw_decode.os, "fsync", side_effect=record_fsync),
                patch.object(raw_decode.os, "replace", side_effect=record_replace),
                patch.object(
                    raw_decode,
                    "_fsync_directory",
                    side_effect=record_directory_fsync,
                ),
            ):
                raw_decode._publish(temporary, destination)

            self.assertEqual(events, ["source fsync", "replace", "directory fsync"])

    def test_cleanup_preserves_a_raced_foreign_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            native, output, receipt = self._paths(root)
            raw = _FakeRaw(np.array([[1]], dtype=np.uint16))
            foreign = root / ".foreign.pgm"
            foreign.write_bytes(b"foreign output")
            real_publish = raw_decode._publish
            publication_count = 0

            def publish_with_race(
                temporary: Path, destination: Path
            ) -> tuple[int, int]:
                nonlocal publication_count
                publication_count += 1
                if publication_count == 1:
                    return real_publish(temporary, destination)
                os.replace(foreign, output)
                raise RawDecodeError("simulated receipt publication failure")

            with (
                patch.dict(sys.modules, {"rawpy": _fake_rawpy(raw)}),
                patch.object(raw_decode, "_publish", side_effect=publish_with_race),
            ):
                with self.assertRaisesRegex(
                    RawDecodeError, "simulated receipt publication failure"
                ):
                    decode_raw(native, output, receipt)

            self.assertEqual(output.read_bytes(), b"foreign output")
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
