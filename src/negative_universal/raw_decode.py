"""Decode one native RAW into an unscaled, undemosaiced PGM mosaic."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


_BAYER_LAYOUTS = frozenset({"rggb", "bggr", "grbg", "gbrg"})
_RECEIPT_SCHEMA = 1
_WRITE_CHUNK_BYTES = 8 * 1024 * 1024


class RawDecodeError(ValueError):
    """The RAW cannot be represented by the fixed-scale mosaic contract."""


def _version(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ".".join(str(part) for part in value)
    return str(value)


def _validate_paths(native_path: Path, output_path: Path, receipt_path: Path) -> None:
    if not native_path.is_file() or native_path.is_symlink():
        raise RawDecodeError(f"native RAW must be a regular file: {native_path}")
    if output_path.suffix.lower() != ".pgm":
        raise RawDecodeError("output path must end in .pgm")
    if receipt_path.name != f"{output_path.stem}.receipt.json":
        raise RawDecodeError(
            f"receipt path must be named {output_path.stem}.receipt.json"
        )
    if native_path.stem != output_path.stem:
        raise RawDecodeError("native RAW, PGM, and receipt must share one stem")

    parents = {
        native_path.parent.resolve(),
        output_path.parent.resolve(),
        receipt_path.parent.resolve(),
    }
    if len(parents) != 1:
        raise RawDecodeError("native RAW, PGM, and receipt must be siblings")
    for path in (output_path, receipt_path):
        if os.path.lexists(path):
            raise RawDecodeError(f"refusing to overwrite existing path: {path}")


def _sensor_layout(raw: Any, numpy: Any) -> str:
    pattern = raw.raw_pattern
    if pattern is None:
        return "mono"
    shape = numpy.asarray(pattern).shape
    if shape == (1, 1):
        return "mono"
    if shape != (2, 2):
        raise RawDecodeError(f"only mono or 2x2 Bayer RAW is supported, got {shape}")

    try:
        description = bytes(raw.color_desc).decode("ascii")
        top = int(raw.sizes.top_margin)
        left = int(raw.sizes.left_margin)
        layout = "".join(
            description[int(raw.raw_color(top + row, left + column))]
            for row in range(2)
            for column in range(2)
        ).lower()
    except (IndexError, TypeError, UnicodeDecodeError, ValueError) as error:
        raise RawDecodeError(
            "cannot resolve the Bayer pattern at the visible origin"
        ) from error
    if layout not in _BAYER_LAYOUTS:
        raise RawDecodeError(f"unsupported Bayer pattern at visible origin: {layout!r}")
    return layout


def _four_levels(value: Any, label: str, *, optional: bool = False) -> list[int] | None:
    if value is None and optional:
        return None
    try:
        levels = [int(level) for level in value]
    except (TypeError, ValueError) as error:
        raise RawDecodeError(f"{label} must contain four integer levels") from error
    if len(levels) != 4 or any(level < 0 for level in levels):
        raise RawDecodeError(f"{label} must contain four non-negative integer levels")
    return levels


def _write_pgm(stream: Any, image: Any, max_value: int, numpy: Any) -> None:
    height, width = image.shape
    stream.write(f"P5\n{width} {height}\n{max_value}\n".encode("ascii"))
    bytes_per_sample = 1 if max_value < 256 else 2
    rows_per_chunk = max(1, _WRITE_CHUNK_BYTES // (width * bytes_per_sample))
    dtype = numpy.dtype("u1" if bytes_per_sample == 1 else ">u2")
    for first_row in range(0, height, rows_per_chunk):
        chunk = numpy.asarray(
            image[first_row : first_row + rows_per_chunk], dtype=dtype, order="C"
        )
        stream.write(chunk.tobytes(order="C"))


def _new_temporary_file(path: Path) -> tuple[Any, Path]:
    stream = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", dir=path.parent, delete=False
    )
    return stream, Path(stream.name)


def _file_identity(path: Path) -> tuple[int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino


def _unlink_if_owned(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        if _file_identity(path) == identity:
            path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(temp_path: Path, destination: Path) -> tuple[int, int]:
    """Atomically publish a fully synced sibling temporary file.

    macOS SMB does not support hard links or exclusive rename flags.  The
    immediate existence check preserves the no-overwrite contract for normal
    single-writer use; ``os.replace`` keeps an interrupted publish from ever
    exposing a partially copied final file.
    """

    with temp_path.open("rb") as source:
        os.fsync(source.fileno())
    if os.path.lexists(destination):
        raise RawDecodeError(f"refusing to overwrite existing path: {destination}")
    try:
        os.replace(temp_path, destination)
    except FileExistsError as error:
        raise RawDecodeError(
            f"refusing to overwrite existing path: {destination}"
        ) from error
    identity = _file_identity(destination)
    try:
        _fsync_directory(destination.parent)
    except BaseException:
        _unlink_if_owned(destination, identity)
        raise
    return identity


def decode_raw(native_path: Path, output_path: Path, receipt_path: Path) -> None:
    """Preserve one native RAW plane as PGM and write its provenance receipt.

    The decoder reads ``raw_image_visible`` directly and never calls LibRaw's
    post-processing path, so no demosaic, white balance, tone curve, rotation,
    black subtraction, or content-dependent scaling is applied.
    """

    native_path = Path(native_path)
    output_path = Path(output_path)
    receipt_path = Path(receipt_path)
    _validate_paths(native_path, output_path, receipt_path)

    try:
        import numpy
        import rawpy
    except ImportError as error:
        raise RawDecodeError("native RAW decoding requires rawpy and numpy") from error

    pgm_temp: Path | None = None
    receipt_temp: Path | None = None
    published_output: tuple[int, int] | None = None
    try:
        with rawpy.imread(str(native_path)) as raw:
            if raw.raw_type != rawpy.RawType.Flat:
                raise RawDecodeError(f"only flat RAW is supported, got {raw.raw_type}")
            image = numpy.asarray(raw.raw_image_visible)
            if image.ndim != 2 or image.dtype.kind != "u" or image.dtype.itemsize != 2:
                raise RawDecodeError(
                    f"raw_image_visible must be a 2D uint16 plane, got {image.shape} {image.dtype}"
                )
            if not image.size:
                raise RawDecodeError("raw_image_visible must not be empty")

            max_value = int(raw.white_level or 0)
            if not 1 <= max_value <= 65535:
                raise RawDecodeError(
                    f"camera white level must be in 1..65535, got {max_value}"
                )
            sample_max = int(image.max())
            if sample_max > max_value:
                raise RawDecodeError(
                    f"RAW sample {sample_max} exceeds camera white level {max_value}"
                )

            height, width = (int(size) for size in image.shape)
            sizes = raw.sizes
            raw_width = int(sizes.raw_width)
            raw_height = int(sizes.raw_height)
            crop_left = int(sizes.left_margin)
            crop_top = int(sizes.top_margin)
            if min(raw_width, raw_height, width, height) <= 0:
                raise RawDecodeError("RAW dimensions must be positive")
            if (
                crop_left < 0
                or crop_top < 0
                or crop_left + width > raw_width
                or crop_top + height > raw_height
            ):
                raise RawDecodeError("visible RAW crop falls outside the full raster")

            receipt = {
                "schema": _RECEIPT_SCHEMA,
                "native_file": native_path.name,
                "decoder": {
                    "name": "rawpy",
                    "version": _version(rawpy.__version__),
                    "libraw_version": _version(rawpy.libraw_version),
                    "operation": "raw_image_visible",
                },
                "raster": {
                    "raw_width": raw_width,
                    "raw_height": raw_height,
                    "crop_left": crop_left,
                    "crop_top": crop_top,
                    "width": width,
                    "height": height,
                    "sensor_layout": _sensor_layout(raw, numpy),
                    "black_levels": _four_levels(
                        raw.black_level_per_channel, "black_levels"
                    ),
                    "white_level": max_value,
                    "camera_white_levels": _four_levels(
                        raw.camera_white_level_per_channel,
                        "camera_white_levels",
                        optional=True,
                    ),
                    "libraw_flip": int(sizes.flip),
                    "orientation_applied": False,
                },
            }

            pgm_stream, pgm_temp = _new_temporary_file(output_path)
            with pgm_stream:
                _write_pgm(pgm_stream, image, max_value, numpy)
                pgm_stream.flush()
                os.fsync(pgm_stream.fileno())

        receipt_stream, receipt_temp = _new_temporary_file(receipt_path)
        with receipt_stream:
            receipt_stream.write(
                (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            receipt_stream.flush()
            os.fsync(receipt_stream.fileno())

        published_output = _publish(pgm_temp, output_path)
        pgm_temp = None
        _publish(receipt_temp, receipt_path)
        receipt_temp = None
    except BaseException:
        _unlink_if_owned(output_path, published_output)
        raise
    finally:
        for temporary in (pgm_temp, receipt_temp):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
