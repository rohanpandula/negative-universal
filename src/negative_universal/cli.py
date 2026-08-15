"""Capture and normalize sequential R/G/B raw sensor frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Sequence


CHANNELS = ("R", "G", "B")
SENSOR_LAYOUTS = frozenset({"mono", "rggb", "bggr", "grbg", "gbrg"})
MANIFEST_NAME = "manifest.json"
DEFAULT_CAPTURE_TIMEOUT_SECONDS = 60.0


class NegativeUniversalError(Exception):
    """An expected input, capture, or calibration failure."""


@dataclass(frozen=True, slots=True)
class ChannelSettings:
    power: float
    exposure_us: int


@dataclass(frozen=True, slots=True)
class PgmImage:
    width: int
    height: int
    max_value: int
    pixels: array

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise NegativeUniversalError("PGM dimensions must be positive")
        if not 1 <= self.max_value <= 65535:
            raise NegativeUniversalError("PGM max value must be in 1..65535")
        if len(self.pixels) != self.width * self.height:
            raise NegativeUniversalError("PGM pixel count does not match dimensions")
        if any(pixel > self.max_value for pixel in self.pixels):
            raise NegativeUniversalError("PGM sample exceeds its declared max value")


def _sensor_layout(value: Any, source: str) -> str:
    if not isinstance(value, str) or value.lower() not in SENSOR_LAYOUTS:
        allowed = ", ".join(sorted(SENSOR_LAYOUTS))
        raise NegativeUniversalError(
            f"{source} sensor_layout must be one of: {allowed}"
        )
    return value.lower()


def _read_header_token(source: Any) -> bytes:
    while True:
        first = source.read(1)
        if not first:
            raise NegativeUniversalError("truncated PGM header")
        if first == b"#":
            source.readline()
            continue
        if first.isspace():
            continue
        break

    token = bytearray(first)
    while True:
        byte = source.read(1)
        if not byte:
            raise NegativeUniversalError("truncated PGM header")
        if byte.isspace():
            if byte == b"\r":
                following = source.read(1)
                if following != b"\n" and following:
                    source.seek(-1, 1)
            return bytes(token)
        if byte == b"#":
            source.readline()
            return bytes(token)
        token.extend(byte)


def read_pgm(path: Path) -> PgmImage:
    with path.open("rb") as source:
        if _read_header_token(source) != b"P5":
            raise NegativeUniversalError(f"{path} is not a binary P5 PGM")
        try:
            width = int(_read_header_token(source))
            height = int(_read_header_token(source))
            max_value = int(_read_header_token(source))
        except ValueError as error:
            raise NegativeUniversalError(f"invalid PGM header in {path}") from error
        raster = source.read()

    sample_count = width * height
    bytes_per_sample = 1 if max_value < 256 else 2
    expected_size = sample_count * bytes_per_sample
    if len(raster) != expected_size:
        raise NegativeUniversalError(
            f"{path} has {len(raster)} raster bytes; expected {expected_size}"
        )

    pixels = array("H")
    if bytes_per_sample == 1:
        pixels.extend(raster)
    else:
        pixels.frombytes(raster)
        if sys.byteorder == "little":
            pixels.byteswap()
    return PgmImage(width, height, max_value, pixels)


def write_pgm(path: Path, image: PgmImage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{image.width} {image.height}\n{image.max_value}\n".encode("ascii")
    if image.max_value < 256:
        raster = bytes(pixel for pixel in image.pixels)
    else:
        samples = array("H", image.pixels)
        if sys.byteorder == "little":
            samples.byteswap()
        raster = samples.tobytes()
    with path.open("wb") as output:
        output.write(header)
        output.write(raster)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_settings(
    path: Path,
) -> tuple[dict[str, ChannelSettings], dict[str, Any], str]:
    try:
        settings_bytes = path.read_bytes()
        payload = json.loads(settings_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NegativeUniversalError(f"cannot read settings from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise NegativeUniversalError("settings must be a JSON object")
    allowed_keys = {*CHANNELS, "_meta"}
    if not set(CHANNELS).issubset(payload) or not set(payload).issubset(allowed_keys):
        raise NegativeUniversalError("settings must contain R, G, B, and optional _meta")
    metadata = payload.get("_meta", {})
    if not isinstance(metadata, dict):
        raise NegativeUniversalError("settings _meta must be a JSON object")
    metadata = dict(metadata)
    configuration_id = metadata.get("configuration_id")
    if not isinstance(configuration_id, str) or not configuration_id.strip():
        raise NegativeUniversalError("settings _meta.configuration_id is required")
    metadata["sensor_layout"] = _sensor_layout(
        metadata.get("sensor_layout", "mono"), "settings _meta"
    )

    settings: dict[str, ChannelSettings] = {}
    for channel in CHANNELS:
        item = payload[channel]
        if not isinstance(item, dict) or set(item) != {"power", "exposure_us"}:
            raise NegativeUniversalError(
                f"{channel} settings must contain power and exposure_us"
            )
        power = item["power"]
        exposure_us = item["exposure_us"]
        if isinstance(power, bool) or not isinstance(power, (int, float)):
            raise NegativeUniversalError(f"{channel} power must be numeric")
        power = float(power)
        if not math.isfinite(power) or not 0.0 <= power <= 1.0:
            raise NegativeUniversalError(f"{channel} power must be in 0..1")
        if isinstance(exposure_us, bool) or not isinstance(exposure_us, int):
            raise NegativeUniversalError(f"{channel} exposure_us must be an integer")
        if exposure_us <= 0:
            raise NegativeUniversalError(f"{channel} exposure_us must be positive")
        settings[channel] = ChannelSettings(power, exposure_us)
    settings_hash = hashlib.sha256(settings_bytes).hexdigest()
    return settings, metadata, settings_hash


@contextmanager
def _staged_output(path: Path) -> Iterator[Path]:
    if path.exists():
        raise NegativeUniversalError(f"refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temporary:
        staging = Path(temporary)
        yield staging
        staging.rename(path)


def _frame_record(path: Path, image: PgmImage) -> dict[str, Any]:
    return {
        "file": path.name,
        "width": image.width,
        "height": image.height,
        "max_value": image.max_value,
        "sha256": sha256_file(path),
    }


def _validate_matching_images(images: dict[str, PgmImage], label: str) -> None:
    first = images[CHANNELS[0]]
    for channel in CHANNELS[1:]:
        image = images[channel]
        if (image.width, image.height, image.max_value) != (
            first.width,
            first.height,
            first.max_value,
        ):
            raise NegativeUniversalError(f"{label} R/G/B images do not match")


def _run_rig(
    command: list[str], *, timeout_seconds: float, action: str
) -> None:
    try:
        subprocess.run(command, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise NegativeUniversalError(
            f"rig timed out while {action} after {timeout_seconds:g} seconds"
        ) from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise NegativeUniversalError(f"rig failed while {action}: {error}") from error


def _turn_lights_off(rig: Path, timeout_seconds: float) -> None:
    _run_rig(
        [str(rig.resolve()), "off"],
        timeout_seconds=min(timeout_seconds, 10.0),
        action="turning all lights off",
    )


def capture_triplet(
    *,
    kind: str,
    settings_path: Path,
    rig: Path,
    output: Path,
    timeout_seconds: float = DEFAULT_CAPTURE_TIMEOUT_SECONDS,
) -> None:
    if kind not in {"dark", "flat", "film"}:
        raise NegativeUniversalError(f"unsupported capture kind: {kind}")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise NegativeUniversalError("capture timeout must be positive and finite")
    settings, rig_metadata, settings_hash = load_settings(settings_path)
    with _staged_output(output) as staging:
        images: dict[str, PgmImage] = {}
        frame_records: dict[str, Any] = {}
        _turn_lights_off(rig, timeout_seconds)

        for channel in CHANNELS:
            channel_settings = settings[channel]
            commanded_power = 0.0 if kind == "dark" else channel_settings.power
            frame_path = (staging / f"{channel}.pgm").resolve()
            command = [
                str(rig.resolve()),
                "capture",
                "--kind",
                kind,
                "--channel",
                channel,
                "--power",
                format(commanded_power, ".17g"),
                "--exposure-us",
                str(channel_settings.exposure_us),
                "--output",
                str(frame_path),
            ]
            capture_error: BaseException | None = None
            try:
                _run_rig(
                    command,
                    timeout_seconds=timeout_seconds,
                    action=f"capturing {channel}",
                )
            except BaseException as error:
                capture_error = error
            finally:
                try:
                    _turn_lights_off(rig, timeout_seconds)
                except NegativeUniversalError as off_error:
                    if capture_error is None:
                        raise
                    if isinstance(capture_error, NegativeUniversalError):
                        raise NegativeUniversalError(
                            f"{capture_error}; lights-off cleanup also failed: {off_error}"
                        ) from off_error
                    capture_error.add_note(
                        f"lights-off cleanup also failed: {off_error}"
                    )
            if capture_error is not None:
                raise capture_error
            if not frame_path.is_file():
                raise NegativeUniversalError(f"rig did not create {frame_path}")
            image = read_pgm(frame_path)
            images[channel] = image
            frame_records[channel] = {
                **_frame_record(frame_path, image),
                "commanded_power": commanded_power,
                "commanded_exposure_us": channel_settings.exposure_us,
            }

        _validate_matching_images(images, kind)
        manifest = {
            "schema": 1,
            "kind": kind,
            "captured_at": datetime.now(UTC).isoformat(),
            "channel_order": list(CHANNELS),
            "rig": str(rig.resolve()),
            "rig_metadata": rig_metadata,
            "settings_file": str(settings_path.resolve()),
            "settings_sha256": settings_hash,
            "capture_timeout_seconds": timeout_seconds,
            "frames": frame_records,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _load_manifest(directory: Path, expected_kind: str) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NegativeUniversalError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise NegativeUniversalError(f"unsupported manifest in {path}")
    if payload.get("kind") != expected_kind:
        raise NegativeUniversalError(f"{path} is not a {expected_kind} capture")
    if payload.get("channel_order") != list(CHANNELS):
        raise NegativeUniversalError(f"invalid channel order in {path}")
    settings_hash = payload.get("settings_sha256")
    if not isinstance(settings_hash, str) or len(settings_hash) != 64:
        raise NegativeUniversalError(f"invalid settings hash in {path}")
    if any(character not in "0123456789abcdef" for character in settings_hash):
        raise NegativeUniversalError(f"invalid settings hash in {path}")
    if not isinstance(payload.get("rig"), str) or not payload["rig"]:
        raise NegativeUniversalError(f"invalid rig path in {path}")
    rig_metadata = payload.get("rig_metadata")
    if not isinstance(rig_metadata, dict):
        raise NegativeUniversalError(f"invalid rig metadata in {path}")
    rig_metadata = dict(rig_metadata)
    configuration_id = rig_metadata.get("configuration_id")
    if not isinstance(configuration_id, str) or not configuration_id.strip():
        raise NegativeUniversalError(f"missing configuration ID in {path}")
    rig_metadata["sensor_layout"] = _sensor_layout(
        rig_metadata.get("sensor_layout", "mono"), str(path)
    )
    payload["rig_metadata"] = rig_metadata
    frames = payload.get("frames")
    if not isinstance(frames, dict) or set(frames) != set(CHANNELS):
        raise NegativeUniversalError(f"{path} must describe R, G, and B frames")
    return payload


def _load_capture(directory: Path, kind: str) -> tuple[dict[str, Any], dict[str, PgmImage]]:
    manifest = _load_manifest(directory, kind)
    images: dict[str, PgmImage] = {}
    for channel in CHANNELS:
        record = manifest["frames"][channel]
        required_fields = {
            "file",
            "width",
            "height",
            "max_value",
            "sha256",
            "commanded_power",
            "commanded_exposure_us",
        }
        if not isinstance(record, dict) or set(record) != required_fields:
            raise NegativeUniversalError(f"invalid {channel} record in {directory}")
        if record["file"] != f"{channel}.pgm":
            raise NegativeUniversalError(f"invalid {channel} record in {directory}")
        for field in ("width", "height", "max_value", "commanded_exposure_us"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise NegativeUniversalError(
                    f"invalid {channel} {field} in {directory}"
                )
        if record["max_value"] > 65535:
            raise NegativeUniversalError(f"invalid {channel} max_value in {directory}")
        power = record["commanded_power"]
        if isinstance(power, bool) or not isinstance(power, (int, float)):
            raise NegativeUniversalError(f"invalid {channel} power in {directory}")
        if not math.isfinite(power) or not 0.0 <= power <= 1.0:
            raise NegativeUniversalError(f"invalid {channel} power in {directory}")
        frame_hash = record["sha256"]
        if not isinstance(frame_hash, str) or len(frame_hash) != 64:
            raise NegativeUniversalError(f"invalid {channel} hash in {directory}")
        if any(character not in "0123456789abcdef" for character in frame_hash):
            raise NegativeUniversalError(f"invalid {channel} hash in {directory}")
        frame_path = directory / record["file"]
        try:
            if sha256_file(frame_path) != record.get("sha256"):
                raise NegativeUniversalError(f"hash mismatch for {frame_path}")
            image = read_pgm(frame_path)
        except OSError as error:
            raise NegativeUniversalError(f"cannot read {frame_path}: {error}") from error
        if (
            record.get("width"),
            record.get("height"),
            record.get("max_value"),
        ) != (image.width, image.height, image.max_value):
            raise NegativeUniversalError(f"metadata mismatch for {frame_path}")
        images[channel] = image
    _validate_matching_images(images, kind)
    return manifest, images


def _validate_capture_settings(
    film: dict[str, Any], dark: dict[str, Any], flat: dict[str, Any]
) -> None:
    settings_hashes = {
        film.get("settings_sha256"),
        dark.get("settings_sha256"),
        flat.get("settings_sha256"),
    }
    if None in settings_hashes or len(settings_hashes) != 1:
        raise NegativeUniversalError("film, dark, and flat settings do not match")
    configuration_ids = {
        film["rig_metadata"].get("configuration_id"),
        dark["rig_metadata"].get("configuration_id"),
        flat["rig_metadata"].get("configuration_id"),
    }
    if None in configuration_ids or len(configuration_ids) != 1:
        raise NegativeUniversalError(
            "film, dark, and flat configuration IDs do not match"
        )
    sensor_layouts = {
        film["rig_metadata"]["sensor_layout"],
        dark["rig_metadata"]["sensor_layout"],
        flat["rig_metadata"]["sensor_layout"],
    }
    if len(sensor_layouts) != 1:
        raise NegativeUniversalError(
            "film, dark, and flat sensor layouts do not match"
        )
    rig_paths = {film.get("rig"), dark.get("rig"), flat.get("rig")}
    if None in rig_paths or len(rig_paths) != 1:
        raise NegativeUniversalError("film, dark, and flat rig adapters do not match")
    for channel in CHANNELS:
        film_frame = film["frames"][channel]
        dark_frame = dark["frames"][channel]
        flat_frame = flat["frames"][channel]
        if film_frame["commanded_exposure_us"] != dark_frame["commanded_exposure_us"]:
            raise NegativeUniversalError(f"{channel} dark exposure does not match film")
        if film_frame["commanded_exposure_us"] != flat_frame["commanded_exposure_us"]:
            raise NegativeUniversalError(f"{channel} flat exposure does not match film")
        if film_frame["commanded_power"] != flat_frame["commanded_power"]:
            raise NegativeUniversalError(f"{channel} flat power does not match film")
        if dark_frame["commanded_power"] != 0.0:
            raise NegativeUniversalError(f"{channel} dark capture must use zero light power")


def _normalize_images(
    *,
    film: Path,
    dark: Path,
    flat: Path,
    film_images: dict[str, PgmImage],
    dark_images: dict[str, PgmImage],
    flat_images: dict[str, PgmImage],
    sensor_layout: str,
    output: Path,
) -> None:
    normalized_images: dict[str, PgmImage] = {}
    normalization_stats: dict[str, dict[str, int]] = {}
    total_invalid = 0
    for channel in CHANNELS:
        film_image = film_images[channel]
        dark_image = dark_images[channel]
        flat_image = flat_images[channel]
        normalized = array("H")
        clipped_low = 0
        clipped_high = 0
        invalid = 0
        for film_value, dark_value, flat_value in zip(
            film_image.pixels, dark_image.pixels, flat_image.pixels, strict=True
        ):
            denominator = flat_value - dark_value
            if denominator <= 0:
                invalid += 1
                normalized.append(0)
                continue
            transmittance = (film_value - dark_value) / denominator
            if transmittance < 0.0:
                clipped_low += 1
                transmittance = 0.0
            elif transmittance > 1.0:
                clipped_high += 1
                transmittance = 1.0
            normalized.append(int(transmittance * 65535.0 + 0.5))

        if invalid:
            total_invalid += invalid
        normalized_images[channel] = PgmImage(
            film_image.width,
            film_image.height,
            65535,
            normalized,
        )
        normalization_stats[channel] = {
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "invalid_denominator": invalid,
        }

    if total_invalid:
        raise NegativeUniversalError(
            f"flat must exceed dark at every pixel; found {total_invalid} invalid samples"
        )

    frame_records: dict[str, Any] = {}
    for channel in CHANNELS:
        output_path = output / f"{channel}.pgm"
        output_image = normalized_images[channel]
        write_pgm(output_path, output_image)
        frame_records[channel] = {
            **_frame_record(output_path, output_image),
            **normalization_stats[channel],
        }

    manifest = {
        "schema": 1,
        "kind": "normalized_transmittance",
        "created_at": datetime.now(UTC).isoformat(),
        "channel_order": list(CHANNELS),
        "sensor_layout": sensor_layout,
        "formula": "clamp((film-dark)/(flat-dark), 0, 1)",
        "output_max_value": 65535,
        "sources": {
            "film_manifest_sha256": sha256_file(film / MANIFEST_NAME),
            "dark_manifest_sha256": sha256_file(dark / MANIFEST_NAME),
            "flat_manifest_sha256": sha256_file(flat / MANIFEST_NAME),
        },
        "frames": frame_records,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_triplets(*, film: Path, dark: Path, flat: Path, output: Path) -> None:
    film_manifest, film_images = _load_capture(film, "film")
    dark_manifest, dark_images = _load_capture(dark, "dark")
    flat_manifest, flat_images = _load_capture(flat, "flat")
    _validate_capture_settings(film_manifest, dark_manifest, flat_manifest)

    reference = film_images["R"]
    for label, images in (("dark", dark_images), ("flat", flat_images)):
        image = images["R"]
        if (image.width, image.height, image.max_value) != (
            reference.width,
            reference.height,
            reference.max_value,
        ):
            raise NegativeUniversalError(f"{label} images do not match film images")

    with _staged_output(output) as staging:
        _normalize_images(
            film=film,
            dark=dark,
            flat=flat,
            film_images=film_images,
            dark_images=dark_images,
            flat_images=flat_images,
            sensor_layout=film_manifest["rig_metadata"]["sensor_layout"],
            output=staging,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="negative-universal")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture an R/G/B triplet")
    capture.add_argument("--kind", choices=("dark", "flat", "film"), required=True)
    capture.add_argument("--settings", type=Path, required=True)
    capture.add_argument("--rig", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_CAPTURE_TIMEOUT_SECONDS,
        help=f"maximum time for one exposure (default: {DEFAULT_CAPTURE_TIMEOUT_SECONDS:g})",
    )

    normalize = subparsers.add_parser(
        "normalize", help="apply matched dark and flat correction"
    )
    normalize.add_argument("--film", type=Path, required=True)
    normalize.add_argument("--dark", type=Path, required=True)
    normalize.add_argument("--flat", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            capture_triplet(
                kind=args.kind,
                settings_path=args.settings,
                rig=args.rig,
                output=args.output,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            normalize_triplets(
                film=args.film,
                dark=args.dark,
                flat=args.flat,
                output=args.output,
            )
    except (NegativeUniversalError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
