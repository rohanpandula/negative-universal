"""Capture and normalize sequential R/G/B raw sensor frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Sequence

from . import invert, probe


CHANNELS = ("R", "G", "B")
SENSOR_LAYOUTS = frozenset({"mono", "rggb", "bggr", "grbg", "gbrg"})
MANIFEST_NAME = "manifest.json"
CAPTURE_MANIFEST_SCHEMA = 2
CAPTURE_RECEIPT_SCHEMA = 1
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


def _roi_argument(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from error
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return parts


def _triplet_argument(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected R,G,B floats") from error
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B floats")
    return parts


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


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_settings(
    path: Path,
) -> tuple[dict[str, ChannelSettings], dict[str, Any], str]:
    try:
        settings_bytes = path.read_bytes()
        payload = json.loads(settings_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NegativeUniversalError(
            f"cannot read settings from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise NegativeUniversalError("settings must be a JSON object")
    allowed_keys = {*CHANNELS, "_meta"}
    if not set(CHANNELS).issubset(payload) or not set(payload).issubset(allowed_keys):
        raise NegativeUniversalError(
            "settings must contain R, G, B, and optional _meta"
        )
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


def _capture_receipt_source(
    *,
    directory: Path,
    channel: str,
    kind: str,
    image: PgmImage,
    expected_layout: str,
    commanded_power: float,
    commanded_exposure_us: int,
) -> dict[str, Any]:
    receipt_path = directory / f"{channel}.receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise NegativeUniversalError(f"missing regular capture receipt: {receipt_path}")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NegativeUniversalError(f"cannot read {receipt_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != CAPTURE_RECEIPT_SCHEMA:
        raise NegativeUniversalError(f"unsupported capture receipt: {receipt_path}")

    native_name = payload.get("native_file")
    if (
        not isinstance(native_name, str)
        or Path(native_name).name != native_name
        or Path(native_name).stem != channel
        or Path(native_name).suffix.upper() not in {".RAF", ".NEF"}
    ):
        raise NegativeUniversalError(f"invalid native file in {receipt_path}")
    native_path = directory / native_name
    if native_path.is_symlink() or not native_path.is_file():
        raise NegativeUniversalError(f"missing regular native RAW: {native_path}")

    decoder = payload.get("decoder")
    decoder_fields = {"name", "version", "libraw_version", "operation"}
    if not isinstance(decoder, dict) or not decoder_fields.issubset(decoder):
        raise NegativeUniversalError(f"invalid decoder metadata in {receipt_path}")
    if (
        decoder.get("name") != "rawpy"
        or decoder.get("operation") != "raw_image_visible"
    ):
        raise NegativeUniversalError(f"unsupported decoder operation in {receipt_path}")
    for field in ("version", "libraw_version"):
        if not isinstance(decoder.get(field), str) or not decoder[field]:
            raise NegativeUniversalError(f"invalid decoder {field} in {receipt_path}")

    raster = payload.get("raster")
    raster_fields = {
        "raw_width",
        "raw_height",
        "crop_left",
        "crop_top",
        "width",
        "height",
        "sensor_layout",
        "black_levels",
        "white_level",
        "camera_white_levels",
        "libraw_flip",
        "orientation_applied",
    }
    if not isinstance(raster, dict) or not raster_fields.issubset(raster):
        raise NegativeUniversalError(f"invalid raster metadata in {receipt_path}")
    for field in ("raw_width", "raw_height", "width", "height", "white_level"):
        value = raster.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NegativeUniversalError(f"invalid raster {field} in {receipt_path}")
    for field in ("crop_left", "crop_top", "libraw_flip"):
        value = raster.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NegativeUniversalError(f"invalid raster {field} in {receipt_path}")
    if raster["white_level"] > 65535:
        raise NegativeUniversalError(f"invalid raster white_level in {receipt_path}")
    if raster["crop_left"] + raster["width"] > raster["raw_width"] or (
        raster["crop_top"] + raster["height"] > raster["raw_height"]
    ):
        raise NegativeUniversalError(f"invalid visible crop in {receipt_path}")
    if raster.get("orientation_applied") is not False:
        raise NegativeUniversalError(
            f"RAW orientation must not be applied in {receipt_path}"
        )
    layout = _sensor_layout(raster.get("sensor_layout"), str(receipt_path))
    if layout != expected_layout:
        raise NegativeUniversalError(f"sensor layout mismatch in {receipt_path}")
    if (raster["width"], raster["height"], raster["white_level"]) != (
        image.width,
        image.height,
        image.max_value,
    ):
        raise NegativeUniversalError(f"PGM metadata mismatch in {receipt_path}")

    black_levels = raster.get("black_levels")
    if (
        not isinstance(black_levels, list)
        or len(black_levels) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= raster["white_level"]
            for value in black_levels
        )
    ):
        raise NegativeUniversalError(f"invalid black levels in {receipt_path}")
    camera_white = raster.get("camera_white_levels")
    if camera_white is not None and (
        not isinstance(camera_white, list)
        or len(camera_white) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 65535
            for value in camera_white
        )
    ):
        raise NegativeUniversalError(f"invalid camera white levels in {receipt_path}")

    capture = payload.get("capture")
    if not isinstance(capture, dict):
        raise NegativeUniversalError(f"missing capture metadata in {receipt_path}")
    expected_capture = {
        "kind": kind,
        "channel": channel,
        "commanded_power": commanded_power,
        "commanded_exposure_us": commanded_exposure_us,
    }
    for field, expected in expected_capture.items():
        if capture.get(field) != expected:
            raise NegativeUniversalError(f"capture {field} mismatch in {receipt_path}")
    camera_port = capture.get("camera_port")
    if not isinstance(camera_port, str) or not camera_port:
        raise NegativeUniversalError(f"invalid camera port in {receipt_path}")
    applied_exposure = capture.get("applied_exposure_us")
    if (
        isinstance(applied_exposure, bool)
        or not isinstance(applied_exposure, int)
        or applied_exposure <= 0
    ):
        raise NegativeUniversalError(f"invalid applied exposure in {receipt_path}")
    gphoto_version = capture.get("gphoto2_version")
    if not isinstance(gphoto_version, str) or not gphoto_version:
        raise NegativeUniversalError(f"invalid gphoto2 version in {receipt_path}")

    camera = payload.get("camera")
    camera_fields = {
        "model",
        "serial_number",
        "iso",
        "image_format",
        "image_format_config",
        "exposure_mode",
        "capture_target",
        "gphoto_driver",
    }
    if not isinstance(camera, dict) or not camera_fields.issubset(camera):
        raise NegativeUniversalError(f"invalid camera metadata in {receipt_path}")
    for field in (
        "model",
        "iso",
        "image_format",
        "image_format_config",
        "exposure_mode",
    ):
        if not isinstance(camera[field], str) or not camera[field]:
            raise NegativeUniversalError(f"invalid camera {field} in {receipt_path}")
    for field in ("serial_number", "capture_target", "gphoto_driver"):
        value = camera[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise NegativeUniversalError(f"invalid camera {field} in {receipt_path}")
    capture_target = camera["capture_target"]
    if capture_target is not None and capture_target.casefold() not in {
        "memory card",
        "card",
    }:
        raise NegativeUniversalError(
            f"camera capture target is not Memory card in {receipt_path}"
        )
    iso_label = camera["iso"].strip()
    if iso_label.casefold().startswith("iso"):
        iso_label = iso_label[3:].strip()
    try:
        fixed_iso = float(iso_label)
    except ValueError as error:
        raise NegativeUniversalError(
            f"camera ISO is not fixed in {receipt_path}"
        ) from error
    if not math.isfinite(fixed_iso) or fixed_iso <= 0.0:
        raise NegativeUniversalError(f"camera ISO is not fixed in {receipt_path}")
    format_name = camera["image_format"].upper()
    if (
        not any(token in format_name for token in ("RAW", "RAF", "NEF"))
        or "+" in format_name
        or "JPEG" in format_name
        or "JPG" in format_name
    ):
        raise NegativeUniversalError(f"camera format is not RAW-only in {receipt_path}")
    if camera["exposure_mode"].casefold() not in {
        "m",
        "manual",
        "manual exposure",
    }:
        raise NegativeUniversalError(f"camera exposure is not manual in {receipt_path}")

    scanlight = payload.get("scanlight")
    if not isinstance(scanlight, dict):
        raise NegativeUniversalError(f"missing ScanLight metadata in {receipt_path}")
    for field in ("port", "hardware_id", "firmware_id", "pwm_code"):
        if field not in scanlight:
            raise NegativeUniversalError(f"missing ScanLight {field} in {receipt_path}")
    if not isinstance(scanlight["port"], str) or not scanlight["port"]:
        raise NegativeUniversalError(f"invalid ScanLight port in {receipt_path}")
    for field in ("hardware_id", "firmware_id", "pwm_code"):
        value = scanlight[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NegativeUniversalError(f"invalid ScanLight {field} in {receipt_path}")
    if scanlight["pwm_code"] > 255:
        raise NegativeUniversalError(f"invalid ScanLight pwm_code in {receipt_path}")
    settle_ms = scanlight.get("settle_ms")
    if (
        isinstance(settle_ms, bool)
        or not isinstance(settle_ms, (int, float))
        or not math.isfinite(settle_ms)
        or not 0.0 <= settle_ms <= 60_000.0
    ):
        raise NegativeUniversalError(f"invalid ScanLight settle_ms in {receipt_path}")
    expected_pwm = int(commanded_power * 255.0 + 0.5)
    if scanlight["pwm_code"] != expected_pwm:
        raise NegativeUniversalError(f"ScanLight pwm_code mismatch in {receipt_path}")

    decode_configuration = {
        "schema": payload["schema"],
        "decoder": decoder,
        "raster": raster,
    }
    device_configuration = {
        **decode_configuration,
        "camera_port": camera_port,
        "gphoto2_version": gphoto_version,
        "camera": camera,
        "scanlight": {
            field: scanlight[field]
            for field in ("port", "hardware_id", "firmware_id", "settle_ms")
        },
    }
    return {
        "file": native_name,
        "bytes": native_path.stat().st_size,
        "sha256": sha256_file(native_path),
        "receipt_file": receipt_path.name,
        "receipt_sha256": sha256_file(receipt_path),
        "decode_configuration_sha256": _sha256_json(decode_configuration),
        "device_configuration_sha256": _sha256_json(device_configuration),
        "applied_exposure_us": applied_exposure,
        "pwm_code": scanlight["pwm_code"],
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


def _run_rig(command: list[str], *, timeout_seconds: float, action: str) -> None:
    try:
        process = subprocess.Popen(command, start_new_session=os.name == "posix")
    except OSError as error:
        raise NegativeUniversalError(f"rig failed while {action}: {error}") from error
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _stop_rig_process(process)
        raise NegativeUniversalError(
            f"rig timed out while {action} after {timeout_seconds:g} seconds"
        ) from error
    except BaseException:
        _stop_rig_process(process)
        raise
    if return_code:
        error = subprocess.CalledProcessError(return_code, command)
        raise NegativeUniversalError(f"rig failed while {action}: {error}") from error


def _stop_rig_process(process: subprocess.Popen[Any]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised on Windows only
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows only
            process.kill()
    except ProcessLookupError:
        return
    process.wait()


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
            receipt_path = (staging / f"{channel}.receipt.json").resolve()
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
            if rig_metadata["sensor_layout"] != "mono":
                command.extend(("--receipt", str(receipt_path)))
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
            native_source = None
            if receipt_path.exists():
                native_source = _capture_receipt_source(
                    directory=staging,
                    channel=channel,
                    kind=kind,
                    image=image,
                    expected_layout=rig_metadata["sensor_layout"],
                    commanded_power=commanded_power,
                    commanded_exposure_us=channel_settings.exposure_us,
                )
            elif rig_metadata["sensor_layout"] != "mono":
                raise NegativeUniversalError(
                    f"rig did not create native RAW receipt {receipt_path}"
                )
            frame_records[channel] = {
                **_frame_record(frame_path, image),
                "commanded_power": commanded_power,
                "commanded_exposure_us": channel_settings.exposure_us,
                "native_source": native_source,
            }

        _validate_matching_images(images, kind)
        native_sources = [
            frame_records[channel]["native_source"] for channel in CHANNELS
        ]
        if any(source is not None for source in native_sources):
            if any(source is None for source in native_sources):
                raise NegativeUniversalError(
                    "R/G/B must either all retain native RAW sources or none may"
                )
            decode_configurations = {
                source["decode_configuration_sha256"] for source in native_sources
            }
            device_configurations = {
                source["device_configuration_sha256"] for source in native_sources
            }
            if len(decode_configurations) != 1 or len(device_configurations) != 1:
                raise NegativeUniversalError(
                    "R/G/B native RAW decode or device configurations do not match"
                )
        manifest = {
            "schema": CAPTURE_MANIFEST_SCHEMA,
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
    if not isinstance(payload, dict) or payload.get("schema") not in {
        1,
        CAPTURE_MANIFEST_SCHEMA,
    }:
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


def _load_capture(
    directory: Path, kind: str
) -> tuple[dict[str, Any], dict[str, PgmImage]]:
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
        if manifest["schema"] == CAPTURE_MANIFEST_SCHEMA:
            required_fields.add("native_source")
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
            raise NegativeUniversalError(
                f"cannot read {frame_path}: {error}"
            ) from error
        if (
            record.get("width"),
            record.get("height"),
            record.get("max_value"),
        ) != (image.width, image.height, image.max_value):
            raise NegativeUniversalError(f"metadata mismatch for {frame_path}")
        if manifest["schema"] == CAPTURE_MANIFEST_SCHEMA:
            if record["native_source"] is None:
                if manifest["rig_metadata"]["sensor_layout"] != "mono":
                    raise NegativeUniversalError(
                        f"Bayer {channel} record must retain its native RAW"
                    )
            else:
                native_source = _capture_receipt_source(
                    directory=directory,
                    channel=channel,
                    kind=kind,
                    image=image,
                    expected_layout=manifest["rig_metadata"]["sensor_layout"],
                    commanded_power=power,
                    commanded_exposure_us=record["commanded_exposure_us"],
                )
                if record["native_source"] != native_source:
                    raise NegativeUniversalError(
                        f"native source metadata mismatch for {frame_path}"
                    )
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
        raise NegativeUniversalError("film, dark, and flat sensor layouts do not match")
    rig_paths = {film.get("rig"), dark.get("rig"), flat.get("rig")}
    if None in rig_paths or len(rig_paths) != 1:
        raise NegativeUniversalError("film, dark, and flat rig adapters do not match")
    all_native_sources = [
        manifest["frames"][channel].get("native_source")
        for manifest in (film, dark, flat)
        for channel in CHANNELS
    ]
    if any(source is not None for source in all_native_sources):
        if any(source is None for source in all_native_sources):
            raise NegativeUniversalError(
                "native RAW captures cannot be mixed with captures lacking provenance"
            )
        for field in (
            "decode_configuration_sha256",
            "device_configuration_sha256",
        ):
            configurations = {source[field] for source in all_native_sources}
            if len(configurations) != 1:
                label = field.removesuffix("_sha256").replace("_", " ")
                raise NegativeUniversalError(
                    f"film, dark, and flat native RAW {label} values do not match"
                )
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
            raise NegativeUniversalError(
                f"{channel} dark capture must use zero light power"
            )
        native_sources = (
            film_frame.get("native_source"),
            dark_frame.get("native_source"),
            flat_frame.get("native_source"),
        )
        if all(source is not None for source in native_sources):
            film_source, dark_source, flat_source = native_sources
            if film_source["applied_exposure_us"] != dark_source["applied_exposure_us"]:
                raise NegativeUniversalError(
                    f"{channel} applied dark exposure does not match film"
                )
            if film_source["applied_exposure_us"] != flat_source["applied_exposure_us"]:
                raise NegativeUniversalError(
                    f"{channel} applied flat exposure does not match film"
                )
            if film_source["pwm_code"] != flat_source["pwm_code"]:
                raise NegativeUniversalError(
                    f"{channel} applied flat PWM code does not match film"
                )
            if dark_source["pwm_code"] != 0:
                raise NegativeUniversalError(
                    f"{channel} dark capture must use zero ScanLight PWM"
                )


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


def _load_normalized(directory: Path) -> tuple[dict[str, Any], dict[str, PgmImage]]:
    """Load a `normalized_transmittance` capture, verifying hashes and geometry."""
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise NegativeUniversalError(f"no {MANIFEST_NAME} in {directory}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NegativeUniversalError(f"unreadable manifest in {directory}") from error
    if payload.get("kind") != "normalized_transmittance":
        raise NegativeUniversalError(
            f"{directory} holds {payload.get('kind')!r}, not 'normalized_transmittance'"
        )
    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise NegativeUniversalError("normalized manifest has no frames")
    images: dict[str, PgmImage] = {}
    for channel in CHANNELS:
        record = frames.get(channel)
        if not isinstance(record, dict):
            raise NegativeUniversalError(f"normalized manifest is missing {channel}")
        path = directory / str(record.get("file", ""))
        if not path.is_file():
            raise NegativeUniversalError(f"missing {channel} frame: {path.name}")
        if sha256_file(path) != record.get("sha256"):
            raise NegativeUniversalError(f"{channel} frame hash does not match")
        image = read_pgm(path)
        if (image.width, image.height) != (record.get("width"), record.get("height")):
            raise NegativeUniversalError(f"{channel} frame size disagrees with manifest")
        if image.max_value != 65535:
            raise NegativeUniversalError(
                f"{channel} frame is not 16-bit transmittance"
            )
        images[channel] = image
    reference = images["R"]
    for channel in CHANNELS[1:]:
        if (images[channel].width, images[channel].height) != (
            reference.width,
            reference.height,
        ):
            raise NegativeUniversalError("triplet frames differ in size")
    return payload, images


def _patch_record(patch: probe.Patch) -> dict[str, Any]:
    return {
        "x": patch.x,
        "y": patch.y,
        "width": patch.width,
        "height": patch.height,
        "transmittance": [round(value, 6) for value in patch.medians],
    }


def invert_triplets(
    *,
    normalized: Path,
    output: Path,
    clear_base: Path | None = None,
    base_roi: tuple[int, int, int, int] | None = None,
    density_offset: float = 0.0,
    gray: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contrast: float = 0.5,
    toe: float = 0.06,
    white_density: float = 0.05,
    black_density: float = 1.7,
    neutral_pull: float = 0.0,
    auto_range: bool = False,
) -> None:
    """Invert a normalized triplet through measured density into a positive."""
    if clear_base is not None and base_roi is not None:
        raise NegativeUniversalError("pass either --clear-base or --base-roi, not both")
    if not -2.0 <= density_offset <= 2.0:
        raise NegativeUniversalError("--density-offset must be in -2..2 decades")
    if not 0.0 <= contrast <= 1.0:
        raise NegativeUniversalError("--contrast must be in 0..1")
    if clear_base is None and base_roi is None and not normalized.is_dir():
        raise NegativeUniversalError(f"not a directory: {normalized}")

    manifest, images = _load_normalized(normalized)
    base_manifest: dict[str, Any] | None = None
    try:
        if clear_base is not None:
            base_manifest, base_images = _load_normalized(clear_base)
            if base_images["R"].width != images["R"].width or (
                base_images["R"].height != images["R"].height
            ):
                raise NegativeUniversalError(
                    "clear-base capture must match the frame geometry"
                )
            estimate = probe.probe_base(base_images, band_fraction=0.05)
        elif base_roi is not None:
            estimate = probe.probe_base(images, roi=base_roi)
        else:
            estimate = probe.probe_base(images)

        if auto_range:
            resolved = invert.sample_densities(images, estimate)
            white_density, black_density = invert.auto_range(resolved)
            print(
                "auto range: "
                f"white {white_density:.3f}, black {black_density:.3f} decades"
            )

        params = invert.RenderParams(
            look=invert.LookParams(
                contrast=contrast,
                toe=toe,
                white_density=white_density,
                black_density=black_density,
            ),
            density_offset=density_offset,
            gray=gray,
            neutral_pull=neutral_pull,
        )
        rendered, stats = invert.render(images, estimate, params)
    except (probe.ProbeError, invert.InvertError) as error:
        raise NegativeUniversalError(str(error)) from error

    base_luma = sum(
        value * weight for value, weight in zip(estimate.base, invert.LUMA, strict=True)
    )
    created_at = datetime.now(UTC).isoformat()
    with _staged_output(output) as staging:
        frame_records: dict[str, Any] = {}
        for channel in CHANNELS:
            path = staging / f"positive_{channel}.pgm"
            write_pgm(
                path,
                PgmImage(
                    images[channel].width,
                    images[channel].height,
                    65535,
                    rendered[channel],
                ),
            )
            frame_records[channel] = {
                "file": path.name,
                "width": images[channel].width,
                "height": images[channel].height,
                "max_value": 65535,
                "sha256": sha256_file(path),
            }
        invert.write_ppm(
            staging / "positive.ppm",
            rendered,
            images["R"].width,
            images["R"].height,
        )
        payload: dict[str, Any] = {
            "schema": 1,
            "kind": "positive_render",
            "created_at": created_at,
            "channel_order": list(CHANNELS),
            "sensor_layout": manifest.get("rig_metadata", {}).get("sensor_layout"),
            "width": images["R"].width,
            "height": images["R"].height,
            "output_max_value": 65535,
            "output_encoding": "srgb",
            "formula": {
                "domain": "optical density, -log10(transmittance)",
                "density": "d = -log10(clamp(T / base_plane(x, y), 1e-4, 1))",
                "correction": "d + gray[channel] + density_offset",
                "look": (
                    "monotone LUT over 0..4 decades, "
                    f"white={white_density}, black={black_density}, "
                    f"contrast={contrast}, toe={toe}"
                ),
            },
            "base": {
                "source": estimate.source,
                "observed_patches": len(estimate.patches),
                "transmittance": [round(value, 6) for value in estimate.base],
                "density": [round(-math.log10(min(value, 1.0)), 6) for value in estimate.base],
                "luma": round(base_luma, 6),
                "plane": [
                    [round(coefficient, 6) for coefficient in axis]
                    for axis in estimate.plane
                ],
                "chosen": [_patch_record(patch) for patch in estimate.chosen],
            },
            "correction": {
                "density_offset": density_offset,
                "gray": list(gray),
                "neutral_pull": neutral_pull,
            },
            "look": {
                "contrast": contrast,
                "toe": toe,
                "white_density": round(white_density, 4),
                "black_density": round(black_density, 4),
                "auto_range": auto_range,
            },
            "stats": {
                "pixels": stats.pixels,
                "clipped_transmittance": stats.clipped_transmittance,
                "density_min": round(stats.density_floor, 4),
                "density_max": round(stats.density_ceiling, 4),
            },
            "sources": {
                "normalized_dir": str(normalized),
                "normalized_manifest_sha256": sha256_file(normalized / MANIFEST_NAME),
            },
            "frames": frame_records,
        }
        if clear_base is not None and base_manifest is not None:
            payload["sources"]["clear_base_dir"] = str(clear_base)
            payload["sources"]["clear_base_manifest_sha256"] = sha256_file(
                clear_base / MANIFEST_NAME
            )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(f"rendered {stats.pixels} px, base {estimate.base[0]:.4f}/"
          f"{estimate.base[1]:.4f}/{estimate.base[2]:.4f} from {estimate.source}")


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

    decode = subparsers.add_parser(
        "decode-raw", help="preserve one native RAW as an undemosaiced PGM"
    )
    decode.add_argument("--input", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--receipt", type=Path, required=True)

    calibrate = subparsers.add_parser(
        "calibrate-clear-base",
        help="recommend exposure from matched clear-film, dark, and flat captures",
    )
    calibrate.add_argument("--clear-base", type=Path, required=True)
    calibrate.add_argument("--dark", type=Path, required=True)
    calibrate.add_argument("--flat", type=Path, required=True)
    calibrate.add_argument("--roi", type=_roi_argument, required=True)
    calibrate.add_argument("--target-base-fraction", type=float, required=True)
    calibrate.add_argument("--flat-ceiling-fraction", type=float, required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    invert_parser = subparsers.add_parser(
        "invert",
        help="invert a normalized triplet through measured density",
    )
    invert_parser.add_argument("--normalized", type=Path, required=True)
    invert_parser.add_argument(
        "--clear-base",
        type=Path,
        help="normalized clear-film capture to measure the base from",
    )
    invert_parser.add_argument(
        "--base-roi",
        type=_roi_argument,
        help="explicit base patch for frames without a border",
    )
    invert_parser.add_argument("--density-offset", type=float, default=0.0)
    invert_parser.add_argument("--gray", type=_triplet_argument, default=(0.0, 0.0, 0.0))
    invert_parser.add_argument("--contrast", type=float, default=0.5)
    invert_parser.add_argument("--toe", type=float, default=0.06)
    invert_parser.add_argument("--white-density", type=float, default=0.05)
    invert_parser.add_argument("--black-density", type=float, default=1.7)
    invert_parser.add_argument(
        "--auto-range",
        action="store_true",
        help="fit white/black to this frame's density percentiles",
    )
    invert_parser.add_argument("--neutral-pull", type=float, default=0.0)
    invert_parser.add_argument("--output", type=Path, required=True)
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
        elif args.command == "normalize":
            normalize_triplets(
                film=args.film,
                dark=args.dark,
                flat=args.flat,
                output=args.output,
            )
        elif args.command == "invert":
            invert_triplets(
                normalized=args.normalized,
                output=args.output,
                clear_base=args.clear_base,
                base_roi=args.base_roi,
                density_offset=args.density_offset,
                gray=args.gray,
                contrast=args.contrast,
                toe=args.toe,
                white_density=args.white_density,
                black_density=args.black_density,
                neutral_pull=args.neutral_pull,
                auto_range=args.auto_range,
            )
        elif args.command == "decode-raw":
            from .raw_decode import RawDecodeError, decode_raw

            try:
                decode_raw(args.input, args.output, args.receipt)
            except RawDecodeError as error:
                raise NegativeUniversalError(str(error)) from error
        else:
            from .calibration import calibrate_clear_base

            calibrate_clear_base(
                args.clear_base,
                args.dark,
                args.flat,
                args.roi,
                args.target_base_fraction,
                args.flat_ceiling_fraction,
                args.output,
            )
    except (NegativeUniversalError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
