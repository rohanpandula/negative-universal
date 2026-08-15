"""First-party ScanLight and gPhoto adapter for one RAW exposure."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping, Sequence


BAUD_RATE = 115200
OFF_PACKET = bytes.fromhex("fe0006000000000000")
PROBE_PACKET = bytes.fromhex("fe0200")
SHUTTER_CONFIG = "/main/capturesettings/shutterspeed"
CAMERA_MODEL_CONFIG = "/main/status/cameramodel"
CAMERA_SERIAL_CONFIG = "/main/status/serialnumber"
ISO_CONFIG = "/main/imgsettings/iso"
EXPOSURE_PROGRAM_CONFIG = "/main/capturesettings/expprogram"
CAPTURE_TARGET_CONFIG = "/main/settings/capturetarget"
IMAGE_FORMAT_CONFIGS = (
    "/main/imgsettings/imageformat",
    "/main/capturesettings/imagequality",
)
KNOWN_SCANLIGHT_V4_HARDWARE_IDS = frozenset({1, 3})
MINIMUM_FIRMWARE_ID = 2
SHUTTER_TOLERANCE = 0.02


class CaptureOneError(RuntimeError):
    """The adapter could not safely complete one exposure."""


@dataclass(frozen=True, slots=True)
class _Config:
    scanlight_port: str
    gphoto_port: str | None
    gphoto_camera: str | None
    gphoto2: str
    settle_seconds: float
    expected_hardware_id: int | None


def _value(
    environment: Mapping[str, str], name: str, *, required: bool = False
) -> str | None:
    value = environment.get(name, "").strip()
    if required and not value:
        raise CaptureOneError(f"{name} is required")
    return value or None


def _config(environment: Mapping[str, str], *, camera: bool) -> _Config:
    scanlight_port = _value(
        environment, "NEGATIVE_UNIVERSAL_SCANLIGHT_PORT", required=True
    )
    gphoto_port = _value(environment, "NEGATIVE_UNIVERSAL_GPHOTO_PORT", required=camera)
    gphoto_camera = _value(environment, "NEGATIVE_UNIVERSAL_GPHOTO_CAMERA")
    gphoto2 = _value(environment, "NEGATIVE_UNIVERSAL_GPHOTO2") or "gphoto2"
    settle_value = _value(environment, "NEGATIVE_UNIVERSAL_SETTLE_MS", required=camera)
    try:
        settle_ms = float(settle_value or "0")
    except ValueError as error:
        raise CaptureOneError("NEGATIVE_UNIVERSAL_SETTLE_MS must be numeric") from error
    if not math.isfinite(settle_ms) or not 0.0 <= settle_ms <= 60_000.0:
        raise CaptureOneError("NEGATIVE_UNIVERSAL_SETTLE_MS must be in 0..60000")

    hardware = _value(
        environment, "NEGATIVE_UNIVERSAL_SCANLIGHT_HW_ID", required=camera
    )
    try:
        expected_hardware_id = None if hardware is None else int(hardware, 0)
    except ValueError as error:
        raise CaptureOneError(
            "NEGATIVE_UNIVERSAL_SCANLIGHT_HW_ID must be an integer"
        ) from error
    if expected_hardware_id is not None and not 0 <= expected_hardware_id <= 65535:
        raise CaptureOneError("NEGATIVE_UNIVERSAL_SCANLIGHT_HW_ID is out of range")
    return _Config(
        scanlight_port=scanlight_port or "",
        gphoto_port=gphoto_port,
        gphoto_camera=gphoto_camera,
        gphoto2=gphoto2,
        settle_seconds=settle_ms / 1000.0,
        expected_hardware_id=expected_hardware_id,
    )


def _open_scanlight(port: str) -> Any:
    try:
        import serial
    except ImportError as error:
        raise CaptureOneError("ScanLight control requires pyserial") from error
    try:
        return serial.Serial(
            port,
            baudrate=BAUD_RATE,
            timeout=0.1,
            write_timeout=1.0,
        )
    except Exception as error:
        raise CaptureOneError(f"cannot open ScanLight port {port}: {error}") from error


def _write(device: Any, packet: bytes) -> None:
    written = device.write(packet)
    if written is not None and written != len(packet):
        raise CaptureOneError(
            f"short ScanLight write: wrote {written} of {len(packet)} bytes"
        )
    device.flush()


def _off(device: Any) -> None:
    _write(device, OFF_PACKET)
    time.sleep(0.05)


@contextmanager
def _safe_scanlight(port: str) -> Iterator[Any]:
    device = _open_scanlight(port)
    primary_error: BaseException | None = None
    try:
        _off(device)
        yield device
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _off(device)
        except BaseException as off_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"ScanLight OFF cleanup also failed: {off_error}")
        finally:
            try:
                device.close()
            except BaseException as close_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"ScanLight close also failed: {close_error}")


def _probe(device: Any, timeout_seconds: float = 2.0) -> tuple[int, int]:
    device.reset_input_buffer()
    _write(device, PROBE_PACKET)
    deadline = time.monotonic() + timeout_seconds
    buffer = bytearray()
    while time.monotonic() < deadline:
        chunk = device.read(256)
        if chunk:
            buffer.extend(chunk)
        while buffer:
            if buffer[0] != 0xFE:
                del buffer[: max(1, buffer.find(b"\xfe"))]
                continue
            if len(buffer) < 3:
                break
            length = buffer[2]
            if length > 128:
                del buffer[0]
                continue
            packet_length = 3 + length
            if len(buffer) < packet_length:
                break
            header = buffer[1]
            payload = bytes(buffer[3:packet_length])
            del buffer[:packet_length]
            if header != 3:
                continue
            if len(payload) != 4:
                raise CaptureOneError("malformed ScanLight firmware response")
            word = int.from_bytes(payload, "big")
            return word & 0xFFFF, word >> 16
    raise CaptureOneError("ScanLight did not answer the firmware probe")


def _validated_probe(device: Any, expected_hardware_id: int | None) -> tuple[int, int]:
    firmware_id, hardware_id = _probe(device)
    if firmware_id < MINIMUM_FIRMWARE_ID:
        raise CaptureOneError(
            f"ScanLight firmware ID {firmware_id} is older than required ID "
            f"{MINIMUM_FIRMWARE_ID}"
        )
    if hardware_id not in KNOWN_SCANLIGHT_V4_HARDWARE_IDS:
        raise CaptureOneError(f"unsupported ScanLight v4 hardware ID {hardware_id}")
    if expected_hardware_id is not None and hardware_id != expected_hardware_id:
        raise CaptureOneError(
            f"ScanLight hardware ID {hardware_id} does not match expected ID "
            f"{expected_hardware_id}"
        )
    return firmware_id, hardware_id


def _color_packet(channel: str, power: float) -> tuple[bytes, int]:
    if channel not in {"R", "G", "B"}:
        raise CaptureOneError(f"unsupported channel: {channel}")
    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise CaptureOneError("power must be finite and in 0..1")
    pwm = int(math.floor(power * 255.0 + 0.5))
    rgb = [0, 0, 0]
    rgb[{"R": 0, "G": 1, "B": 2}[channel]] = pwm
    return bytes([0xFE, 0x00, 0x06, *rgb, 0, 0, 0]), pwm


def _run_gphoto(
    command: list[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise CaptureOneError(
            f"gphoto2 timed out after {timeout_seconds:g} seconds"
        ) from error
    except FileNotFoundError as error:
        raise CaptureOneError(f"gphoto2 executable not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail[-1000:]}" if detail else ""
        raise CaptureOneError(
            f"gphoto2 failed with exit code {error.returncode}{suffix}"
        ) from error


def _camera_command(config: _Config) -> list[str]:
    if config.gphoto_port is None:
        raise CaptureOneError("NEGATIVE_UNIVERSAL_GPHOTO_PORT is required")
    command = [config.gphoto2, "--port", config.gphoto_port]
    if config.gphoto_camera is not None:
        command.extend(("--camera", config.gphoto_camera))
    return command


def _duration_us(label: str) -> int:
    value = label.strip().lower()
    multiplier = 1.0
    if value.endswith("m"):
        multiplier = 60.0
        value = value[:-1]
    elif value.endswith("s"):
        value = value[:-1]
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            seconds = float(numerator) / float(denominator)
        else:
            seconds = float(value)
    except (ValueError, ZeroDivisionError) as error:
        raise CaptureOneError(f"unsupported shutter value: {label!r}") from error
    seconds *= multiplier
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise CaptureOneError(f"invalid shutter value: {label!r}")
    microseconds = int(math.floor(seconds * 1_000_000.0 + 0.5))
    if microseconds <= 0:
        raise CaptureOneError(f"invalid shutter value: {label!r}")
    return microseconds


def _shutter_config(text: str) -> tuple[bool, str, list[tuple[int, str]]]:
    readonly_match = re.search(r"^Readonly:\s*([01])\s*$", text, re.MULTILINE)
    current_match = re.search(r"^Current:\s*(.+?)\s*$", text, re.MULTILINE)
    if readonly_match is None or current_match is None:
        raise CaptureOneError("cannot parse gphoto2 shutter configuration")
    choices = [
        (int(match.group(1)), match.group(2).strip())
        for match in re.finditer(r"^Choice:\s*(\d+)\s+(.+?)\s*$", text, re.MULTILINE)
    ]
    return readonly_match.group(1) == "1", current_match.group(1).strip(), choices


def _matches_exposure(actual_us: int, requested_us: int) -> bool:
    return abs(actual_us - requested_us) <= max(1.0, requested_us * SHUTTER_TOLERANCE)


def _get_shutter(config: _Config) -> tuple[bool, str, list[tuple[int, str]]]:
    result = _run_gphoto(
        [*_camera_command(config), "--get-config", SHUTTER_CONFIG],
        timeout_seconds=30.0,
    )
    return _shutter_config(result.stdout)


def _current_config(config: _Config, path: str) -> str:
    result = _run_gphoto(
        [*_camera_command(config), "--get-config", path], timeout_seconds=30.0
    )
    match = re.search(r"^Current:\s*(.*?)\s*$", result.stdout, re.MULTILINE)
    if match is None:
        raise CaptureOneError(f"cannot parse gphoto2 camera configuration {path}")
    return match.group(1).strip()


def _memory_card_target(config: _Config) -> str:
    result = _run_gphoto(
        [*_camera_command(config), "--get-config", CAPTURE_TARGET_CONFIG],
        timeout_seconds=30.0,
    )
    readonly, current, choices = _shutter_config(result.stdout)
    memory_card = [
        (index, label)
        for index, label in choices
        if label.casefold() in {"memory card", "card"}
    ]
    if not memory_card:
        raise CaptureOneError("camera exposes no Memory card capture target")
    if readonly:
        if current.casefold() not in {"memory card", "card"}:
            raise CaptureOneError(
                f"camera capture target is read-only and not Memory card: {current!r}"
            )
        return current
    index, _ = memory_card[0]
    _run_gphoto(
        [
            *_camera_command(config),
            "--set-config-index",
            f"{CAPTURE_TARGET_CONFIG}={index}",
        ],
        timeout_seconds=30.0,
    )
    result = _run_gphoto(
        [*_camera_command(config), "--get-config", CAPTURE_TARGET_CONFIG],
        timeout_seconds=30.0,
    )
    _, current, _ = _shutter_config(result.stdout)
    if current.casefold() not in {"memory card", "card"}:
        raise CaptureOneError(
            f"camera did not apply Memory card capture target, got {current!r}"
        )
    return current


def _camera_state(config: _Config) -> dict[str, str | None]:
    result = _run_gphoto(
        [*_camera_command(config), "--list-config"], timeout_seconds=30.0
    )
    available = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("/")
    }
    required = (CAMERA_MODEL_CONFIG, ISO_CONFIG, EXPOSURE_PROGRAM_CONFIG)
    missing = [path for path in required if path not in available]
    image_format_config = next(
        (path for path in IMAGE_FORMAT_CONFIGS if path in available), None
    )
    if image_format_config is None:
        missing.append("RAW image format")
    if missing:
        raise CaptureOneError(
            f"camera is missing required gphoto2 configuration: {', '.join(missing)}"
        )

    model = _current_config(config, CAMERA_MODEL_CONFIG)
    serial_number = (
        _current_config(config, CAMERA_SERIAL_CONFIG) or None
        if CAMERA_SERIAL_CONFIG in available
        else None
    )
    iso = _current_config(config, ISO_CONFIG)
    image_format = _current_config(config, image_format_config)
    exposure_mode = _current_config(config, EXPOSURE_PROGRAM_CONFIG)
    capture_target = (
        _memory_card_target(config) if CAPTURE_TARGET_CONFIG in available else None
    )
    if not model:
        raise CaptureOneError("camera model is empty")
    try:
        fixed_iso = float(re.sub(r"^iso\s*", "", iso, flags=re.IGNORECASE))
    except ValueError as error:
        raise CaptureOneError(f"camera ISO must be fixed, got {iso!r}") from error
    if not math.isfinite(fixed_iso) or fixed_iso <= 0.0:
        raise CaptureOneError(f"camera ISO must be fixed, got {iso!r}")
    format_name = image_format.upper()
    if (
        not any(token in format_name for token in ("RAW", "RAF", "NEF"))
        or "+" in format_name
        or "JPEG" in format_name
        or "JPG" in format_name
    ):
        raise CaptureOneError(
            f"camera image format must be RAW-only, got {image_format!r}"
        )
    if exposure_mode.casefold() not in {"m", "manual", "manual exposure"}:
        raise CaptureOneError(
            f"camera exposure mode must be manual, got {exposure_mode!r}"
        )
    return {
        "model": model,
        "serial_number": serial_number,
        "iso": iso,
        "image_format": image_format,
        "image_format_config": image_format_config,
        "exposure_mode": exposure_mode,
        "capture_target": capture_target,
        "gphoto_driver": config.gphoto_camera,
    }


def _configure_shutter(config: _Config, requested_us: int) -> int:
    readonly, current, choices = _get_shutter(config)
    if not readonly:
        usable: list[tuple[int, int, str]] = []
        for index, label in choices:
            try:
                usable.append((abs(_duration_us(label) - requested_us), index, label))
            except CaptureOneError:
                continue
        if not usable:
            raise CaptureOneError("gphoto2 reported no usable shutter choices")
        _, index, label = min(usable)
        selected_us = _duration_us(label)
        if not _matches_exposure(selected_us, requested_us):
            raise CaptureOneError(
                f"requested {requested_us} us is not a supported camera shutter speed"
            )
        _run_gphoto(
            [
                *_camera_command(config),
                "--set-config-index",
                f"{SHUTTER_CONFIG}={index}",
            ],
            timeout_seconds=30.0,
        )
        _, current, _ = _get_shutter(config)

    applied_us = _duration_us(current)
    if not _matches_exposure(applied_us, requested_us):
        mode = "read-only " if readonly else ""
        raise CaptureOneError(
            f"camera {mode}shutter is {current} ({applied_us} us), not requested "
            f"{requested_us} us"
        )
    return applied_us


def _gphoto_version(config: _Config) -> str:
    result = _run_gphoto([config.gphoto2, "--version"], timeout_seconds=10.0)
    match = re.search(r"^gphoto2\s+([^\s]+)", result.stdout, re.MULTILINE)
    if match is None:
        raise CaptureOneError("cannot parse gphoto2 version")
    return match.group(1)


def _download_raw(config: _Config, directory: Path, exposure_us: int) -> Path:
    filename = directory / "capture.%C"
    _run_gphoto(
        [
            *_camera_command(config),
            "--filename",
            str(filename),
            "--force-overwrite",
            "--keep",
            "--capture-image-and-download",
        ],
        timeout_seconds=max(60.0, exposure_us / 1_000_000.0 + 60.0),
    )
    files = [path for path in directory.iterdir() if path.is_file()]
    if len(files) != 1 or files[0].suffix.lower() not in {".raf", ".nef"}:
        names = ", ".join(sorted(path.name for path in files)) or "none"
        raise CaptureOneError(
            f"camera must download exactly one RAW-only RAF or NEF; got {names}"
        )
    return files[0]


def _augment_receipt(
    path: Path,
    *,
    capture: dict[str, Any],
    camera: dict[str, Any],
    scanlight: dict[str, Any],
) -> None:
    from .raw_decode import _file_identity, _fsync_directory, _unlink_if_owned

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureOneError(f"cannot read decoder receipt {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise CaptureOneError("decoder returned an invalid receipt")
    payload["capture"] = capture
    payload["camera"] = camera
    payload["scanlight"] = scanlight
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        identity = _file_identity(path)
        try:
            _fsync_directory(path.parent)
        except BaseException:
            _unlink_if_owned(path, identity)
            raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_output(output: Path, receipt: Path) -> None:
    if not output.is_absolute() or not receipt.is_absolute():
        raise CaptureOneError("output and receipt paths must be absolute")
    if output.suffix.lower() != ".pgm":
        raise CaptureOneError("output path must end in .pgm")
    expected_receipt = output.with_name(f"{output.stem}.receipt.json")
    if receipt != expected_receipt:
        raise CaptureOneError(f"receipt path must be {expected_receipt}")
    if not output.parent.is_dir():
        raise CaptureOneError(f"output directory does not exist: {output.parent}")
    for path in (
        output,
        receipt,
        output.with_suffix(".RAF"),
        output.with_suffix(".NEF"),
    ):
        if os.path.lexists(path):
            raise CaptureOneError(f"refusing to overwrite existing path: {path}")


def capture_frame(
    *,
    kind: str,
    channel: str,
    power: float,
    exposure_us: int,
    output: Path,
    receipt: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    if kind not in {"film", "flat", "dark"}:
        raise CaptureOneError(f"unsupported capture kind: {kind}")
    if (
        isinstance(exposure_us, bool)
        or not isinstance(exposure_us, int)
        or exposure_us <= 0
    ):
        raise CaptureOneError("exposure-us must be a positive integer")
    packet, pwm_code = _color_packet(channel, power)
    if kind == "dark" and power != 0.0:
        raise CaptureOneError("dark capture power must be zero")
    output = Path(output)
    receipt = (
        Path(receipt)
        if receipt is not None
        else output.with_name(f"{output.stem}.receipt.json")
    )
    _validate_output(output, receipt)
    config = _config(os.environ if environment is None else environment, camera=True)

    downloaded: Path | None = None
    with TemporaryDirectory(
        prefix=f".{output.stem}.capture.", dir=output.parent
    ) as temporary:
        with _safe_scanlight(config.scanlight_port) as light:
            firmware_id, hardware_id = _validated_probe(
                light, config.expected_hardware_id
            )
            version = _gphoto_version(config)
            applied_exposure_us = _configure_shutter(config, exposure_us)
            camera_state = _camera_state(config)
            _write(light, packet)
            time.sleep(config.settle_seconds)
            downloaded = _download_raw(config, Path(temporary), exposure_us)

        native = output.with_suffix(downloaded.suffix.upper())
        from .raw_decode import _publish, decode_raw

        _publish(downloaded, native)
        decode_raw(native, output, receipt)
        _augment_receipt(
            receipt,
            capture={
                "kind": kind,
                "channel": channel,
                "commanded_power": power,
                "commanded_exposure_us": exposure_us,
                "applied_exposure_us": applied_exposure_us,
                "camera_port": config.gphoto_port,
                "gphoto2_version": version,
            },
            camera=camera_state,
            scanlight={
                "port": config.scanlight_port,
                "hardware_id": hardware_id,
                "firmware_id": firmware_id,
                "pwm_code": pwm_code,
                "settle_ms": config.settle_seconds * 1000.0,
            },
        )


def turn_off(environment: Mapping[str, str] | None = None) -> None:
    config = _config(os.environ if environment is None else environment, camera=False)
    with _safe_scanlight(config.scanlight_port):
        pass


def probe(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    config = _config(os.environ if environment is None else environment, camera=False)
    with _safe_scanlight(config.scanlight_port) as light:
        firmware_id, hardware_id = _validated_probe(light, config.expected_hardware_id)
    return {
        "port": config.scanlight_port,
        "baud_rate": BAUD_RATE,
        "firmware_id": firmware_id,
        "hardware_id": hardware_id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capture-one")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("off", help="switch every ScanLight channel off")
    subparsers.add_parser("probe", help="verify the configured ScanLight")
    capture = subparsers.add_parser("capture", help="capture one native RAW plane")
    capture.add_argument("--kind", choices=("film", "flat", "dark"), required=True)
    capture.add_argument("--channel", choices=("R", "G", "B"), required=True)
    capture.add_argument("--power", type=float, required=True)
    capture.add_argument("--exposure-us", type=int, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "off":
            turn_off()
        elif args.command == "probe":
            print(json.dumps(probe(), sort_keys=True))
        else:
            capture_frame(
                kind=args.kind,
                channel=args.channel,
                power=args.power,
                exposure_us=args.exposure_us,
                output=args.output,
                receipt=args.receipt,
            )
    except Exception as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
