"""Clear-film exposure calibration for matched sequential R/G/B captures."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from .cli import (
    CHANNELS,
    MANIFEST_NAME,
    NegativeUniversalError,
    PgmImage,
    _load_capture,
    _staged_output,
    _validate_capture_settings,
    sha256_file,
)


def _fraction(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NegativeUniversalError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise NegativeUniversalError(f"{name} must be between 0 and 1")
    return value


def _checked_roi(roi: Sequence[int], image: PgmImage) -> tuple[int, int, int, int]:
    try:
        values = tuple(roi)
    except TypeError as error:
        raise NegativeUniversalError(
            "ROI must be four integers: x, y, width, height"
        ) from error
    if (
        isinstance(roi, (str, bytes))
        or len(values) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        )
    ):
        raise NegativeUniversalError("ROI must be four integers: x, y, width, height")
    x, y, width, height = values
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise NegativeUniversalError("ROI origin must be nonnegative and size positive")
    if x + width > image.width or y + height > image.height:
        raise NegativeUniversalError("ROI exceeds the capture dimensions")
    return x, y, width, height


def _site_pattern(sensor_layout: str) -> tuple[str, ...]:
    if sensor_layout == "mono":
        return ("mono",)
    labels: list[str] = []
    green = 0
    for site in sensor_layout.upper():
        if site == "G":
            green += 1
            labels.append(f"G{green}")
        else:
            labels.append(site)
    return tuple(labels)


def _p99(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise NegativeUniversalError("ROI contains no usable samples")
    rank = (99 * len(ordered) + 99) // 100
    return ordered[rank - 1]


def _validate_camera_white_levels(
    captures: Iterable[tuple[Path, dict[str, Any], dict[str, PgmImage]]],
) -> dict[str, int]:
    observed = {channel: set() for channel in CHANNELS}
    for directory, manifest, images in captures:
        for channel in CHANNELS:
            pgm_max = images[channel].max_value
            source = manifest["frames"][channel].get("native_source")
            if source is None:
                observed[channel].add(pgm_max)
                continue
            receipt_path = directory / source["receipt_file"]
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                levels = receipt["raster"]["camera_white_levels"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise NegativeUniversalError(
                    f"cannot read camera white levels from {receipt_path}: {error}"
                ) from error
            if levels is None:
                observed[channel].add(pgm_max)
                continue
            if (
                not isinstance(levels, list)
                or len(levels) != 4
                or any(
                    isinstance(level, bool) or not isinstance(level, int)
                    for level in levels
                )
                or len(set(levels)) != 1
                or not 0 < levels[0] <= pgm_max
            ):
                raise NegativeUniversalError(
                    f"{channel} camera white levels in {receipt_path} must all "
                    f"be equal and no greater than the PGM maximum {pgm_max}"
                )
            observed[channel].add(levels[0])

    saturation_levels: dict[str, int] = {}
    for channel, levels in observed.items():
        if len(levels) != 1:
            raise NegativeUniversalError(
                f"{channel} camera saturation level does not match across "
                "clear base, dark, and flat"
            )
        saturation_levels[channel] = next(iter(levels))
    return saturation_levels


def _channel_statistics(
    *,
    channel: str,
    clear_base: PgmImage,
    dark: PgmImage,
    flat: PgmImage,
    roi: tuple[int, int, int, int],
    sensor_layout: str,
    target_base_fraction: float,
    flat_ceiling_fraction: float,
    saturation_level: int,
    frame: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pattern = _site_pattern(sensor_layout)
    site_names = tuple(dict.fromkeys(pattern))
    samples: dict[str, dict[str, Any]] = {
        site: {
            "base": [],
            "flat": [],
            "transmittance": [],
            "clipped_low": 0,
            "clipped_high": 0,
            "clear_base_at_or_above_saturation": 0,
            "flat_at_or_above_saturation": 0,
        }
        for site in site_names
    }
    x0, y0, width, height = roi

    # ponytail: exact ROI quantiles stay in memory; use streaming histograms only
    # if real calibration ROIs prove too large for this simple implementation.
    for y in range(y0, y0 + height):
        row = y * clear_base.width
        for x in range(x0, x0 + width):
            index = row + x
            site = "mono" if sensor_layout == "mono" else pattern[(y % 2) * 2 + x % 2]
            base_value = clear_base.pixels[index]
            dark_value = dark.pixels[index]
            flat_value = flat.pixels[index]
            if saturation_level <= dark_value:
                raise NegativeUniversalError(
                    f"{channel} dark reaches the saturation level inside the ROI"
                )
            flat_denominator = flat_value - dark_value
            if flat_denominator <= 0:
                raise NegativeUniversalError(
                    f"{channel} flat must exceed dark at every ROI sample"
                )

            full_well = saturation_level - dark_value
            base_fraction = (base_value - dark_value) / full_well
            flat_fraction = flat_denominator / full_well
            transmittance = (base_value - dark_value) / flat_denominator
            record = samples[site]
            record["base"].append(base_fraction)
            record["flat"].append(flat_fraction)
            record["transmittance"].append(min(1.0, max(0.0, transmittance)))
            record["clipped_low"] += transmittance < 0.0
            record["clipped_high"] += transmittance > 1.0
            record["clear_base_at_or_above_saturation"] += (
                base_value >= saturation_level
            )
            record["flat_at_or_above_saturation"] += flat_value >= saturation_level

    if sensor_layout != "mono" and any(
        not samples[site]["base"] for site in site_names
    ):
        raise NegativeUniversalError("Bayer ROI must contain every CFA site class")

    meter_sites = (
        ("mono",)
        if sensor_layout == "mono"
        else {
            "R": ("R",),
            "G": ("G1", "G2"),
            "B": ("B",),
        }[channel]
    )
    base_level = median(
        chain.from_iterable(samples[site]["base"] for site in meter_sites)
    )
    site_flat_highs = {site: _p99(record["flat"]) for site, record in samples.items()}
    flat_high = max(site_flat_highs.values())
    flat_ceiling_sites = [
        site for site, site_high in site_flat_highs.items() if site_high == flat_high
    ]
    if base_level <= 0.0:
        raise NegativeUniversalError(
            f"{channel} clear-base signal must exceed dark on its meter sites"
        )
    if flat_high <= 0.0:
        raise NegativeUniversalError(f"{channel} flat signal must exceed dark")

    base_multiplier = target_base_fraction / base_level
    flat_multiplier = flat_ceiling_fraction / flat_high
    ideal_multiplier = min(base_multiplier, flat_multiplier)
    native_source = frame.get("native_source")
    applied_exposure = (
        native_source["applied_exposure_us"] if native_source is not None else None
    )
    basis_exposure = applied_exposure or frame["commanded_exposure_us"]
    recommended_exposure = max(1, int(basis_exposure * ideal_multiplier + 0.5))
    acquisition = {
        "meter_sites": list(meter_sites),
        "base_fraction_median": base_level,
        "flat_fraction_max_site_p99": flat_high,
        "flat_ceiling_sites": flat_ceiling_sites,
        "ideal_exposure_time_multiplier": ideal_multiplier,
        "rounded_recommendation_multiplier": recommended_exposure / basis_exposure,
        "limited_by": "clear_base" if base_multiplier <= flat_multiplier else "flat",
        "current": {
            "commanded_power": frame["commanded_power"],
            "commanded_exposure_us": frame["commanded_exposure_us"],
            "applied_exposure_us": applied_exposure,
            "emitted_pwm_code": (
                native_source["pwm_code"] if native_source is not None else None
            ),
        },
        "recommendation_basis": {
            "exposure_us": basis_exposure,
            "source": (
                "applied_exposure_us"
                if applied_exposure is not None
                else "commanded_exposure_us"
            ),
        },
        "recommendation": {
            "commanded_power": frame["commanded_power"],
            "commanded_exposure_us": recommended_exposure,
        },
    }
    diagnostics = {
        site: {
            "sample_count": len(record["base"]),
            "base_fraction_median": median(record["base"]),
            "flat_fraction_p99": site_flat_highs[site],
            "transmittance_median": median(record["transmittance"]),
            "clipped_low": record["clipped_low"],
            "clipped_high": record["clipped_high"],
            "clear_base_at_or_above_saturation": record[
                "clear_base_at_or_above_saturation"
            ],
            "flat_at_or_above_saturation": record["flat_at_or_above_saturation"],
        }
        for site, record in samples.items()
    }
    return acquisition, diagnostics


def calibrate_clear_base(
    clear_base: Path,
    dark: Path,
    flat: Path,
    roi: Sequence[int],
    target_base_fraction: float,
    flat_ceiling_fraction: float,
    output: Path,
) -> None:
    """Recommend per-illumination exposure from a matched clear-film capture."""

    target_base_fraction = _fraction("target_base_fraction", target_base_fraction)
    flat_ceiling_fraction = _fraction("flat_ceiling_fraction", flat_ceiling_fraction)
    if target_base_fraction > flat_ceiling_fraction:
        raise NegativeUniversalError(
            "target_base_fraction must not exceed flat_ceiling_fraction"
        )

    clear_manifest, clear_images = _load_capture(Path(clear_base), "film")
    dark_manifest, dark_images = _load_capture(Path(dark), "dark")
    flat_manifest, flat_images = _load_capture(Path(flat), "flat")
    _validate_capture_settings(clear_manifest, dark_manifest, flat_manifest)
    saturation_levels = _validate_camera_white_levels(
        (
            (Path(clear_base), clear_manifest, clear_images),
            (Path(dark), dark_manifest, dark_images),
            (Path(flat), flat_manifest, flat_images),
        )
    )

    reference = clear_images["R"]
    checked_roi = _checked_roi(roi, reference)
    sensor_layout = clear_manifest["rig_metadata"]["sensor_layout"]
    if sensor_layout != "mono" and (checked_roi[2] < 2 or checked_roi[3] < 2):
        raise NegativeUniversalError("Bayer ROI must be at least 2 by 2 samples")
    for channel in CHANNELS:
        expected = (reference.width, reference.height, reference.max_value)
        for label, image in (
            ("clear base", clear_images[channel]),
            ("dark", dark_images[channel]),
            ("flat", flat_images[channel]),
        ):
            if (image.width, image.height, image.max_value) != expected:
                raise NegativeUniversalError(
                    f"{channel} {label} image does not match the clear-base images"
                )

    acquisition: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    with _staged_output(Path(output)) as staging:
        for channel in CHANNELS:
            acquisition[channel], diagnostics[channel] = _channel_statistics(
                channel=channel,
                clear_base=clear_images[channel],
                dark=dark_images[channel],
                flat=flat_images[channel],
                roi=checked_roi,
                sensor_layout=sensor_layout,
                target_base_fraction=target_base_fraction,
                flat_ceiling_fraction=flat_ceiling_fraction,
                saturation_level=saturation_levels[channel],
                frame=clear_manifest["frames"][channel],
            )

        manifest = {
            "schema": 1,
            "kind": "clear_base_exposure_calibration",
            "created_at": datetime.now(UTC).isoformat(),
            "channel_order": list(CHANNELS),
            "configuration_id": clear_manifest["rig_metadata"]["configuration_id"],
            "settings_sha256": clear_manifest["settings_sha256"],
            "sensor_layout": sensor_layout,
            "pgm_max_value": reference.max_value,
            "channel_saturation_levels": saturation_levels,
            "saturation_level_policy": (
                "four_equal_camera_white_levels_at_or_below_pgm_max_value_"
                "else_pgm_max_value_matching_clear_base_dark_flat"
            ),
            "roi": dict(zip(("x", "y", "width", "height"), checked_roi, strict=True)),
            "targets": {
                "base_median_fraction": target_base_fraction,
                "flat_p99_ceiling_fraction": flat_ceiling_fraction,
            },
            "percentile_method": "nearest_rank",
            "formula": {
                "base_fraction": ("(clear_base-dark)/(channel_saturation_level-dark)"),
                "flat_fraction": "(flat-dark)/(channel_saturation_level-dark)",
                "transmittance": "clamp((clear_base-dark)/(flat-dark), 0, 1)",
                "ideal_exposure_time_multiplier": (
                    "min(target_base_fraction/base_median, "
                    "flat_ceiling_fraction/max_site_flat_p99)"
                ),
                "rounded_recommendation_multiplier": (
                    "recommended_commanded_exposure_us/recommendation_basis_exposure_us"
                ),
            },
            "sources": {
                "clear_base_manifest_sha256": sha256_file(
                    Path(clear_base) / MANIFEST_NAME
                ),
                "dark_manifest_sha256": sha256_file(Path(dark) / MANIFEST_NAME),
                "flat_manifest_sha256": sha256_file(Path(flat) / MANIFEST_NAME),
            },
            "acquisition": acquisition,
            "diagnostics": diagnostics,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
