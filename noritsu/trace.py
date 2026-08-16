"""Deterministic, incremental image-stage tracing.

``StageTrace`` records the observable output of each image-processing stage.  It
is deliberately independent of the processing pipeline: callers provide an RGB
snapshot and the evidence that describes the stage.  Measurements are made on
the same quantized 16-bit pixels that are written to the checkpoint TIFFs.

When an output directory is supplied, only the current full-resolution frame is
held by the recorder.  Previous frames and the optional reference are read from
their checkpoint TIFFs as needed; only small previews and JSON-compatible
records accumulate in memory.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


PathLike = Union[str, Path]
ImageLike = Union[np.ndarray, PathLike]

_CHANNELS = ("r", "g", "b")
_LUMA_WEIGHTS = np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)
_U16_MAX = 65535.0
_HIST_VALUES = np.arange(65536, dtype=np.float64)


def _json_value(value: Any) -> Any:
    """Return a deterministic, strict-JSON representation of *value*."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("trace metadata must not contain NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError("trace metadata is not JSON serializable: %r" % (type(value),))


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return value[:80] or "stage"


def _quantize_rgb(
    image: np.ndarray,
    expected_range: Optional[Tuple[float, float]] = None,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Convert an HxWx3 image to RGB uint16 using round-half-up quantization."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            "stage image must have shape (height, width, 3), got %r" % (array.shape,)
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("stage image dimensions must be non-zero")

    if array.dtype == np.uint16:
        return np.ascontiguousarray(array)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array.astype(np.uint16) * np.uint16(257))
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            "stage image must be float, uint8, or uint16; got %s" % array.dtype
        )

    if expected_range is None:
        expected_range = (0.0, 1.0)
    low, high = (float(expected_range[0]), float(expected_range[1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("float expected_range must be finite and increasing")

    result = np.empty(array.shape, dtype=np.uint16)
    for start in range(0, array.shape[0], chunk_rows):
        stop = min(start + chunk_rows, array.shape[0])
        chunk = np.asarray(array[start:stop])
        if not np.isfinite(chunk).all():
            raise ValueError("stage image contains NaN or infinity")
        scaled = np.clip((chunk - low) / (high - low), 0.0, 1.0) * _U16_MAX
        result[start:stop] = np.floor(scaled + 0.5).astype(np.uint16)
    return result


def _load_image(image: ImageLike) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() in (".tif", ".tiff"):
            return np.asarray(tifffile.imread(str(path)))
        with Image.open(str(path)) as opened:
            return np.asarray(opened.convert("RGB"))
    return np.asarray(image)


def _write_tiff(path: Path, pixels: np.ndarray) -> None:
    """Write a metadata-free, memory-mappable 16-bit RGB TIFF."""

    tifffile.imwrite(
        str(path),
        np.asarray(pixels, dtype=np.uint16),
        photometric="rgb",
        metadata=None,
        software=False,
        datetime=False,
        compression=None,
        contiguous=True,
        byteorder="<",
    )


def _open_tiff(path: Path) -> np.ndarray:
    try:
        return tifffile.memmap(str(path), mode="r")
    except (ValueError, TypeError):
        return np.asarray(tifffile.imread(str(path)))


def _sha256_u16(pixels: np.ndarray, chunk_rows: int = 256) -> str:
    digest = hashlib.sha256()
    for start in range(0, pixels.shape[0], chunk_rows):
        chunk = np.asarray(pixels[start : start + chunk_rows], dtype="<u2")
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _hist_quantile(histogram: np.ndarray, percentile: float) -> float:
    """Linear percentile from an integer histogram, matching NumPy's default."""

    count = int(histogram.sum())
    if count == 0:
        raise ValueError("cannot find a percentile of an empty histogram")
    rank = (count - 1) * (float(percentile) / 100.0)
    lower_rank = int(math.floor(rank))
    upper_rank = int(math.ceil(rank))
    cumulative = np.cumsum(histogram)
    lower = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    fraction = rank - lower_rank
    return float(lower + (upper - lower) * fraction)


def _channel_statistics(pixels: np.ndarray, chunk_rows: int = 256) -> Dict[str, Any]:
    histograms = [np.zeros(65536, dtype=np.int64) for _ in _CHANNELS]
    for start in range(0, pixels.shape[0], chunk_rows):
        chunk = np.asarray(pixels[start : start + chunk_rows])
        for channel in range(3):
            histograms[channel] += np.bincount(
                chunk[..., channel].reshape(-1), minlength=65536
            )

    output: Dict[str, Any] = {}
    for label, histogram in zip(_CHANNELS, histograms):
        count = int(histogram.sum())
        populated = np.flatnonzero(histogram)
        total = float(np.dot(histogram.astype(np.float64), _HIST_VALUES))
        mean_u16 = total / count
        variance_u16 = float(
            np.dot(histogram.astype(np.float64), (_HIST_VALUES - mean_u16) ** 2) / count
        )
        low_count = int(histogram[0])
        high_count = int(histogram[-1])
        output[label] = {
            "min": float(populated[0] / _U16_MAX),
            "max": float(populated[-1] / _U16_MAX),
            "mean": float(mean_u16 / _U16_MAX),
            "std": float(math.sqrt(max(0.0, variance_u16)) / _U16_MAX),
            "p01": float(_hist_quantile(histogram, 1.0) / _U16_MAX),
            "p50": float(_hist_quantile(histogram, 50.0) / _U16_MAX),
            "p99": float(_hist_quantile(histogram, 99.0) / _U16_MAX),
            "clipping": {
                "low_count": low_count,
                "high_count": high_count,
                "either_count": low_count + high_count,
                "low_fraction": float(low_count / count),
                "high_fraction": float(high_count / count),
                "either_fraction": float((low_count + high_count) / count),
            },
        }
    return output


def _native_statistics(
    image: np.ndarray,
    expected_range: Optional[Tuple[float, float]],
    chunk_rows: int = 256,
) -> Dict[str, Any]:
    """Describe source values before display quantization or clipping."""

    array = np.asarray(image)
    channels = []
    for channel in range(3):
        count = 0
        finite_count = 0
        nonfinite_count = 0
        below_count = 0
        above_count = 0
        minimum = math.inf
        maximum = -math.inf
        total = 0.0
        total_square = 0.0
        for start in range(0, array.shape[0], chunk_rows):
            values = np.asarray(
                array[start : start + chunk_rows, :, channel], dtype=np.float64
            )
            count += int(values.size)
            finite = np.isfinite(values)
            nonfinite_count += int(values.size - np.count_nonzero(finite))
            if not finite.any():
                continue
            valid = values[finite]
            finite_count += int(valid.size)
            minimum = min(minimum, float(valid.min()))
            maximum = max(maximum, float(valid.max()))
            total += float(valid.sum(dtype=np.float64))
            total_square += float(np.dot(valid, valid))
            if expected_range is not None:
                below_count += int(np.count_nonzero(valid < expected_range[0]))
                above_count += int(np.count_nonzero(valid > expected_range[1]))
        if finite_count:
            mean = total / finite_count
            variance = max(0.0, total_square / finite_count - mean * mean)
            std = math.sqrt(variance)
        else:
            minimum = maximum = mean = std = None
        channels.append(
            {
                "name": _CHANNELS[channel],
                "count": count,
                "finite_count": finite_count,
                "nonfinite_count": nonfinite_count,
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "std": std,
                "below_expected_count": below_count,
                "above_expected_count": above_count,
                "out_of_expected_count": below_count + above_count,
                "out_of_expected_fraction": float(
                    (below_count + above_count) / finite_count
                )
                if finite_count
                else None,
            }
        )
    return {
        "dtype": str(array.dtype),
        "expected_range": list(expected_range) if expected_range is not None else None,
        "channels": channels,
    }


def _empty_comparison(reason: str) -> Dict[str, Any]:
    return {"status": "not_available", "reason": reason}


def _is_unexecuted_status(status: str) -> bool:
    """Return whether a checkpoint is only a carried-forward placeholder.

    Disabled/default-off stages have a defensible zero effect.  A blocked
    stage does not: its pixels merely repeat the prior checkpoint because the
    missing parameters prevented execution.  Keeping those cases distinct is
    essential to avoid reporting a fabricated zero delta for an unknown
    effect.
    """

    value = str(status).lower()
    return (
        value.startswith("blocked_")
        or value.startswith("not_executed_")
        or value.endswith("_pending")
    )


def _identical_comparison(
    shape: Sequence[int], value_range: Tuple[float, float]
) -> Dict[str, Any]:
    metric = {
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs": 0.0,
        "p95_abs": 0.0,
        "p99_abs": 0.0,
        "psnr_db": None,
        "psnr_status": "identical",
    }
    return {
        "status": "ok",
        "current_shape": [int(value) for value in shape],
        "other_shape": [int(value) for value in shape],
        "normalization_range": [float(value_range[0]), float(value_range[1])],
        "overall": dict(metric),
        "channels": {channel: dict(metric) for channel in _CHANNELS},
        "signed_bias": {channel: 0.0 for channel in _CHANNELS},
        "changed_pixel_count": 0,
        "changed_pixel_fraction": 0.0,
        "luma_ncc": None,
        "luma_ncc_status": "not_computed_identical_by_stage_contract",
    }


def _shape_list(pixels: np.ndarray) -> Sequence[int]:
    return [int(item) for item in pixels.shape]


def _metric_values(histogram: np.ndarray, normalization_span: float) -> Dict[str, Any]:
    count = int(histogram.sum())
    values = _HIST_VALUES
    histogram_float = histogram.astype(np.float64)
    mae = float(np.dot(histogram_float, values) / count / normalization_span)
    mse = float(
        np.dot(histogram_float, values * values) / count / (normalization_span**2)
    )
    rmse = float(math.sqrt(max(0.0, mse)))
    populated = np.flatnonzero(histogram)
    max_abs = float(populated[-1] / normalization_span)
    p95_abs = float(_hist_quantile(histogram, 95.0) / normalization_span)
    p99_abs = float(_hist_quantile(histogram, 99.0) / normalization_span)
    if rmse == 0.0:
        psnr = None
        psnr_status = "identical"
    else:
        psnr = float(20.0 * math.log10(1.0 / rmse))
        psnr_status = "finite"
    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "p95_abs": p95_abs,
        "p99_abs": p99_abs,
        "psnr_db": psnr,
        "psnr_status": psnr_status,
    }


def _merge_covariance(
    state: Tuple[int, float, float, float, float, float],
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[int, float, float, float, float, float]:
    """Merge a luma chunk into an online bivariate covariance state."""

    count, mean_x, mean_y, m2_x, m2_y, covariance = state
    chunk_count = int(x.size)
    chunk_mean_x = float(x.mean(dtype=np.float64))
    chunk_mean_y = float(y.mean(dtype=np.float64))
    centered_x = x - chunk_mean_x
    centered_y = y - chunk_mean_y
    chunk_m2_x = float(np.dot(centered_x, centered_x))
    chunk_m2_y = float(np.dot(centered_y, centered_y))
    chunk_covariance = float(np.dot(centered_x, centered_y))
    if count == 0:
        return (
            chunk_count,
            chunk_mean_x,
            chunk_mean_y,
            chunk_m2_x,
            chunk_m2_y,
            chunk_covariance,
        )

    combined_count = count + chunk_count
    delta_x = chunk_mean_x - mean_x
    delta_y = chunk_mean_y - mean_y
    weight = count * chunk_count / combined_count
    return (
        combined_count,
        mean_x + delta_x * chunk_count / combined_count,
        mean_y + delta_y * chunk_count / combined_count,
        m2_x + chunk_m2_x + delta_x * delta_x * weight,
        m2_y + chunk_m2_y + delta_y * delta_y * weight,
        covariance + chunk_covariance + delta_x * delta_y * weight,
    )


def compare_u16_rgb(
    current: np.ndarray,
    other: np.ndarray,
    chunk_rows: int = 128,
    value_range: Tuple[float, float] = (0.0, 65535.0),
) -> Dict[str, Any]:
    """Compare two quantized RGB frames without allocating a full-size diff."""

    current = np.asarray(current)
    other = np.asarray(other)
    for label, array in (("current", current), ("other", other)):
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{label} must have shape (height, width, 3)")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{label} dimensions must be non-zero")
        if array.dtype != np.uint16:
            raise TypeError(f"{label} must be uint16")

    low, high = (float(value_range[0]), float(value_range[1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("comparison value range must be finite and increasing")
    normalization_span = high - low

    if tuple(current.shape) != tuple(other.shape):
        return {
            "status": "shape_mismatch",
            "current_shape": _shape_list(current),
            "other_shape": _shape_list(other),
            "reason": "pixel metrics require identical height, width, and channel count",
        }

    channel_histograms = [np.zeros(65536, dtype=np.int64) for _ in _CHANNELS]
    overall_histogram = np.zeros(65536, dtype=np.int64)
    covariance_state = (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    signed_sum = np.zeros(3, dtype=np.int64)
    changed_pixels = 0
    pixel_count = 0

    for start in range(0, current.shape[0], chunk_rows):
        current_chunk = np.asarray(current[start : start + chunk_rows])
        other_chunk = np.asarray(other[start : start + chunk_rows])
        difference = np.abs(
            current_chunk.astype(np.int32) - other_chunk.astype(np.int32)
        )
        signed = current_chunk.astype(np.int32) - other_chunk.astype(np.int32)
        signed_sum += signed.reshape(-1, 3).sum(axis=0, dtype=np.int64)
        changed_pixels += int(np.count_nonzero(np.any(difference != 0, axis=2)))
        pixel_count += int(difference.shape[0] * difference.shape[1])
        overall_histogram += np.bincount(difference.reshape(-1), minlength=65536)
        for channel in range(3):
            channel_histograms[channel] += np.bincount(
                difference[..., channel].reshape(-1), minlength=65536
            )

        current_luma = np.tensordot(
            current_chunk.astype(np.float64), _LUMA_WEIGHTS, axes=([-1], [0])
        ).reshape(-1)
        other_luma = np.tensordot(
            other_chunk.astype(np.float64), _LUMA_WEIGHTS, axes=([-1], [0])
        ).reshape(-1)
        covariance_state = _merge_covariance(covariance_state, current_luma, other_luma)

    _, _, _, m2_current, m2_other, covariance = covariance_state
    denominator = math.sqrt(max(0.0, m2_current) * max(0.0, m2_other))
    if denominator == 0.0:
        luma_ncc = None
        luma_status = "zero_variance"
    else:
        luma_ncc = float(np.clip(covariance / denominator, -1.0, 1.0))
        luma_status = "ok"

    return {
        "status": "ok",
        "current_shape": _shape_list(current),
        "other_shape": _shape_list(other),
        "normalization_range": [low, high],
        "overall": _metric_values(overall_histogram, normalization_span),
        "channels": {
            label: _metric_values(histogram, normalization_span)
            for label, histogram in zip(_CHANNELS, channel_histograms)
        },
        "signed_bias": {
            label: float(value / pixel_count / normalization_span)
            for label, value in zip(_CHANNELS, signed_sum)
        },
        "changed_pixel_count": changed_pixels,
        "changed_pixel_fraction": float(changed_pixels / pixel_count),
        "luma_ncc": luma_ncc,
        "luma_ncc_status": luma_status,
    }


def _sample_step(shape: Sequence[int], maximum: Tuple[int, int]) -> int:
    height, width = int(shape[0]), int(shape[1])
    maximum_width, maximum_height = maximum
    return max(1, int(math.ceil(max(width / maximum_width, height / maximum_height))))


def _u16_to_u8(
    pixels: np.ndarray, value_range: Tuple[float, float] = (0.0, 65535.0)
) -> np.ndarray:
    low, high = (float(value_range[0]), float(value_range[1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("preview value range must be finite and increasing")
    values = (np.asarray(pixels, dtype=np.float64) - low) / (high - low)
    return np.floor(np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _preview_image(
    pixels: np.ndarray,
    maximum: Tuple[int, int],
    value_range: Tuple[float, float] = (0.0, 65535.0),
) -> Image.Image:
    step = _sample_step(pixels.shape, maximum)
    sampled = np.asarray(pixels[::step, ::step])
    return Image.fromarray(_u16_to_u8(sampled, value_range), mode="RGB")


def _checkpoint_value_range(
    source: np.ndarray, expected_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Translate a native expected range into written-checkpoint units."""

    if np.issubdtype(source.dtype, np.floating):
        return (0.0, _U16_MAX)
    if source.dtype == np.uint8:
        return (
            float(expected_range[0]) * 257.0,
            float(expected_range[1]) * 257.0,
        )
    return (float(expected_range[0]), float(expected_range[1]))


def _heatmap(values: np.ndarray) -> np.ndarray:
    """Map [0,1] scalars through a fixed black-blue-cyan-yellow-red palette."""

    positions = np.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float32)
    colors = np.array(
        (
            (0, 0, 0),
            (22, 36, 145),
            (0, 190, 220),
            (255, 225, 40),
            (255, 35, 15),
        ),
        dtype=np.float32,
    )
    flattened = np.clip(values, 0.0, 1.0).reshape(-1)
    output = np.empty((flattened.size, 3), dtype=np.float32)
    for channel in range(3):
        output[:, channel] = np.interp(flattened, positions, colors[:, channel])
    return np.floor(output.reshape(values.shape + (3,)) + 0.5).astype(np.uint8)


def _write_diff_previews(
    current: np.ndarray,
    other: np.ndarray,
    rgb_path: Path,
    heatmap_path: Path,
    maximum: Tuple[int, int],
    amplification: float,
    normalization_span: float = _U16_MAX,
) -> None:
    step = _sample_step(current.shape, maximum)
    current_sample = np.asarray(current[::step, ::step], dtype=np.int32)
    other_sample = np.asarray(other[::step, ::step], dtype=np.int32)
    absolute = np.abs(current_sample - other_sample).astype(np.float32) / float(
        normalization_span
    )
    amplified = np.clip(absolute * amplification, 0.0, 1.0)
    rgb = np.floor(amplified * 255.0 + 0.5).astype(np.uint8)
    scalar = amplified.max(axis=2)
    Image.fromarray(rgb, mode="RGB").save(str(rgb_path), format="PNG")
    Image.fromarray(_heatmap(scalar), mode="RGB").save(str(heatmap_path), format="PNG")


def _write_signed_diff_preview(
    current: np.ndarray,
    other: np.ndarray,
    path: Path,
    maximum: Tuple[int, int],
    amplification: float,
    normalization_span: float = _U16_MAX,
) -> None:
    """Write signed BT.709 luma difference: blue negative, red positive."""

    step = _sample_step(current.shape, maximum)
    current_sample = np.asarray(current[::step, ::step], dtype=np.float64)
    other_sample = np.asarray(other[::step, ::step], dtype=np.float64)
    signed = np.tensordot(
        current_sample - other_sample, _LUMA_WEIGHTS, axes=([-1], [0])
    )
    signed = np.clip(signed / float(normalization_span) * amplification, -1.0, 1.0)
    output = np.zeros(signed.shape + (3,), dtype=np.uint8)
    output[..., 0] = np.floor(np.clip(signed, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    output[..., 2] = np.floor(np.clip(-signed, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    output[..., 1] = np.floor((1.0 - np.abs(signed)) * 24 + 0.5).astype(np.uint8)
    Image.fromarray(output, mode="RGB").save(str(path), format="PNG")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_comparison_artifacts(
    current: np.ndarray,
    other: np.ndarray,
    output_dir: PathLike,
    *,
    prefix: str = "comparison",
    preview_size: Tuple[int, int] = (640, 480),
    diff_amplification: float = 8.0,
    value_range: Tuple[float, float] = (0.0, 65535.0),
) -> Dict[str, Any]:
    """Write a standalone numerical and visual comparison of two RGB frames."""

    current = np.asarray(current)
    other = np.asarray(other)
    metrics = compare_u16_rgb(current, other, value_range=value_range)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = _slug(prefix)
    paths = {
        "current_preview_png": destination / f"{stem}_current.png",
        "other_preview_png": destination / f"{stem}_other.png",
        "absolute_diff_rgb_png": destination / f"{stem}_diff_rgb.png",
        "absolute_diff_heatmap_png": destination / f"{stem}_diff_heatmap.png",
        "signed_luma_diff_png": destination / f"{stem}_diff_signed.png",
    }
    _preview_image(current, preview_size, value_range).save(
        str(paths["current_preview_png"]), format="PNG"
    )
    _preview_image(other, preview_size, value_range).save(
        str(paths["other_preview_png"]), format="PNG"
    )
    span = float(value_range[1]) - float(value_range[0])
    _write_diff_previews(
        current,
        other,
        paths["absolute_diff_rgb_png"],
        paths["absolute_diff_heatmap_png"],
        preview_size,
        diff_amplification,
        span,
    )
    _write_signed_diff_preview(
        current,
        other,
        paths["signed_luma_diff_png"],
        preview_size,
        diff_amplification,
        span,
    )
    return {
        "metrics": metrics,
        "artifacts": {key: path.name for key, path in paths.items()},
    }


class StageTrace:
    """Incrementally record deterministic RGB pipeline checkpoints.

    Args:
        trace_dir: Destination for TIFFs, visual diffs, and reports.  If omitted,
            metrics are still returned but the immediately previous frame must
            remain in memory.
        reference: Optional RGB array or image path.  Every compatible stage is
            compared with this reference.
        reference_name: Human-readable reference label for the manifest/sheet.
        preview_size: Maximum ``(width, height)`` of individual preview images.
        diff_amplification: Fixed multiplier used only for visual diff images.
    """

    schema_version = 2

    def __init__(
        self,
        trace_dir: Optional[PathLike] = None,
        reference: Optional[ImageLike] = None,
        reference_name: str = "reference",
        reference_domain: str = "display_rgb",
        run_provenance: Optional[Mapping[str, Any]] = None,
        preview_size: Tuple[int, int] = (640, 480),
        diff_amplification: float = 8.0,
    ) -> None:
        if len(preview_size) != 2 or min(preview_size) <= 0:
            raise ValueError("preview_size must contain two positive integers")
        if not math.isfinite(diff_amplification) or diff_amplification <= 0:
            raise ValueError("diff_amplification must be a positive finite number")

        self.trace_dir = Path(trace_dir).resolve() if trace_dir is not None else None
        self.preview_size = (int(preview_size[0]), int(preview_size[1]))
        self.diff_amplification = float(diff_amplification)
        self.reference_name = str(reference_name)
        self.reference_domain = str(reference_domain)
        self.run_provenance = _json_value(run_provenance or {})
        self.stages = []
        self._previous_path: Optional[Path] = None
        self._previous_q: Optional[np.ndarray] = None
        self._previous_descriptor: Optional[
            Tuple[str, str, str, Tuple[float, float]]
        ] = None
        self._reference_path: Optional[Path] = None
        self._reference_q: Optional[np.ndarray] = None
        self._reference_record: Optional[Dict[str, Any]] = None
        self._input_path: Optional[Path] = None
        self._input_q: Optional[np.ndarray] = None
        self._input_descriptor: Optional[Tuple[str, str, str, Tuple[float, float]]] = (
            None
        )
        self._finalized = False

        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            for dirname in ("checkpoints", "previews", "diffs"):
                (self.trace_dir / dirname).mkdir(exist_ok=True)

        if reference is not None:
            loaded = _load_image(reference)
            reference_q = _quantize_rgb(loaded)
            self._reference_record = self._describe_frame(
                reference_q, str(loaded.dtype)
            )
            self._reference_record["name"] = self.reference_name
            self._reference_record["domain"] = self.reference_domain
            if isinstance(reference, (str, Path)):
                self._reference_record["source"] = str(Path(reference).resolve())
            if self.trace_dir is not None:
                self._reference_path = self.trace_dir / "reference.tif"
                _write_tiff(self._reference_path, reference_q)
                reference_preview = self.trace_dir / "previews" / "reference.png"
                _preview_image(reference_q, self.preview_size).save(
                    str(reference_preview), format="PNG"
                )
                self._reference_record["artifacts"] = {
                    "checkpoint_tiff": _relative(self._reference_path, self.trace_dir),
                    "preview_png": _relative(reference_preview, self.trace_dir),
                }
            else:
                self._reference_q = reference_q

    def __enter__(self) -> "StageTrace":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.finalize()

    @property
    def manifest(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "quantization": {
                "dtype": "uint16",
                "range": [0, 65535],
                "rounding": "round-half-up",
                "metrics_domain": "checkpoint pixels normalized to [0,1]",
            },
            "luma_weights": [float(value) for value in _LUMA_WEIGHTS],
            "preview": {
                "maximum_size": [self.preview_size[0], self.preview_size[1]],
                "diff_amplification": self.diff_amplification,
            },
            "run_provenance": self.run_provenance,
            "reference": self._reference_record,
            "stages": self.stages,
        }
        if (
            self.trace_dir is not None
            and (self.trace_dir / "contact_sheet.png").exists()
        ):
            result["contact_sheet"] = "contact_sheet.png"
        return result

    def _describe_frame(self, pixels: np.ndarray, source_dtype: str) -> Dict[str, Any]:
        height, width, channels = pixels.shape
        return {
            "dimensions": {
                "height": int(height),
                "width": int(width),
                "channels": int(channels),
                "shape": [int(height), int(width), int(channels)],
            },
            "source_dtype": source_dtype,
            "pixel_hash": {
                "algorithm": "sha256",
                "encoding": "uint16-little-endian-rgb-row-major",
                "value": _sha256_u16(pixels),
            },
            "statistics": {
                "domain": "normalized_u16",
                "channels": _channel_statistics(pixels),
            },
        }

    def _other_pixels(
        self, path: Optional[Path], array: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if path is not None:
            return _open_tiff(path)
        return array

    def add(
        self,
        name: str,
        image: np.ndarray,
        *,
        status: Optional[str] = None,
        applied: bool = True,
        skipped_reason: Optional[str] = None,
        provenance: Any = None,
        confidence: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        domain: str = "display_rgb",
        encoding: str = "normalized_or_u16_rgb",
        channel_order: str = "RGB",
        expected_range: Optional[Tuple[float, float]] = None,
        reference_comparable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Record one ordered stage snapshot and return its manifest record."""

        if not name or not str(name).strip():
            raise ValueError("stage name must not be empty")
        if str(channel_order).upper() != "RGB":
            raise ValueError(
                "StageTrace checkpoints must be RGB; convert working BGR/YCC/Luv data "
                "to an explicitly documented RGB visualization before recording"
            )
        if not applied and not skipped_reason:
            raise ValueError("a skipped stage must include skipped_reason")
        if applied and skipped_reason:
            raise ValueError("an applied stage cannot include skipped_reason")
        if status is None:
            status = "applied" if applied else "skipped"

        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("stage image must have shape (height, width, 3)")
        if expected_range is None:
            expected_range = (
                (0.0, 1.0)
                if np.issubdtype(source.dtype, np.floating)
                else (0.0, float(np.iinfo(source.dtype).max))
            )
        checkpoint_range = _checkpoint_value_range(source, expected_range)
        descriptor = (
            str(domain),
            str(encoding),
            str(channel_order).upper(),
            checkpoint_range,
        )
        if reference_comparable is None:
            reference_comparable = (
                str(domain) == self.reference_domain
                and str(channel_order).upper() == "RGB"
            )
        pixels = _quantize_rgb(source, expected_range=expected_range)
        index = len(self.stages)
        basename = "%03d_%s" % (index, _slug(str(name)))
        record: Dict[str, Any] = {
            "index": index,
            "name": str(name),
            "status": str(status),
            "applied": bool(applied),
            "skipped_reason": str(skipped_reason) if skipped_reason else None,
            "provenance": _json_value(provenance),
            "confidence": _json_value(confidence),
            "params": _json_value(params or {}),
            "domain": str(domain),
            "encoding": str(encoding),
            "channel_order": str(channel_order),
            "reference_comparable": bool(reference_comparable),
            "checkpoint_value_range": list(checkpoint_range),
            "native_statistics": _native_statistics(source, expected_range),
        }
        record.update(self._describe_frame(pixels, str(source.dtype)))
        record["artifacts"] = {}

        checkpoint_path: Optional[Path] = None
        if self.trace_dir is not None:
            checkpoint_path = self.trace_dir / "checkpoints" / (basename + ".tif")
            preview_path = self.trace_dir / "previews" / (basename + ".png")
            _write_tiff(checkpoint_path, pixels)
            _preview_image(pixels, self.preview_size, checkpoint_range).save(
                str(preview_path), format="PNG"
            )
            record["artifacts"].update(
                {
                    "checkpoint_tiff": _relative(checkpoint_path, self.trace_dir),
                    "preview_png": _relative(preview_path, self.trace_dir),
                }
            )

        previous = self._other_pixels(self._previous_path, self._previous_q)
        if previous is None:
            record["difference_from_previous"] = _empty_comparison("first stage")
        elif _is_unexecuted_status(str(status)):
            record["difference_from_previous"] = _empty_comparison(
                "stage was not executed; checkpoint pixels are carried forward"
            )
        elif descriptor != self._previous_descriptor:
            record["difference_from_previous"] = _empty_comparison(
                "stage domain, encoding, channel order, or value range changed"
            )
        else:
            record["difference_from_previous"] = compare_u16_rgb(
                pixels, previous, value_range=checkpoint_range
            )
            if (
                self.trace_dir is not None
                and record["difference_from_previous"]["status"] == "ok"
            ):
                rgb_path = (
                    self.trace_dir / "diffs" / (basename + "_vs_previous_rgb.png")
                )
                heatmap_path = (
                    self.trace_dir / "diffs" / (basename + "_vs_previous_heatmap.png")
                )
                _write_diff_previews(
                    pixels,
                    previous,
                    rgb_path,
                    heatmap_path,
                    self.preview_size,
                    self.diff_amplification,
                    checkpoint_range[1] - checkpoint_range[0],
                )
                record["artifacts"]["previous_diff_rgb_png"] = _relative(
                    rgb_path, self.trace_dir
                )
                record["artifacts"]["previous_diff_heatmap_png"] = _relative(
                    heatmap_path, self.trace_dir
                )
                signed_path = (
                    self.trace_dir / "diffs" / (basename + "_vs_previous_signed.png")
                )
                _write_signed_diff_preview(
                    pixels,
                    previous,
                    signed_path,
                    self.preview_size,
                    self.diff_amplification,
                    checkpoint_range[1] - checkpoint_range[0],
                )
                record["artifacts"]["previous_diff_signed_png"] = _relative(
                    signed_path, self.trace_dir
                )
        del previous

        reference = self._other_pixels(self._reference_path, self._reference_q)
        if not reference_comparable:
            record["difference_from_reference"] = _empty_comparison(
                "stage domain/encoding is not comparable with the machine reference"
            )
        elif reference is None:
            record["difference_from_reference"] = _empty_comparison(
                "no reference supplied"
            )
        else:
            record["difference_from_reference"] = compare_u16_rgb(pixels, reference)
            if (
                self.trace_dir is not None
                and record["difference_from_reference"]["status"] == "ok"
            ):
                rgb_path = (
                    self.trace_dir / "diffs" / (basename + "_vs_reference_rgb.png")
                )
                heatmap_path = (
                    self.trace_dir / "diffs" / (basename + "_vs_reference_heatmap.png")
                )
                _write_diff_previews(
                    pixels,
                    reference,
                    rgb_path,
                    heatmap_path,
                    self.preview_size,
                    self.diff_amplification,
                )
                record["artifacts"]["reference_diff_rgb_png"] = _relative(
                    rgb_path, self.trace_dir
                )
                record["artifacts"]["reference_diff_heatmap_png"] = _relative(
                    heatmap_path, self.trace_dir
                )
                signed_path = (
                    self.trace_dir / "diffs" / (basename + "_vs_reference_signed.png")
                )
                _write_signed_diff_preview(
                    pixels,
                    reference,
                    signed_path,
                    self.preview_size,
                    self.diff_amplification,
                )
                record["artifacts"]["reference_diff_signed_png"] = _relative(
                    signed_path, self.trace_dir
                )
        del reference

        input_pixels = self._other_pixels(self._input_path, self._input_q)
        if input_pixels is None:
            record["difference_from_input"] = _empty_comparison("first stage")
        elif descriptor != self._input_descriptor:
            record["difference_from_input"] = _empty_comparison(
                "stage domain, encoding, channel order, or value range differs from the first stage"
            )
        else:
            record["difference_from_input"] = compare_u16_rgb(
                pixels, input_pixels, value_range=checkpoint_range
            )
        del input_pixels

        # Every stage keeps its own numbered checkpoint, but an unchanged frame
        # need not consume another full-resolution TIFF's worth of disk.  Hash
        # equality is also valid for blocked carried-forward checkpoints whose
        # unknown stage effect is deliberately reported as N/A rather than 0.
        # A hard link preserves the stage artifact and byte identity.  If the
        # filesystem cannot link, retain the independently written checkpoint.
        if (
            checkpoint_path is not None
            and self._previous_path is not None
            and self.stages
            and record["pixel_hash"]["value"]
            == self.stages[-1]["pixel_hash"]["value"]
            and record["dimensions"]["shape"] == self.stages[-1]["dimensions"]["shape"]
        ):
            try:
                checkpoint_path.unlink()
                os.link(self._previous_path, checkpoint_path)
                record["artifacts"]["checkpoint_storage"] = "hardlink_to_previous"
            except OSError:
                if not checkpoint_path.exists():
                    _write_tiff(checkpoint_path, pixels)
                record["artifacts"]["checkpoint_storage"] = "independent_file"

        self.stages.append(record)
        if self.trace_dir is not None:
            self._previous_path = checkpoint_path
            self._previous_q = None
            self._previous_descriptor = descriptor
            if self._input_path is None:
                self._input_path = checkpoint_path
                self._input_descriptor = descriptor
            self._write_reports()
        else:
            self._previous_q = pixels
            self._previous_descriptor = descriptor
            if self._input_q is None:
                self._input_q = pixels
                self._input_descriptor = descriptor
        self._finalized = False
        return record

    record = add

    def skip(
        self,
        name: str,
        image: np.ndarray,
        *,
        reason: str,
        provenance: Any = None,
        confidence: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        domain: str = "display_rgb",
        encoding: str = "normalized_or_u16_rgb",
        channel_order: str = "RGB",
        expected_range: Optional[Tuple[float, float]] = None,
        reference_comparable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Record a deliberately skipped stage while preserving the image."""

        return self.add(
            name,
            image,
            status="skipped",
            applied=False,
            skipped_reason=reason,
            provenance=provenance,
            confidence=confidence,
            params=params,
            domain=domain,
            encoding=encoding,
            channel_order=channel_order,
            expected_range=expected_range,
            reference_comparable=reference_comparable,
        )

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> str:
        try:
            os.link(source, destination)
            return "hardlink_to_previous"
        except OSError:
            shutil.copyfile(source, destination)
            return "copy_of_previous"

    def unchanged(
        self,
        name: str,
        *,
        reason: str,
        status: str = "skipped",
        applied: bool = False,
        provenance: Any = None,
        confidence: Any = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a guaranteed no-op without rescanning a full-size frame.

        The previous checkpoint and preview remain independently addressable
        stage artifacts.  Proven identities/default-off stages receive exact
        zero metrics.  Blocked stages receive N/A because their carried-forward
        pixels are not evidence that the unavailable stage has zero effect.
        """

        if not name or not str(name).strip():
            raise ValueError("stage name must not be empty")
        if not reason:
            raise ValueError("an unchanged stage must include a reason")
        if self.trace_dir is None or self._previous_path is None or not self.stages:
            raise RuntimeError("unchanged() requires a disk-backed previous stage")

        previous_record = self.stages[-1]
        index = len(self.stages)
        basename = "%03d_%s" % (index, _slug(str(name)))
        checkpoint_path = self.trace_dir / "checkpoints" / (basename + ".tif")
        preview_path = self.trace_dir / "previews" / (basename + ".png")
        previous_preview = self.trace_dir / previous_record["artifacts"]["preview_png"]
        checkpoint_storage = self._link_or_copy(self._previous_path, checkpoint_path)
        preview_storage = self._link_or_copy(previous_preview, preview_path)

        record: Dict[str, Any] = {
            key: copy.deepcopy(previous_record[key])
            for key in (
                "dimensions",
                "source_dtype",
                "pixel_hash",
                "statistics",
                "native_statistics",
                "domain",
                "encoding",
                "channel_order",
                "reference_comparable",
                "checkpoint_value_range",
            )
        }
        record.update(
            {
                "index": index,
                "name": str(name),
                "status": str(status),
                "applied": bool(applied),
                "skipped_reason": None if applied else str(reason),
                "provenance": _json_value(provenance),
                "confidence": _json_value(confidence),
                "params": _json_value(
                    {
                        **(params or {}),
                        **({"unchanged_reason": str(reason)} if applied else {}),
                    }
                ),
                "difference_from_previous": (
                    _empty_comparison(
                        "stage was not executed; checkpoint pixels are carried forward"
                    )
                    if _is_unexecuted_status(str(status))
                    else _identical_comparison(
                        record["dimensions"]["shape"],
                        tuple(record["checkpoint_value_range"]),
                    )
                ),
                "difference_from_reference": copy.deepcopy(
                    previous_record["difference_from_reference"]
                ),
                "difference_from_input": copy.deepcopy(
                    previous_record["difference_from_input"]
                ),
                "artifacts": {
                    "checkpoint_tiff": _relative(checkpoint_path, self.trace_dir),
                    "preview_png": _relative(preview_path, self.trace_dir),
                    "checkpoint_storage": checkpoint_storage,
                    "preview_storage": preview_storage,
                },
            }
        )

        if not _is_unexecuted_status(str(status)):
            preview = Image.open(str(preview_path))
            try:
                zero = Image.new("RGB", preview.size, color=(0, 0, 0))
            finally:
                preview.close()
            for suffix, artifact_key in (
                ("rgb", "previous_diff_rgb_png"),
                ("heatmap", "previous_diff_heatmap_png"),
                ("signed", "previous_diff_signed_png"),
            ):
                path = (
                    self.trace_dir
                    / "diffs"
                    / (basename + "_vs_previous_" + suffix + ".png")
                )
                zero.save(str(path), format="PNG")
                record["artifacts"][artifact_key] = _relative(path, self.trace_dir)

        self.stages.append(record)
        self._previous_path = checkpoint_path
        self._previous_q = None
        self._write_reports()
        self._finalized = False
        return record

    def _write_json(self) -> None:
        assert self.trace_dir is not None
        path = self.trace_dir / "manifest.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                self.manifest,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")

    @staticmethod
    def _comparison_csv(
        record: Dict[str, Any], prefix: str, row: Dict[str, Any]
    ) -> None:
        comparison = record[prefix]
        column_prefix = prefix.removeprefix("difference_from_")
        row[column_prefix + "_status"] = comparison.get("status")
        if comparison.get("status") != "ok":
            row[column_prefix + "_reason"] = comparison.get("reason")
            return
        row[column_prefix + "_luma_ncc"] = comparison.get("luma_ncc")
        row[column_prefix + "_luma_ncc_status"] = comparison.get("luma_ncc_status")
        row[column_prefix + "_changed_pixel_fraction"] = comparison.get(
            "changed_pixel_fraction"
        )
        for channel in _CHANNELS:
            row["%s_%s_signed_bias" % (column_prefix, channel)] = comparison.get(
                "signed_bias", {}
            ).get(channel)
        for scope, values in (("overall", comparison["overall"]),) + tuple(
            (channel, comparison["channels"][channel]) for channel in _CHANNELS
        ):
            for metric in ("mae", "rmse", "max_abs", "p95_abs", "p99_abs", "psnr_db"):
                row["%s_%s_%s" % (column_prefix, scope, metric)] = values.get(metric)
            row["%s_%s_psnr_status" % (column_prefix, scope)] = values.get(
                "psnr_status"
            )

    def _csv_rows(self) -> Tuple[Sequence[str], Sequence[Dict[str, Any]]]:
        base_fields = [
            "index",
            "name",
            "status",
            "applied",
            "skipped_reason",
            "provenance",
            "confidence",
            "params",
            "domain",
            "encoding",
            "channel_order",
            "reference_comparable",
            "native_statistics",
            "height",
            "width",
            "channels",
            "source_dtype",
            "sha256_u16_le",
        ]
        statistic_fields = []
        for channel in _CHANNELS:
            statistic_fields.extend(
                "%s_%s" % (channel, metric)
                for metric in ("min", "max", "mean", "std", "p01", "p50", "p99")
            )
            statistic_fields.extend(
                "%s_clip_%s" % (channel, metric)
                for metric in (
                    "low_count",
                    "high_count",
                    "either_count",
                    "low_fraction",
                    "high_fraction",
                    "either_fraction",
                )
            )
        comparison_fields = []
        for prefix in ("previous", "input", "reference"):
            comparison_fields.extend(
                (
                    prefix + "_status",
                    prefix + "_reason",
                    prefix + "_luma_ncc",
                    prefix + "_luma_ncc_status",
                    prefix + "_changed_pixel_fraction",
                )
            )
            comparison_fields.extend(
                "%s_%s_signed_bias" % (prefix, channel) for channel in _CHANNELS
            )
            for scope in ("overall",) + _CHANNELS:
                comparison_fields.extend(
                    "%s_%s_%s" % (prefix, scope, metric)
                    for metric in (
                        "mae",
                        "rmse",
                        "max_abs",
                        "p95_abs",
                        "p99_abs",
                        "psnr_db",
                        "psnr_status",
                    )
                )
        fields = base_fields + statistic_fields + comparison_fields

        rows = []
        for record in self.stages:
            dimensions = record["dimensions"]
            row: Dict[str, Any] = {
                "index": record["index"],
                "name": record["name"],
                "status": record["status"],
                "applied": record["applied"],
                "skipped_reason": record["skipped_reason"],
                "provenance": json.dumps(
                    record["provenance"], sort_keys=True, separators=(",", ":")
                ),
                "confidence": json.dumps(
                    record["confidence"], sort_keys=True, separators=(",", ":")
                ),
                "params": json.dumps(
                    record["params"], sort_keys=True, separators=(",", ":")
                ),
                "domain": record["domain"],
                "encoding": record["encoding"],
                "channel_order": record["channel_order"],
                "reference_comparable": record["reference_comparable"],
                "native_statistics": json.dumps(
                    record["native_statistics"], sort_keys=True, separators=(",", ":")
                ),
                "height": dimensions["height"],
                "width": dimensions["width"],
                "channels": dimensions["channels"],
                "source_dtype": record["source_dtype"],
                "sha256_u16_le": record["pixel_hash"]["value"],
            }
            for channel in _CHANNELS:
                statistics = record["statistics"]["channels"][channel]
                for metric in ("min", "max", "mean", "std", "p01", "p50", "p99"):
                    row["%s_%s" % (channel, metric)] = statistics[metric]
                for metric, value in statistics["clipping"].items():
                    row["%s_clip_%s" % (channel, metric)] = value
            self._comparison_csv(record, "difference_from_previous", row)
            self._comparison_csv(record, "difference_from_input", row)
            self._comparison_csv(record, "difference_from_reference", row)
            rows.append(row)
        return fields, rows

    def _write_csv(self) -> None:
        assert self.trace_dir is not None
        fields, rows = self._csv_rows()
        with (self.trace_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _write_reports(self) -> None:
        self._write_csv()
        self._write_json()

    @staticmethod
    def _mae_label(comparison: Dict[str, Any], prefix: str) -> str:
        if comparison.get("status") == "ok":
            return "%s %.6f" % (prefix, comparison["overall"]["mae"])
        return "%s %s" % (prefix, comparison.get("status", "n/a"))

    def _write_contact_sheet(self) -> Optional[Path]:
        if self.trace_dir is None or not self.stages:
            return None
        cards = []
        if self._reference_record is not None:
            reference_preview = self._reference_record.get("artifacts", {}).get(
                "preview_png"
            )
            if reference_preview:
                cards.append(
                    (
                        self.trace_dir / reference_preview,
                        "REFERENCE: " + self.reference_name,
                        "comparison target",
                    )
                )
        for record in self.stages:
            previous_text = self._mae_label(
                record["difference_from_previous"], "prev MAE"
            )
            reference_text = self._mae_label(
                record["difference_from_reference"], "ref MAE"
            )
            cards.append(
                (
                    self.trace_dir / record["artifacts"]["preview_png"],
                    "%03d %s [%s]"
                    % (record["index"], record["name"], record["status"]),
                    previous_text + " | " + reference_text,
                )
            )

        columns = min(3, len(cards))
        rows = int(math.ceil(len(cards) / columns))
        padding = 12
        label_height = 38
        card_width = self.preview_size[0] + padding * 2
        card_height = self.preview_size[1] + label_height + padding * 2
        sheet = Image.new(
            "RGB",
            (columns * card_width, rows * card_height),
            color=(24, 24, 24),
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for index, (preview_path, title, subtitle) in enumerate(cards):
            column = index % columns
            row = index // columns
            left = column * card_width
            top = row * card_height
            with Image.open(str(preview_path)) as opened:
                preview = opened.convert("RGB")
                image_left = (
                    left + padding + (self.preview_size[0] - preview.width) // 2
                )
                image_top = top + padding + (self.preview_size[1] - preview.height) // 2
                sheet.paste(preview, (image_left, image_top))
            text_top = top + padding + self.preview_size[1] + 4
            draw.text(
                (left + padding, text_top), title, fill=(245, 245, 245), font=font
            )
            draw.text(
                (left + padding, text_top + 15),
                subtitle,
                fill=(180, 180, 180),
                font=font,
            )
        output = self.trace_dir / "contact_sheet.png"
        sheet.save(str(output), format="PNG")
        return output

    def finalize(self) -> Dict[str, Any]:
        """Write the contact sheet and final reports; safe to call repeatedly."""

        if self.trace_dir is not None and not self._finalized:
            self._write_contact_sheet()
            self._write_reports()
        self._finalized = True
        return self.manifest

    close = finalize


__all__ = ["StageTrace", "compare_u16_rgb", "write_comparison_artifacts"]
