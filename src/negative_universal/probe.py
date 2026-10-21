"""Clear-film base probing from the frame border.

The base of a negative is the clear film beside the picture — it is the thing
that divides out, so it has to be found, not invented. This module tiles the
border into candidate patches, scores each one on how much it looks like bare
film (flat, neutral, bright), and keeps the best few. Patches that landed on
image content score badly and drop out, which is what makes the ranking worth
having over a single hand-placed box.

The same patches double as illumination samples: their positions and per
channel medians are least-squares fitted to a plane, so falloff across the
gate is measured on bare film instead of on the picture.

Only the border ring is searched. A frame with no border (full-bleed scan)
needs an explicit ROI, which `probe_base` accepts as an override.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence

CHANNELS = ("R", "G", "B")

# Ranking weights. Chroma spread dominates: a patch with a colour cast is not
# bare film no matter how quiet it is.
_W_CHROMA = 12.0
_W_NOISE = 6.0
_W_LUMA = 2.0
_MIN_LUMA = 0.02
_MAX_SAMPLES = 2048


class ProbeError(Exception):
    """Raised when no usable base patch exists."""


@dataclass(frozen=True)
class Patch:
    """One candidate base patch, sampled at low resolution."""

    x: int
    y: int
    width: int
    height: int
    medians: tuple[float, float, float]
    noise: float
    luma: float

    @property
    def chroma(self) -> float:
        return max(self.medians) - min(self.medians)

    def cost(self, target_luma: float) -> float:
        relative_chroma = self.chroma / max(self.luma, _MIN_LUMA)
        relative_noise = self.noise / max(self.luma, _MIN_LUMA)
        luma_gap = abs(self.luma - target_luma)
        return (
            _W_CHROMA * relative_chroma
            + _W_NOISE * relative_noise
            + _W_LUMA * luma_gap
        )


@dataclass(frozen=True)
class BaseEstimate:
    """Chosen clear-film base plus the illumination planes fitted through it.

    `plane[channel]` is `(offset, x_slope, y_slope)` in normalised coordinates
    (`u`, `v` in -1..1), scaled so the plane averages 1.0 over the frame.
    """

    base: tuple[float, float, float]
    plane: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    patches: tuple[Patch, ...]
    chosen: tuple[Patch, ...]
    source: str


def _patch_at(
    image: Mapping[str, object], x: int, y: int, width: int, height: int
) -> Patch:
    frame_width = image["R"].width  # type: ignore[attr-defined]
    frame_height = image["R"].height  # type: ignore[attr-defined]
    x = min(max(x, 0), frame_width - 1)
    y = min(max(y, 0), frame_height - 1)
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    medians: list[float] = []
    squares: list[float] = []
    stride_x = max(1, width // 32)
    stride_y = max(1, height // 32)
    for channel in CHANNELS:
        pixels = image[channel].pixels  # type: ignore[attr-defined]
        samples = array("f")
        for row_index in range(y, y + height, stride_y):
            row = row_index * frame_width
            for column in range(x, x + width, stride_x):
                samples.append(pixels[row + column] / 65535.0)
                if len(samples) >= _MAX_SAMPLES:
                    break
        if not samples:
            raise ProbeError("empty probe patch")
        mean = sum(samples) / len(samples)
        variance = sum(value * value for value in samples) / len(samples) - mean * mean
        squares.append(max(0.0, variance) ** 0.5)
        ordered = sorted(samples)
        medians.append(ordered[len(ordered) // 2])
    noise = sum(squares) / 3.0
    luma = 0.299 * medians[0] + 0.587 * medians[1] + 0.114 * medians[2]
    return Patch(
        x,
        y,
        width,
        height,
        (medians[0], medians[1], medians[2]),
        noise,
        luma,
    )


def border_candidates(
    image: Mapping[str, object], *, band_fraction: float = 0.02, tiles: int = 8
) -> list[Patch]:
    """Tile the border ring outside the picture into candidate patches."""
    frame_width = image["R"].width  # type: ignore[attr-defined]
    frame_height = image["R"].height  # type: ignore[attr-defined]
    band = max(6, int(round(min(frame_width, frame_height) * band_fraction)))
    band = min(band, max(6, min(frame_width, frame_height) // 4))
    step_x = max(1, (frame_width - 2 * band) // tiles)
    step_y = max(1, (frame_height - 2 * band) // tiles)
    patches: list[Patch] = []
    for index in range(tiles):
        x = band + index * step_x
        y = band + index * step_y
        patches.append(_patch_at(image, x, 0, step_x, band))
        patches.append(_patch_at(image, x, frame_height - band, step_x, band))
        patches.append(_patch_at(image, 0, y, band, step_y))
        patches.append(_patch_at(image, frame_width - band, y, band, step_y))
    return patches


def _fit_plane(samples: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float]:
    """Least-squares `a + b*u + c*v` with normalised coordinates, mean 1.0."""
    if len(samples) < 3:
        return (1.0, 0.0, 0.0)
    count = len(samples)
    su = sum(sample[0] for sample in samples)
    sv = sum(sample[1] for sample in samples)
    suu = sum(sample[0] * sample[0] for sample in samples)
    svv = sum(sample[1] * sample[1] for sample in samples)
    suv = sum(sample[0] * sample[1] for sample in samples)
    sy = sum(sample[2] for sample in samples)
    suy = sum(sample[0] * sample[2] for sample in samples)
    svy = sum(sample[1] * sample[2] for sample in samples)
    # normal equations, solved by Cramer's rule; fall back to flat on a
    # degenerate (collinear) layout
    matrix = [[count, su, sv], [su, suu, suv], [sv, suv, svv]]
    rhs = [sy, suy, svy]
    det = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(det) < 1e-12:
        return (max(1e-6, sy / count), 0.0, 0.0)
    solution: list[float] = []
    for column in range(3):
        substituted = [list(row) for row in matrix]
        for row_index in range(3):
            substituted[row_index][column] = rhs[row_index]
        numerator = (
            substituted[0][0]
            * (substituted[1][1] * substituted[2][2] - substituted[1][2] * substituted[2][1])
            - substituted[0][1]
            * (substituted[1][0] * substituted[2][2] - substituted[1][2] * substituted[2][0])
            + substituted[0][2]
            * (substituted[1][0] * substituted[2][1] - substituted[1][1] * substituted[2][0])
        )
        solution.append(numerator / det)
    offset, x_slope, y_slope = solution
    centre = offset  # u = v = 0 at the frame centre
    if centre <= 1e-6:
        return (1.0, 0.0, 0.0)
    return (1.0, x_slope / centre, y_slope / centre)


def _side(patch: Patch, width: int, height: int) -> str:
    distances = {
        "top": patch.y,
        "bottom": height - (patch.y + patch.height),
        "left": patch.x,
        "right": width - (patch.x + patch.width),
    }
    return min(distances, key=distances.__getitem__)


def probe_base(
    image: Mapping[str, object],
    *,
    keep: int = 5,
    band_fraction: float = 0.02,
    roi: tuple[int, int, int, int] | None = None,
) -> BaseEstimate:
    """Rank border patches and return the base plus its illumination planes.

    `roi` is `(x, y, width, height)` in normalised-image pixels, for frames
    with no clear border.
    """
    if roi is not None:
        x, y, width, height = roi
        if width <= 0 or height <= 0:
            raise ProbeError("base ROI must have positive width and height")
        frame_width = image["R"].width  # type: ignore[attr-defined]
        frame_height = image["R"].height  # type: ignore[attr-defined]
        if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise ProbeError("base ROI falls outside the frame")
        patches = [_patch_at(image, x, y, width, height)]
        chosen = tuple(patches)
    else:
        patches = border_candidates(image, band_fraction=band_fraction)
        usable = [patch for patch in patches if patch.luma >= _MIN_LUMA]
        if not usable:
            raise ProbeError(
                "no clear-film patch on the border; pass an explicit base ROI"
            )
        lumas = sorted(patch.luma for patch in usable)
        target = lumas[min(len(lumas) - 1, int(len(lumas) * 0.9))]
        frame_width = image["R"].width  # type: ignore[attr-defined]
        frame_height = image["R"].height  # type: ignore[attr-defined]
        # One or two winners per side: a base picked only from the top edge
        # would leave the falloff plane degenerate.
        per_side = max(1, keep // 4)
        buckets: dict[str, list[Patch]] = {"top": [], "bottom": [], "left": [], "right": []}
        for patch in usable:
            buckets[_side(patch, frame_width, frame_height)].append(patch)
        chosen_patches: list[Patch] = []
        for side in ("top", "bottom", "left", "right"):
            ranked = sorted(buckets[side], key=lambda patch: patch.cost(target))
            chosen_patches.extend(ranked[:per_side])
        chosen = tuple(chosen_patches)

    frame_width = image["R"].width  # type: ignore[attr-defined]
    frame_height = image["R"].height  # type: ignore[attr-defined]
    base = base_from_patches(chosen)
    if min(base) <= 0.0:
        raise ProbeError("clear-film base is black")
    samples = [
        (
            (patch.x + patch.width / 2) / frame_width * 2.0 - 1.0,
            (patch.y + patch.height / 2) / frame_height * 2.0 - 1.0,
            patch.medians[channel] / base[channel],
        )
        for patch in chosen
        for channel in range(3)
    ]
    plane = tuple(_fit_plane(samples) for _ in range(3))
    return BaseEstimate(
        base=(base[0], base[1], base[2]),
        plane=(plane[0], plane[1], plane[2]),
        patches=tuple(patches),
        chosen=chosen,
        source="roi" if roi is not None else "border",
    )


def base_from_patches(patches: Iterable[Patch]) -> tuple[float, float, float]:
    collected = tuple(patches)
    if not collected:
        raise ProbeError("no patches supplied")
    return tuple(
        median(patch.medians[channel] for patch in collected) for channel in range(3)
    )
