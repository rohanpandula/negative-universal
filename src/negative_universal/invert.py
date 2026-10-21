"""Density-space inversion: transmittance -> optical density -> print.

A C-41 negative is measured as transmittance and converted to optical density
`d = -log10(T)`. Density is the domain the whole contract is built on: exposure
is an additive offset, a channel imbalance is an additive per-channel offset,
contrast is a scale about a pivot, and none of those touch what the scanner
measured until the look is applied at the very end.

The clear-film base is divided out as transmittance, which makes the base
exactly `d = 0` — no separate subtraction step and no chance of the base being
applied with the wrong sign. Falloff measured on the base patches is folded
into the same division as a per-pixel plane.

Everything above the look is calibration. The look is a monotone LUT in
density with a fixed domain (`DENSITY_CEILING` = 4.0, i.e. 4 decades of
transmittance), so a look can be swapped without moving a single density.
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import dataclass
from math import log10
from typing import Mapping, Sequence

from .probe import CHANNELS, BaseEstimate, ProbeError

# One decade per unit; 4.0 covers T = 1e-4, comfortably past any film's
# maximum density and enough headroom that clipping is a look decision.
DENSITY_CEILING = 4.0
# Highlights thinner than the base have negative density, so the LUT domain
# starts below zero rather than being clipped at it.
DENSITY_FLOOR = -0.6
DENSITY_SPAN = DENSITY_CEILING - DENSITY_FLOOR
MIN_TRANSMITTANCE = 1e-4
LUT_SIZE = 4096
LUMA = (0.299, 0.587, 0.114)


class InvertError(Exception):
    """Raised when a render cannot be produced."""


@dataclass(frozen=True)
class LookParams:
    """A print curve expressed purely in density.

    `white_density` is the density printed as paper white and `black_density`
    is the deepest density; everything outside is clipped. `contrast` steepens
    the shoulder, `toe` lifts the shadow end without moving black.
    """

    contrast: float = 0.5
    toe: float = 0.06
    white_density: float = 0.05
    black_density: float = 1.7

    def validate(self) -> None:
        if not 0.0 <= self.contrast <= 1.0:
            raise InvertError("contrast must be in 0..1")
        if not 0.0 <= self.toe <= 0.5:
            raise InvertError("toe must be in 0..0.5")
        if self.black_density <= self.white_density:
            raise InvertError("black density must exceed white density")
        if self.black_density > DENSITY_CEILING:
            raise InvertError(f"black density must be at most {DENSITY_CEILING}")


@dataclass(frozen=True)
class RenderParams:
    """Per-frame correction plus the look. Correction first, look last."""

    look: LookParams = LookParams()
    density_offset: float = 0.0
    gray: tuple[float, float, float] = (0.0, 0.0, 0.0)
    neutral_pull: float = 0.0

    def validate(self) -> None:
        self.look.validate()
        if not -2.0 <= self.density_offset <= 2.0:
            raise InvertError("density offset must be in -2..2 decades")
        if len(self.gray) != 3:
            raise InvertError("gray balance needs three channel offsets")
        for offset in self.gray:
            if not -1.0 <= offset <= 1.0:
                raise InvertError("gray offsets must be in -1..1 decades")
        if not 0.0 <= self.neutral_pull <= 1.0:
            raise InvertError("neutral pull must be in 0..1")


@dataclass(frozen=True)
class RenderStats:
    clipped_transmittance: int
    density_floor: float
    density_ceiling: float
    pixels: int


def _srgb(value: float) -> float:
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * value ** (1 / 2.4) - 0.055


def build_display_lut(look: LookParams, *, max_value: int = 65535) -> tuple[float, ...]:
    """Map a density index (0..LUT_SIZE, 4 decades) to an encoded sample.

    The curve is a normalised shoulder: `y(0) = 0`, `y(1) = 1`, monotone for
    every gamma, so black and white land exactly on the endpoints.
    """
    look.validate()
    gamma = 1.0 + 2.0 * look.contrast
    shoulder = 0.35
    span = look.black_density - look.white_density
    values: list[float] = []
    for index in range(LUT_SIZE + 1):
        density = DENSITY_FLOOR + DENSITY_SPAN * index / LUT_SIZE
        position = (density - look.white_density) / span
        position = min(max(position, 0.0), 1.0)
        if position <= 0.0:
            rendered = 0.0
        elif position >= 1.0:
            rendered = 1.0
        else:
            powered = position**gamma
            rendered = powered * (1.0 + shoulder) / (powered + shoulder)
            if look.toe:
                rendered += look.toe * (rendered**0.5 - rendered)
        values.append(round(_srgb(min(max(rendered, 0.0), 1.0)) * max_value))
    values[0] = 0.0
    values[-1] = float(max_value)
    return tuple(values)


def build_density_table() -> tuple[float, ...]:
    """Density for every 16-bit transmittance sample, clamped at the floor.

    One table build replaces a `log10` per pixel per channel.
    """
    floor = MIN_TRANSMITTANCE
    return tuple(
        -log10(max(value / 65535.0, floor)) for value in range(65536)
    )


def _plane_value(coefficients: Sequence[float], u: float, v: float) -> float:
    offset, x_slope, y_slope = coefficients
    value = offset + x_slope * u + y_slope * v
    return min(max(value, 0.5), 2.0)


def sample_densities(
    images: Mapping[str, object],
    estimate: BaseEstimate,
    *,
    target_samples: int = 200_000,
) -> dict[str, list[float]]:
    """Per-channel density samples for auto-ranging (stride subsampled)."""
    width = images["R"].width
    height = images["R"].height
    stride = max(1, int((width * height / max(target_samples, 1)) ** 0.5))
    table = build_density_table()
    anchors = tuple(value * 65535.0 for value in estimate.base)
    planes = estimate.plane
    samples: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    for y in range(0, height, stride):
        v = (y + 0.5) / height * 2.0 - 1.0
        for x in range(0, width, stride):
            u = (x + 0.5) / width * 2.0 - 1.0
            index = y * width + x
            for channel_index, channel in enumerate(CHANNELS):
                pixels = images[channel].pixels
                corrected = (pixels[index]) / (
                    anchors[channel_index] * _plane_value(planes[channel_index], u, v)
                )
                if corrected >= 1.0:
                    density = -log10(corrected)
                else:
                    density = table[int(corrected * 65535.0 + 0.5)]
                samples[channel].append(density)
    return samples


def auto_range(samples: Mapping[str, Sequence[float]]) -> tuple[float, float]:
    """Paper white and black from the frame's own density percentiles."""
    pooled = sorted(value for values in samples.values() for value in values)
    if not pooled:
        raise InvertError("no density samples to auto-range on")
    # p2/p98 rather than the extremes: a scan's black surround and a few
    # specular holes would otherwise drag the whole frame to mud.
    white = pooled[int(len(pooled) * 0.02)]
    black = pooled[min(len(pooled) - 1, int(len(pooled) * 0.98))]
    white = min(max(white, DENSITY_FLOOR + 0.05), 0.5)
    black = min(2.6, max(black, white + 0.8))
    return white, black


def render(
    images: Mapping[str, object],
    estimate: BaseEstimate,
    params: RenderParams,
    *,
    max_value: int = 65535,
) -> tuple[dict[str, object], RenderStats]:
    """Render a normalized triplet to a positive, per channel.

    `images` are 16-bit `normalized_transmittance` frames; `estimate` comes
    from `probe.probe_base` so the base and its falloff planes are measured
    rather than assumed.
    """
    params.validate()
    if not 1 <= max_value <= 65535:
        raise InvertError("max value must be in 1..65535")
    reference = images["R"]
    width = reference.width
    height = reference.height
    for channel in CHANNELS:
        image = images[channel]
        if (image.width, image.height, image.max_value) != (width, height, 65535):
            raise InvertError(f"{channel} frame does not match the triplet geometry")

    density_table = build_density_table()
    display = build_display_lut(params.look, max_value=max_value)
    scale = LUT_SIZE / DENSITY_SPAN
    base_luma = sum(
        coefficient * luma
        for coefficient, luma in zip(estimate.base, LUMA, strict=True)
    )
    if base_luma <= 0.0:
        raise ProbeError("clear-film base is black")
    base_density = -log10(min(max(base_luma, MIN_TRANSMITTANCE), 1.0))
    anchors = tuple(base * 65535.0 for base in estimate.base)
    shifts = tuple(
        params.gray[channel] + params.density_offset for channel in range(3)
    )
    pull = params.neutral_pull
    largest = float(max_value)

    outputs = {channel: array("H") for channel in CHANNELS}
    pixels_r = images["R"].pixels
    pixels_g = images["G"].pixels
    pixels_b = images["B"].pixels
    plane_r, plane_g, plane_b = estimate.plane
    clips = 0
    lowest = DENSITY_CEILING
    highest = 0.0

    for y in range(height):
        v = (y + 0.5) / height * 2.0 - 1.0
        row = y * width
        out_r = outputs["R"]
        out_g = outputs["G"]
        out_b = outputs["B"]
        for x in range(width):
            u = (x + 0.5) / width * 2.0 - 1.0
            index = row + x
            densities: list[float] = []
            for channel, pixels, plane, target in (
                (0, pixels_r, plane_r, out_r),
                (1, pixels_g, plane_g, out_g),
                (2, pixels_b, plane_b, out_b),
            ):
                transmittance = pixels[index] / 65535.0
                if transmittance > 1.0:
                    clips += 1
                corrected = (transmittance * 65535.0) / (
                    anchors[channel] * _plane_value(plane, u, v)
                )
                if corrected >= 1.0:
                    # above the base, so the density is negative: the table is
                    # only indexed on transmittance, so take the log directly
                    density = -log10(corrected)
                else:
                    density = density_table[int(corrected * 65535.0 + 0.5)]
                density += shifts[channel]
                densities.append(density)
                position = (density - DENSITY_FLOOR) * scale
                if position <= 0.0:
                    target.append(0)
                    continue
                if position >= LUT_SIZE:
                    target.append(int(display[LUT_SIZE]))
                    continue
                position += 0.5
                integer = int(position)
                fraction = position - integer
                low = display[integer]
                high = display[integer + 1]
                target.append(int(low + (high - low) * fraction))

            luma = (
                LUMA[0] * densities[0]
                + LUMA[1] * densities[1]
                + LUMA[2] * densities[2]
            )
            lowest = min(lowest, luma)
            highest = max(highest, luma)

            if pull:
                # Shadow neutrality: pull against the density-remapped luma,
                # strongest at the base, vanishing into the highlights.
                weight = pull * min(max((luma - base_density) / 1.5, 0.0), 1.0)
                if weight > 0.0:
                    luma_sample = (
                        LUMA[0] * out_r[index]
                        + LUMA[1] * out_g[index]
                        + LUMA[2] * out_b[index]
                    )
                    for target in (out_r, out_g, out_b):
                        value = target[index]
                        target[index] = int(
                            min(
                                max(
                                    luma_sample + (value - luma_sample) * (1.0 - weight),
                                    0.0,
                                ),
                                largest,
                            )
                        )

    return outputs, RenderStats(
        clipped_transmittance=clips,
        density_floor=lowest,
        density_ceiling=highest,
        pixels=width * height,
    )


def write_ppm(path, outputs: Mapping[str, object], width: int, height: int) -> None:
    """Write a 16-bit binary PPM so the render is viewable without a decoder."""
    header = f"P6\n{width} {height}\n65535\n".encode("ascii")
    samples = array("H")
    for index in range(width * height):
        samples.append(outputs["R"][index])
        samples.append(outputs["G"][index])
        samples.append(outputs["B"][index])
    if sys.byteorder == "little":
        samples.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(header)
        output.write(samples.tobytes())
