"""Installer-derived Noritsu LS-600/RD5X processing primitives.

This module intentionally separates exact kernels from the parameters that a
scanner or EZ Controller job supplies at runtime.  A function is only called
"exact" when its integer order, clipping, table indexing, and channel order are
visible in the shipped x86 implementation.  Callers remain responsible for
labelling inferred or externally fitted parameters in their trace.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


U12_MAX = 4095
U16_MAX = 65535
GAMMA_SEAMS = (8000, 60000)
GAMMA_ANCHOR_X = 0x5F00
UO_RAW_X = (
    0, 1600, 2400, 3200, 4000, 4800, 5600, 6400, 7200, 8000, 8800, 9600,
    11200, 12800, 14400, 16000, 17600, 19200, 20800, 22400, 24000, 25600,
    27200, 28800, 30400, 32000, 33600, 35200, 36800, 38400, 40000, 41600,
    43200, 44800, 46400, 48000, 49600, 51200, 52800, 54400, 56000, 57600,
    59200, 60800, 62400, 64000, 65535,
)
UO_RAW_Y = (
    UO_RAW_X,
    (
        0, 16, 24, 32, 40, 768, 2016, 3360, 4644, 6000, 7260, 8640,
        10640, 12672, 14400, 16000, 17600, 19200, 20800, 22400, 24000,
        25600, 27200, 28800, 30400, 32000, 33600, 35200, 36800, 38400,
        40000, 41600, 43200, 44800, 46400, 48000, 49600, 51200, 52800,
        54400, 56000, 57600, 59200, 60800, 62400, 64000, 65535,
    ),
    (
        0, 16, 24, 32, 960, 2280, 3612, 4774, 5940, 6980, 7929, 8928,
        10864, 12672, 14400, 16000, 17600, 19200, 20800, 22400, 24000,
        25600, 27200, 28800, 30400, 32000, 33600, 35200, 36800, 38400,
        40000, 41600, 43200, 44800, 46400, 48000, 49600, 51200, 52800,
        54400, 56000, 57600, 59200, 60800, 62400, 64000, 65535,
    ),
    (
        0, 16, 24, 768, 1980, 3240, 4452, 5568, 6624, 7520, 8448, 9408,
        11088, 12800, 14400, 16000, 17600, 19200, 20800, 22400, 24000,
        25600, 27200, 28800, 30400, 32000, 33600, 35200, 36800, 38400,
        40000, 41600, 43200, 44800, 46400, 48000, 49600, 51200, 52800,
        54400, 56000, 57600, 59200, 60800, 62400, 64000, 65535,
    ),
)
Q10000_FLOAT = np.float32(9.999999747378752e-05)
CCD_NOISE_RESPONSE_SCALE_F32 = np.float32(3.9215687138494104e-05)
CCD_NOISE_EDGE_SCALE_F32 = np.float32(0.00015259021893143654)
ONE_THIRD_F32 = np.float32(0.3333333432674408)
GRAIN_CURVE_TO_Q15_F64 = np.float64(8.001953601953602)
GRAIN_BYTE_TO_Q15_F64 = np.float64(128.50196078431372)
YCC_CORRECTION_SCALE_F32 = np.array(
    (0x3883126F,), dtype=np.uint32
).view(np.float32)[0]
# Exact float words at UniRdImgLib.dll VA 0x1018d848.  The index bits are
# ``C0 terminal, C1 terminal, Y terminal``; terminal cells span 127/511
# integer steps instead of the ordinary 128/512.
HUE_SAT_S0_INTERPOLATION_SCALE_F32 = np.array(
    (
        0x34000000,
        0x34004020,
        0x34010204,
        0x340142A5,
        0x34010204,
        0x340142A5,
        0x34020610,
        0x34024734,
    ),
    dtype=np.uint32,
).view(np.float32)

# Static words copied from the group-0/type-1 normal-film CorrectServ bank.
# They are tuples so callers cannot mutate shared process state.  The forward
# matrix is at NEGA_135_COL_calc.dat +0x1662.  The inverse matrix and its
# normal-path variant byte (1) are at +0xf7a6 and +0xf7b8 respectively.
LS600_G0_TYPE1_RGB_TO_YCC_Q10000 = (
    3733,
    2533,
    3734,
    -1340,
    4200,
    -2860,
    5000,
    -1840,
    -3160,
)
LS600_G0_TYPE1_YCC_TO_RGB_Q10000 = (
    10000,
    -3334,
    12441,
    10000,
    12439,
    -3332,
    10000,
    -9109,
    -9107,
)
LS600_G0_TYPE1_YCC_TO_RGB_VARIANT = 1

# ``RD5X_FILM_SHARP_PARA`` is copied verbatim into the final process-control
# block.  In the captured normal-film PCB it begins at absolute offset
# ``+0x3b2`` and occupies exactly 0x154 bytes.
FILM_SHARP_PARAMETER_BLOCK_SIZE = 0x154
FILM_SHARP_PARAMETER_PCB_OFFSET = 0x3B2


def _as_u16_rgb(image: np.ndarray, name: str = "image") -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if array.dtype != np.uint16:
        raise TypeError(f"{name} must be uint16")
    return array


def _as_u8_rgb(image: np.ndarray, name: str = "image") -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8")
    return array


def _signed_q12(values: np.ndarray, name: str) -> np.ndarray:
    """Interpret an integer coefficient array as the installer's signed i16."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer signed-Q12 coefficients")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < -32768 or maximum > 65535:
            raise ValueError(f"{name} values must fit a signed/encoded 16-bit word")
    # Scanner CSV/binary loaders may expose the two's-complement words as
    # uint16.  Round-trip through uint16 so 0xffff is -1, never +65535.
    return array.astype(np.uint16, copy=False).view(np.int16)


def _q10000_coefficients(matrix_q10000: np.ndarray) -> np.ndarray:
    """Convert nine signed words as the vendor's x87 setup functions do."""

    words = _signed_q12(matrix_q10000, "matrix_q10000")
    if words.shape != (9,):
        raise ValueError("matrix_q10000 must contain exactly 9 words")
    # The DLL loads its single-precision 1/10000 constant into x87, performs
    # the multiplication there, then stores each coefficient as float32.
    return np.asarray(
        words.astype(np.float64) * float(Q10000_FLOAT), dtype=np.float32
    )


def _require_positive_chunk_rows(chunk_rows: int) -> int:
    if not isinstance(chunk_rows, (int, np.integer)) or int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be a positive integer")
    return int(chunk_rows)


def _trunc_divide_signed(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Vectorized signed integer division with x86 ``idiv`` truncation."""

    top = np.asarray(numerator, dtype=np.int64)
    bottom = np.asarray(denominator, dtype=np.int64)
    if np.any(bottom == 0):
        raise ZeroDivisionError("vendor integer division denominator is zero")
    quotient = np.abs(top) // np.abs(bottom)
    return np.where((top < 0) != (bottom < 0), -quotient, quotient)


def _trunc_divide_scalar(numerator: int, denominator: int) -> int:
    """Scalar signed division with x86 ``idiv`` truncation."""

    if denominator == 0:
        raise ZeroDivisionError("vendor integer division denominator is zero")
    quotient = abs(int(numerator)) // abs(int(denominator))
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


def _wrap_signed_i32(value: int) -> int:
    """Return the signed low word produced by a 32-bit x86 integer operation."""

    return (int(value) + 0x80000000) % 0x100000000 - 0x80000000


def _wrap_signed_i16(value: int) -> int:
    return (int(value) + 0x8000) % 0x10000 - 0x8000


def _mul_f32(values: np.ndarray, coefficient: np.float32) -> np.ndarray:
    return np.multiply(values, coefficient, dtype=np.float32)


def _add_f32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.add(left, right, dtype=np.float32)


def _pack_signed_f32_to_u16(values: np.ndarray) -> np.ndarray:
    """Model ``cvtps2dq; packssdw; pxor 0x8000`` for finite values."""

    rounded = np.rint(np.asarray(values, dtype=np.float32)).astype(np.int64)
    return (np.clip(rounded, -32768, 32767) + 32768).astype(np.uint16)


def apply_trimming_u16(
    image_rgb: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Port the visible three lanes of ``RD5X_TRIMMING_PARA``.

    The structure consumed by ``FmScalingS0/S2_Do`` and ``FmRotation_Do`` is
    four signed 32-bit values in ``x, y, width, height`` order.  The native
    crop helper at VA ``0x100d1d80`` copies exactly that rectangle.  Its image
    object has a fourth storage word, but the color stages use only the first
    three words; this API therefore preserves the repository's RGB shape.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    values = (x, y, width, height)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("x, y, width, and height must be integers")
    left, top, columns, rows = (int(value) for value in values)
    if left < 0 or top < 0 or columns < 1 or rows < 1:
        raise ValueError("crop origin must be nonnegative and size must be positive")
    if left + columns > image.shape[1] or top + rows > image.shape[0]:
        raise ValueError("crop rectangle exceeds the source image")
    return image[top : top + rows, left : left + columns].copy()


def _box_prefilter_u16(image_rgb: np.ndarray, factor: int) -> np.ndarray:
    """Model the shared S0/S2 integer prefilter at ``0x100d12f0``.

    The DLL extends the last row and column, sums ``factor`` squared uint16
    pixels in signed 32-bit accumulators, converts the sums to float32,
    multiplies by a float32 reciprocal, and rounds with ``cvtps2dq``.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    if not isinstance(factor, (int, np.integer)) or not 1 <= int(factor) <= 255:
        raise ValueError("pre_scale must be a byte-sized integer in 1..255")
    step = int(factor)
    if step == 1:
        return image.copy()

    output_height = (image.shape[0] + step - 1) // step
    output_width = (image.shape[1] + step - 1) // step
    pad_height = output_height * step - image.shape[0]
    pad_width = output_width * step - image.shape[1]
    padded = np.pad(
        image,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge",
    )
    sums = padded.reshape(
        output_height, step, output_width, step, 3
    ).sum(axis=(1, 3), dtype=np.uint64)
    # Native accumulation is signed i32.  Normal controller factors stay in
    # range, while this wrap also defines the binary's behavior for all byte
    # values accepted by the structure.
    sums_i32 = (sums & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32)
    reciprocal = np.divide(
        np.float32(1.0), np.float32(step * step), dtype=np.float32
    )
    averaged = np.multiply(sums_i32.astype(np.float32), reciprocal, dtype=np.float32)
    rounded = np.rint(averaged).astype(np.int64)
    return (rounded & 0xFFFF).astype(np.uint16)


def _scaling_axis_plan(
    input_length: int,
    output_length: int,
    sampling: int,
    taps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the exact index/float32-weight plan used by S0 or S2."""

    if input_length < 1 or output_length < 1:
        raise ValueError("scaling dimensions must be positive")
    if sampling not in (1, 2):
        raise ValueError("sampling must be 1 or 2")
    sample_count = output_length * sampling

    if sample_count < input_length:
        denominator = 2 * sample_count
        numerators = (
            (2 * np.arange(sample_count, dtype=np.int64) + 1) * input_length
            - sample_count
        )
    elif sample_count == 1:
        # The DLL's x87 expression is 0/0 only for a 1 -> 1 axis.  Every
        # source value on that axis is the same pixel, so defining coordinate
        # zero preserves its observable result without propagating a NaN.
        denominator = 1
        numerators = np.zeros(1, dtype=np.int64)
    else:
        denominator = sample_count - 1
        numerators = (
            np.arange(sample_count, dtype=np.int64) * (input_length - 1)
        )

    bases = numerators // denominator
    fractions = numerators - bases * denominator
    bases = np.clip(bases, 0, input_length - 1)

    if taps == 2:
        indexes = np.stack(
            (bases, np.minimum(bases + 1, input_length - 1)), axis=1
        )
        fraction = fractions.astype(np.float64) / float(denominator)
        inverse_sampling = 1.0 / float(sampling)
        weights = np.stack(
            ((1.0 - fraction) * inverse_sampling,
             fraction * inverse_sampling),
            axis=1,
        ).astype(np.float32)
    elif taps == 4:
        indexes = np.stack(
            (
                np.maximum(bases - 1, 0),
                bases,
                np.minimum(bases + 1, input_length - 1),
                np.minimum(bases + 2, input_length - 1),
            ),
            axis=1,
        )
        quantized = (fractions * 4096) // denominator
        distances = np.stack(
            (
                4096 + quantized,
                quantized,
                4096 - quantized,
                8192 - quantized,
            ),
            axis=1,
        )
        positions = distances.astype(np.float64) / 4096.0
        weights_f64 = np.zeros(positions.shape, dtype=np.float64)
        inner = distances < 4096
        weights_f64[inner] = (
            1.0 - 2.0 * positions[inner] ** 2 + positions[inner] ** 3
        )
        outer = (distances >= 4096) & (distances < 8192)
        weights_f64[outer] = (
            4.0
            - 8.0 * positions[outer]
            + 5.0 * positions[outer] ** 2
            - positions[outer] ** 3
        )
        weights = (weights_f64 / float(sampling)).astype(np.float32)
    else:
        raise ValueError("taps must be 2 or 4")

    return (
        indexes.reshape(output_length, sampling, taps),
        weights.reshape(output_length, sampling, taps),
    )


def _separable_scaling_u16(
    image_rgb: np.ndarray,
    output_width: int,
    output_height: int,
    *,
    sampling: int,
    taps: int,
) -> np.ndarray:
    image = _as_u16_rgb(image_rgb, "image_rgb")
    if not isinstance(output_width, (int, np.integer)) or int(output_width) < 1:
        raise ValueError("output_width must be a positive integer")
    if not isinstance(output_height, (int, np.integer)) or int(output_height) < 1:
        raise ValueError("output_height must be a positive integer")
    width = int(output_width)
    height = int(output_height)
    horizontal_indexes, horizontal_weights = _scaling_axis_plan(
        image.shape[1], width, int(sampling), taps
    )
    vertical_indexes, vertical_weights = _scaling_axis_plan(
        image.shape[0], height, int(sampling), taps
    )

    horizontal = np.zeros((image.shape[0], width, 3), dtype=np.float32)
    for sample in range(int(sampling)):
        for tap in range(taps):
            source = image[:, horizontal_indexes[:, sample, tap], :].astype(
                np.float32
            )
            term = np.multiply(
                source,
                horizontal_weights[:, sample, tap][None, :, None],
                dtype=np.float32,
            )
            horizontal = np.add(horizontal, term, dtype=np.float32)
    horizontal = np.clip(horizontal, 0.0, float(U16_MAX)).astype(
        np.float32, copy=False
    )

    vertical = np.zeros((height, width, 3), dtype=np.float32)
    for sample in range(int(sampling)):
        # The sampling=2 vertical workers do not keep one live eight-term
        # accumulator.  S0 ``0x100d0f30`` and S2 ``0x100d5920`` first build
        # one four-/two-tap float32 sum for each sample, then add that complete
        # sample sum to the prior output accumulator.  This grouping is
        # observably different by one output code at float32 rounding edges.
        sample_sum = np.zeros_like(vertical)
        for tap in range(taps):
            source = horizontal[vertical_indexes[:, sample, tap], :, :]
            term = np.multiply(
                source,
                vertical_weights[:, sample, tap][:, None, None],
                dtype=np.float32,
            )
            sample_sum = np.add(sample_sum, term, dtype=np.float32)
        vertical = np.add(sample_sum, vertical, dtype=np.float32)
    vertical = np.clip(vertical, 0.0, float(U16_MAX)).astype(
        np.float32, copy=False
    )
    # ``cvtps2dq`` uses the default nearest-even mode.  The following shuffle
    # stores the low words of lanes 0..2 and discards the native fourth lane.
    return np.rint(vertical).astype(np.uint16)


def apply_scaling_s0_u16(
    image_rgb: np.ndarray,
    output_width: int,
    output_height: int,
    *,
    pre_scale: int = 1,
    sampling: int = 1,
    trim: Tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Port ``FmScalingS0_Do`` (VA ``0x100d50d0``) for visible RGB.

    ``pre_scale`` is byte ``RD5X_SCALING_PARA+0x10``.  Values above one run
    the edge-replicated integer box prefilter.  ``sampling`` is byte ``+0x11``
    and selects one or two samples per output pixel.  S0 then uses the DLL's
    1/4096-quantized cubic-convolution kernel (cubic parameter ``a=-1``).
    Optional ``trim`` is the preceding ``x, y, width, height`` structure.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    if trim is not None:
        if len(trim) != 4:
            raise ValueError("trim must be (x, y, width, height)")
        image = apply_trimming_u16(image, *trim)
    image = _box_prefilter_u16(image, pre_scale)
    return _separable_scaling_u16(
        image,
        output_width,
        output_height,
        sampling=int(sampling),
        taps=4,
    )


def apply_scaling_s2_u16(
    image_rgb: np.ndarray,
    output_width: int,
    output_height: int,
    *,
    pre_scale: int = 1,
    sampling: int = 1,
    trim: Tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Port ``FmScalingS2_Do`` (VA ``0x100d6e30``) for visible RGB.

    Parameter interpretation and the optional prefilter match S0, while the
    final separable interpolation is the exact bilinear S2 worker at VAs
    ``0x100d56d0``/``0x100d5920``.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    if trim is not None:
        if len(trim) != 4:
            raise ValueError("trim must be (x, y, width, height)")
        image = apply_trimming_u16(image, *trim)
    image = _box_prefilter_u16(image, pre_scale)
    return _separable_scaling_u16(
        image,
        output_width,
        output_height,
        sampling=int(sampling),
        taps=2,
    )


def apply_rotation_orthogonal_u16(
    image_rgb: np.ndarray,
    angle_tenths: int,
    *,
    trim: Tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Port ``FmRotation_Do``'s exact 0/90/180/270/360-degree branches.

    ``RD5X_ROTATION_PARA`` stores clockwise tenths of a degree.  The binary
    uses dedicated copy workers at ``0x1009c1c0``, ``0x1009d560``, and
    ``0x1009e8e0`` for the three nonzero orthogonal rotations.  Arbitrary
    angles use a separate fixed-point break-table ABI and are intentionally
    not accepted by this primitive.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    if not isinstance(angle_tenths, (int, np.integer)):
        raise TypeError("angle_tenths must be an integer")
    angle = int(angle_tenths) % 3600
    if angle not in (0, 900, 1800, 2700):
        raise ValueError("orthogonal rotation requires 0, 900, 1800, or 2700")
    rotated = np.rot90(image, k=-(angle // 900)).copy()
    if trim is not None:
        if len(trim) != 4:
            raise ValueError("trim must be (x, y, width, height)")
        rotated = apply_trimming_u16(rotated, *trim)
    return rotated


def _rotation_fixed_coordinates(
    length: int,
    angle_radians: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``FmRotation``'s shared 10-bit fixed-point sine/cosine arrays.

    ``FUN_10099e60`` (VA ``0x10099e60``) evaluates positions relative to
    ``length / 2``.  The sine remains on the x87 stack while cosine is first
    stored as float32.  Both products are rounded half away from zero before
    being stored as signed i32.  The returned arrays are ``-sin`` then
    ``cos``, matching the two BaseBuf allocations consumed by the row worker.
    """

    if not isinstance(length, (int, np.integer)) or int(length) < 1:
        raise ValueError("length must be a positive integer")
    count = int(length)
    positions = np.arange(count, dtype=np.float64) - count * 0.5
    sine = math.sin(float(angle_radians))
    cosine_f32 = np.float32(math.cos(float(angle_radians)))

    sine_products = positions * sine * 1024.0
    cosine_products = positions * float(cosine_f32) * 1024.0

    def rounded_away(values: np.ndarray) -> np.ndarray:
        adjusted = np.where(values < 0.0, values - 0.5, values + 0.5)
        return np.trunc(adjusted).astype(np.int32)

    return -rounded_away(sine_products), rounded_away(cosine_products)


def _rotation_break_row(status: np.ndarray, width: int) -> np.ndarray:
    """Reproduce ``FUN_1009a040``'s four-transition row state machine."""

    result = np.full(4, int(width), dtype=np.int32)
    state = 0
    # The status of a rotated convex row is piecewise constant.  Feeding only
    # each run's first pixel is equivalent to the native per-pixel state
    # machine and avoids a Python loop across a full scanner-width row.
    changes = np.flatnonzero(status[1:] != status[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), changes))
    for index in starts:
        pixel = int(index)
        value = int(status[pixel])
        if state == 0:
            if value == 1:
                result[0] = pixel
                state = 1
            elif value == 0:
                result[0] = pixel
                result[1] = pixel
                state = 2
        elif state == 1:
            if value == 0:
                result[1] = pixel
                state = 2
            elif value == 2:
                result[1] = pixel
                result[2] = pixel
                result[3] = pixel
                state = 4
        elif state == 2:
            if value == 1:
                result[2] = pixel
                state = 3
            elif value == 2:
                result[2] = pixel
                result[3] = pixel
                state = 4
        elif state == 3 and value != 1:
            result[3] = pixel
            state = 4
        if state == 4:
            break
    return result


def apply_rotation_arbitrary_u16(
    image_rgb: np.ndarray,
    angle_tenths: int,
    *,
    trim: Tuple[int, int, int, int] | None = None,
    chunk_rows: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port the arbitrary-angle uint16 ``FmRotation_Do`` array worker.

    The worker uses centered inverse rotation, signed 10-bit fixed-point
    coordinates, float32 bilinear interpolation, and nearest-even SSE packing.
    It also returns the native ``T_ROT2BRK_POSTBL_tag`` payload as an
    ``(height, 4)`` int32 array.  That table is the exact input to the optional
    Blank Hex/Oct halo and boundary-mean branch.

    ``trim`` is the native ``x, y, width, height`` rectangle in the full
    rotated bounding canvas.  Orthogonal angles have dedicated copy workers
    and must be sent to :func:`apply_rotation_orthogonal_u16`.

    The shipped binary obtains sine from x87 ``fsin``.  Python's platform
    libm is numerically equivalent for normal scanner dimensions, but a
    contrived coordinate exactly on a half-integer rounding boundary remains
    an architecture-level x87 transcendental ambiguity.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("arbitrary rotation requires source dimensions >= 3")
    if not isinstance(angle_tenths, (int, np.integer)):
        raise TypeError("angle_tenths must be an integer")
    rows_per_chunk = _require_positive_chunk_rows(chunk_rows)
    angle = int(angle_tenths) % 3600
    if angle in (0, 900, 1800, 2700):
        raise ValueError("orthogonal angles use apply_rotation_orthogonal_u16")

    radians = float(angle) * 0.0017453292519943296
    sine = math.sin(radians)
    cosine = math.cos(radians)
    source_height, source_width = image.shape[:2]
    full_width = int(
        math.trunc(abs(cosine) * source_width + abs(sine) * source_height + 0.999)
    )
    full_height = int(
        math.trunc(abs(cosine) * source_height + abs(sine) * source_width + 0.999)
    )

    if trim is None:
        left = top = 0
        width, height = full_width, full_height
    else:
        if len(trim) != 4:
            raise ValueError("trim must be (x, y, width, height)")
        if any(not isinstance(value, (int, np.integer)) for value in trim):
            raise TypeError("trim values must be integers")
        left, top, width, height = (int(value) for value in trim)
        if left < 0 or top < 0 or width < 1 or height < 1:
            raise ValueError("trim origin must be nonnegative and size positive")
        if left + width > full_width or top + height > full_height:
            raise ValueError("trim rectangle exceeds the rotated bounding canvas")

    coordinate_length = max(full_width, full_height)
    sine_coordinates, cosine_coordinates = _rotation_fixed_coordinates(
        coordinate_length, radians
    )
    x_start = coordinate_length // 2 - full_width // 2 + left
    y_start = coordinate_length // 2 - full_height // 2 + top
    sine_x = sine_coordinates[x_start : x_start + width].astype(np.int64)
    cosine_x = cosine_coordinates[x_start : x_start + width].astype(np.int64)

    output = np.empty((height, width, 3), dtype=np.uint16)
    breaks = np.full((height, 4), width, dtype=np.int32)
    scale = np.float32(0.0009765625)
    source_x_center = source_width // 2
    source_y_center = source_height // 2

    for row_start in range(0, height, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, height)
        sine_y = sine_coordinates[
            y_start + row_start : y_start + row_end
        ].astype(np.int64)
        cosine_y = cosine_coordinates[
            y_start + row_start : y_start + row_end
        ].astype(np.int64)

        fixed_x = cosine_x[None, :] - sine_y[:, None]
        fixed_y = sine_x[None, :] + cosine_y[:, None]
        source_x = (fixed_x >> 10) + source_x_center
        source_y = (fixed_y >> 10) + source_y_center
        fraction_x = (fixed_x & 0x3FF).astype(np.int32)
        fraction_y = (fixed_y & 0x3FF).astype(np.int32)
        status = np.zeros(source_x.shape, dtype=np.uint8)

        low_x = source_x < 0
        status[low_x] = np.where(source_x[low_x] == -1, 1, 2)
        source_x[low_x] = 1
        fraction_x[low_x] = 1024 - fraction_x[low_x]
        high_x = source_x >= source_width - 1
        status[high_x] = np.where(source_x[high_x] == source_width - 1, 1, 2)
        source_x[high_x] = source_width - 2
        fraction_x[high_x] = 1024 - fraction_x[high_x]

        low_y = source_y < 0
        y_low_status = np.where(source_y == -1, 1, 2).astype(np.uint8)
        status[low_y] = np.where(
            status[low_y] == 2, 2, y_low_status[low_y]
        )
        source_y[low_y] = 1
        fraction_y[low_y] = 1024 - fraction_y[low_y]
        high_y = source_y >= source_height - 1
        y_high_status = np.where(source_y == source_height - 1, 1, 2).astype(
            np.uint8
        )
        status[high_y] = np.where(
            status[high_y] == 2, 2, y_high_status[high_y]
        )
        source_y[high_y] = source_height - 2
        fraction_y[high_y] = 1024 - fraction_y[high_y]

        p00 = image[source_y, source_x].astype(np.float32)
        p01 = image[source_y, source_x + 1].astype(np.float32)
        p10 = image[source_y + 1, source_x].astype(np.float32)
        p11 = image[source_y + 1, source_x + 1].astype(np.float32)
        fx = fraction_x.astype(np.float32)[..., None]
        fy = fraction_y.astype(np.float32)[..., None]
        inv_x = (1024 - fraction_x).astype(np.float32)[..., None]
        inv_y = (1024 - fraction_y).astype(np.float32)[..., None]

        upper = np.add(
            np.multiply(p00, inv_x, dtype=np.float32),
            np.multiply(p01, fx, dtype=np.float32),
            dtype=np.float32,
        )
        upper = np.multiply(upper, scale, dtype=np.float32)
        lower = np.add(
            np.multiply(p10, inv_x, dtype=np.float32),
            np.multiply(p11, fx, dtype=np.float32),
            dtype=np.float32,
        )
        lower = np.multiply(lower, scale, dtype=np.float32)
        interpolated = np.add(
            np.multiply(upper, inv_y, dtype=np.float32),
            np.multiply(lower, fy, dtype=np.float32),
            dtype=np.float32,
        )
        interpolated = np.multiply(interpolated, scale, dtype=np.float32)
        output[row_start:row_end] = np.rint(interpolated).astype(np.uint16)

        for offset, row_status in enumerate(status):
            breaks[row_start + offset] = _rotation_break_row(row_status, width)

    return output, breaks


def _blank_color_rgb(color_rgb_u8: Sequence[int]) -> np.ndarray:
    color = np.asarray(color_rgb_u8)
    if color.shape != (3,) or not np.issubdtype(color.dtype, np.integer):
        raise ValueError("color_rgb_u8 must contain three integer values")
    if int(color.min()) < 0 or int(color.max()) > 255:
        raise ValueError("color_rgb_u8 values must fit unsigned bytes")
    return color.astype(np.uint8)


def _rotation_breaks_i32(
    rotation_breaks: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    breaks = np.asarray(rotation_breaks)
    if breaks.shape != (height, 4) or not np.issubdtype(
        breaks.dtype, np.integer
    ):
        raise ValueError("rotation_breaks must be an integer (height, 4) array")
    if breaks.size and (int(breaks.min()) < 0 or int(breaks.max()) > width):
        raise ValueError("rotation break positions must lie in 0..image width")
    return breaks.astype(np.int32, copy=False)


def _blank_rotation_postbreak(
    output_rgb: np.ndarray,
    image_rgb: np.ndarray,
    left: int,
    top: int,
    color_rgb: np.ndarray,
    rotation_breaks: np.ndarray,
) -> np.ndarray:
    """Port the shared Blank halo and boundary 3x3-mean workers.

    The Hex implementation is at VAs ``0x10012b90``/``0x10012e50`` and the
    byte-identical Oct topology at ``0x10013f50``/``0x10014190``.  Native
    CImage rows expose a three-pixel halo.  An explicit padded array gives the
    same visible result without exposing CImage's allocator ABI.
    """

    height, width = image_rgb.shape[:2]
    breaks = _rotation_breaks_i32(rotation_breaks, height, width)
    padding = 3
    work = np.empty(
        (height + padding * 2, width + padding * 2, 3), dtype=image_rgb.dtype
    )
    work[...] = color_rgb
    work[padding : padding + height, padding : padding + width] = image_rgb

    # ``FUN_10012b90``/``FUN_10013f50`` preserve only [break0, break3) on
    # each rotated row and fill everything else, including the three-pixel
    # CImage halo, with the requested blank color.
    for row, (first, _inside, _outside, last) in enumerate(breaks):
        first_clamped = min(max(int(first), 0), width)
        last_clamped = min(max(int(last), 0), width)
        core = work[padding + row, padding : padding + width]
        core[:first_clamped] = color_rgb
        core[last_clamped:] = color_rgb

    output_rgb[top : top + height, left : left + width] = work[
        padding : padding + height, padding : padding + width
    ]

    # Precompute the integer /9 means for every center the DLL can address:
    # x=-1..width and y=-1..height.  The native worker always reads its
    # prefilled source image, never a previously smoothed destination pixel.
    means_sum = np.zeros((height + 2, width + 2, 3), dtype=np.uint32)
    for y_offset in range(3):
        for x_offset in range(3):
            means_sum += work[
                1 + y_offset : 1 + y_offset + height + 2,
                1 + x_offset : 1 + x_offset + width + 2,
            ].astype(np.uint32)
    means = (means_sum // np.uint32(9)).astype(image_rgb.dtype)

    smooth = np.zeros((height + 2, width + 2), dtype=bool)
    for row, (first, inside, outside, last) in enumerate(breaks):
        first = int(first)
        inside = int(inside)
        outside = int(outside)
        last = int(last)
        ranges = []
        if first != inside:
            ranges.append(
                (
                    min(max(first - 1, -1), width),
                    min(max(inside + 1, -1), width),
                )
            )
        if outside != last:
            ranges.append(
                (
                    min(max(outside - 1, 0), width + 1),
                    min(max(last + 1, 0), width + 1),
                )
            )
        for start, stop in ranges:
            if start < stop:
                # Center coordinates are shifted by one in this mask.
                smooth[row : row + 3, start + 1 : stop + 1] = True

    center_y, center_x = np.nonzero(smooth)
    destination_y = top + center_y - 1
    destination_x = left + center_x - 1
    visible = (
        (destination_y >= 0)
        & (destination_y < output_rgb.shape[0])
        & (destination_x >= 0)
        & (destination_x < output_rgb.shape[1])
    )
    output_rgb[destination_y[visible], destination_x[visible]] = means[
        center_y[visible], center_x[visible]
    ]
    return output_rgb


def apply_blank_oct_color_u8(
    image_rgb: np.ndarray,
    output_width: int,
    output_height: int,
    left: int,
    top: int,
    color_rgb_u8: Sequence[int],
    *,
    rotation_breaks: np.ndarray | None = None,
) -> np.ndarray:
    """Port ``FmBlankOctColor_Do`` (VA ``0x100145d0``).

    The parameter block at ``+0x10/+0x14`` supplies the left/top placement;
    bytes ``+0x18..+0x1a`` are R, G, B.  The native four-byte image stores BGRX,
    so returning repository-standard RGB reverses only that storage detail.
    Supplying ``rotation_breaks`` enables its exact halo-fill and boundary
    3x3-mean branch.
    """

    image = _as_u8_rgb(image_rgb, "image_rgb")
    values = (output_width, output_height, left, top)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("output dimensions and placement must be integers")
    width, height, x, y = (int(value) for value in values)
    if width < 1 or height < 1 or x < 0 or y < 0:
        raise ValueError("output size must be positive and placement nonnegative")
    if x + image.shape[1] > width or y + image.shape[0] > height:
        raise ValueError("placed image exceeds the blank output canvas")
    color = _blank_color_rgb(color_rgb_u8)
    output = np.empty((height, width, 3), dtype=np.uint8)
    output[...] = color
    if rotation_breaks is None:
        output[y : y + image.shape[0], x : x + image.shape[1]] = image
    else:
        _blank_rotation_postbreak(
            output, image, x, y, color, rotation_breaks
        )
    return output


def apply_blank_hex_color_u16(
    image_rgb: np.ndarray,
    output_width: int,
    output_height: int,
    left: int,
    top: int,
    color_rgb_u8: Sequence[int],
    *,
    rotation_breaks: np.ndarray | None = None,
) -> np.ndarray:
    """Port ``FmBlankHexColor_Do`` (VA ``0x10013290``).

    Hex expands each supplied color byte as ``(byte << 8) | 0xff``; this is
    intentionally different from the later generic 8-to-16 conversion's
    multiplication by 257.  Source uint16 pixels are copied unchanged.
    Supplying ``rotation_breaks`` enables its exact halo-fill and boundary
    3x3-mean branch.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    values = (output_width, output_height, left, top)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("output dimensions and placement must be integers")
    width, height, x, y = (int(value) for value in values)
    if width < 1 or height < 1 or x < 0 or y < 0:
        raise ValueError("output size must be positive and placement nonnegative")
    if x + image.shape[1] > width or y + image.shape[0] > height:
        raise ValueError("placed image exceeds the blank output canvas")
    color_u8 = _blank_color_rgb(color_rgb_u8)
    color_u16 = (color_u8.astype(np.uint16) << 8) | np.uint16(0x00FF)
    output = np.empty((height, width, 3), dtype=np.uint16)
    output[...] = color_u16
    if rotation_breaks is None:
        output[y : y + image.shape[0], x : x + image.shape[1]] = image
    else:
        _blank_rotation_postbreak(
            output, image, x, y, color_u16, rotation_breaks
        )
    return output


def compose_ccd_gain_lut(source_lut: np.ndarray, gain_q12: int) -> np.ndarray:
    """Reproduce UniRdImgLib ``sub_1005eb10`` for one 4096-entry LUT."""

    source = np.asarray(source_lut, dtype=np.uint16)
    if source.shape != (4096,):
        raise ValueError("source_lut must contain 4096 uint16 entries")
    if not 0 <= int(gain_q12) <= 0xFFFF:
        raise ValueError("gain_q12 must fit an unsigned 16-bit value")
    indexes = (np.arange(4096, dtype=np.uint32) * int(gain_q12)) >> 12
    indexes = np.minimum(indexes, 4095)
    return source[indexes].copy()


def build_ccd_noise_response_lut(
    levels_u8: Sequence[int],
    breakpoints_u16: Sequence[int],
    gain_u8: int,
) -> np.ndarray:
    """Port UniRdImgLib ``sub_1007e190`` at VA ``0x1007e190``.

    ``FmCCDNoiseClearS0_Do`` (VA ``0x1007f780``) calls this helper once for
    each R, G, B channel.  Four byte-sized response levels are interpolated
    at four word-sized breakpoints after the DLL shifts each breakpoint
    right by eight.  The interpolated integer is multiplied by ``gain_u8``
    before the x87 multiplication by the DLL's float32 constant at
    ``0x101948f0`` (``1 / 25500`` rounded to float32).

    This is the complete response-table kernel, not the complete CCD-noise
    stage.  The latter also uses the image-band setup at ``0x1007e350`` and
    rolling-window worker at ``0x1007ee80``.
    """

    levels = np.asarray(levels_u8)
    breakpoints = np.asarray(breakpoints_u16)
    if levels.shape != (4,) or not np.issubdtype(levels.dtype, np.integer):
        raise ValueError("levels_u8 must contain four integer values")
    if breakpoints.shape != (4,) or not np.issubdtype(
        breakpoints.dtype, np.integer
    ):
        raise ValueError("breakpoints_u16 must contain four integer values")
    if levels.size and (int(levels.min()) < 0 or int(levels.max()) > 0xFF):
        raise ValueError("levels_u8 values must fit unsigned bytes")
    if breakpoints.size and (
        int(breakpoints.min()) < 0 or int(breakpoints.max()) > U16_MAX
    ):
        raise ValueError("breakpoints_u16 values must fit unsigned words")
    if not 0 <= int(gain_u8) <= 0xFF:
        raise ValueError("gain_u8 must fit an unsigned byte")

    x = breakpoints.astype(np.int64) >> 8
    y = levels.astype(np.int64)
    integer_response = np.empty(256, dtype=np.int64)
    cursor = 0

    # These inclusive/exclusive endpoints follow the three IDIV loops in
    # sub_1007e190, including their behavior for equal/reversed breakpoints.
    if cursor <= int(x[0]):
        stop = min(int(x[0]) + 1, 256)
        integer_response[cursor:stop] = y[0]
        cursor = stop
    for segment in range(3):
        segment_stop = int(x[segment + 1])
        inclusive = segment == 2
        stop = min(segment_stop + int(inclusive), 256)
        if int(x[segment]) < segment_stop and cursor < stop:
            positions = np.arange(cursor, stop, dtype=np.int64)
            numerator = (positions - int(x[segment])) * int(
                y[segment + 1] - y[segment]
            )
            denominator = segment_stop - int(x[segment])
            integer_response[cursor:stop] = y[segment] + _trunc_divide_signed(
                numerator, denominator
            )
            cursor = stop
    if cursor < 256:
        integer_response[cursor:] = y[3]

    # The integer product is exact; multiplying its float64 representation by
    # the binary-exact float32 constant models x87 FMUL followed by FSTP f32.
    scaled = (
        integer_response.astype(np.float64)
        * int(gain_u8)
        * float(CCD_NOISE_RESPONSE_SCALE_F32)
    )
    return np.asarray(scaled, dtype=np.float32)


def build_ccd_noise_response_luts_rgb(
    levels_rgb_u8: np.ndarray,
    breakpoints_rgb_u16: np.ndarray,
    gains_rgb_u8: Sequence[int],
) -> np.ndarray:
    """Build the three ``sub_1007e190`` tables in an explicit RGB API.

    The packed vendor structure stores R, G, B triplets, while the native
    image worker visits B, G, R lanes.  Naming both the input and output RGB
    here removes that otherwise implicit reversal.
    """

    levels = np.asarray(levels_rgb_u8)
    breakpoints = np.asarray(breakpoints_rgb_u16)
    gains = np.asarray(gains_rgb_u8)
    if levels.shape != (3, 4):
        raise ValueError("levels_rgb_u8 must have shape (3, 4)")
    if breakpoints.shape != (3, 4):
        raise ValueError("breakpoints_rgb_u16 must have shape (3, 4)")
    if gains.shape != (3,) or not np.issubdtype(gains.dtype, np.integer):
        raise ValueError("gains_rgb_u8 must contain three integer values")
    return np.stack(
        [
            build_ccd_noise_response_lut(levels[channel], breakpoints[channel], gain)
            for channel, gain in enumerate(gains)
        ]
    )


def _reflect_without_edge(indices: np.ndarray, length: int) -> np.ndarray:
    """Map coordinates as ``sub_1007dfc0``'s vendor halo copies do."""

    if length < 2:
        raise ValueError("vendor reflection requires an axis of at least 2 pixels")
    period = 2 * (length - 1)
    folded = np.mod(np.asarray(indices, dtype=np.int64), period)
    return np.where(folded < length, folded, period - folded)


def _ccd_box_means_with_one_pixel_halo(
    channel: np.ndarray,
    start: int,
    stop: int,
    window_size: int,
) -> np.ndarray:
    """Recover ``sub_1007e350``'s centered float32 box means for one band."""

    radius = int(window_size) // 2
    height, width = channel.shape
    y_coordinates = np.arange(start - radius - 1, stop + radius + 1)
    x_coordinates = np.arange(-radius - 1, width + radius + 1)
    reflected = channel[
        np.ix_(
            _reflect_without_edge(y_coordinates, height),
            _reflect_without_edge(x_coordinates, width),
        )
    ]
    integral = np.pad(
        reflected.astype(np.int64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )
    size = int(window_size)
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    reciprocal = np.float32(1.0 / (size * size))
    return np.multiply(sums.astype(np.float32), reciprocal, dtype=np.float32)


def apply_ccd_noise_clear_s0(
    image_rgb: np.ndarray,
    response_luts_rgb_f32: np.ndarray,
    window_sizes_rgb: Sequence[int],
    edge_sensitivity_rgb_u8: Sequence[int],
    edge_limits_rgb_u8: Sequence[int],
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Port the complete software pixel stage ``FmCCDNoiseClearS0_Do``.

    Provenance is UniRdImgLib VA ``0x1007f780`` and its local helpers:
    ``sub_1007dfc0`` (reflection halo), ``sub_1007e350`` (box means), and
    ``sub_1007ee80`` (edge-adaptive blend).  The API is RGB even though the
    DLL's native lanes are B, G, R; all parameter triplets are reordered with
    the pixels, so the lane-independent arithmetic is unchanged.

    The worker forms one centered mean per channel using its odd 3..15 window,
    estimates edge strength from six first/second differences in the mean
    image, blends the source toward an edge-adaptive mean, then applies the
    recovered response table.  Every float operation below is forced through
    float32 in the SSE instruction order, and both ``cvtps2dq`` conversions
    use nearest-even rounding.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    response = np.asarray(response_luts_rgb_f32)
    windows = np.asarray(window_sizes_rgb)
    sensitivity = np.asarray(edge_sensitivity_rgb_u8)
    limits = np.asarray(edge_limits_rgb_u8)
    if response.shape != (3, 256) or response.dtype != np.float32:
        raise ValueError(
            "response_luts_rgb_f32 must have shape (3, 256) and dtype float32"
        )
    for name, values in (
        ("window_sizes_rgb", windows),
        ("edge_sensitivity_rgb_u8", sensitivity),
        ("edge_limits_rgb_u8", limits),
    ):
        if values.shape != (3,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain three integer values")
    if any(int(value) not in (3, 5, 7, 9, 11, 13, 15) for value in windows):
        raise ValueError("CCD-noise window sizes must be odd values in 3..15")
    for name, values in (
        ("edge_sensitivity_rgb_u8", sensitivity),
        ("edge_limits_rgb_u8", limits),
    ):
        if int(values.min()) < 0 or int(values.max()) > 0xFF:
            raise ValueError(f"{name} values must fit unsigned bytes")
    rows = _require_positive_chunk_rows(chunk_rows)

    maximum_radius = int(windows.max()) // 2
    if image.shape[0] <= maximum_radius + 1 or image.shape[1] <= maximum_radius + 1:
        raise ValueError(
            "image axes must exceed the largest CCD-noise radius plus one"
        )

    output = np.empty_like(image)
    sensitivity_f32 = np.asarray(
        sensitivity.astype(np.float64) * float(CCD_NOISE_EDGE_SCALE_F32),
        dtype=np.float32,
    )
    limit_f32 = limits.astype(np.float32)
    window_areas_f32 = np.asarray(windows.astype(np.int64) ** 2, dtype=np.float32)
    source_f32 = image.astype(np.float32)

    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        means = np.stack(
            [
                _ccd_box_means_with_one_pixel_halo(
                    image[..., channel], start, stop, int(windows[channel])
                )
                for channel in range(3)
            ],
            axis=2,
        )
        upper = means[:-2]
        center = means[1:-1]
        lower = means[2:]

        vertical = np.abs(
            np.subtract(upper[:, 1:-1], lower[:, 1:-1], dtype=np.float32)
        )
        horizontal = np.abs(
            np.subtract(center[:, :-2], center[:, 2:], dtype=np.float32)
        )
        diagonal_down = np.abs(
            np.subtract(upper[:, :-2], lower[:, 2:], dtype=np.float32)
        )
        diagonal_up = np.abs(
            np.subtract(upper[:, 2:], lower[:, :-2], dtype=np.float32)
        )
        axial_second = np.abs(
            np.subtract(
                np.subtract(
                    _add_f32(upper[:, 1:-1], lower[:, 1:-1]),
                    center[:, :-2],
                    dtype=np.float32,
                ),
                center[:, 2:],
                dtype=np.float32,
            )
        )
        diagonal_second = np.abs(
            np.subtract(
                np.subtract(
                    _add_f32(upper[:, :-2], lower[:, 2:]),
                    lower[:, :-2],
                    dtype=np.float32,
                ),
                upper[:, 2:],
                dtype=np.float32,
            )
        )

        first_differences = _add_f32(
            _add_f32(_add_f32(vertical, horizontal), diagonal_down), diagonal_up
        )
        first_differences = _mul_f32(first_differences, ONE_THIRD_F32)
        second_differences = _mul_f32(
            _add_f32(axial_second, diagonal_second), np.float32(0.5)
        )
        edge = _add_f32(first_differences, second_differences)
        edge = np.multiply(edge, sensitivity_f32, dtype=np.float32)
        edge = np.add(edge, np.float32(1.0), dtype=np.float32)
        edge = np.minimum(np.maximum(edge, np.float32(0.0)), limit_f32)

        source = source_f32[start:stop]
        local_mean = center[:, 1:-1]
        mean_indexes = np.rint(local_mean).astype(np.int64) >> 8
        lane_response = np.empty_like(local_mean)
        for channel in range(3):
            lane_response[..., channel] = response[channel][
                mean_indexes[..., channel]
            ]

        edge_minus_one = np.subtract(edge, np.float32(1.0), dtype=np.float32)
        numerator = _add_f32(
            np.multiply(source, edge_minus_one, dtype=np.float32),
            np.multiply(local_mean, window_areas_f32, dtype=np.float32),
        )
        denominator = np.add(
            edge_minus_one, window_areas_f32, dtype=np.float32
        )
        adaptive_mean = np.divide(numerator, denominator, dtype=np.float32)
        correction = np.multiply(
            np.subtract(source, adaptive_mean, dtype=np.float32),
            lane_response,
            dtype=np.float32,
        )
        corrected = np.subtract(source, correction, dtype=np.float32)
        corrected = np.minimum(
            np.maximum(corrected, np.float32(0.0)), np.float32(U16_MAX)
        )
        output[start:stop] = np.rint(corrected).astype(np.uint16)
    return output


def build_grain_s4_variance_lut(
    curve_u16: np.ndarray,
    base_divisor_u32: int,
    range_divisor_u32: int,
) -> np.ndarray:
    """Port Grain S4's 256-entry setup helper at VA ``0x1006c910``.

    The helper is called directly by ``FmGrainS4_Do`` at ``0x100739af``.
    It converts the caller-supplied 0..4095 common curve to a double-precision
    multiplier table using the two unsigned 32-bit divisors stored at offsets
    ``+0`` and ``+4`` of ``RD5X_GRAIN_PARA``.

    :func:`apply_grain_s4_pixel_kernel` ports the pure per-pixel tail of
    ``sub_10072e20``; :func:`apply_grain_s4` also recovers the complete
    reflected spatial-statistics pass from ``sub_10072850``/``sub_10072e20``.
    """

    curve = np.asarray(curve_u16)
    if curve.shape != (256,) or curve.dtype != np.uint16:
        raise ValueError("curve_u16 must have shape (256,) and dtype uint16")
    for name, value in (
        ("base_divisor_u32", base_divisor_u32),
        ("range_divisor_u32", range_divisor_u32),
    ):
        if not 1 <= int(value) <= 0xFFFFFFFF:
            raise ValueError(f"{name} must be a nonzero unsigned 32-bit value")

    scaled_curve = np.minimum(
        curve.astype(np.float64) * GRAIN_CURVE_TO_Q15_F64,
        np.float64(32768.0),
    )
    base = np.float64(32768.0) / np.float64(int(base_divisor_u32))
    reciprocal = np.float64(1.0) / np.float64(int(range_divisor_u32))
    return np.asarray(
        base - (np.float64(32768.0) - scaled_curve) * reciprocal,
        dtype=np.float64,
    )


def apply_grain_s4_pixel_kernel(
    image_rgb: np.ndarray,
    local_sums_rgbm: np.ndarray,
    local_square_sums_rgbm: np.ndarray,
    variance_lut_f64: np.ndarray,
    *,
    window_size: int,
    color_offset_u8: int,
    luminance_offset_u8: int,
) -> np.ndarray:
    """Port the pure pixel tail of Grain S4 ``sub_10072e20``.

    ``local_sums_rgbm`` and ``local_square_sums_rgbm`` are the exact rolling
    sums for R, G, B, and ``M=floor((R+G+B)/3)`` over the selected square
    window.  :func:`apply_grain_s4` constructs those statistics with the
    recovered ``sub_10072850`` reflection convention.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    sums = np.asarray(local_sums_rgbm)
    square_sums = np.asarray(local_square_sums_rgbm)
    lut = np.asarray(variance_lut_f64)
    expected_shape = image.shape[:2] + (4,)
    if sums.shape != expected_shape or not np.issubdtype(sums.dtype, np.integer):
        raise ValueError("local_sums_rgbm must have integer shape (H, W, 4)")
    if square_sums.shape != expected_shape or not (
        np.issubdtype(square_sums.dtype, np.integer)
        or np.issubdtype(square_sums.dtype, np.floating)
    ):
        raise ValueError(
            "local_square_sums_rgbm must have numeric shape (H, W, 4)"
        )
    if lut.shape != (256,) or lut.dtype != np.float64:
        raise ValueError("variance_lut_f64 must have shape (256,) and dtype float64")
    if int(window_size) not in (3, 5, 7, 9, 11, 13, 15):
        raise ValueError("window_size must be one of 3, 5, 7, 9, 11, 13, 15")
    for name, value in (
        ("color_offset_u8", color_offset_u8),
        ("luminance_offset_u8", luminance_offset_u8),
    ):
        if not 0 <= int(value) <= 0xFF:
            raise ValueError(f"{name} must fit an unsigned byte")

    source = np.empty(expected_shape, dtype=np.int64)
    source[..., :3] = image.astype(np.int64)
    source[..., 3] = source[..., :3].sum(axis=2) // 3
    count = int(window_size) ** 2
    half = count >> 1
    sums_i64 = sums.astype(np.int64, copy=False)
    means = (sums_i64 + half) // count
    reciprocal = np.float64(1.0) / np.float64(count)
    floating_means = sums_i64.astype(np.float64) * reciprocal
    variances = (
        square_sums.astype(np.float64) * reciprocal
        - floating_means * floating_means
    )
    maximum_variance = variances.max(axis=2)
    response = lut[source[..., 3] >> 8] * maximum_variance
    color_weight = np.trunc(
        np.clip(
            response + int(color_offset_u8) * GRAIN_BYTE_TO_Q15_F64,
            0.0,
            32768.0,
        )
    ).astype(np.int64)
    luminance_weight = np.trunc(
        np.clip(
            response + int(luminance_offset_u8) * GRAIN_BYTE_TO_Q15_F64,
            0.0,
            32768.0,
        )
    ).astype(np.int64)

    color_blend = (
        (32768 - color_weight[..., None]) * means[..., :3]
        + color_weight[..., None] * source[..., :3]
        + 0x4000
    ) >> 15
    target_mean = (
        (32768 - luminance_weight) * means[..., 3]
        + luminance_weight * source[..., 3]
        + 0x4000
    ) >> 15
    correction = target_mean - color_blend.sum(axis=2) // 3
    return np.clip(color_blend + correction[..., None], 0, U16_MAX).astype(
        np.uint16
    )


def _grain_s4_local_statistics(
    image_rgb: np.ndarray,
    start: int,
    stop: int,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port the integer/double rolling state of ``sub_10072850/10072e20``."""

    radius = int(window_size) // 2
    height, width = image_rgb.shape[:2]
    y_coordinates = np.arange(start - radius, stop + radius)
    x_coordinates = np.arange(-radius, width + radius)
    rgb = image_rgb[
        np.ix_(
            _reflect_without_edge(y_coordinates, height),
            _reflect_without_edge(x_coordinates, width),
            np.arange(3),
        )
    ].astype(np.int64)
    samples = np.empty(rgb.shape[:2] + (4,), dtype=np.int64)
    samples[..., :3] = rgb
    samples[..., 3] = rgb.sum(axis=2) // 3

    size = int(window_size)

    def window_sum(values: np.ndarray) -> np.ndarray:
        integral = np.pad(
            values.cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0), (0, 0)),
        )
        return (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )

    sums = window_sum(samples)
    # Each square and every N<=15 window sum is below 2**53, so the DLL's
    # double square/sum ring and this exact int64 accumulation convert to the
    # same binary64 values in the pure pixel tail.
    square_sums = window_sum(samples * samples)
    return sums, square_sums


def apply_grain_s4(
    image_rgb: np.ndarray,
    curve_u16: np.ndarray,
    base_divisor_u32: int,
    range_divisor_u32: int,
    *,
    window_size: int,
    color_offset_u8: int,
    luminance_offset_u8: int,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Port the complete software stage ``FmGrainS4_Do`` at ``0x100738f0``.

    ``sub_10072850`` seeds reflected ring rows and exact integer/double sums;
    ``sub_10072e20`` advances those sums and applies the pixel kernel.  Since
    integer addition and the stored squared integers are exact throughout the
    supported 3..15 windows, centered integral sums are mathematically and
    numerically identical to the ring order.  Reflection follows the explicit
    halo copies in those helpers (edge samples are not repeated).
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    if int(window_size) not in (3, 5, 7, 9, 11, 13, 15):
        raise ValueError("window_size must be one of 3, 5, 7, 9, 11, 13, 15")
    rows = _require_positive_chunk_rows(chunk_rows)
    radius = int(window_size) // 2
    if image.shape[0] <= radius or image.shape[1] <= radius:
        raise ValueError("image axes must exceed the Grain S4 window radius")
    response = build_grain_s4_variance_lut(
        curve_u16, base_divisor_u32, range_divisor_u32
    )
    output = np.empty_like(image)
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        sums, square_sums = _grain_s4_local_statistics(
            image, start, stop, int(window_size)
        )
        output[start:stop] = apply_grain_s4_pixel_kernel(
            image[start:stop],
            sums,
            square_sums,
            response,
            window_size=int(window_size),
            color_offset_u8=color_offset_u8,
            luminance_offset_u8=luminance_offset_u8,
        )
    return output


def build_auto_contrast_s0_luts(
    x_breakpoints: Sequence[int],
    y_values: Sequence[int],
    blend_q9: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the exact Auto-contrast S0 tables from ``sub_1005bef0``.

    The first half of the helper at VA ``0x1005bef0`` expands five signed
    32-bit X/Y controls into a 65,536-entry intercept table and assigns one
    signed Q9 blend coefficient to each of the four segments.  Floating
    interpolation is converted with the shipped MSVC ``_ftol2_sse`` helper
    at ``0x1016e6c0``, i.e. truncation toward zero.
    """

    x = np.asarray(x_breakpoints)
    y = np.asarray(y_values)
    blend = _signed_q12(np.asarray(blend_q9), "blend_q9")
    if x.shape != (5,) or y.shape != (5,):
        raise ValueError("x_breakpoints and y_values must each contain five values")
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(
        y.dtype, np.integer
    ):
        raise TypeError("auto-contrast X/Y controls must be integers")
    if blend.shape != (4,):
        raise ValueError("blend_q9 must contain four signed/encoded words")
    if np.any(x < 0) or np.any(x > U16_MAX) or np.any(np.diff(x) <= 0):
        raise ValueError(
            "x_breakpoints must be strictly increasing values in 0..65535"
        )
    if y.size and (
        int(y.min()) < np.iinfo(np.int32).min
        or int(y.max()) > np.iinfo(np.int32).max
    ):
        raise ValueError("y_values must fit signed 32-bit words")

    positions = np.arange(65536, dtype=np.int64)
    segments = np.searchsorted(x[1:].astype(np.int64), positions, side="left")
    segments = np.minimum(segments, 3)
    slopes = (y[1:].astype(np.float64) - y[:-1].astype(np.float64)) / (
        x[1:].astype(np.float64) - x[:-1].astype(np.float64)
    )
    interpolated = y.astype(np.float64)[segments] + (
        positions - x.astype(np.int64)[segments]
    ) * slopes[segments]
    intercept = np.trunc(interpolated).astype(np.int64)
    intercept[positions <= int(x[0])] = int(y[0])
    intercept[positions >= int(x[4])] = int(y[4])
    return intercept.astype(np.int32), blend[segments].copy()


def apply_auto_contrast_s0_luts(
    luminance_u16: np.ndarray,
    local_index_u16: np.ndarray,
    intercept_i32: np.ndarray,
    blend_q9_i16: np.ndarray,
) -> np.ndarray:
    """Port Auto-contrast S0's final luminance worker in ``sub_1005bef0``.

    The spatial pass supplies ``local_index_u16``.  For each luminance sample
    the DLL indexes the two recovered tables, applies the signed Q9 difference
    blend with an arithmetic right shift, and saturates to an unsigned word.
    :func:`apply_auto_contrast_s0` composes this worker with the recovered
    downsample, two-pass local filter, and bilinear map expansion.
    """

    luminance = np.asarray(luminance_u16)
    local_index = np.asarray(local_index_u16)
    intercept = np.asarray(intercept_i32)
    blend = np.asarray(blend_q9_i16)
    if luminance.dtype != np.uint16 or luminance.ndim != 2:
        raise ValueError("luminance_u16 must be a 2D uint16 array")
    if local_index.shape != luminance.shape or local_index.dtype != np.uint16:
        raise ValueError("local_index_u16 must match luminance_u16 as uint16")
    if intercept.shape != (65536,) or intercept.dtype != np.int32:
        raise ValueError("intercept_i32 must have shape (65536,) and dtype int32")
    if blend.shape != (65536,) or blend.dtype != np.int16:
        raise ValueError("blend_q9_i16 must have shape (65536,) and dtype int16")

    index = local_index.astype(np.int64)
    result = intercept[index].astype(np.int64) + (
        (luminance.astype(np.int64) - index) * blend[index].astype(np.int64)
        >> 9
    )
    return np.clip(result, 0, U16_MAX).astype(np.uint16)


def prepare_auto_contrast_s0_curve(
    inner_x_u16: Sequence[int],
    inner_y_u16: Sequence[int],
    extrapolation_slopes_q9: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Port ``sub_1005aa40``'s raw curve-control preparation.

    The packed parameter stores only the three interior X/Y points at offsets
    ``+0x46..+0x50``.  The first and fourth signed Q9 slopes at
    ``+0x52..+0x58`` extrapolate the endpoint Y values at X=0 and X=65535.
    The signed division is x86 truncation toward zero, including for negative
    outer slopes.  The two middle slope words are copied by the DLL but are
    not consulted by the S0 LUT builder.
    """

    x_inner = np.asarray(inner_x_u16)
    y_inner = np.asarray(inner_y_u16)
    slopes = _signed_q12(
        np.asarray(extrapolation_slopes_q9), "extrapolation_slopes_q9"
    )
    if x_inner.shape != (3,) or y_inner.shape != (3,):
        raise ValueError("inner_x_u16 and inner_y_u16 must contain three values")
    if not np.issubdtype(x_inner.dtype, np.integer) or not np.issubdtype(
        y_inner.dtype, np.integer
    ):
        raise TypeError("Auto-contrast interior X/Y controls must be integers")
    if slopes.shape != (4,):
        raise ValueError("extrapolation_slopes_q9 must contain four words")
    if (
        int(x_inner.min()) <= 0
        or int(x_inner.max()) >= U16_MAX
        or np.any(np.diff(x_inner.astype(np.int64)) <= 0)
    ):
        raise ValueError(
            "inner_x_u16 must be strictly increasing inside 0..65535"
        )
    if int(y_inner.min()) < 0 or int(y_inner.max()) > U16_MAX:
        raise ValueError("inner_y_u16 values must fit unsigned words")

    x = np.array(
        [0, *x_inner.astype(np.int64).tolist(), U16_MAX], dtype=np.int32
    )
    y = np.empty(5, dtype=np.int32)
    y[1:4] = y_inner.astype(np.int32)
    first_product = np.array(
        [int(slopes[0]) * int(x_inner[0])], dtype=np.int64
    )
    last_product = np.array(
        [int(slopes[3]) * (U16_MAX - int(x_inner[2]))], dtype=np.int64
    )
    denominator = np.array([512], dtype=np.int64)
    y[0] = int(y[1]) - int(
        _trunc_divide_signed(first_product, denominator)[0]
    )
    y[4] = int(y[3]) + int(
        _trunc_divide_signed(last_product, denominator)[0]
    )
    return x, y


def expand_auto_contrast_s0_kernel(
    kernel_quadrant_i8: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Expand Auto-contrast S0's packed quadrant as ``sub_1005aa40`` does.

    The raw ``RD5X_AUTO_CONTRAST_PARA`` stores the upper-left quadrant,
    including the centre row and column, in eight-byte rows at offset ``+2``.
    ``sub_1005aa40`` (VA ``0x1005aa40``) mirrors it into the internal 15-byte
    row layout consumed by ``sub_1005b6b0``.  This array API omits only the
    unused row padding.

    The size-specific native filters zero-extend each expanded weight before
    converting it to float.  Valid installer kernels are therefore the
    non-negative signed-byte domain represented here.
    """

    size = int(kernel_size)
    if size not in (5, 7, 9):
        raise ValueError("Auto-contrast S0 kernel_size must be 5, 7, or 9")
    quadrant = np.asarray(kernel_quadrant_i8)
    half = size // 2
    expected = (half + 1, half + 1)
    if quadrant.shape != expected or not np.issubdtype(
        quadrant.dtype, np.integer
    ):
        raise ValueError(
            f"kernel_quadrant_i8 must have integer shape {expected}"
        )
    if quadrant.size and (
        int(quadrant.min()) < 0 or int(quadrant.max()) > 0x7F
    ):
        raise ValueError(
            "kernel_quadrant_i8 values must be non-negative signed bytes"
        )

    weights = np.empty((size, size), dtype=np.int16)
    compact = quadrant.astype(np.int16, copy=False)
    for row in range(half + 1):
        for column in range(half + 1):
            value = compact[row, column]
            weights[row, column] = value
            weights[row, size - 1 - column] = value
            weights[size - 1 - row, column] = value
            weights[size - 1 - row, size - 1 - column] = value
    return weights


def _auto_contrast_s0_axis_map_sse(
    source_length: int,
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one ``sub_1005ad90/1005ba50`` SSE coordinate table."""

    if source_length < 2 or target_length < 2:
        raise ValueError("Auto-contrast coordinate-map axes must be at least 2")
    scale = np.float32(
        np.float64(source_length - 1) / np.float64(target_length - 1)
    )
    coordinates = np.arange(target_length, dtype=np.float32)
    mapped = np.multiply(coordinates, scale, dtype=np.float32)
    integer = np.trunc(mapped).astype(np.int64)
    if np.any(integer < 0) or np.any(integer >= source_length):
        raise ValueError("Auto-contrast coordinate map escaped the source axis")
    fraction = np.subtract(mapped, integer.astype(np.float32), dtype=np.float32)
    return integer, fraction


def _auto_contrast_s0_axis_map_x87(
    source_length: int,
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the vertical map whose multiply remains in x87 precision."""

    if source_length < 2 or target_length < 2:
        raise ValueError("Auto-contrast coordinate-map axes must be at least 2")
    scale = np.float32(
        np.float64(source_length - 1) / np.float64(target_length - 1)
    )
    # The DLL reloads the stored float32 scale into x87, multiplies by an
    # integer row number there, calls _ftol2_sse, and only then stores the
    # fractional remainder as float32.  A float32 is dyadic, so this product
    # is exact in binary64 for all practical CImage dimensions.
    mapped = np.arange(target_length, dtype=np.float64) * np.float64(scale)
    integer = np.trunc(mapped).astype(np.int64)
    if np.any(integer < 0) or np.any(integer >= source_length):
        raise ValueError("Auto-contrast coordinate map escaped the source axis")
    fraction = np.asarray(mapped - integer, dtype=np.float32)
    return integer, fraction


def _auto_contrast_s0_horizontal_sample(
    row_u16: np.ndarray,
    indexes: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """Apply the native float32 left/right interpolation to one row."""

    right = np.minimum(indexes + 1, row_u16.shape[0] - 1)
    left_values = row_u16[indexes].astype(np.float32)
    right_values = row_u16[right].astype(np.float32)
    left_weights = np.subtract(
        np.float32(1.0), fractions, dtype=np.float32
    )
    return _add_f32(
        np.multiply(left_values, left_weights, dtype=np.float32),
        np.multiply(right_values, fractions, dtype=np.float32),
    )


def downsample_auto_contrast_s0_luminance(
    luminance_u16: np.ndarray,
    downsample_factor: int,
) -> np.ndarray:
    """Port Auto-contrast S0's local-map setup at ``0x1005ad90..0x1005b380``.

    Divisible image dimensions take ``sub_1005b1c0``'s integer block-average
    path.  Otherwise ``sub_1005ad90`` first stretches the image onto the
    next factor-aligned canvas with endpoint-aligned bilinear sampling, then
    averages each factor square.  X coordinates and interpolation use SSE
    float32; Y coordinate multiplication follows the visible x87 order.
    Both native exits use ``cvttps2dq`` truncation.
    """

    luminance = np.asarray(luminance_u16)
    if luminance.dtype != np.uint16 or luminance.ndim != 2:
        raise ValueError("luminance_u16 must be a 2D uint16 array")
    if luminance.shape[0] < 2 or luminance.shape[1] < 2:
        raise ValueError("Auto-contrast S0 requires both image axes to exceed 1")
    if not isinstance(downsample_factor, (int, np.integer)):
        raise TypeError("downsample_factor must be an integer")
    factor = int(downsample_factor)
    if factor not in (2, 4, 6, 8, 10, 12, 14, 16):
        raise ValueError(
            "downsample_factor must be an even value in 2..16"
        )

    height, width = luminance.shape
    padded_height = ((height + factor - 1) // factor) * factor
    padded_width = ((width + factor - 1) // factor) * factor
    output_height = padded_height // factor
    output_width = padded_width // factor
    reciprocal = np.float32(1.0 / (factor * factor))

    if padded_height == height and padded_width == width:
        sums = (
            luminance.astype(np.int64)
            .reshape(output_height, factor, output_width, factor)
            .sum(axis=(1, 3), dtype=np.int64)
        )
        averaged = np.multiply(
            sums.astype(np.float32), reciprocal, dtype=np.float32
        )
        return np.trunc(averaged).astype(np.uint16)

    x_index, x_fraction = _auto_contrast_s0_axis_map_sse(width, padded_width)
    y_index, y_fraction = _auto_contrast_s0_axis_map_x87(
        height, padded_height
    )
    output = np.empty((output_height, output_width), dtype=np.uint16)
    for output_row in range(output_height):
        accumulated = np.zeros(output_width, dtype=np.float32)
        for within_row in range(factor):
            padded_row = output_row * factor + within_row
            top = int(y_index[padded_row])
            bottom = min(top + 1, height - 1)
            vertical_fraction = y_fraction[padded_row]
            top_values = _auto_contrast_s0_horizontal_sample(
                luminance[top], x_index, x_fraction
            )
            bottom_values = _auto_contrast_s0_horizontal_sample(
                luminance[bottom], x_index, x_fraction
            )
            upper_weight = np.float32(
                np.subtract(
                    np.float32(1.0), vertical_fraction, dtype=np.float32
                )
            )
            samples = _add_f32(
                np.multiply(top_values, upper_weight, dtype=np.float32),
                np.multiply(
                    bottom_values, vertical_fraction, dtype=np.float32
                ),
            ).reshape(output_width, factor)

            # sub_1005ad90 accumulates the even and odd members of each
            # factor-wide group in separate SSE lanes, horizontally adds the
            # lanes, then adds that row to the persistent output accumulator.
            even = np.zeros(output_width, dtype=np.float32)
            odd = np.zeros(output_width, dtype=np.float32)
            for column in range(0, factor, 2):
                even = _add_f32(even, samples[:, column])
                odd = _add_f32(odd, samples[:, column + 1])
            accumulated = _add_f32(accumulated, _add_f32(even, odd))
        averaged = np.multiply(accumulated, reciprocal, dtype=np.float32)
        output[output_row] = np.trunc(averaged).astype(np.uint16)
    return output


def filter_auto_contrast_s0_local_map(
    local_map_u16: np.ndarray,
    kernel_weights_i16: np.ndarray,
    spatial_strength_u8: int,
    clamp_radius_u16: int,
) -> np.ndarray:
    """Port Auto-contrast S0's fixed two-pass local filter.

    ``sub_1005b6b0`` (VA ``0x1005b6b0``) builds row/offset pairs and invokes
    ``sub_1005ce40``, ``sub_1005d150``, or ``sub_1005d4f0`` for 5x5, 7x7,
    or 9x9 kernels.  Each pass reflects without repeating the edge, clamps
    every neighbour to centre +/- ``clamp_radius_u16``, forms the weighted
    float32 mean, and blends it toward the centre by strength/128.  The DLL
    alternates its two temporary images, so exactly two passes are applied.

    The native parameter kernels use non-negative signed-byte weights whose
    total is at most 256.  Under that observed contract all weighted integer
    partial sums remain below 2**24, making this row-major accumulation
    bit-identical to the size-specific SSE grouping.
    """

    local_map = np.asarray(local_map_u16)
    weights = np.asarray(kernel_weights_i16)
    if local_map.dtype != np.uint16 or local_map.ndim != 2:
        raise ValueError("local_map_u16 must be a 2D uint16 array")
    if weights.shape not in ((5, 5), (7, 7), (9, 9)) or not np.issubdtype(
        weights.dtype, np.integer
    ):
        raise ValueError(
            "kernel_weights_i16 must have integer 5x5, 7x7, or 9x9 shape"
        )
    if weights.size and (
        int(weights.min()) < 0 or int(weights.max()) > 0x7F
    ):
        raise ValueError("Auto-contrast kernel weights must be in 0..127")
    total_weight = int(weights.astype(np.int64).sum())
    if not 1 <= total_weight <= 256:
        raise ValueError("Auto-contrast kernel-weight total must be in 1..256")
    for name, value, maximum in (
        ("spatial_strength_u8", spatial_strength_u8, 0xFF),
        ("clamp_radius_u16", clamp_radius_u16, U16_MAX),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= maximum
        ):
            domain = "byte" if maximum == 0xFF else "word"
            raise ValueError(f"{name} must fit an unsigned {domain}")

    size = int(weights.shape[0])
    radius = size // 2
    if local_map.shape[0] <= radius or local_map.shape[1] <= radius:
        raise ValueError(
            "local-map axes must exceed the Auto-contrast kernel radius"
        )
    weights_f32 = weights.astype(np.float32)
    reciprocal = np.float32(1.0 / total_weight)
    strength = int(spatial_strength_u8)
    threshold = np.float32(int(clamp_radius_u16))
    current = local_map.copy()

    for _pass in range(2):
        height, width = current.shape
        reflected = current[
            np.ix_(
                _reflect_without_edge(
                    np.arange(-radius, height + radius), height
                ),
                _reflect_without_edge(
                    np.arange(-radius, width + radius), width
                ),
            )
        ]
        center_i64 = current.astype(np.int64)
        center_f32 = current.astype(np.float32)
        lower = np.subtract(center_f32, threshold, dtype=np.float32)
        upper = np.add(center_f32, threshold, dtype=np.float32)
        weighted = np.zeros(current.shape, dtype=np.float32)
        for row in range(size):
            for column in range(size):
                if not int(weights[row, column]):
                    continue
                sample = reflected[
                    row : row + height, column : column + width
                ].astype(np.float32)
                clipped = np.minimum(np.maximum(sample, lower), upper)
                contribution = np.multiply(
                    clipped, weights_f32[row, column], dtype=np.float32
                )
                weighted = _add_f32(weighted, contribution)
        average = np.trunc(
            np.multiply(weighted, reciprocal, dtype=np.float32)
        ).astype(np.int64)
        delta = average - center_i64
        blended = center_i64 + ((delta * strength) >> 7)
        current = np.clip(blended, 0, U16_MAX).astype(np.uint16)
    return current


def resize_auto_contrast_s0_local_index(
    local_map_u16: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Port ``sub_1005ba50``'s endpoint-aligned local-map expansion."""

    local_map = np.asarray(local_map_u16)
    if local_map.dtype != np.uint16 or local_map.ndim != 2:
        raise ValueError("local_map_u16 must be a 2D uint16 array")
    if len(output_shape) != 2:
        raise ValueError("output_shape must be a (height, width) pair")
    output_height, output_width = (int(value) for value in output_shape)
    if output_height < 2 or output_width < 2:
        raise ValueError("Auto-contrast output axes must be at least 2")
    if local_map.shape == (output_height, output_width):
        return local_map.copy()
    if local_map.shape[0] < 2 or local_map.shape[1] < 2:
        raise ValueError("Auto-contrast local-map axes must be at least 2")

    x_index, x_fraction = _auto_contrast_s0_axis_map_sse(
        local_map.shape[1], output_width
    )
    y_index, y_fraction = _auto_contrast_s0_axis_map_x87(
        local_map.shape[0], output_height
    )
    output = np.empty((output_height, output_width), dtype=np.uint16)
    for row in range(output_height):
        top = int(y_index[row])
        bottom = min(top + 1, local_map.shape[0] - 1)
        vertical_fraction = y_fraction[row]
        upper = _auto_contrast_s0_horizontal_sample(
            local_map[top], x_index, x_fraction
        )
        lower = _auto_contrast_s0_horizontal_sample(
            local_map[bottom], x_index, x_fraction
        )
        upper_weight = np.float32(
            np.subtract(
                np.float32(1.0), vertical_fraction, dtype=np.float32
            )
        )
        interpolated = _add_f32(
            np.multiply(upper, upper_weight, dtype=np.float32),
            np.multiply(lower, vertical_fraction, dtype=np.float32),
        )
        output[row] = np.trunc(interpolated).astype(np.uint16)
    return output


def apply_auto_contrast_s0(
    luminance_u16: np.ndarray,
    kernel_quadrant_i8: np.ndarray,
    x_breakpoints: Sequence[int],
    y_values: Sequence[int],
    blend_q9: Sequence[int],
    *,
    downsample_factor: int,
    kernel_size: int,
    spatial_strength_u8: int,
    clamp_radius_u16: int,
) -> np.ndarray:
    """Port complete ``FmAutoContrastS0_Do`` at VA ``0x1005c350``.

    This composes the exact prepared-parameter kernels recovered from
    ``sub_1005aa40`` (symmetric weight expansion), ``sub_1005ad90`` /
    ``sub_1005b1c0`` / ``sub_1005b380`` (local-map downsample),
    ``sub_1005b6b0`` plus its 5/7/9 filters (two spatial passes), and
    ``sub_1005ba50`` / ``sub_1005bef0`` (map expansion and Q9 application).
    The API accepts the unpacked controls so no scanner/job parameter is
    guessed or baked into the software kernel.
    """

    luminance = np.asarray(luminance_u16)
    if luminance.dtype != np.uint16 or luminance.ndim != 2:
        raise ValueError("luminance_u16 must be a 2D uint16 array")
    weights = expand_auto_contrast_s0_kernel(
        kernel_quadrant_i8, kernel_size
    )
    local_map = downsample_auto_contrast_s0_luminance(
        luminance, downsample_factor
    )
    filtered = filter_auto_contrast_s0_local_map(
        local_map,
        weights,
        spatial_strength_u8,
        clamp_radius_u16,
    )
    local_index = resize_auto_contrast_s0_local_index(
        filtered, luminance.shape
    )
    intercept, blend = build_auto_contrast_s0_luts(
        x_breakpoints, y_values, blend_q9
    )
    return apply_auto_contrast_s0_luts(
        luminance, local_index, intercept, blend
    )


def _normalize_hue_sat_s0_chroma(
    chroma0: np.ndarray,
    chroma1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project chroma onto the DLL's -2048..2047 interpolation square."""

    first = np.asarray(chroma0, dtype=np.int64).copy()
    second = np.asarray(chroma1, dtype=np.int64).copy()
    outside = (first < -2048) | (first > 2047)
    if np.any(outside):
        edge = np.where(first[outside] > 2047, 2047, -2048)
        second[outside] = _trunc_divide_signed(
            edge * second[outside], first[outside]
        )
        first[outside] = edge

    outside = (second < -2048) | (second > 2047)
    if np.any(outside):
        edge = np.where(second[outside] > 2047, 2047, -2048)
        first[outside] = _trunc_divide_signed(
            edge * first[outside], second[outside]
        )
        second[outside] = edge
    return first, second


def _interpolate_hue_sat_s0_corrections(
    luminance: np.ndarray,
    chroma0: np.ndarray,
    chroma1: np.ndarray,
    correction_table_f32: np.ndarray,
) -> np.ndarray:
    """Run the eight-corner SSE arithmetic in ``sub_100539a0``."""

    c0_position = chroma0 + 2048
    c1_position = chroma1 + 2048
    c0_index = c0_position >> 7
    c1_index = c1_position >> 7
    y_index = luminance >> 9
    c0_fraction = chroma0 & 0x7F
    c1_fraction = chroma1 & 0x7F
    y_fraction = luminance & 0x1FF
    c0_complement = 128 - c0_fraction - (c0_index == 31)
    c1_complement = 128 - c1_fraction - (c1_index == 31)
    y_complement = 512 - y_fraction - (y_index == 7)

    c0_fraction_f32 = c0_fraction.astype(np.float32)
    c1_fraction_f32 = c1_fraction.astype(np.float32)
    c0_complement_f32 = c0_complement.astype(np.float32)
    c1_complement_f32 = c1_complement.astype(np.float32)
    y_fraction_f32 = y_fraction.astype(np.float32)
    y_complement_f32 = y_complement.astype(np.float32)

    # 0x10053b26..0x10053b7c forms these four chroma products first, then
    # multiplies them by the upper/lower Y weights.  Keeping both float32
    # operations explicit preserves the shipped SSE rounding points.
    fraction_fraction = np.multiply(
        c1_fraction_f32, c0_fraction_f32, dtype=np.float32
    )
    fraction_complement = np.multiply(
        c1_fraction_f32, c0_complement_f32, dtype=np.float32
    )
    complement_fraction = np.multiply(
        c1_complement_f32, c0_fraction_f32, dtype=np.float32
    )
    complement_complement = np.multiply(
        c1_complement_f32, c0_complement_f32, dtype=np.float32
    )

    upper_weights = (
        np.multiply(
            y_fraction_f32, complement_complement, dtype=np.float32
        ),
        np.multiply(
            y_fraction_f32, fraction_complement, dtype=np.float32
        ),
        np.multiply(
            y_fraction_f32, fraction_fraction, dtype=np.float32
        ),
        np.multiply(
            y_fraction_f32, complement_fraction, dtype=np.float32
        ),
    )
    lower_weights = (
        np.multiply(
            y_complement_f32, complement_complement, dtype=np.float32
        ),
        np.multiply(
            y_complement_f32, fraction_complement, dtype=np.float32
        ),
        np.multiply(
            y_complement_f32, fraction_fraction, dtype=np.float32
        ),
        np.multiply(
            y_complement_f32, complement_fraction, dtype=np.float32
        ),
    )
    upper = y_index + 1
    next_c0 = c0_index + 1
    next_c1 = c1_index + 1
    upper_corners = (
        correction_table_f32[upper, c0_index, c1_index],
        correction_table_f32[upper, c0_index, next_c1],
        correction_table_f32[upper, next_c0, next_c1],
        correction_table_f32[upper, next_c0, c1_index],
    )
    lower_corners = (
        correction_table_f32[y_index, c0_index, c1_index],
        correction_table_f32[y_index, c0_index, next_c1],
        correction_table_f32[y_index, next_c0, next_c1],
        correction_table_f32[y_index, next_c0, c1_index],
    )

    correction = np.multiply(
        upper_corners[0], upper_weights[0][..., None], dtype=np.float32
    )
    for corner, weight in zip(
        upper_corners[1:], upper_weights[1:], strict=True
    ):
        correction = _add_f32(
            correction,
            np.multiply(corner, weight[..., None], dtype=np.float32),
        )
    for corner, weight in zip(lower_corners, lower_weights, strict=True):
        correction = _add_f32(
            correction,
            np.multiply(corner, weight[..., None], dtype=np.float32),
        )

    scale_index = (
        (c0_index == 31).astype(np.int64) * 4
        + (c1_index == 31).astype(np.int64) * 2
        + (y_index == 7).astype(np.int64)
    )
    correction = np.multiply(
        correction,
        HUE_SAT_S0_INTERPOLATION_SCALE_F32[scale_index, None],
        dtype=np.float32,
    )
    # cvttps2dq truncates each finite correction toward zero.
    return np.trunc(correction).astype(np.int64)


def _apply_hue_sat_s0_zero_saturation(
    image_ycc: np.ndarray,
    correction_table_i16: np.ndarray,
    total_balance_i16: np.ndarray,
) -> np.ndarray:
    """Port the saturation-zero centre-line helper at VA ``0x10054290``."""

    flat = image_ycc.reshape(-1, 3)
    luminance = flat[:, 2].astype(np.int64)
    y_index = luminance >> 9
    fraction = luminance & 0x1FF
    current = correction_table_i16[y_index, 16, 16].astype(np.int64)
    following = correction_table_i16[y_index + 1, 16, 16].astype(np.int64)
    correction = np.empty_like(current)
    ordinary = y_index < 7
    if np.any(ordinary):
        ordinary_fraction = fraction[ordinary, None]
        numerator = (
            following[ordinary] * ordinary_fraction
            + current[ordinary] * (512 - ordinary_fraction)
        )
        correction[ordinary] = numerator >> 9
    terminal = ~ordinary
    if np.any(terminal):
        terminal_fraction = fraction[terminal, None]
        numerator = (
            following[terminal] * terminal_fraction
            + current[terminal] * (511 - terminal_fraction)
        )
        correction[terminal] = _trunc_divide_signed(numerator, 511)

    output = np.empty_like(flat)
    output[:, 0] = np.clip(
        32768 + correction[:, 1] + int(total_balance_i16[0]),
        0,
        U16_MAX,
    ).astype(np.uint16)
    output[:, 1] = np.clip(
        32768 + correction[:, 2] + int(total_balance_i16[1]),
        0,
        U16_MAX,
    ).astype(np.uint16)
    output[:, 2] = np.bitwise_and(
        luminance + correction[:, 0], U16_MAX
    ).astype(np.uint16)
    return output.reshape(image_ycc.shape)


def apply_hue_saturation_s0(
    image_ycc_u16: np.ndarray,
    correction_table_i16: np.ndarray,
    saturation_u8: int,
    total_balance_i16: Sequence[int] = (0, 0),
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Port complete ``FmHueSatCorrectionS0_Do`` at VA ``0x10055c10``.

    The native four-word pixel is exposed as the three active words in exact
    ``[C0, C1, Y]`` order.  Chroma is centred on 32768 and the internal Y
    domain is 0..4095.  ``correction_table_i16`` is the runtime lattice at
    ``COM_PARA_FOR_ALL + 8`` with shape ``[9 Y, 33 C0, 33 C1, 3]`` and
    correction channels ``[delta-Y, delta-C0, delta-C1]``.

    This explicit lattice is the exact terminal software boundary: wrapper
    ``sub_10055890`` copies its 9*33*33 signed triplets to padded float32
    storage and never derives them from hue-slope controls.  The core at
    ``0x100539a0`` projects chroma to its square, performs ordered float32
    trilinear interpolation, and wraps the corrected words.  Saturations
    1..99 pre-scale chroma in ``sub_10053ca0``; values above 100 post-scale
    through ``sub_10053c60``; zero uses the integer centre-line helper at
    ``0x10054290`` and its two signed balance words.
    """

    image = _as_u16_rgb(image_ycc_u16, "image_ycc_u16")
    table = _signed_q12(
        np.asarray(correction_table_i16), "correction_table_i16"
    )
    if table.shape != (9, 33, 33, 3):
        raise ValueError(
            "correction_table_i16 must have shape (9, 33, 33, 3)"
        )
    balance = _signed_q12(
        np.asarray(total_balance_i16), "total_balance_i16"
    )
    if balance.shape != (2,):
        raise ValueError("total_balance_i16 must contain two words")
    if (
        not isinstance(saturation_u8, (int, np.integer))
        or not 0 <= int(saturation_u8) <= 0xFF
    ):
        raise ValueError("saturation_u8 must fit an unsigned byte")
    rows = _require_positive_chunk_rows(chunk_rows)
    if image.size and np.any(image[..., 2] > U12_MAX):
        raise ValueError("Hue/Saturation S0 luminance must be in 0..4095")
    if not image.size:
        return image.copy()

    saturation = int(saturation_u8)
    if saturation == 0:
        return _apply_hue_sat_s0_zero_saturation(image, table, balance)

    table_f32 = table.astype(np.float32)
    scale_q10 = (saturation * 1024) // 100
    output = np.empty_like(image)
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        flat = image[start:stop].reshape(-1, 3)
        chroma0 = flat[:, 0].astype(np.int64) - 32768
        chroma1 = flat[:, 1].astype(np.int64) - 32768
        luminance = flat[:, 2].astype(np.int64)
        if saturation < 100:
            chroma0 = (chroma0 * scale_q10) >> 10
            chroma1 = (chroma1 * scale_q10) >> 10
        chroma0, chroma1 = _normalize_hue_sat_s0_chroma(
            chroma0, chroma1
        )
        correction = _interpolate_hue_sat_s0_corrections(
            luminance, chroma0, chroma1, table_f32
        )
        corrected = np.empty_like(flat)
        corrected[:, 0] = np.bitwise_and(
            chroma0 + correction[:, 1] + 32768, U16_MAX
        ).astype(np.uint16)
        corrected[:, 1] = np.bitwise_and(
            chroma1 + correction[:, 2] + 32768, U16_MAX
        ).astype(np.uint16)
        corrected[:, 2] = np.bitwise_and(
            luminance + correction[:, 0], U16_MAX
        ).astype(np.uint16)
        if saturation > 100:
            for channel in (0, 1):
                signed = corrected[:, channel].astype(np.int64) - 32768
                mapped = ((signed * scale_q10) >> 10) + 32768
                corrected[:, channel] = np.clip(
                    mapped, 0, U16_MAX
                ).astype(np.uint16)
        output[start:stop] = corrected.reshape(stop - start, image.shape[1], 3)
    return output


def _expand_film_sharp_symmetric_kernel(
    packed_i16: np.ndarray,
    size: int,
    packed_stride: int,
) -> np.ndarray:
    """Expand the quadrant table used by FilmSharp's convolution wrappers."""

    if size < 1 or size % 2 == 0 or size > packed_stride * 2 - 1:
        raise ValueError(
            "FilmSharp filter size must be odd and fit its packed quadrant"
        )
    radius = size // 2
    expanded = np.empty((size, size), dtype=np.int16)
    for output_y, delta_y in enumerate(range(-radius, radius + 1)):
        packed_y = radius - abs(delta_y)
        for output_x, delta_x in enumerate(range(-radius, radius + 1)):
            packed_x = radius - abs(delta_x)
            expanded[output_y, output_x] = packed_i16[
                packed_y * packed_stride + packed_x
            ]
    return expanded


def decode_film_sharp_parameter_block(
    parameter_block: bytes,
    *,
    offset: int = 0,
) -> dict[str, object]:
    """Decode the exact 0x154-byte ``RD5X_FILM_SHARP_PARA`` structure.

    The final RD5X PCB stores this structure at absolute ``+0x3b2``.  The
    nominal full-scan capture whose SHA-256 begins ``a22aa8b6`` has filter
    sizes ``(5, 1, 13, 11, 5)``; the decoder is parameterized and also
    expands every packed quadrant into the full symmetric kernel consumed by
    the installed workers.

    Offsets were established from ``FmFilmSharp_Do`` at VA ``0x100b9a30``
    and its five convolution wrappers at ``0x100b4f80``, ``0x100b54a0``,
    ``0x100b5e70``, ``0x100b6ed0``, and ``0x100b7590``.  All words are
    little-endian; convolution coefficients and response gain are signed.
    """

    if not isinstance(offset, (int, np.integer)) or int(offset) < 0:
        raise ValueError("offset must be a nonnegative integer")
    try:
        raw = bytes(parameter_block)
    except (TypeError, ValueError) as exc:
        raise TypeError("parameter_block must be bytes-like") from exc
    base = int(offset)
    if len(raw) < base + FILM_SHARP_PARAMETER_BLOCK_SIZE:
        raise ValueError(
            "parameter_block does not contain a complete 0x154-byte "
            "RD5X_FILM_SHARP_PARA structure"
        )

    def u16(relative_offset: int) -> int:
        return int.from_bytes(
            raw[base + relative_offset : base + relative_offset + 2],
            "little",
            signed=False,
        )

    filter_layout = (
        # size byte, packed words, packed stride, scale word
        (0x000, 0x002, 3, 0x014),
        (0x016, 0x018, 3, 0x02A),
        (0x02C, 0x02E, 7, 0x090),
        (0x092, 0x094, 6, 0x0DC),
        (0x0DE, 0x0E0, 5, 0x112),
    )
    sizes: list[int] = []
    scales: list[int] = []
    packed_kernels: list[np.ndarray] = []
    expanded_kernels: list[np.ndarray] = []
    for size_offset, kernel_offset, stride, scale_offset in filter_layout:
        size = raw[base + size_offset]
        packed = np.frombuffer(
            raw,
            dtype="<i2",
            count=stride * stride,
            offset=base + kernel_offset,
        ).astype(np.int16, copy=True)
        sizes.append(size)
        scales.append(u16(scale_offset))
        packed_kernels.append(packed.reshape(stride, stride))
        expanded_kernels.append(
            _expand_film_sharp_symmetric_kernel(packed, size, stride)
        )

    response_gain = int.from_bytes(
        raw[base + 0x152 : base + 0x154], "little", signed=True
    )
    return {
        "filter_sizes": tuple(sizes),
        "filter_scales": tuple(scales),
        "packed_kernels_i16": tuple(packed_kernels),
        "kernels_i16": tuple(expanded_kernels),
        "blend_u16": u16(0x114),
        "mean_radius_u8": raw[base + 0x116],
        "feature_radius_u8": raw[base + 0x117],
        "envelope_controls_u16": (
            u16(0x118),
            u16(0x11A),
            u16(0x11C),
            u16(0x11E),
        ),
        "response_controls_u16": (
            u16(0x120),
            u16(0x122),
            u16(0x124),
            u16(0x126),
        ),
        "tone_x_u16": np.frombuffer(
            raw, dtype="<u2", count=10, offset=base + 0x128
        ).astype(np.uint16, copy=True),
        "tone_y_u16": np.frombuffer(
            raw, dtype="<u2", count=10, offset=base + 0x13C
        ).astype(np.uint16, copy=True),
        "response_clamp_u16": u16(0x150),
        "response_gain_i16": response_gain,
    }


def _film_sharp_convolve_reflect(
    values: np.ndarray,
    kernel_i16: np.ndarray,
    scale_u16: int,
    *,
    unsigned_output: bool,
    chunk_rows: int,
) -> np.ndarray:
    """Run FilmSharp's ordered float32 convolution with reflected framing."""

    source = np.asarray(values)
    kernel = np.asarray(kernel_i16)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("FilmSharp convolution input must be a 2D integer array")
    if (
        kernel.ndim != 2
        or kernel.shape[0] != kernel.shape[1]
        or kernel.shape[0] % 2 == 0
        or kernel.dtype != np.int16
    ):
        raise ValueError("kernel_i16 must be a square odd-sized int16 array")
    if not isinstance(scale_u16, (int, np.integer)) or not 1 <= int(scale_u16) <= U16_MAX:
        raise ValueError("FilmSharp convolution scale must be in 1..65535")
    rows = _require_positive_chunk_rows(chunk_rows)
    output_dtype = np.uint16 if unsigned_output else np.int16
    if source.size == 0:
        return np.empty(source.shape, dtype=output_dtype)

    radius = kernel.shape[0] // 2
    padded = np.pad(source, ((radius, radius), (radius, radius)), mode="reflect")
    reciprocal = np.divide(
        np.float32(1.0), np.float32(int(scale_u16)), dtype=np.float32
    )
    output = np.empty(source.shape, dtype=output_dtype)
    width = source.shape[1]
    nonzero_terms = [
        (kernel_y, kernel_x, np.float32(int(kernel[kernel_y, kernel_x])))
        for kernel_y in range(kernel.shape[0])
        for kernel_x in range(kernel.shape[1])
        if kernel[kernel_y, kernel_x] != 0
    ]

    for start in range(0, source.shape[0], rows):
        stop = min(start + rows, source.shape[0])
        accumulator = np.zeros((stop - start, width), dtype=np.float32)
        for kernel_y, kernel_x, coefficient in nonzero_terms:
            samples = padded[
                start + kernel_y : stop + kernel_y,
                kernel_x : kernel_x + width,
            ].astype(np.float32, copy=False)
            accumulator = _add_f32(
                accumulator, _mul_f32(samples, coefficient)
            )
        scaled = _mul_f32(accumulator, reciprocal)
        if unsigned_output:
            output[start:stop] = np.clip(
                scaled, np.float32(0.0), np.float32(U16_MAX)
            ).astype(np.uint16)
            continue

        # ``cvttps2dq`` produces 0x80000000 for NaN/out-of-range values,
        # after which ``packssdw`` saturates that sentinel to -32768.
        valid = np.isfinite(scaled) & (
            scaled >= np.float32(-2147483648.0)
        ) & (scaled < np.float32(2147483648.0))
        integers = np.full(scaled.shape, -2147483648, dtype=np.int64)
        integers[valid] = np.trunc(scaled[valid]).astype(np.int64)
        output[start:stop] = np.clip(integers, -32768, 32767).astype(np.int16)
    return output


def _film_sharp_ycc_to_rg_u16(
    auxiliary_u16: np.ndarray,
    chroma0_i16: np.ndarray,
    chroma1_i16: np.ndarray,
    matrix_q10000: Sequence[int],
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Port the first two output lanes of worker VA ``0x100b5a00``."""

    matrix_words = _signed_q12(
        np.asarray(matrix_q10000), "matrix_q10000"
    )
    if matrix_words.shape != (9,):
        raise ValueError("matrix_q10000 must contain exactly 9 words")
    coefficients = np.divide(
        matrix_words[:6].astype(np.float32),
        np.float32(10000.0),
        dtype=np.float32,
    )
    rows = _require_positive_chunk_rows(chunk_rows)
    red = np.empty(auxiliary_u16.shape, dtype=np.uint16)
    green = np.empty(auxiliary_u16.shape, dtype=np.uint16)
    for start in range(0, auxiliary_u16.shape[0], rows):
        stop = min(start + rows, auxiliary_u16.shape[0])
        y = auxiliary_u16[start:stop].astype(np.float32)
        c0 = chroma0_i16[start:stop].astype(np.float32)
        c1 = chroma1_i16[start:stop].astype(np.float32)
        first = _add_f32(
            _add_f32(
                _mul_f32(y, coefficients[0]),
                _mul_f32(c0, coefficients[1]),
            ),
            _mul_f32(c1, coefficients[2]),
        )
        second = _add_f32(
            _add_f32(
                _mul_f32(y, coefficients[3]),
                _mul_f32(c0, coefficients[4]),
            ),
            _mul_f32(c1, coefficients[5]),
        )
        red[start:stop] = np.clip(first, 0, U16_MAX).astype(np.uint16)
        green[start:stop] = np.clip(second, 0, U16_MAX).astype(np.uint16)
    return red, green


def _film_sharp_scale_q8(values: np.ndarray, factor: int) -> np.ndarray:
    return (np.asarray(values, dtype=np.int64) * int(factor) + 128) >> 8


def _film_sharp_directional_response_i16(
    green_i16: np.ndarray,
    red_i16: np.ndarray,
    blend_u16: int,
    *,
    chunk_rows: int,
) -> np.ndarray:
    """Port FilmSharp's eight-direction worker at VA ``0x100b8450``."""

    if green_i16.shape != red_i16.shape or green_i16.dtype != np.int16 or red_i16.dtype != np.int16:
        raise ValueError("directional inputs must be matching 2D int16 arrays")
    if not isinstance(blend_u16, (int, np.integer)) or not 0 <= int(blend_u16) <= U16_MAX:
        raise ValueError("blend_u16 must fit an unsigned 16-bit word")
    rows = _require_positive_chunk_rows(chunk_rows)
    if green_i16.size == 0:
        return np.empty(green_i16.shape, dtype=np.int16)

    blend = int(blend_u16)
    other = 256 - blend
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    green_pad = np.pad(green_i16, ((2, 2), (2, 2)), mode="reflect")
    red_pad = np.pad(red_i16, ((2, 2), (2, 2)), mode="reflect")
    output = np.empty(green_i16.shape, dtype=np.int16)
    width = green_i16.shape[1]

    def shifted(
        padded: np.ndarray,
        start: int,
        stop: int,
        delta_y: int,
        delta_x: int,
    ) -> np.ndarray:
        return padded[
            start + 2 + delta_y : stop + 2 + delta_y,
            2 + delta_x : 2 + delta_x + width,
        ].astype(np.int64)

    for start in range(0, green_i16.shape[0], rows):
        stop = min(start + rows, green_i16.shape[0])
        green0 = green_i16[start:stop].astype(np.int64)
        red0 = red_i16[start:stop].astype(np.int64)
        best_absolute = np.full(green0.shape, np.iinfo(np.int64).max, dtype=np.int64)
        best_direction = np.zeros(green0.shape, dtype=np.uint8)

        for index, (delta_y, delta_x) in enumerate(directions):
            green_one = shifted(
                green_pad, start, stop, delta_y, delta_x
            )
            green_two = shifted(
                green_pad, start, stop, delta_y * 2, delta_x * 2
            )
            residual = green0 - _film_sharp_scale_q8(
                green_two, blend
            ) - _film_sharp_scale_q8(green_one, other)
            absolute = np.abs(residual)
            better = absolute < best_absolute
            best_absolute[better] = absolute[better]
            best_direction[better] = index

        candidate = green0.copy()
        force_zero = green0 == 0
        for index, (delta_y, delta_x) in enumerate(directions):
            selected = best_direction == index
            if not np.any(selected):
                continue
            red_one = shifted(red_pad, start, stop, delta_y, delta_x)
            red_two = shifted(red_pad, start, stop, delta_y * 2, delta_x * 2)
            red_side = _film_sharp_scale_q8(
                red_two, blend
            ) + _film_sharp_scale_q8(red_one, other)
            red_nonzero = red0 != 0
            red_opposite = (red_side < 0) != (red0 < 0)
            force_zero |= selected & red_nonzero & red_opposite
            ratio_mask = (
                selected
                & red_nonzero
                & ~red_opposite
                & (np.abs(red_side) <= np.abs(red0))
            )
            if np.any(ratio_mask):
                product = red_side[ratio_mask] * green0[ratio_mask]
                product = (product + 0x80000000) % 0x100000000 - 0x80000000
                candidate[ratio_mask] = _trunc_divide_signed(
                    product, red0[ratio_mask]
                )

            green_one = shifted(
                green_pad, start, stop, -delta_y, -delta_x
            )
            green_two = shifted(
                green_pad, start, stop, -delta_y * 2, -delta_x * 2
            )
            opposite_side = _film_sharp_scale_q8(
                green_two, blend
            ) + _film_sharp_scale_q8(green_one, other)
            green_opposite = (opposite_side < 0) != (green0 < 0)
            force_zero |= selected & green_opposite
            replace = (
                selected
                & ~force_zero
                & (np.abs(opposite_side) < np.abs(candidate))
            )
            candidate[replace] = opposite_side[replace]

        candidate[force_zero] = 0
        output[start:stop] = np.clip(candidate, -32768, 32767).astype(np.int16)
    return output


def _film_sharp_box_mean_abs_i16(
    values_i16: np.ndarray,
    radius: int,
    *,
    chunk_rows: int,
) -> np.ndarray:
    """Port the three reflected mean-absolute workers used by FilmSharp."""

    values = np.asarray(values_i16)
    if values.ndim != 2 or values.dtype != np.int16:
        raise ValueError("mean-absolute input must be a 2D int16 array")
    if not isinstance(radius, (int, np.integer)) or not 0 <= int(radius) <= 4:
        raise ValueError("FilmSharp mean radius must be in 0..4")
    rows = _require_positive_chunk_rows(chunk_rows)
    radius = int(radius)
    absolute = np.abs(values.astype(np.int32))
    if radius == 0:
        return absolute.astype(np.uint16).view(np.int16)
    if values.size == 0:
        return np.empty(values.shape, dtype=np.int16)

    padded = np.pad(absolute, ((radius, radius), (radius, radius)), mode="reflect")
    window = radius * 2 + 1
    reciprocal = np.divide(
        np.float32(1.0), np.float32(window * window), dtype=np.float32
    )
    output = np.empty(values.shape, dtype=np.int16)
    for start in range(0, values.shape[0], rows):
        stop = min(start + rows, values.shape[0])
        region = padded[start : stop + radius * 2]
        horizontal_prefix = np.pad(
            np.cumsum(region, axis=1, dtype=np.int64),
            ((0, 0), (1, 0)),
            mode="constant",
        )
        horizontal = (
            horizontal_prefix[:, window:] - horizontal_prefix[:, :-window]
        )
        vertical_prefix = np.pad(
            np.cumsum(horizontal, axis=0, dtype=np.int64),
            ((1, 0), (0, 0)),
            mode="constant",
        )
        sums = vertical_prefix[window:] - vertical_prefix[:-window]
        means = _mul_f32(sums.astype(np.float32), reciprocal).astype(np.int64)
        output[start:stop] = np.bitwise_and(means, U16_MAX).astype(
            np.uint16
        ).view(np.int16)
    return output


def _build_film_sharp_spatial_features_from_parameters(
    image_c1c2y: np.ndarray,
    parameters: dict[str, object],
    matrix_q10000: Sequence[int],
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = _as_u16_rgb(image_c1c2y, "image_c1c2y")
    rows = _require_positive_chunk_rows(chunk_rows)
    kernels = parameters["kernels_i16"]
    scales = parameters["filter_scales"]
    sizes = parameters["filter_sizes"]
    if not isinstance(kernels, tuple) or not isinstance(scales, tuple) or not isinstance(sizes, tuple):
        raise TypeError("decoded FilmSharp parameters have an invalid layout")

    first_kernel = kernels[0]
    if sizes[0] == 1 and int(first_kernel[0, 0]) == scales[0]:
        auxiliary = image[..., 2].copy()
    else:
        auxiliary = _film_sharp_convolve_reflect(
            image[..., 2],
            first_kernel,
            scales[0],
            unsigned_output=True,
            chunk_rows=rows,
        )

    centered0 = np.bitwise_xor(image[..., 0], np.uint16(0x8000)).view(
        np.int16
    )
    centered1 = np.bitwise_xor(image[..., 1], np.uint16(0x8000)).view(
        np.int16
    )
    second_kernel = kernels[1]
    if sizes[1] == 1 and int(second_kernel[0, 0]) == scales[1]:
        filtered0 = centered0
        filtered1 = centered1
    else:
        filtered0 = _film_sharp_convolve_reflect(
            centered0,
            second_kernel,
            scales[1],
            unsigned_output=False,
            chunk_rows=rows,
        )
        filtered1 = _film_sharp_convolve_reflect(
            centered1,
            second_kernel,
            scales[1],
            unsigned_output=False,
            chunk_rows=rows,
        )

    red, green = _film_sharp_ycc_to_rg_u16(
        auxiliary,
        filtered0,
        filtered1,
        matrix_q10000,
        chunk_rows=rows,
    )
    red_filtered = _film_sharp_convolve_reflect(
        red,
        kernels[2],
        scales[2],
        unsigned_output=False,
        chunk_rows=rows,
    )
    del red
    green_filtered = _film_sharp_convolve_reflect(
        green,
        kernels[2],
        scales[2],
        unsigned_output=False,
        chunk_rows=rows,
    )
    del green
    combined = _film_sharp_directional_response_i16(
        green_filtered,
        red_filtered,
        int(parameters["blend_u16"]),
        chunk_rows=rows,
    )
    del red_filtered

    mean_radius = int(parameters["mean_radius_u8"])
    green_mean = _film_sharp_box_mean_abs_i16(
        green_filtered, mean_radius, chunk_rows=rows
    )
    del green_filtered
    combined_mean = _film_sharp_box_mean_abs_i16(
        combined, mean_radius, chunk_rows=rows
    )
    del combined

    features = np.empty(image.shape[:2] + (2,), dtype=np.uint16)
    denominator = green_mean.astype(np.int64)
    numerator = combined_mean.view(np.uint16).astype(np.int64) * 0x7FFF
    ratio = np.zeros(denominator.shape, dtype=np.int64)
    nonzero = denominator != 0
    ratio[nonzero] = _trunc_divide_signed(
        numerator[nonzero], denominator[nonzero]
    )
    ratio[ratio > U16_MAX] = U16_MAX
    features[..., 0] = np.bitwise_and(ratio, U16_MAX).astype(np.uint16)
    del green_mean, combined_mean, ratio

    signed_feature = _film_sharp_convolve_reflect(
        auxiliary,
        kernels[3],
        scales[3],
        unsigned_output=False,
        chunk_rows=rows,
    )
    feature_mean = _film_sharp_box_mean_abs_i16(
        signed_feature,
        int(parameters["feature_radius_u8"]),
        chunk_rows=rows,
    )
    features[..., 1] = feature_mean.view(np.uint16)
    detail = _film_sharp_convolve_reflect(
        auxiliary,
        kernels[4],
        scales[4],
        unsigned_output=False,
        chunk_rows=rows,
    )
    return auxiliary, features, detail


def build_film_sharp_spatial_features(
    image_c1c2y: np.ndarray,
    parameter_block: bytes,
    matrix_q10000: Sequence[int] = LS600_G0_TYPE1_YCC_TO_RGB_Q10000,
    *,
    parameter_offset: int = 0,
    chunk_rows: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build all three upstream arrays consumed by FilmSharp's compositor.

    This is the complete selected spatial graph below ``FmFilmSharp_Do``:
    auxiliary-Y and chroma filtering, two-lane YCC-to-R/G, the 13x13 R/G
    filter, eight-direction nonlinear response, three reflected mean-absolute
    windows, feature ratio, 11x11 feature filtering, and 5x5 signed detail.
    It returns ``(auxiliary_Y, feature_indexes, signed_detail)`` in exactly the
    public shapes required by :func:`apply_film_sharp_final_ycc`.
    """

    parameters = decode_film_sharp_parameter_block(
        parameter_block, offset=parameter_offset
    )
    return _build_film_sharp_spatial_features_from_parameters(
        image_c1c2y,
        parameters,
        matrix_q10000,
        chunk_rows=chunk_rows,
    )


def apply_film_sharp_full_ycc(
    image_c1c2y: np.ndarray,
    parameter_block: bytes,
    matrix_q10000: Sequence[int] = LS600_G0_TYPE1_YCC_TO_RGB_Q10000,
    *,
    parameter_offset: int = 0,
    chunk_rows: int = 64,
) -> np.ndarray:
    """Run the complete parameterized FilmSharp stage from YCC to YCC.

    The normal full-scan target can pass its complete final PCB together with
    ``parameter_offset=FILM_SHARP_PARAMETER_PCB_OFFSET``.  This function uses
    no fitted controls: all five kernels, two radii, nonlinear blend, and
    three LUT families come directly from the 0x154-byte captured structure.
    """

    parameters = decode_film_sharp_parameter_block(
        parameter_block, offset=parameter_offset
    )
    auxiliary, features, detail = (
        _build_film_sharp_spatial_features_from_parameters(
            image_c1c2y,
            parameters,
            matrix_q10000,
            chunk_rows=chunk_rows,
        )
    )
    envelope = build_film_sharp_envelope_lut_u16(
        *parameters["envelope_controls_u16"]
    )
    response = build_film_sharp_response_lut_i16(
        *parameters["response_controls_u16"],
        parameters["response_clamp_u16"],
        parameters["response_gain_i16"],
    )
    tone = build_film_sharp_tone_lut_u16(
        parameters["tone_x_u16"], parameters["tone_y_u16"]
    )
    return apply_film_sharp_final_ycc(
        image_c1c2y,
        auxiliary,
        features,
        detail,
        envelope,
        response,
        tone,
        chunk_rows=chunk_rows,
    )


def build_film_sharp_envelope_lut_u16(
    center_u16: int,
    outer_value_u16: int,
    half_width_u16: int,
    inner_value_u16: int,
) -> np.ndarray:
    """Port FilmSharp's 65,536-word LUT builder at VA ``0x100b4b90``.

    ``FmFilmSharp_Do`` reaches this helper through final wrapper
    ``sub_100b97f0``.  The four arguments are the unsigned words at offsets
    ``+0x118``, ``+0x11a``, ``+0x11c``, and ``+0x11e`` of the packed
    ``RD5X_FILM_SHARP_PARA`` block.  The transition uses the DLL's two
    sequential signed 64-bit divisions (first by ``half_width_u16``, then by
    two); combining those divisions would change negative rounding.
    """

    parameters = (
        ("center_u16", center_u16),
        ("outer_value_u16", outer_value_u16),
        ("half_width_u16", half_width_u16),
        ("inner_value_u16", inner_value_u16),
    )
    for name, value in parameters:
        if not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= U16_MAX:
            raise ValueError(f"{name} must be an unsigned 16-bit integer")

    center = int(center_u16)
    outer = int(outer_value_u16)
    half_width = int(half_width_u16)
    inner = int(inner_value_u16)
    output = np.empty(65536, dtype=np.uint16)
    cursor = max(center - half_width, 0)
    output[:cursor] = np.uint16(inner)
    stop = min(center + half_width, 65536)

    if cursor < stop:
        positions = np.arange(cursor, stop, dtype=np.int64)
        numerator = (positions - center + half_width) * (outer - inner)
        ramp = _trunc_divide_signed(numerator, half_width)
        ramp = _trunc_divide_signed(ramp, 2) + inner
        output[cursor:stop] = ramp.astype(np.uint16)
        cursor = stop
    output[cursor:] = np.uint16(outer)
    return output


def build_film_sharp_tone_lut_u16(
    x_breakpoints_u16: Sequence[int],
    y_values_u16: Sequence[int],
) -> np.ndarray:
    """Port FilmSharp's ten-point LUT builder at VA ``0x100b4d20``.

    The controls are the ten unsigned X words at parameter offset ``+0x128``
    and ten unsigned Y words at ``+0x13c``.  Segment products and their sum
    retain x86 signed-32-bit wrap before ``idiv``; each quotient is stored as
    its low 16 bits.  Reversed/equal controls are not normalized here: the
    native worker simply skips any segment whose endpoint is not ahead of its
    current cursor, and this port preserves that behavior.
    """

    x_raw = np.asarray(x_breakpoints_u16)
    y_raw = np.asarray(y_values_u16)
    if x_raw.shape != (10,) or y_raw.shape != (10,):
        raise ValueError("FilmSharp tone controls must each contain ten values")
    if not np.issubdtype(x_raw.dtype, np.integer) or not np.issubdtype(
        y_raw.dtype, np.integer
    ):
        raise TypeError("FilmSharp tone controls must be integers")
    if (
        int(x_raw.min()) < 0
        or int(x_raw.max()) > U16_MAX
        or int(y_raw.min()) < 0
        or int(y_raw.max()) > U16_MAX
    ):
        raise ValueError("FilmSharp tone controls must fit unsigned 16-bit words")

    x = x_raw.astype(np.int64, copy=False)
    y = y_raw.astype(np.int64, copy=False)
    output = np.empty(65536, dtype=np.uint16)
    cursor = int(x[0])
    output[:cursor] = np.uint16(y[0])

    for index in range(1, 10):
        stop = int(x[index])
        if cursor < stop:
            positions = np.arange(cursor, stop, dtype=np.int64)
            numerator = (positions - int(x[index - 1])) * int(y[index])
            numerator += (stop - positions) * int(y[index - 1])
            numerator = (numerator + 0x80000000) % 0x100000000 - 0x80000000
            values = _trunc_divide_signed(
                numerator, stop - int(x[index - 1])
            )
            output[cursor:stop] = values.astype(np.uint16)
            cursor = stop
    output[cursor:] = np.uint16(y[9])
    return output


def build_film_sharp_response_lut_i16(
    transition_center_u16: int,
    outer_slope_u16: int,
    transition_half_width_u16: int,
    inner_slope_u16: int,
    clamp_limit_u16: int,
    gain_i16: int,
) -> np.ndarray:
    """Port FilmSharp's signed response-table builder at VA ``0x100b48a0``.

    The arguments map in order to packed parameter offsets ``+0x120``,
    ``+0x122``, ``+0x124``, ``+0x126``, ``+0x150``, and ``+0x152``.
    Products and running accumulators wrap as x86 signed 32-bit values;
    divisions truncate toward zero.  The middle and outer regions also
    reproduce the native signed-16 narrowing between division by 65,535 and
    multiplication by ``gain_i16``.

    ``sub_100b97f0`` allocates 65,537 words around a centered pointer, but
    compositor ``sub_100b96c0`` can index only signed-int16 values
    ``-32768..32767``.  This function therefore returns the complete active
    65,536-entry domain, indexed as ``signed_value + 32768``; the native
    allocation's unobservable ``+32768`` word is intentionally omitted.
    """

    unsigned_parameters = (
        ("transition_center_u16", transition_center_u16),
        ("outer_slope_u16", outer_slope_u16),
        ("transition_half_width_u16", transition_half_width_u16),
        ("inner_slope_u16", inner_slope_u16),
        ("clamp_limit_u16", clamp_limit_u16),
    )
    for name, value in unsigned_parameters:
        if not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= U16_MAX:
            raise ValueError(f"{name} must be an unsigned 16-bit integer")
    if (
        not isinstance(gain_i16, (int, np.integer))
        or not -32768 <= int(gain_i16) <= U16_MAX
    ):
        raise ValueError("gain_i16 must be a signed or encoded 16-bit integer")

    center = int(transition_center_u16)
    outer_slope = int(outer_slope_u16)
    half_width = int(transition_half_width_u16)
    inner_slope = int(inner_slope_u16)
    limit = int(clamp_limit_u16)
    gain = _wrap_signed_i16(int(gain_i16))
    if center - half_width > 32768:
        raise ValueError(
            "FilmSharp response parameters exceed the native centered allocation"
        )
    output = np.empty(65536, dtype=np.int16)

    def scaled_response(accumulator: int, *, narrow_quotient: bool) -> int:
        quotient = _trunc_divide_scalar(accumulator, U16_MAX)
        if narrow_quotient:
            quotient = _wrap_signed_i16(quotient)
        product = _wrap_signed_i32(quotient * gain)
        return _trunc_divide_scalar(product, 0x7FF)

    def stored(value: int) -> np.int16:
        clipped = max(-limit, min(int(value), limit))
        return np.int16(_wrap_signed_i16(clipped))

    def write_pair(index: int, value: int) -> None:
        if index < 32768:
            output[32768 + index] = stored(value)
        output[32768 - index] = stored(_wrap_signed_i32(-value))

    index = 0
    left_edge = center - half_width
    if left_edge >= 0:
        accumulator = 0
        while index <= left_edge:
            write_pair(index, scaled_response(accumulator, narrow_quotient=False))
            accumulator = _wrap_signed_i32(accumulator + inner_slope)
            index += 1

    right_edge = min(center + half_width, 32768)
    if index <= right_edge:
        slope_delta = outer_slope - inner_slope
        accumulator = _wrap_signed_i32(
            (index - center + half_width) * slope_delta
        )
        while index <= right_edge:
            interpolated_slope = _trunc_divide_scalar(accumulator, half_width)
            interpolated_slope = (
                _trunc_divide_scalar(interpolated_slope, 2) + inner_slope
            )
            product = _wrap_signed_i32(interpolated_slope * index)
            write_pair(index, scaled_response(product, narrow_quotient=True))
            accumulator = _wrap_signed_i32(accumulator + slope_delta)
            index += 1

    if index < 32768:
        accumulator = _wrap_signed_i32(index * outer_slope)
        while index < 32768:
            write_pair(index, scaled_response(accumulator, narrow_quotient=True))
            accumulator = _wrap_signed_i32(accumulator + outer_slope)
            index += 1

    negative_endpoint = _wrap_signed_i32(-32768 * outer_slope)
    output[0] = stored(
        scaled_response(negative_endpoint, narrow_quotient=True)
    )
    return output


def apply_film_sharp_final_ycc(
    image_c1c2y: np.ndarray,
    auxiliary_luminance_u16: np.ndarray,
    feature_indexes_u16: np.ndarray,
    signed_detail_i16: np.ndarray,
    envelope_lut_u16: np.ndarray,
    response_lut_i16: np.ndarray,
    tone_lut_u16: np.ndarray,
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Port FilmSharp's exact final compositor at VA ``0x100b96c0``.

    Wrapper ``sub_100b97f0`` maps the original four-word YCC image, a
    two-word feature-index image, and a signed-detail image before calling
    this worker.  The compositor reads its baseline from the original image's
    *fourth* (auxiliary) word and writes only the visible Y word; C1 and C2 are
    unchanged.  Requiring ``auxiliary_luminance_u16`` separately preserves
    that ABI instead of assuming it duplicates visible Y.

    The two feature lanes select the ``0x100b4b90`` envelope LUT and
    ``0x100b4d20`` tone LUT.  Each multiplication is followed separately by
    signed division by 65,535, then the signed result selects the active
    ``0x100b48a0`` response domain.  These arrays are runtime outputs of the
    unresolved upstream spatial graph and are therefore all explicit here.
    """

    image = _as_u16_rgb(image_c1c2y, "image_c1c2y")
    auxiliary = np.asarray(auxiliary_luminance_u16)
    features = np.asarray(feature_indexes_u16)
    detail = np.asarray(signed_detail_i16)
    envelope = np.asarray(envelope_lut_u16)
    response = np.asarray(response_lut_i16)
    tone = np.asarray(tone_lut_u16)
    shape = image.shape[:2]
    if auxiliary.shape != shape or auxiliary.dtype != np.uint16:
        raise ValueError(
            "auxiliary_luminance_u16 must match the image as a 2D uint16 array"
        )
    if features.shape != shape + (2,) or features.dtype != np.uint16:
        raise ValueError(
            "feature_indexes_u16 must have shape (H, W, 2) and dtype uint16"
        )
    if detail.shape != shape or detail.dtype != np.int16:
        raise ValueError("signed_detail_i16 must match the image as a 2D int16 array")
    if envelope.shape != (65536,) or envelope.dtype != np.uint16:
        raise ValueError("envelope_lut_u16 must have shape (65536,) and dtype uint16")
    if response.shape != (65536,) or response.dtype != np.int16:
        raise ValueError("response_lut_i16 must have shape (65536,) and dtype int16")
    if tone.shape != (65536,) or tone.dtype != np.uint16:
        raise ValueError("tone_lut_u16 must have shape (65536,) and dtype uint16")
    rows = _require_positive_chunk_rows(chunk_rows)

    output = image.copy()
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        feature_chunk = features[start:stop]
        first = envelope[feature_chunk[..., 0]].astype(np.int64)
        first *= detail[start:stop].astype(np.int64)
        first = _trunc_divide_signed(first, U16_MAX)
        second = tone[feature_chunk[..., 1]].astype(np.int64)
        second *= first
        response_indexes = _trunc_divide_signed(second, U16_MAX) + 32768
        correction = response[response_indexes].astype(np.int64)
        luminance = auxiliary[start:stop].astype(np.int64) - correction
        output[start:stop, ..., 2] = np.clip(luminance, 0, U16_MAX).astype(
            np.uint16
        )
    return output


def apply_film_lut(raw_bgr: np.ndarray, lut_rgb: np.ndarray) -> np.ndarray:
    """Apply the exact RD5X 3x4096 Nega film LUT and return RGB uint16.

    LS-600 FULL dumps store B, G, R slots.  ``sub_1005ebe0`` applies the third
    supplied table to slot 0, the second to slot 1, and the first to slot 2.
    The CorrectServ generic payload is R, G, B, so the explicit output reorder
    below is both the inversion and the BGR-to-RGB conversion.
    """

    raw = _as_u16_rgb(raw_bgr, "raw_bgr")
    lut = np.asarray(lut_rgb, dtype=np.uint16)
    if lut.shape != (3, 4096):
        raise ValueError("lut_rgb must have shape (3, 4096)")
    if raw.size and int(raw.max()) > U12_MAX:
        raise ValueError("film LUT input exceeds the recovered 12-bit 0..4095 domain")

    output = np.empty(raw.shape, dtype=np.uint16)
    output[..., 0] = lut[0][raw[..., 2]]
    output[..., 1] = lut[1][raw[..., 1]]
    output[..., 2] = lut[2][raw[..., 0]]
    return output


def _piecewise_linear_u16_lut(
    x_values: np.ndarray, y_values: np.ndarray, count: int
) -> np.ndarray:
    """Build one 65536-entry table with ``sub_100859e0`` integer order."""

    x = np.asarray(x_values)
    y = np.asarray(y_values)
    if x.shape != (20,) or y.shape != (20,):
        raise ValueError("each RGB-to-YCC control array must contain 20 values")
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(
        y.dtype, np.integer
    ):
        raise TypeError("RGB-to-YCC controls must be integer values")
    if not 1 <= int(count) <= 20:
        raise ValueError("each RGB-to-YCC control count must be in 1..20")

    active_x = x[: int(count)].astype(np.int64)
    active_y = y[: int(count)].astype(np.int64)
    if (
        np.any(active_x < 0)
        or np.any(active_x > U16_MAX)
        or np.any(active_y < 0)
        or np.any(active_y > U16_MAX)
    ):
        raise ValueError("RGB-to-YCC controls must fit unsigned 16-bit words")
    if np.any(np.diff(active_x) <= 0):
        raise ValueError("active RGB-to-YCC X controls must be strictly increasing")

    output = np.empty(65536, dtype=np.uint16)
    first_x = int(active_x[0])
    output[: first_x + 1] = np.uint16(active_y[0])
    cursor = first_x + 1
    for index in range(1, int(count)):
        next_x = int(active_x[index])
        if cursor <= next_x:
            positions = np.arange(cursor, next_x + 1, dtype=np.int64)
            # The DLL forms both differences in the opposite direction and
            # uses signed IDIV.  Keeping that form preserves truncation toward
            # zero for descending as well as ascending control segments.
            numerator = (int(active_y[index - 1]) - int(active_y[index])) * (
                positions - int(active_x[index - 1])
            )
            denominator = int(active_x[index - 1]) - next_x
            values = int(active_y[index - 1]) + _trunc_divide_signed(
                numerator, denominator
            )
            output[cursor : next_x + 1] = values.astype(np.uint16)
            cursor = next_x + 1
    if cursor <= U16_MAX:
        output[cursor:] = np.uint16(active_y[-1])
    return output


def build_rgb_to_ycc_luts_u16(
    x_controls_rgb: np.ndarray,
    y_controls_rgb: np.ndarray,
    counts_rgb: Sequence[int],
) -> np.ndarray:
    """Build the three exact S1 input-transfer LUTs in native B, G, R order.

    ``RD5X_RGB_TO_YCC_PARA`` stores three 20-word X arrays at +0x1c,
    +0x44, +0x6c and three Y arrays at +0x94, +0xbc, +0xe4.  Those arrays
    and their count bytes at +0x18..+0x1a are R, G, B.  The setup routine
    ``sub_100859e0`` deliberately reverses them for the image's B, G, R
    slots, which is why this function's result is labelled ``luts_bgr``.
    """

    x = np.asarray(x_controls_rgb)
    y = np.asarray(y_controls_rgb)
    counts = np.asarray(counts_rgb)
    if x.shape != (3, 20) or y.shape != (3, 20):
        raise ValueError("x_controls_rgb and y_controls_rgb must have shape (3, 20)")
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts_rgb must contain three integer counts")

    output = np.empty((3, 65536), dtype=np.uint16)
    for rgb_channel in range(3):
        bgr_channel = 2 - rgb_channel
        output[bgr_channel] = _piecewise_linear_u16_lut(
            x[rgb_channel], y[rgb_channel], int(counts[rgb_channel])
        )
    return output


def apply_rgb_to_ycc_s1(
    image_rgb: np.ndarray,
    matrix_q10000: np.ndarray,
    input_offsets_rgb: np.ndarray,
    transfer_luts_bgr: np.ndarray,
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Apply the exact RD5X ``FmRgbToYccS1`` pixel kernel.

    The RGB input/output API hides the DLL's interleaved B, G, R, padding
    storage.  The returned channels are the vendor's ``C1, C2, Y`` slots.
    ``matrix_q10000`` is the nine-word flat vendor matrix.  The three signed
    input/anchor offsets are R, G, B.  The transfer LUTs are B, G, R as
    returned by :func:`build_rgb_to_ycc_luts_u16`.

    The kernel and matrix are recovered, but the normal job's transfer
    controls and offsets are outputs of SpecialPcb's runtime calculation.
    Requiring them here prevents a static CorrectServ curve from being
    silently presented as a runtime-derived job block when none was supplied.
    """

    image = _as_u16_rgb(image_rgb, "image_rgb")
    coefficients = _q10000_coefficients(np.asarray(matrix_q10000))
    offsets = _signed_q12(input_offsets_rgb, "input_offsets_rgb")
    if offsets.shape != (3,):
        raise ValueError("input_offsets_rgb must contain three R, G, B words")
    luts = np.asarray(transfer_luts_bgr)
    if luts.shape != (3, 65536) or luts.dtype != np.uint16:
        raise ValueError("transfer_luts_bgr must have shape (3, 65536) and dtype uint16")
    rows = _require_positive_chunk_rows(chunk_rows)

    # sub_100859e0 computes this in x87 from stored float32 coefficients and
    # then rounds it once to float32.  Products are exact in float64 for the
    # signed-i16/float32 operand ranges used here.
    anchor = np.float32(
        float(offsets[0]) * float(coefficients[0])
        + float(offsets[1]) * float(coefficients[1])
        + float(offsets[2]) * float(coefficients[2])
        - 32768.0
    )

    output = np.empty(image.shape, dtype=np.uint16)
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        native = image[start:stop, :, ::-1]
        raw_b = native[..., 0].astype(np.float32)
        raw_g = native[..., 1].astype(np.float32)
        raw_r = native[..., 2].astype(np.float32)
        lut_b = luts[0][native[..., 0]].astype(np.float32)
        lut_g = luts[1][native[..., 1]].astype(np.float32)
        lut_r = luts[2][native[..., 2]].astype(np.float32)

        c1 = _add_f32(
            _add_f32(_mul_f32(lut_b, coefficients[5]), _mul_f32(lut_g, coefficients[4])),
            _mul_f32(lut_r, coefficients[3]),
        )
        c2 = _add_f32(
            _add_f32(_mul_f32(lut_b, coefficients[8]), _mul_f32(lut_g, coefficients[7])),
            _mul_f32(lut_r, coefficients[6]),
        )
        luminance = _add_f32(
            _add_f32(
                _add_f32(
                    _mul_f32(raw_b, coefficients[2]),
                    _mul_f32(raw_g, coefficients[1]),
                ),
                _mul_f32(raw_r, coefficients[0]),
            ),
            anchor,
        )
        output[start:stop, ..., 0] = _pack_signed_f32_to_u16(c1)
        output[start:stop, ..., 1] = _pack_signed_f32_to_u16(c2)
        output[start:stop, ..., 2] = _pack_signed_f32_to_u16(luminance)
    return output


def _validate_ycc_correction_curve(
    x_values: Sequence[int],
    y_values: Sequence[int],
    count: int,
    *,
    y_signed: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate one packed 20-point curve without changing word semantics."""

    x = np.asarray(x_values)
    y = np.asarray(y_values)
    if x.shape != (20,) or y.shape != (20,):
        raise ValueError("YCC-correction control arrays must contain 20 values")
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(
        y.dtype, np.integer
    ):
        raise TypeError("YCC-correction controls must be integer values")
    if not isinstance(count, (int, np.integer)) or not 1 <= int(count) <= 20:
        raise ValueError("YCC-correction control count must be in 1..20")
    active = int(count)
    x_words = x.astype(np.uint16, copy=False)
    if np.any(x.astype(np.int64) < 0) or np.any(x.astype(np.int64) > U16_MAX):
        raise ValueError("YCC-correction X controls must fit unsigned words")
    if np.any(np.diff(x_words[:active].astype(np.int64)) <= 0):
        raise ValueError("active YCC-correction X controls must strictly increase")
    if y_signed:
        y_words = _signed_q12(y, "YCC-correction signed controls")
    else:
        if np.any(y.astype(np.int64) < 0) or np.any(y.astype(np.int64) > U16_MAX):
            raise ValueError("YCC-correction Y controls must fit unsigned words")
        y_words = y.astype(np.uint16, copy=False)
    return x_words, y_words, active


def _wrapped_i32_products(values: np.ndarray) -> np.ndarray:
    """Return signed low dwords after the builders' two-operand ``imul``."""

    words = np.asarray(values, dtype=np.int64)
    return ((words + 0x80000000) % 0x100000000) - 0x80000000


def _build_ycc_correction_piecewise_i32(
    x_values: np.ndarray,
    y_values: np.ndarray,
    count: int,
) -> np.ndarray:
    """Common inclusive piecewise interpolation used by 0x10087580/87840."""

    x = x_values[:count].astype(np.int64)
    y = y_values[:count].astype(np.int64)
    output = np.empty(65536, dtype=np.int32)
    first_x = int(x[0])
    output[: first_x + 1] = int(y[0])
    cursor = first_x + 1
    for index in range(1, count):
        next_x = int(x[index])
        if cursor <= next_x:
            positions = np.arange(cursor, next_x + 1, dtype=np.int64)
            products = _wrapped_i32_products(
                (positions - int(x[index - 1]))
                * int(y[index] - y[index - 1])
            )
            increments = _trunc_divide_signed(
                products, np.array([next_x - int(x[index - 1])], dtype=np.int64)
            )
            output[cursor : next_x + 1] = (
                increments + int(y[index - 1])
            ).astype(np.int32)
            cursor = next_x + 1
    if cursor <= U16_MAX:
        output[cursor:] = int(y[-1])
    return output


def build_ycc_correction_y_lut_u16(
    x_values_u16: Sequence[int],
    y_values_q4_u16: Sequence[int],
    count: int,
) -> np.ndarray:
    """Port Y-table builder ``sub_10087580`` at VA ``0x10087580``.

    Twenty packed unsigned X/Y words begin at parameter offsets ``+0`` and
    ``+0x28``; count byte ``+0x50`` selects the active prefix.  The builder
    interpolates in signed 32-bit integer arithmetic, arithmetic-shifts Q4
    values right four, and clamps each result to the 12-bit range.
    """

    x, y, active = _validate_ycc_correction_curve(
        x_values_u16, y_values_q4_u16, count, y_signed=False
    )
    interpolated = _build_ycc_correction_piecewise_i32(x, y, active)
    return np.clip(interpolated.astype(np.int64) >> 4, 0, U12_MAX).astype(
        np.uint16
    )


def build_ycc_correction_chroma_scale_f32(
    x_values_u16: Sequence[int],
    scale_values_u16: Sequence[int],
    count: int,
) -> np.ndarray:
    """Port chroma-scale builder ``sub_10087740`` at VA ``0x10087740``.

    The shared X controls are at packed offset ``+0x52`` and scale words at
    ``+0xca``.  Each interpolated unsigned integer is multiplied in x87 by
    the exact float32 word ``0x3883126f`` and rounded once on storage to f32.
    """

    x, scale, active = _validate_ycc_correction_curve(
        x_values_u16, scale_values_u16, count, y_signed=False
    )
    active_x = x[:active].astype(np.int64)
    active_scale = scale[:active].astype(np.int64)
    interpolated = np.empty(65536, dtype=np.int32)
    first_x = int(active_x[0])
    interpolated[: first_x + 1] = int(active_scale[0])
    cursor = first_x + 1
    for index in range(1, active):
        next_x = int(active_x[index])
        if cursor <= next_x:
            positions = np.arange(cursor, next_x + 1, dtype=np.int64)
            # Preserve 0x100877f8..0x10087806 exactly: both differences are
            # reversed before signed 32-bit IMUL/IDIV.
            products = _wrapped_i32_products(
                int(active_scale[index - 1] - active_scale[index])
                * (positions - int(active_x[index - 1]))
            )
            increments = _trunc_divide_signed(
                products,
                np.array(
                    [int(active_x[index - 1] - active_x[index])],
                    dtype=np.int64,
                ),
            )
            interpolated[cursor : next_x + 1] = (
                increments + int(active_scale[index - 1])
            ).astype(np.int32)
            cursor = next_x + 1
    if cursor <= U16_MAX:
        interpolated[cursor:] = int(active_scale[-1])
    return np.asarray(
        interpolated.astype(np.float64) * float(YCC_CORRECTION_SCALE_F32),
        dtype=np.float32,
    )


def build_ycc_correction_chroma_offsets_i32(
    x_values_u16: Sequence[int],
    offset_values_i16: np.ndarray,
    count: int,
) -> np.ndarray:
    """Port both lanes of ``sub_10087840`` at VA ``0x10087840``.

    Signed offset arrays at packed offsets ``+0x7a`` and ``+0xa2`` share the
    X controls/count with the chroma-scale curve.  Native output interleaves
    two signed dwords per luminance index, exposed here as ``[65536, 2]``.
    """

    offsets = np.asarray(offset_values_i16)
    if offsets.shape != (2, 20):
        raise ValueError("offset_values_i16 must have shape (2, 20)")
    lanes = []
    for lane in range(2):
        x, values, active = _validate_ycc_correction_curve(
            x_values_u16, offsets[lane], count, y_signed=True
        )
        lanes.append(_build_ycc_correction_piecewise_i32(x, values, active))
    return np.stack(lanes, axis=1).astype(np.int32, copy=False)


def build_ycc_correction_tables(
    y_x_u16: Sequence[int],
    y_values_q4_u16: Sequence[int],
    y_count: int,
    chroma_x_u16: Sequence[int],
    chroma_offsets_i16: np.ndarray,
    chroma_scale_u16: Sequence[int],
    chroma_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three arrays consumed by :func:`apply_ycc_correction`."""

    return (
        build_ycc_correction_y_lut_u16(y_x_u16, y_values_q4_u16, y_count),
        build_ycc_correction_chroma_scale_f32(
            chroma_x_u16, chroma_scale_u16, chroma_count
        ),
        build_ycc_correction_chroma_offsets_i32(
            chroma_x_u16, chroma_offsets_i16, chroma_count
        ),
    )


def apply_ycc_correction(
    image_c1c2y: np.ndarray,
    y_lut_u16: np.ndarray,
    chroma_scale_f32: np.ndarray,
    chroma_offsets_i32: np.ndarray,
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Apply the exact table-driven ``FmYccCorrection_Do`` worker.

    The three tables correspond to ``sub_10087580``, ``sub_10087740``, and
    ``sub_10087840``.  They are required explicitly because the selected
    normal-job tables depend on ``tagSCN_ORD_OUT`` and are not stored in the
    captured PRM or the static CorrectServ bank.
    """

    image = _as_u16_rgb(image_c1c2y, "image_c1c2y")
    y_lut = np.asarray(y_lut_u16)
    scale = np.asarray(chroma_scale_f32, dtype=np.float32)
    offsets = np.asarray(chroma_offsets_i32)
    if y_lut.shape != (65536,) or y_lut.dtype != np.uint16:
        raise ValueError("y_lut_u16 must have shape (65536,) and dtype uint16")
    if scale.shape != (65536,):
        raise ValueError("chroma_scale_f32 must have shape (65536,)")
    if offsets.shape != (65536, 2) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("chroma_offsets_i32 must have integer shape (65536, 2)")
    if offsets.size and (
        int(offsets.min()) < np.iinfo(np.int32).min
        or int(offsets.max()) > np.iinfo(np.int32).max
    ):
        raise ValueError("chroma_offsets_i32 values must fit signed 32-bit words")
    rows = _require_positive_chunk_rows(chunk_rows)
    offset_words = offsets.astype(np.int64, copy=False)

    output = np.empty_like(image)
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        chunk = image[start:stop]
        y_index = chunk[..., 2]
        signed = np.empty(chunk.shape[:2] + (2,), dtype=np.int64)
        signed[..., 0] = (
            chunk[..., 0].astype(np.int64) + offset_words[y_index, 0] - 32768
        )
        signed[..., 1] = (
            chunk[..., 1].astype(np.int64) + offset_words[y_index, 1] - 32768
        )
        scaled = np.trunc(
            signed.astype(np.float32) * scale[y_index, None]
        ).astype(np.int64)
        candidates = scaled + 32768
        fits = np.all((candidates >= 0) & (candidates <= U16_MAX), axis=2)

        corrected = candidates.copy()
        overflow = ~fits
        if np.any(overflow):
            pair = scaled[overflow]
            maximum = np.maximum(pair[:, 0], pair[:, 1])
            minimum = np.minimum(pair[:, 0], pair[:, 1])
            negative_limit = -minimum
            negative_dominates = maximum <= negative_limit

            normalized = np.empty_like(pair)
            if np.any(negative_dominates):
                selected = pair[negative_dominates]
                denominator = negative_limit[negative_dominates]
                normalized[negative_dominates] = (
                    _trunc_divide_signed(selected * 32768, denominator[:, None])
                    + 32768
                )
            positive_dominates = ~negative_dominates
            if np.any(positive_dominates):
                selected = pair[positive_dominates]
                denominator = maximum[positive_dominates]
                normalized[positive_dominates] = (
                    _trunc_divide_signed(selected * 32767, denominator[:, None])
                    + 32768
                )
            corrected[overflow] = normalized

        output[start:stop, ..., :2] = corrected.astype(np.uint16)
        output[start:stop, ..., 2] = y_lut[y_index]
    return output


def _fused_gamma_control_arrays(
    x_controls_u16: np.ndarray,
    y_controls_u8: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate one packed ``RD5X_GAM_LUT_OCT_PARA`` channel."""

    x_source = np.asarray(x_controls_u16)
    y_source = np.asarray(y_controls_u8)
    if x_source.shape != (20,) or y_source.shape != (20,):
        raise ValueError("fused-gamma X and Y controls must each contain 20 values")
    if not np.issubdtype(x_source.dtype, np.integer) or not np.issubdtype(
        y_source.dtype, np.integer
    ):
        raise TypeError("fused-gamma controls must be integer values")
    if not isinstance(count, (int, np.integer)) or not 2 <= int(count) <= 20:
        raise ValueError("fused-gamma count must be in the native range 2..20")
    if x_source.size and (
        int(x_source.min()) < 0 or int(x_source.max()) > U16_MAX
    ):
        raise ValueError("fused-gamma X controls must fit unsigned 16-bit words")
    if y_source.size and (
        int(y_source.min()) < 0 or int(y_source.max()) > 0xFF
    ):
        raise ValueError("fused-gamma Y controls must fit unsigned bytes")

    active_count = int(count)
    x = x_source.astype(np.uint16, copy=False)[:active_count].astype(np.float64)
    y = y_source.astype(np.uint8, copy=False)[:active_count].astype(np.float64)
    if np.any(np.diff(x) <= 0):
        raise ValueError("active fused-gamma X controls must be strictly increasing")
    return x, y, active_count


def _fused_gamma_sixth_derivatives(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Return the natural-cubic M/6 coefficients built by ``0x1016baf0``."""

    count = x.size
    intervals = np.empty(count - 1, dtype=np.float64)
    slopes = np.empty(count - 1, dtype=np.float64)
    for index in range(count - 1):
        intervals[index] = x[index + 1] - x[index]
        slopes[index] = (y[index + 1] - y[index]) / intervals[index]

    diagonal = np.zeros(count, dtype=np.float64)
    right_hand = np.zeros(count, dtype=np.float64)
    coefficients = np.zeros(count, dtype=np.float64)
    if count == 2:
        return coefficients

    diagonal[1] = 2.0 * (intervals[0] + intervals[1])
    right_hand[1] = slopes[1] - slopes[0]
    for index in range(2, count - 1):
        factor = intervals[index - 1] / diagonal[index - 1]
        diagonal[index] = (
            2.0 * (intervals[index - 1] + intervals[index])
            - factor * intervals[index - 1]
        )
        right_hand[index] = (
            slopes[index]
            - slopes[index - 1]
            - factor * right_hand[index - 1]
        )

    coefficients[count - 2] = (
        right_hand[count - 2] / diagonal[count - 2]
    )
    for index in range(count - 3, 0, -1):
        coefficients[index] = (
            right_hand[index]
            - intervals[index] * coefficients[index + 1]
        ) / diagonal[index]
    return coefficients


def _build_fused_gamma_lut(
    x_controls_u16: np.ndarray,
    y_controls_u8: np.ndarray,
    count: int,
    *,
    output_u16: bool,
) -> np.ndarray:
    """Shared transcription of private builders ``0x100810c0/0x10080980``."""

    x, y, active_count = _fused_gamma_control_arrays(
        x_controls_u16, y_controls_u8, count
    )
    if output_u16:
        # ``0x10080980`` multiplies each byte by 257.0, adds 0.5, truncates,
        # and stores it as a double before constructing the spline.  Integer
        # byte inputs make the stored result exactly y*257.
        y = y * 257.0
        maximum = U16_MAX
        dtype = np.dtype("<u2")
    else:
        maximum = 0xFF
        dtype = np.dtype(np.uint8)

    sixth_derivatives = _fused_gamma_sixth_derivatives(x, y)
    domain = np.arange(65536, dtype=np.float64)
    intervals = np.searchsorted(x, domain, side="right") - 1
    intervals = np.clip(intervals, 0, active_count - 2)
    widths = x[intervals + 1] - x[intervals]
    distance = domain - x[intervals]

    # ``0x1016c000`` evaluates the natural cubic with M/6 coefficients in
    # this operation order.  Its caller adds 0.5 then truncates toward zero.
    values = y[intervals] + distance * (
        (y[intervals + 1] - y[intervals]) / widths
        - (
            2.0 * sixth_derivatives[intervals]
            + sixth_derivatives[intervals + 1]
        )
        * widths
        + 3.0 * sixth_derivatives[intervals] * distance
        + (
            sixth_derivatives[intervals + 1]
            - sixth_derivatives[intervals]
        )
        * distance
        * distance
        / widths
    )
    rounded = np.trunc(values + 0.5)
    rounded = np.minimum(np.maximum(rounded, 0.0), float(maximum))

    # Preserve the native comparison order, including its observable result
    # for unusual descending endpoint controls; this is not merely min/max.
    first = y[0]
    last = y[-1]
    rounded = np.where(
        last <= np.maximum(rounded, first),
        last,
        np.where(rounded <= first, first, rounded),
    )
    rounded[domain < x[0]] = first
    rounded[domain >= x[-1]] = last
    return rounded.astype(dtype)


def build_fused_gamma_lut_u8(
    x_controls_u16: np.ndarray,
    y_controls_u8: np.ndarray,
    count: int,
) -> np.ndarray:
    """Build one exact 65,536-byte fused-gamma LUT.

    This ports private builder ``sub_100810c0`` from UniRdImgLib.dll.  It
    allocates the natural-cubic workspace through ``0x1016ba10``, constructs
    coefficients at ``0x1016baf0``, evaluates them at ``0x1016c000``, then
    performs the wrapper's rounding, clipping, and endpoint fills.  The
    implementation is byte-identical to the installed DLL for the captured
    final-PCB curve and four independent conformance curves.
    """

    return _build_fused_gamma_lut(
        x_controls_u16, y_controls_u8, count, output_u16=False
    )


def build_fused_gamma_lut_u16(
    x_controls_u16: np.ndarray,
    y_controls_u8: np.ndarray,
    count: int,
) -> np.ndarray:
    """Build one exact 65,536-word alternative-depth fused-gamma LUT.

    This is private builder ``sub_10080980``.  It shares the cubic math with
    the selected byte builder but expands each Y control by exactly 257 before
    interpolation.  The fixed owner job does *not* select this branch; it is
    retained as the exact ``FmYccToRgbS0_16bit_Do`` alternative.
    """

    return _build_fused_gamma_lut(
        x_controls_u16, y_controls_u8, count, output_u16=True
    )


def _fused_gamma_rgb_arrays(
    x_controls_rgb: np.ndarray,
    y_controls_rgb_u8: np.ndarray,
    counts_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the packed RGB arrays before the wrapper's BGR reversal."""

    x = np.asarray(x_controls_rgb)
    y = np.asarray(y_controls_rgb_u8)
    counts = np.asarray(counts_rgb)
    if x.shape != (3, 20) or y.shape != (3, 20):
        raise ValueError("fused-gamma RGB controls must have shape (3, 20)")
    if counts.shape != (3,):
        raise ValueError("counts_rgb must contain three values")
    if not np.issubdtype(counts.dtype, np.integer):
        raise TypeError("counts_rgb must contain integer values")
    return x, y, counts


def build_fused_gamma_luts_bgr(
    x_controls_rgb: np.ndarray,
    y_controls_rgb_u8: np.ndarray,
    counts_rgb: np.ndarray,
) -> np.ndarray:
    """Build the selected wrapper's B, G, R array of uint8 gamma LUTs.

    Final-PCB ``RD5X_GAM_LUT_OCT_PARA`` stores its three channels in R, G, B
    order.  ``FmYccToRgbS0_Do`` at VA ``0x1008d2a0`` invokes the byte builder
    in B, G, R order and passes those pointers to variant-1 worker
    ``0x100894e0``.  The explicit reversal here produces the exact array order
    required by :func:`apply_ycc_to_rgb_s0_variant1_u8`.
    """

    x, y, counts = _fused_gamma_rgb_arrays(
        x_controls_rgb, y_controls_rgb_u8, counts_rgb
    )
    return np.stack(
        [
            build_fused_gamma_lut_u8(x[channel], y[channel], counts[channel])
            for channel in (2, 1, 0)
        ]
    )


def build_fused_gamma_luts_bgr_u16(
    x_controls_rgb: np.ndarray,
    y_controls_rgb_u8: np.ndarray,
    counts_rgb: np.ndarray,
) -> np.ndarray:
    """Build the unselected 16-bit wrapper's B, G, R gamma LUT array."""

    x, y, counts = _fused_gamma_rgb_arrays(
        x_controls_rgb, y_controls_rgb_u8, counts_rgb
    )
    return np.stack(
        [
            build_fused_gamma_lut_u16(x[channel], y[channel], counts[channel])
            for channel in (2, 1, 0)
        ]
    )


def _ycc_to_rgb_s0_variant1_indexes(
    chunk_c1c2y: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Return variant-1 BGR gamma indexes in the workers' operation order."""

    c1 = np.subtract(chunk_c1c2y[..., 0].astype(np.float32), np.float32(32768.0))
    c2 = np.subtract(chunk_c1c2y[..., 1].astype(np.float32), np.float32(32768.0))
    luminance = chunk_c1c2y[..., 2].astype(np.float32)

    # sub_10091be0 lays the coefficient vectors out as B, G, R.  Both
    # variant-1 workers add C2+C1 first, then add the luminance contribution.
    native = np.empty(chunk_c1c2y.shape, dtype=np.float32)
    native[..., 0] = _add_f32(
        _mul_f32(luminance, coefficients[6]),
        _add_f32(_mul_f32(c2, coefficients[8]), _mul_f32(c1, coefficients[7])),
    )
    native[..., 1] = _add_f32(
        _mul_f32(luminance, coefficients[3]),
        _add_f32(_mul_f32(c2, coefficients[5]), _mul_f32(c1, coefficients[4])),
    )
    native[..., 2] = _add_f32(
        _mul_f32(luminance, coefficients[0]),
        _add_f32(_mul_f32(c2, coefficients[2]), _mul_f32(c1, coefficients[1])),
    )

    rounded = np.rint(native).astype(np.int64)
    valid = np.all((rounded >= 0) & (rounded <= U12_MAX), axis=2)
    indexes = rounded.copy()
    invalid = ~valid
    if np.any(invalid):
        work = native[invalid].copy()
        rounded_bad = rounded[invalid]
        minimum = rounded_bad.min(axis=1)
        maximum = rounded_bad.max(axis=1)
        equal = maximum == minimum

        if np.any(equal):
            level = np.clip(maximum[equal], 0, U12_MAX).astype(np.float32)
            work[equal] = level[:, None]

        unequal = ~equal
        if np.any(unequal):
            selected = work[unequal]
            selected_minimum = minimum[unequal].copy()
            selected_maximum = maximum[unequal].copy()
            too_high = selected_maximum > U12_MAX
            if np.any(too_high):
                low = selected_minimum[too_high].astype(np.float32)
                span = (selected_maximum[too_high] - selected_minimum[too_high]).astype(
                    np.float64
                )
                reciprocal = np.asarray(1.0 / span, dtype=np.float32)
                adjusted = np.subtract(
                    selected[too_high], low[:, None], dtype=np.float32
                )
                adjusted = np.multiply(
                    adjusted,
                    np.subtract(np.float32(U12_MAX), low, dtype=np.float32)[:, None],
                    dtype=np.float32,
                )
                adjusted = np.multiply(
                    adjusted, reciprocal[:, None], dtype=np.float32
                )
                selected[too_high] = np.add(
                    adjusted, low[:, None], dtype=np.float32
                )
                selected_maximum[too_high] = U12_MAX

            too_low = selected_minimum < 0
            if np.any(too_low):
                low = selected_minimum[too_low].astype(np.float32)
                high = selected_maximum[too_low].astype(np.float32)
                span = (selected_maximum[too_low] - selected_minimum[too_low]).astype(
                    np.float64
                )
                reciprocal = np.asarray(1.0 / span, dtype=np.float32)
                adjusted = np.subtract(
                    selected[too_low], low[:, None], dtype=np.float32
                )
                adjusted = np.multiply(adjusted, high[:, None], dtype=np.float32)
                selected[too_low] = np.multiply(
                    adjusted, reciprocal[:, None], dtype=np.float32
                )
            work[unequal] = selected

        work = np.maximum(
            np.minimum(work, np.float32(U12_MAX)), np.float32(0.0)
        )
        indexes[invalid] = np.rint(work).astype(np.int64)
    return indexes


def apply_ycc_to_rgb_s0_variant1_u16(
    image_c1c2y: np.ndarray,
    matrix_q10000: np.ndarray,
    gamma_luts_bgr: np.ndarray,
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Apply normal variant 1 of ``FmYccToRgbS0_16bit_Do``.

    The installer wrapper at ``0x100917e0`` builds three 65536-word LUTs with
    ``sub_10080980`` and dispatches variant byte 1 to worker ``0x1008da40``.
    That worker transforms in native B, G, R order, gamut-compresses indexes
    to 0..4095 using rounded extrema and the original float32 triples, reads
    uint16 LUT entries, and stores a 16-bit result.  This function returns the
    visible channels as RGB uint16 (the native image's fourth lane is padding).

    Group-0/type-1 has an exact static inverse matrix, but its gamma controls
    are a 2017 runtime-built PCB output.  This primitive therefore accepts
    completed B, G, R LUTs and does not invent a substitute curve.
    """

    image = _as_u16_rgb(image_c1c2y, "image_c1c2y")
    coefficients = _q10000_coefficients(np.asarray(matrix_q10000))
    gamma = np.asarray(gamma_luts_bgr)
    if gamma.shape != (3, 65536) or gamma.dtype != np.uint16:
        raise ValueError("gamma_luts_bgr must have shape (3, 65536) and dtype uint16")
    rows = _require_positive_chunk_rows(chunk_rows)
    output = np.empty_like(image)

    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        chunk = image[start:stop]
        indexes = _ycc_to_rgb_s0_variant1_indexes(chunk, coefficients)
        native_output = np.empty(chunk.shape, dtype=np.uint16)
        for channel in range(3):
            native_output[..., channel] = gamma[channel][indexes[..., channel]]
        output[start:stop] = native_output[..., ::-1]
    return output


def apply_ycc_to_rgb_s0_variant1_u8(
    image_c1c2y: np.ndarray,
    matrix_q10000: np.ndarray,
    gamma_luts_bgr: np.ndarray,
    *,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Apply normal variant 1 of the byte-output ``FmYccToRgbS0_Do``.

    The installer wrapper at ``0x1008d2a0`` builds three 65536-byte LUTs with
    ``sub_100810c0`` and dispatches variant byte 1 to worker ``0x100894e0``.
    The worker's float32 matrix, rounded-extrema gamut compression, and BGR
    indexing order match the 16-bit variant-1 worker; only the LUT element and
    output lane widths differ.  This function returns the three visible lanes
    as RGB uint8, omitting the native image's unused fourth byte.

    As with the 16-bit wrapper, the completed fused-gamma LUTs are runtime
    controller state, so callers must supply them rather than substituting an
    inferred curve.
    """

    image = _as_u16_rgb(image_c1c2y, "image_c1c2y")
    coefficients = _q10000_coefficients(np.asarray(matrix_q10000))
    gamma = np.asarray(gamma_luts_bgr)
    if gamma.shape != (3, 65536) or gamma.dtype != np.uint8:
        raise ValueError("gamma_luts_bgr must have shape (3, 65536) and dtype uint8")
    rows = _require_positive_chunk_rows(chunk_rows)
    output = np.empty(image.shape, dtype=np.uint8)

    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        chunk = image[start:stop]
        indexes = _ycc_to_rgb_s0_variant1_indexes(chunk, coefficients)
        native_output = np.empty(chunk.shape, dtype=np.uint8)
        for channel in range(3):
            native_output[..., channel] = gamma[channel][indexes[..., channel]]
        output[start:stop] = native_output[..., ::-1]
    return output


def apply_dark_offset(raw: np.ndarray, dark_offset: np.ndarray) -> np.ndarray:
    """Saturating 4-channel dark-offset subtraction from CreateImage.dll."""

    values = np.asarray(raw)
    offsets = np.asarray(dark_offset)
    if values.ndim != 3 or values.shape[2] != 4 or values.dtype != np.uint16:
        raise ValueError("raw must be HxWx4 uint16 B,G,R,IR")
    if offsets.shape not in ((values.shape[1], 4), values.shape):
        raise ValueError("dark_offset must be Wx4 or HxWx4")
    expanded = offsets[None, ...] if offsets.ndim == 2 else offsets
    return np.clip(
        values.astype(np.int32) - expanded.astype(np.int32), 0, U12_MAX
    ).astype(np.uint16)


def apply_digital_gain(raw: np.ndarray, gain_q12: np.ndarray) -> np.ndarray:
    """Apply CreateImage.dll's per-position signed Q12 digital-gain kernel."""

    values = np.asarray(raw)
    gains = _signed_q12(gain_q12, "gain_q12")
    if values.ndim != 3 or values.shape[2] != 4 or values.dtype != np.uint16:
        raise ValueError("raw must be HxWx4 uint16 B,G,R,IR")
    if gains.shape not in ((values.shape[1], 4), values.shape):
        raise ValueError("gain_q12 must be Wx4 or HxWx4")
    expanded = gains[None, ...] if gains.ndim == 2 else gains
    result = (values.astype(np.int64) * expanded.astype(np.int64)) >> 12
    return np.clip(result, 0, U12_MAX).astype(np.uint16)


def apply_ccd_linear(raw: np.ndarray, ccd_linear: np.ndarray) -> np.ndarray:
    """Apply four exact 4096-entry CreateImage CCD-linearization tables."""

    values = np.asarray(raw)
    tables = np.asarray(ccd_linear, dtype=np.uint16)
    if values.ndim != 3 or values.shape[2] != 4 or values.dtype != np.uint16:
        raise ValueError("raw must be HxWx4 uint16 B,G,R,IR")
    if tables.shape != (4, 4096):
        raise ValueError("ccd_linear must have shape (4, 4096)")
    if values.size and int(values.max()) > U12_MAX:
        raise ValueError("CCD-linear input exceeds 4095")
    output = np.empty_like(values)
    for channel in range(4):
        output[..., channel] = tables[channel][values[..., channel]]
    return output


def apply_color_matrix_q12(raw: np.ndarray, matrix_q12: np.ndarray) -> np.ndarray:
    """Apply CreateImage's per-CCD-position signed 4x4 Q12 color matrix."""

    values = np.asarray(raw)
    matrices = _signed_q12(matrix_q12, "matrix_q12")
    if values.ndim != 3 or values.shape[2] != 4 or values.dtype != np.uint16:
        raise ValueError("raw must be HxWx4 uint16 B,G,R,IR")
    if matrices.shape not in ((values.shape[1], 4, 4), values.shape[:2] + (4, 4)):
        raise ValueError("matrix_q12 must be Wx4x4 or HxWx4x4")
    expanded = matrices[None, ...] if matrices.ndim == 3 else matrices
    # The recovered kernel accumulates signed products, arithmetic-shifts by
    # 12, and only then clips each output slot.
    result = np.einsum(
        "hwk,hwok->hwo",
        values.astype(np.int64),
        np.broadcast_to(expanded, values.shape[:2] + (4, 4)).astype(np.int64),
        optimize=True,
    )
    result >>= 12
    return np.clip(result, 0, U12_MAX).astype(np.uint16)


def apply_scanner_preprocess(
    raw_bgri: np.ndarray,
    *,
    dark_offset: np.ndarray,
    ccd_linear: np.ndarray,
    color_matrix_q12: np.ndarray,
) -> np.ndarray:
    """Run CreateImage dispatcher op12's exact fused preprocessing chain.

    Dispatcher op12 is dark offset -> four CCD-linear LUTs -> per-position
    4x4 color matrix.  The digital-gain kernel is dispatcher op4, an
    alternative main operation rather than an optional step inside op12; call
    :func:`apply_digital_gain` independently only when captured dispatcher
    state selects op4.
    """

    output = apply_dark_offset(raw_bgri, dark_offset)
    output = apply_ccd_linear(output, ccd_linear)
    return apply_color_matrix_q12(output, color_matrix_q12)


def apply_ir_misregistration_u16(
    image_bgri: np.ndarray,
    horizontal_offsets_i32: np.ndarray,
    vertical_offsets_i32: np.ndarray,
) -> np.ndarray:
    """Port ``CIpmBeforeImageConvs::misregMMF2``'s array rewrite.

    The CreateImage worker at VA ``0x100514b0`` first saves the IR lane from
    the interleaved ``B,G,R,IR`` uint16 raster.  Its write loop at
    ``0x10051ad3..0x10051cd3`` then sends source pixel ``(y, x)`` to::

        destination_x = clip(x + pMapIR[x], 0, width - 1)
        destination_y = clip(y - pMap[x], 0, height - 1)

    Only the IR word at byte offset ``+6`` is rewritten; B, G, and R remain
    byte-for-byte unchanged.  Writes occur in source row-major order, so a
    later source wins when calibration offsets collide at one destination.
    Destinations receiving no write retain their original IR value because
    the native MMF is modified in place.

    ``horizontal_offsets_i32`` and ``vertical_offsets_i32`` are the selected
    image-width slices of the per-unit maps stored at object offsets ``+0xc8``
    and ``+0x5438``.  The worker maps three overlap rows on each side of an
    MMF band; accepting vertical offsets outside ``[-3, 3]`` would therefore
    expose its band-size-dependent failure-domain behavior rather than the
    calibrated normal domain, so this portable primitive rejects them.
    """

    image = np.asarray(image_bgri)
    if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint16:
        raise ValueError("image_bgri must be HxWx4 uint16 B,G,R,IR")

    def signed_i32_map(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != (image.shape[1],) or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise ValueError(f"{name} must have integer shape (width,)")
        if array.size:
            minimum = int(array.min())
            maximum = int(array.max())
            if minimum < -(1 << 31) or maximum > 0xFFFFFFFF:
                raise ValueError(f"{name} values must fit signed/encoded i32")
        return array.astype(np.uint32, copy=False).view(np.int32)

    horizontal = signed_i32_map(
        horizontal_offsets_i32, "horizontal_offsets_i32"
    ).astype(np.int64)
    vertical = signed_i32_map(
        vertical_offsets_i32, "vertical_offsets_i32"
    ).astype(np.int64)
    if vertical.size and (int(vertical.min()) < -3 or int(vertical.max()) > 3):
        raise ValueError("vertical_offsets_i32 must stay in the native -3..3 domain")

    height, width = image.shape[:2]
    destination_x = np.clip(
        np.arange(width, dtype=np.int64) + horizontal, 0, width - 1
    )
    source_ir = image[..., 3]
    output = image.copy()
    destination_ir = output[..., 3].reshape(-1)

    for source_y in range(height):
        destination_y = np.clip(source_y - vertical, 0, height - 1)
        destination_indexes = destination_y * width + destination_x

        # The normal maps are one-to-one away from clipped edges.  Resolve
        # the exceptional duplicate destinations explicitly so NumPy's
        # advanced-assignment duplicate semantics cannot change native
        # "last source wins" behavior.
        if np.unique(destination_indexes).size == width:
            destination_ir[destination_indexes] = source_ir[source_y]
            continue
        _, reverse_first = np.unique(
            destination_indexes[::-1], return_index=True
        )
        last_sources = width - 1 - reverse_first
        destination_ir[destination_indexes[last_sources]] = source_ir[
            source_y, last_sources
        ]

    return output


def _interp_control_fraction(
    y: np.ndarray, x: np.ndarray, position: int
) -> Tuple[int, int]:
    if position <= int(x[0]):
        return int(y[0]), 1
    if position >= int(x[-1]):
        return int(y[-1]), 1
    index = int(np.searchsorted(x, position, side="left"))
    x0, x1 = int(x[index - 1]), int(x[index])
    y0, y1 = int(y[index - 1]), int(y[index])
    denominator = x1 - x0
    numerator = y0 * denominator + (position - x0) * (y1 - y0)
    return numerator, denominator


def _interp_control(y: np.ndarray, x: np.ndarray, position: int) -> float:
    numerator, denominator = _interp_control_fraction(y, x, position)
    return numerator / denominator


def _interp_control_trunc(y: np.ndarray, x: np.ndarray, position: int) -> int:
    numerator, denominator = _interp_control_fraction(y, x, position)
    return _trunc_rational_x87(numerator, denominator)


def _interp_control_difference_trunc(
    first_y: np.ndarray,
    first_x: np.ndarray,
    second_y: np.ndarray,
    second_x: np.ndarray,
    position: int,
) -> int:
    # RD5X_Calc_Gamma_InDat stores each interpolation to binary64 before the
    # caller reloads both into x87, subtracts, and calls _ftol2_sse.  Keeping
    # those two stores is observable when an exact-in-real-arithmetic integer
    # difference lands infinitesimally below the integer in binary64.
    return _x87_trunc(
        _interp_control(first_y, first_x, position)
        - _interp_control(second_y, second_x, position)
    )


def _x87_trunc(value: float) -> int:
    """Model SpecialPcb's x87-to-long ``_ftol2_sse`` conversion."""

    return int(np.trunc(float(value)))


def _trunc_rational_x87(numerator: int, denominator: int) -> int:
    """Truncate an exact rational as SpecialPcb's ``_ftol2_sse`` does."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return _trunc_divide_scalar(int(numerator), int(denominator))


def make_uo_raw_sub2(uo_mode: int, value: int) -> int:
    """Port ``pcb_makeUORawSub2`` at SpecialPcb VA ``0x101156a0``.

    The 47-point X axis at ``0x1016de00`` and four rows at ``0x1016dec0``
    are embedded in the DLL.  Mode zero is the identity; modes one through
    three apply the corresponding forward under/over-exposure curve.  The
    x87 result is converted by ``_ftol2_sse`` with truncation toward zero;
    like the binary, out-of-range values pass through unchanged.
    """

    mode = int(uo_mode)
    position = int(value)
    if mode not in range(len(UO_RAW_Y)):
        raise ValueError("uo_mode must be 0, 1, 2, or 3")
    if mode == 0 or position < UO_RAW_X[0] or position > UO_RAW_X[-1]:
        return position
    index = int(np.searchsorted(UO_RAW_X, position, side="left"))
    if index == 0:
        return int(UO_RAW_Y[mode][0])
    x0, x1 = UO_RAW_X[index - 1], UO_RAW_X[index]
    y0, y1 = UO_RAW_Y[mode][index - 1], UO_RAW_Y[mode][index]
    numerator = y0 * (x1 - x0) + (position - x0) * (y1 - y0)
    return _trunc_rational_x87(numerator, x1 - x0)


def apply_uo_raw_sub1(y_values: Sequence[int], uo_mode: int) -> np.ndarray:
    """Port ``pcb_makeUORawSub1`` at SpecialPcb VA ``0x10116490``.

    This is the inverse table mapping applied to the 20 negative-tone output
    ordinates after ``pcb_MakeGammaCurve2``.  The DLL walks high to low so its
    below-range continuation can reuse the already converted next ordinate;
    that order is preserved explicitly.
    """

    values = np.asarray(y_values, dtype=np.int64)
    if values.shape != (20,):
        raise ValueError("y_values must contain exactly 20 values")
    mode = int(uo_mode)
    if mode not in range(len(UO_RAW_Y)):
        raise ValueError("uo_mode must be 0, 1, 2, or 3")
    output = values.copy()
    if mode == 0:
        return output

    row = UO_RAW_Y[mode]
    for point in range(19, -1, -1):
        value = int(output[point])
        if value < row[0]:
            if point == 19:
                raise ValueError(
                    "the vendor's below-range continuation is undefined at an endpoint"
                )
            output[point] = (
                int(values[point])
                - int(values[point + 1])
                + int(output[point + 1])
            )
            continue
        if value > row[-1]:
            continue
        index = int(np.searchsorted(row, value, side="left"))
        if index == 0:
            mapped = UO_RAW_X[0]
        else:
            y0, y1 = row[index - 1], row[index]
            x0, x1 = UO_RAW_X[index - 1], UO_RAW_X[index]
            numerator = x0 * (y1 - y0) + (value - y0) * (x1 - x0)
            mapped = _trunc_rational_x87(numerator, y1 - y0)
        output[point] = min(max(mapped, 0), U16_MAX)
    return output


def make_gamma_curve2(
    y_values: Sequence[int],
    x_values: Sequence[int],
    low_ratio: int = 100,
    mid_ratio: int = 100,
    high_ratio: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port ``pcb_MakeGammaCurve2`` from SpecialPcb.dll at VA 0x10117340.

    The three integer ratios are percentages.  The function scales output
    slope below 8000, from 8000 to 60000, and above 60000; stitches the regions;
    reanchors at input 0x5f00; and smooths changed seams.  It returns ``(y,x)``
    int64 arrays in the same order as the vendor function's output pointers.
    """

    y = np.asarray(y_values, dtype=np.int64)
    x = np.asarray(x_values, dtype=np.int64)
    if y.shape != (20,) or x.shape != (20,):
        raise ValueError("gamma controls must each contain exactly 20 values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("gamma X controls must be strictly increasing")

    scaled = []
    for ratio in (low_ratio, mid_ratio, high_ratio):
        current = y.copy()
        current[1:] = [
            _x87_trunc((int(value) - int(y[0])) * int(ratio) * 0.01 + int(y[0]))
            for value in y[1:]
        ]
        scaled.append(current)
    low, mid, high = scaled

    out_y: list[int] = []
    out_x: list[int] = []
    accumulated = 0
    for value_y, value_x in zip(low, x):
        if int(value_x) >= GAMMA_SEAMS[0]:
            break
        out_y.append(int(value_y - low[0]))
        out_x.append(int(value_x))

    accumulated += _interp_control_trunc(low, x, GAMMA_SEAMS[0]) - int(low[0])
    mid_base = _interp_control_trunc(mid, x, GAMMA_SEAMS[0])
    for value_y, value_x in zip(mid, x):
        if GAMMA_SEAMS[0] <= int(value_x) < GAMMA_SEAMS[1]:
            out_y.append(int(value_y) - mid_base + accumulated)
            out_x.append(int(value_x))

    accumulated += _interp_control_trunc(mid, x, GAMMA_SEAMS[1]) - mid_base
    high_base = _interp_control_trunc(high, x, GAMMA_SEAMS[1])
    for value_y, value_x in zip(high, x):
        if int(value_x) >= GAMMA_SEAMS[1]:
            out_y.append(int(value_y) - high_base + accumulated)
            out_x.append(int(value_x))

    if len(out_y) != 20:
        raise ValueError("vendor seam construction did not preserve 20 control points")
    output_y = np.asarray(out_y, dtype=np.int64)
    output_x = np.asarray(out_x, dtype=np.int64)
    anchor_shift = _interp_control_difference_trunc(
        y, x, output_y, output_x, GAMMA_ANCHOR_X
    )
    output_y += anchor_shift

    if int(low_ratio) != int(mid_ratio):
        index = int(np.flatnonzero(output_x > GAMMA_SEAMS[0])[0])
        linear = _interp_control_trunc(
            output_y[[index - 1, index + 1]],
            output_x[[index - 1, index + 1]],
            int(output_x[index]),
        )
        delta = linear - int(output_y[index])
        output_y[index] += int(np.trunc(delta / 2.0))
    if int(mid_ratio) != int(high_ratio):
        index = int(np.flatnonzero(output_x < GAMMA_SEAMS[1])[-1])
        linear = _interp_control_trunc(
            output_y[[index - 1, index + 1]],
            output_x[[index - 1, index + 1]],
            int(output_x[index]),
        )
        delta = linear - int(output_y[index])
        output_y[index] += int(np.trunc(delta / 2.0))
    return output_y, output_x


def anchor_gamma_curve(
    y_values: Sequence[int],
    x_values: Sequence[int],
    target: int = 32767,
    position: int = GAMMA_ANCHOR_X,
) -> np.ndarray:
    """Apply ``pcb_makeNegaTone2``'s final constant anchor shift."""

    y = np.asarray(y_values, dtype=np.int64).copy()
    x = np.asarray(x_values, dtype=np.int64)
    y += int(target) - _interp_control_trunc(y, x, int(position))
    return y


def make_nega_tone2(
    y_values: Sequence[int],
    x_values: Sequence[int],
    target_anchor: int,
    uo_anchor: int,
    uo_mode: int = 0,
    low_ratio: int = 100,
    mid_ratio: int = 100,
    high_ratio: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port the full ``pcb_makeNegaTone2`` at SpecialPcb VA ``0x10121a50``.

    The worker first anchors the source curve through ``pcb_makeUORawSub2``,
    constructs the three-region gamma curve, applies the inverse UO table when
    selected, and finally reanchors the result at input ``0x5f00``.  All four
    embedded UO modes are implemented; no controller-derived mode is assumed.
    """

    y = np.asarray(y_values, dtype=np.int64).copy()
    x = np.asarray(x_values, dtype=np.int64).copy()
    if y.shape != (20,) or x.shape != (20,):
        raise ValueError("gamma controls must each contain exactly 20 values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("gamma X controls must be strictly increasing")

    source_anchor = _interp_control_trunc(y, x, GAMMA_ANCHOR_X)
    y += make_uo_raw_sub2(uo_mode, int(uo_anchor)) - source_anchor
    y, x = make_gamma_curve2(y, x, low_ratio, mid_ratio, high_ratio)
    if int(uo_mode) > 0:
        y = apply_uo_raw_sub1(y, uo_mode)
    y = anchor_gamma_curve(y, x, int(target_anchor), GAMMA_ANCHOR_X)
    return y, x


def apply_gamma_controls_u16(
    image_rgb: np.ndarray,
    x_controls: np.ndarray,
    y_controls: np.ndarray,
) -> np.ndarray:
    """Interpolate three recovered control curves over an RGB uint16 image."""

    image = _as_u16_rgb(image_rgb)
    x = np.asarray(x_controls)
    y = np.asarray(y_controls)
    if x.shape != (3, 20) or y.shape != (3, 20):
        raise ValueError("x_controls and y_controls must have shape (3, 20)")
    output = np.empty_like(image)
    for channel in range(3):
        values = np.interp(image[..., channel], x[channel], y[channel])
        output[..., channel] = np.clip(np.rint(values), 0, U16_MAX).astype(np.uint16)
    return output


def quantize_oct_to_hex(image_rgb: np.ndarray) -> np.ndarray:
    """Round to 8-bit precision and expand exactly to a 16-bit TIFF range."""

    image = _as_u16_rgb(image_rgb)
    octets = (image.astype(np.uint32) * 255 + 32767) // U16_MAX
    return expand_u8_to_u16(octets.astype(np.uint8))


def expand_u8_to_u16(image_rgb_u8: np.ndarray) -> np.ndarray:
    """Expand a native byte-output checkpoint to exact 16-bit TIFF codes.

    ``MdConv8To16_Do`` at VA ``0x10136ad0`` dispatches inner worker
    ``0x10135770``.  For each visible channel it computes
    ``(value * 0xffff + 0x80) / 0xff`` with unsigned integer division.  Since
    ``value`` is an octet, this is exactly ``value * 257``.  The captured
    owner job selects byte ``FmYccToRgbS0_Do`` directly (packed descriptor
    ``dstType=0`` and propagated depth flag zero); this helper represents the
    separate lossless storage expansion visible in the machine TIFF, not a
    hidden 16-to-8 conversion in that selected color worker.
    """

    image = _as_u8_rgb(image_rgb_u8, "image_rgb_u8")
    return (image.astype(np.uint16) * np.uint16(257)).astype(np.uint16)


def fit_monotone_reference_lut(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    sample_step: int = 2,
) -> np.ndarray:
    """Fit an explicit external-reference 3x65536 monotone transfer LUT.

    This is not a recovered vendor stage.  It is provided to measure how much
    of the remaining final-output gap can be explained by a one-dimensional
    tone transfer when an aligned legacy machine output is available.
    """

    source = _as_u16_rgb(source_rgb, "source_rgb")
    target = _as_u16_rgb(target_rgb, "target_rgb")
    if source.shape != target.shape:
        raise ValueError("source and target must be pixel-aligned and have equal shape")
    if sample_step < 1:
        raise ValueError("sample_step must be positive")

    def isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        levels: list[float] = []
        masses: list[float] = []
        starts: list[int] = []
        ends: list[int] = []
        for index, (value, weight) in enumerate(zip(values, weights)):
            levels.append(float(value))
            masses.append(float(weight))
            starts.append(index)
            ends.append(index + 1)
            while len(levels) >= 2 and levels[-2] > levels[-1]:
                mass = masses[-2] + masses[-1]
                level = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / mass
                levels[-2:] = [level]
                masses[-2:] = [mass]
                ends[-2:] = [ends[-1]]
                starts.pop()
        result = np.empty(values.shape, dtype=np.float64)
        for level, start, end in zip(levels, starts, ends):
            result[start:end] = level
        return result

    lut = np.empty((3, 65536), dtype=np.uint16)
    domain = np.arange(65536)
    sampled_source = source[::sample_step, ::sample_step]
    sampled_target = target[::sample_step, ::sample_step]
    for channel in range(3):
        src = sampled_source[..., channel].reshape(-1)
        dst = sampled_target[..., channel].reshape(-1).astype(np.float64)
        counts = np.bincount(src, minlength=65536).astype(np.float64)
        sums = np.bincount(src, weights=dst, minlength=65536)
        observed = np.flatnonzero(counts)
        means = sums[observed] / counts[observed]
        monotone = isotonic(means, counts[observed])
        fitted = np.interp(domain, observed, monotone)
        lut[channel] = np.clip(np.rint(fitted), 0, U16_MAX).astype(np.uint16)
    return lut


def apply_rgb_lut_u16(image_rgb: np.ndarray, lut_rgb: np.ndarray) -> np.ndarray:
    """Apply a 3x65536 uint16 RGB lookup table."""

    image = _as_u16_rgb(image_rgb)
    lut = np.asarray(lut_rgb, dtype=np.uint16)
    if lut.shape != (3, 65536):
        raise ValueError("lut_rgb must have shape (3, 65536)")
    output = np.empty_like(image)
    for channel in range(3):
        output[..., channel] = lut[channel][image[..., channel]]
    return output
