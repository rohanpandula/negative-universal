"""Decode the native ``RD5X_PCB_PARA_DATA`` produced by SpecialPcb.dll.

The offsets in this module are the packed x86 ABI, not a reconstructed or
normalized parameter format.  They come from the installed binaries:

* ``SpecialPcb!RD5X_GetSpecialPcb`` (VA ``0x1000a100``) clears and fills an
  exact ``0x1493e``-byte output object;
* ``SpecialPcb!sub_100083f0`` writes the named blocks below; and
* the corresponding UniRdImgLib dispatchers consume those same byte offsets.

Keeping the decoder separate from :mod:`noritsu.vendor` makes the evidence
boundary explicit: this module only unpacks captured bytes, while ``vendor``
contains the recovered pixel arithmetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FINAL_PCB_SIZE = 0x1493E
COM_HEADER_SIZE = 0x18
COM_GRAIN_CURVE_SIZE = 256 * 2
COM_HUE_LATTICE_SHAPE = (9, 33, 33, 3)
COM_HUE_LATTICE_SIZE = int(np.prod(COM_HUE_LATTICE_SHAPE)) * 2
IMAGE_DESCRIPTOR_PREFIX_SIZE = 0x40
IMAGE_DESCRIPTOR_DST_TYPE_OFFSET = 0x1C

# RD5X has no selector at +0x20.  SpecialPcb's ExecSW mapping fills +0x10..1f
# and +0x21..31; UniRdImgLib!RD5X_DoMainProcForFilm consumes the same layout.
SELECTOR_OFFSETS = tuple(range(0x10, 0x20)) + tuple(range(0x21, 0x32))

CCD_NOISE_S0_OFFSET = 0x3C
GRAIN_OFFSET = 0xE0
TRIMMING_OFFSET = 0x178
SCALING_OFFSET = 0x188
RGB_TO_YCC_OFFSET = 0x1AC
AUTO_CONTRAST_OFFSET = 0x2BA
HUE_SATURATION_OFFSET = 0x5F8
YCC_CORRECTION_OFFSET = 0x506
YCC_TO_RGB_OFFSET = 0x6D6
FUSED_GAMMA_OFFSET = 0x6EC


def read_final_pcb(path: str | Path) -> bytes:
    """Read and size-check one native final-PCB capture."""

    return _coerce_final_pcb(Path(path))


def _coerce_final_pcb(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> bytes:
    if isinstance(pcb, (str, Path)):
        data = Path(pcb).read_bytes()
    elif isinstance(pcb, bytes):
        data = pcb
    elif isinstance(pcb, (bytearray, memoryview)):
        data = bytes(pcb)
    else:
        raise TypeError("final PCB must be bytes-like or a filesystem path")
    if len(data) != FINAL_PCB_SIZE:
        raise ValueError(
            f"RD5X_PCB_PARA_DATA must be exactly 0x{FINAL_PCB_SIZE:x} bytes"
        )
    return data


def _array(data: bytes, offset: int, dtype: str, count: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.dtype(dtype), count=count, offset=offset).copy()


def _coerce_exact_blob(
    blob: bytes | bytearray | memoryview | str | Path,
    expected_size: int,
    name: str,
) -> bytes:
    if isinstance(blob, (str, Path)):
        data = Path(blob).read_bytes()
    elif isinstance(blob, bytes):
        data = blob
    elif isinstance(blob, (bytearray, memoryview)):
        data = bytes(blob)
    else:
        raise TypeError(f"{name} must be bytes-like or a filesystem path")
    if len(data) != expected_size:
        raise ValueError(f"{name} must be exactly {expected_size} bytes")
    return data


def decode_com_grain_curve_u16(
    blob: bytes | bytearray | memoryview | str | Path,
) -> np.ndarray:
    """Decode the 256-word Grain curve referenced by ``COM_PARA_FOR_ALL+4``.

    ``RD5X_DoMainProcForFilm`` copies the first 0x18 bytes of
    ``COM_PARA_FOR_FILM`` into its local common header.  ``FmGrainS4_Do`` at
    VA ``0x100738f0`` and the other Grain workers dereference header ``+4`` as
    exactly 256 native little-endian unsigned words.  The capture hook records
    that pointee, not its process-specific address.
    """

    data = _coerce_exact_blob(
        blob, COM_GRAIN_CURVE_SIZE, "COM Grain curve"
    )
    return np.frombuffer(data, dtype="<u2", count=256).copy()


def read_grain_curve(
    path: str | Path,
) -> np.ndarray:
    """Read one captured ``COM_PARA_FOR_ALL+4`` Grain curve."""

    return decode_com_grain_curve_u16(path)


def decode_com_hue_lattice_i16(
    blob: bytes | bytearray | memoryview | str | Path,
) -> np.ndarray:
    """Decode the Hue/Saturation S0 lattice at ``COM_PARA_FOR_ALL+8``.

    ``FmHueSatCorrectionS0_Do`` at VA ``0x10055c10`` ultimately consumes
    ``9 * 33 * 33`` signed triplets from this pointee.  Axes are native
    ``[Y, C0, C1, correction]`` and correction channels are
    ``[delta-Y, delta-C0, delta-C1]``, matching
    :func:`vendor.apply_hue_saturation_s0` without an inferred transpose or
    channel swap.
    """

    data = _coerce_exact_blob(
        blob, COM_HUE_LATTICE_SIZE, "COM Hue/Saturation lattice"
    )
    return np.frombuffer(data, dtype="<i2").reshape(COM_HUE_LATTICE_SHAPE).copy()


def read_hue_saturation_lattice(
    path: str | Path,
) -> np.ndarray:
    """Read one captured ``COM_PARA_FOR_ALL+8`` Hue/Saturation lattice."""

    return decode_com_hue_lattice_i16(path)


def read_com_header(path: str | Path) -> bytes:
    """Read the captured 0x18-byte common header without decoding pointers.

    Its six dwords are process-local values and must not be reused as stable
    addresses.  Exact-size validation still makes the header useful capture
    provenance alongside its separately dumped pointees.
    """

    return _coerce_exact_blob(path, COM_HEADER_SIZE, "COM_PARA_FOR_ALL header")


def decode_image_descriptor_prefix(
    blob: bytes | bytearray | memoryview | str | Path,
) -> dict[str, int]:
    """Decode stable fields from the captured packed file-image descriptor.

    The canonical Unify hook captured exactly the first 0x40 bytes of its
    ninth argument.  File ``DoUnifyProcForFilm`` reads source dimensions at
    VA ``0x101454ac/0x101454c2``, destination type at ``0x10145541``, and
    destination dimensions at ``0x10145583/0x1014558e``.  Their packed
    offsets are ``+0x0a/+0x0e``, ``+0x1c``, and ``+0x1e/+0x22`` respectively.
    Unnamed pointer and state fields in the prefix are intentionally omitted.
    """

    data = _coerce_exact_blob(
        blob, IMAGE_DESCRIPTOR_PREFIX_SIZE, "Unify image-descriptor prefix"
    )
    return {
        "size_bytes": len(data),
        "source_width": int(_array(data, 0x0A, "<i4", 1)[0]),
        "source_height": int(_array(data, 0x0E, "<i4", 1)[0]),
        "dst_type": int(
            _array(data, IMAGE_DESCRIPTOR_DST_TYPE_OFFSET, "<u2", 1)[0]
        ),
        "destination_width": int(_array(data, 0x1E, "<i4", 1)[0]),
        "destination_height": int(_array(data, 0x22, "<i4", 1)[0]),
    }


def read_image_descriptor_prefix(path: str | Path) -> dict[str, int]:
    """Read and decode one captured 0x40-byte Unify descriptor prefix."""

    return decode_image_descriptor_prefix(path)


def decode_geometry(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, int]:
    """Decode the source and destination dimensions copied into ``+0..+0xf``.

    ``RD5X_GetSpecialPcb`` copies CalcInput dwords ``+8,+0xc,+0x10,+0x14``
    here in order.  ImgCorrectDLL logs them as source ``[w,h]`` and destination
    ``[w,h]`` immediately after the size check at VA ``0x100fc3a0``.
    """

    data = _coerce_final_pcb(pcb)
    source_width, source_height, destination_width, destination_height = (
        int(value) for value in _array(data, 0, "<u4", 4)
    )
    return {
        "source_width": source_width,
        "source_height": source_height,
        "destination_width": destination_width,
        "destination_height": destination_height,
    }


def decode_effective_selectors(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, int]:
    """Return the final post-rewrite RD5X execution selectors."""

    data = _coerce_final_pcb(pcb)
    return {f"0x{offset:02x}": data[offset] for offset in SELECTOR_OFFSETS}


def decode_scaling_trimming_parameters(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Decode the exact crop and S0/S2 scaling arguments.

    ``RD5X_DoMainProcForFilm`` at VA ``0x100487e0`` passes final-PCB
    ``+0x178`` as ``RD5X_TRIMMING_PARA`` and ``+0x188`` as
    ``RD5X_SCALING_PARA`` to the selected scaling worker.  The former is four
    signed dwords in ``x, y, width, height`` order.  ``FmScalingS0_Do`` at
    ``0x100d50d0`` and ``FmScalingS2_Do`` at ``0x100d6e30`` read input and
    output dimensions from scaling dwords ``+0..+0xc``, followed by the
    byte-sized prefilter and sampling controls at ``+0x10/+0x11``.  The main
    dispatcher sends selector values 1..4 to S0 unchanged and 5..8 to S2
    after subtracting four.  Thus selector 2 selects S0; the independent
    sampling byte may also equal 2.

    ``trim``, ``output_width``, ``output_height``, ``pre_scale``, and
    ``sampling`` can be passed directly to :func:`vendor.apply_scaling_s0_u16`
    or :func:`vendor.apply_scaling_s2_u16`.  The input dimensions and final
    selectors are retained so callers can validate the native dispatch rather
    than infer it from the dimensions.
    """

    data = _coerce_final_pcb(pcb)
    trim = tuple(
        int(value) for value in _array(data, TRIMMING_OFFSET, "<i4", 4)
    )
    input_width, input_height, output_width, output_height = (
        int(value) for value in _array(data, SCALING_OFFSET, "<u4", 4)
    )
    return {
        "trim": trim,
        "input_width": input_width,
        "input_height": input_height,
        "output_width": output_width,
        "output_height": output_height,
        "pre_scale": data[SCALING_OFFSET + 0x10],
        "sampling": data[SCALING_OFFSET + 0x11],
        "trimming_selector": data[0x1A],
        "scaling_selector": data[0x1B],
    }


def decode_ccd_noise_s0_parameters(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, np.ndarray]:
    """Decode arguments for the complete CCD-noise S0 vendor worker.

    UniRdImgLib ``FmCCDNoiseClearS0_Do`` at VA ``0x1007f780`` reads the three
    window bytes at ``+0..+2``.  Its calls at ``0x1007fbbb..0x1007fcbb`` read
    RGB-interleaved response gains, four level triplets, and four breakpoint
    triplets.  ``sub_1007ee80`` reads edge sensitivity at ``+0x2a..+0x2c``
    and edge limits at ``+0x2d..+0x2f``.  The returned names match the explicit
    RGB APIs in :mod:`noritsu.vendor`.
    """

    data = _coerce_final_pcb(pcb)
    base = CCD_NOISE_S0_OFFSET
    levels = _array(data, base + 0x06, "u1", 12).reshape(4, 3).T.copy()
    breakpoints = (
        _array(data, base + 0x12, "<u2", 12).reshape(4, 3).T.copy()
    )
    return {
        "levels_rgb_u8": levels,
        "breakpoints_rgb_u16": breakpoints,
        "gains_rgb_u8": _array(data, base + 0x03, "u1", 3),
        "window_sizes_rgb": _array(data, base + 0x00, "u1", 3),
        "edge_sensitivity_rgb_u8": _array(data, base + 0x2A, "u1", 3),
        "edge_limits_rgb_u8": _array(data, base + 0x2D, "u1", 3),
    }


def decode_grain_s4_controls(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, int]:
    """Decode the packed Grain S4 controls at final-PCB ``+0xe0``.

    ``FmGrainS4_Do`` at VA ``0x100738f0`` and helpers ``0x1006c910`` /
    ``0x10072e20`` consume these fields.  The 256-word variance curve is not
    part of this block; it is reached separately through ``COM_PARA_FOR_ALL
    +4`` and therefore is intentionally not invented here.
    """

    data = _coerce_final_pcb(pcb)
    base = GRAIN_OFFSET
    divisors = _array(data, base, "<u4", 2)
    return {
        "base_divisor_u32": int(divisors[0]),
        "range_divisor_u32": int(divisors[1]),
        "color_offset_u8": data[base + 8],
        "luminance_offset_u8": data[base + 9],
        "window_size": data[base + 0x0A],
    }


def decode_rgb_to_ycc_s1_parameters(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, np.ndarray]:
    """Decode all array arguments needed by the recovered RGB-to-YCC S1 path.

    The packed ``RD5X_RGB_TO_YCC_PARA`` at final-PCB ``+0x1ac`` contains the
    signed 3x3 Q10000 matrix at ``+0``, signed RGB input offsets at ``+0x12``,
    count bytes at ``+0x18``, three 20-word X arrays at ``+0x1c/+0x44/+0x6c``,
    and three 20-word Y arrays at ``+0x94/+0xbc/+0xe4``.  These locations are
    read by UniRdImgLib's setup routine ``sub_100859e0`` and its S1 wrapper.
    """

    data = _coerce_final_pcb(pcb)
    base = RGB_TO_YCC_OFFSET
    return {
        "matrix_q10000": _array(data, base + 0x00, "<i2", 9),
        "input_offsets_rgb": _array(data, base + 0x12, "<i2", 3),
        "counts_rgb": _array(data, base + 0x18, "u1", 3),
        "x_controls_rgb": _array(data, base + 0x1C, "<u2", 60).reshape(3, 20),
        "y_controls_rgb": _array(data, base + 0x94, "<u2", 60).reshape(3, 20),
    }


def decode_auto_contrast_s0_parameters(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Decode packed AutoContrast S0 controls when that selector is active.

    ``sub_1005aa40`` reads the downsample factor and odd kernel size at
    ``+0/+1``; the upper-left kernel quadrant uses eight-byte rows from ``+2``.
    It then reads strength ``+0x42``, clamp radius ``+0x44``, three interior
    X/Y words at ``+0x46..+0x50``, and four S0 Q9 slope words at ``+0x52..+0x58``.
    An inactive, zero-filled block is rejected instead of being presented as
    callable parameters.
    """

    data = _coerce_final_pcb(pcb)
    base = AUTO_CONTRAST_OFFSET
    downsample_factor = data[base]
    kernel_size = data[base + 1]
    if kernel_size not in (5, 7, 9):
        raise ValueError(
            "final PCB does not contain an active AutoContrast S0 kernel"
        )
    half = kernel_size // 2
    quadrant = np.empty((half + 1, half + 1), dtype=np.int8)
    for row in range(half + 1):
        quadrant[row] = _array(
            data, base + 2 + row * 8, "i1", half + 1
        )
    return {
        "downsample_factor": downsample_factor,
        "kernel_size": kernel_size,
        "kernel_quadrant_i8": quadrant,
        "spatial_strength_u8": data[base + 0x42],
        "clamp_radius_u16": int(_array(data, base + 0x44, "<u2", 1)[0]),
        "inner_x_u16": _array(data, base + 0x46, "<u2", 3),
        "inner_y_u16": _array(data, base + 0x4C, "<u2", 3),
        "extrapolation_slopes_q9": _array(data, base + 0x52, "<i2", 4),
    }


def decode_hue_saturation_s0_controls(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Decode final-PCB controls for Hue/Saturation S0.

    The wrapper at VA ``0x10055c10`` dispatches on saturation byte ``+0``;
    the zero-saturation helper at ``0x10054290`` reads signed TotalBal words
    ``+2/+4``.  The 9x33x33x3 correction lattice is a separate pointee at
    ``COM_PARA_FOR_ALL +8`` and is deliberately reported as absent here.
    """

    data = _coerce_final_pcb(pcb)
    base = HUE_SATURATION_OFFSET
    return {
        "saturation_u8": data[base],
        "total_balance_i16": _array(data, base + 2, "<i2", 2),
        "correction_lattice_source": "COM_PARA_FOR_ALL+8",
    }


def decode_ycc_correction_controls(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Decode controls for the three exact YCC-correction table builders.

    UniRdImgLib ``FmYccCorrection_Do`` at VA ``0x100890a0`` passes this block
    to builders ``0x10087580``, ``0x10087740``, and ``0x10087840``.  The first
    curve has independent X/Y arrays and count; chroma scale and both signed
    chroma-offset curves share the second X array/count.
    """

    data = _coerce_final_pcb(pcb)
    base = YCC_CORRECTION_OFFSET
    return {
        "y_x_u16": _array(data, base + 0x00, "<u2", 20),
        "y_values_q4_u16": _array(data, base + 0x28, "<u2", 20),
        "y_count": data[base + 0x50],
        "chroma_count": data[base + 0x51],
        "chroma_x_u16": _array(data, base + 0x52, "<u2", 20),
        "chroma_offsets_i16": np.stack(
            (
                _array(data, base + 0x7A, "<i2", 20),
                _array(data, base + 0xA2, "<i2", 20),
            )
        ),
        "chroma_scale_u16": _array(data, base + 0xCA, "<u2", 20),
    }


def decode_ycc_to_rgb_gamma_controls(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, np.ndarray]:
    """Decode the selected fused YCC-to-RGB matrix and gamma controls.

    ``RD5X_DoMainProcForFilm`` at VA ``0x100487e0`` passes final-PCB
    ``+0x6d6`` and ``+0x6ec`` to the selected YCC-to-RGB wrapper.  The former
    is its signed 3x3 Q10000 matrix.  The packed ``RD5X_GAM_LUT_OCT_PARA``
    contains RGB count bytes at ``+0..+2``, three 20-word X arrays at
    ``+4/+0x2c/+0x54``, and three 20-byte Y arrays at
    ``+0x7c/+0x90/+0xa4``.

    The structure itself is RGB ordered.  ``FmYccToRgbS0_Do`` at
    ``0x1008d2a0`` deliberately builds and consumes the channels in B, G, R
    order, so :func:`vendor.build_fused_gamma_luts_bgr` performs that explicit
    reversal rather than hiding it in this ABI decoder.
    """

    data = _coerce_final_pcb(pcb)
    base = FUSED_GAMMA_OFFSET
    return {
        "matrix_q10000": _array(data, YCC_TO_RGB_OFFSET, "<i2", 9),
        "counts_rgb": _array(data, base + 0x00, "u1", 3),
        "x_controls_rgb": _array(data, base + 0x04, "<u2", 60).reshape(3, 20),
        "y_controls_rgb_u8": _array(data, base + 0x7C, "u1", 60).reshape(3, 20),
    }


def decode_final_pcb_summary(
    pcb: bytes | bytearray | memoryview | str | Path,
) -> dict[str, Any]:
    """Return a JSON-safe summary without manufacturing inactive parameters."""

    data = _coerce_final_pcb(pcb)
    selectors = decode_effective_selectors(data)
    ccd = decode_ccd_noise_s0_parameters(data)
    rgb_to_ycc = decode_rgb_to_ycc_s1_parameters(data)
    hue = decode_hue_saturation_s0_controls(data)
    ycc = decode_ycc_correction_controls(data)
    ycc_to_rgb_gamma = decode_ycc_to_rgb_gamma_controls(data)

    def lists(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }

    summary: dict[str, Any] = {
        "size_bytes": len(data),
        "geometry": decode_geometry(data),
        "effective_selectors": selectors,
        "scaling_trimming": decode_scaling_trimming_parameters(data),
        "ccd_noise_s0": lists(ccd),
        "rgb_to_ycc_s1": lists(rgb_to_ycc),
        "hue_saturation_s0": lists(hue),
        "ycc_correction": lists(ycc),
        "ycc_to_rgb_gamma": lists(ycc_to_rgb_gamma),
    }
    if selectors["0x18"] == 5:
        summary["grain_s4"] = decode_grain_s4_controls(data)
    else:
        summary["grain_s4"] = {
            "selected": False,
            "effective_selector": selectors["0x18"],
        }
    if selectors["0x1f"] == 1:
        summary["auto_contrast_s0"] = lists(
            decode_auto_contrast_s0_parameters(data)
        )
    else:
        summary["auto_contrast_s0"] = {
            "selected": False,
            "effective_selector": selectors["0x1f"],
        }
    return summary
