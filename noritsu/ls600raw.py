"""Load Noritsu LS-600 'FULL...RAW' scan frames.

Recovered layout (from two LS-600 RAW samples provided by an LS-600 owner):
- Headerless raster of 3 x uint16 LE per pixel (B, G, R slots).
- 12-bit scanner data (0..4095) stored in the low bits.
- No dimensions anywhere in the file: our FULL frames are 4042 x 6391, and
  detect_dims() recovers other frame sizes from the byte count alone.
Returns float32 (H, W, 3) in [0,1], channels BGR, linear (no gamma, no WB).
"""

import os

import numpy as np


WIDTH = 4042
HEIGHT = 6391

# (width, height) pairs confirmed against real dumps. Width is the CCD-line
# axis (fixed by the hardware), height is the film-transport axis.
KNOWN_DIMS = [(4042, 6391)]

_BYTES_PER_PIXEL = 6  # 3 channels x uint16

# Bounds for the blind factorization tier: both axes must look like a film
# scanner raster, and a 35mm frame never gets longer than ~4.5:1.
_MIN_DIM = 2000
_MAX_DIM = 16000
_MAX_ASPECT = 4.5


def detect_dims(size):
    """Recover (width, height) of a headerless dump from its byte count alone.

    Three tiers, strictest first; a tier only answers when exactly one
    candidate survives it, otherwise the next tier runs:
    1. exact match against KNOWN_DIMS
    2. one axis matches a known width/height (kept in its known role) and the
       byte count divides cleanly, e.g. a 4042-wide frame of a novel length
    3. blind factorization within scanner-plausible bounds; the pair is
       oriented portrait (width < height) because that is how the transport
       writes every frame we have seen
    Returns (width, height) or None when the size is ambiguous or not a raster.
    """
    if size <= 0 or size % _BYTES_PER_PIXEL:
        return None
    px = size // _BYTES_PER_PIXEL

    for w, h in KNOWN_DIMS:
        if w * h == px:
            return (w, h)

    cands = set()
    for w, h in KNOWN_DIMS:
        if px % w == 0 and 1000 <= px // w <= 65535:
            cands.add((w, px // w))
        if px % h == 0 and 1000 <= px // h <= 65535:
            cands.add((px // h, h))
    if len(cands) == 1:
        return cands.pop()

    pairs = set()
    for a in range(_MIN_DIM, int(px**0.5) + 1):
        if px % a:
            continue
        b = px // a
        if b > _MAX_DIM or b / a > _MAX_ASPECT:
            continue
        pairs.add((a, b))  # portrait: width <= height
    if len(pairs) == 1:
        return pairs.pop()
    return None


def is_ls600_raw(path):
    """True if *path* looks like a headerless LS-600 dump we can decode."""
    if os.path.splitext(str(path))[1].lower() != ".raw":
        return False
    try:
        return detect_dims(os.path.getsize(str(path))) is not None
    except OSError:
        return False


def load_ls600_raw_u16(path, width=None, height=None):
    """Decode a FULL dump without scaling, preserving its BGR uint16 slots."""

    if width is None or height is None:
        dims = detect_dims(os.path.getsize(str(path)))
        if dims is None:
            raise ValueError(
                f"cannot infer dimensions of {path} from its size; "
                "pass width= and height= explicitly"
            )
        width, height = dims
    data = np.fromfile(path, dtype="<u2")
    need = width * height * 3
    if data.size < need:
        raise ValueError(
            f"{path}: expected {need} uint16 samples for {width}x{height}, "
            f"found {data.size}"
        )
    return data[:need].reshape(height, width, 3)


def load_ls600_raw(path, width=None, height=None, maxv=None):
    """Decode a FULL*.RAW dump to float32 (H, W, 3) BGR in [0,1].

    With no arguments the dimensions come from detect_dims() and the container
    depth is sensed from the data (12-bit unless any sample exceeds 4095).
    Pass explicit width/height/maxv to override either.
    """
    arr = load_ls600_raw_u16(path, width=width, height=height).astype(np.float32)
    if maxv is None:
        maxv = 4095.0 if arr.max() <= 4095 else 65535.0
    return np.clip(arr / maxv, 0, 1).astype(np.float32)


# --- LS-600 machine geometry transform ---
# Recovered nominal rectangle plus a separately-labelled reference registration.
# The LS-600 crops the 6391x4042 portrait raw (BGR, 12-bit) to a sub-rect
# then resamples to 6048x4011 output. Derivation from TrzCorFile UsImRct:
#   THUM (252x399) UsImRct=(1,10,251,388) => crop 250x378
#   THUM is ~1/16 downsampled FULL: y0=16, x0=168, h=4008, w=6048
#   Final resize 4008x6048 -> 4011x6048 (INTER_AREA)
CROP_Y0 = 16
CROP_X0 = 168
CROP_H = 4008
CROP_W = 6048
TIF_H = 4011
TIF_W = 6048

# The primary Portra 400 machine pair aligns more closely at this rectangle.
# This is external-reference registration, not a claim about the proprietary
# transform.  Keep it separate from the Trz-derived nominal constants above.
REFERENCE_CROP_Y0 = 17
REFERENCE_CROP_X0 = 171


def align_to_machine(
    image,
    *,
    rotation_k=None,
    crop_y=CROP_Y0,
    crop_x=CROP_X0,
    crop_h=CROP_H,
    crop_w=CROP_W,
    output_h=TIF_H,
    output_w=TIF_W,
):
    """Rotate, crop, and area-resample an LS-600 raster.

    Args:
        image: an HxWx3 array in any three-channel domain.
        rotation_k: explicit ``numpy.rot90`` count, or ``None`` when already
            landscape. Rotation is job/export state and cannot be inferred from
            ``ScnCrg`` alone.
        crop_*: rectangle in the rotated image. Defaults are the Trz-derived
            nominal transform; reference-fitting code may explicitly pass the
            separately documented empirical registration.

    Returns:
        An ``(output_h, output_w, 3)`` array with the input dtype.
    """
    import cv2

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if rotation_k is not None:
        array = np.rot90(array, int(rotation_k))
    stop_y = int(crop_y) + int(crop_h)
    stop_x = int(crop_x) + int(crop_w)
    if min(crop_y, crop_x, crop_h, crop_w, output_h, output_w) < 0:
        raise ValueError("geometry values must be non-negative")
    if stop_y > array.shape[0] or stop_x > array.shape[1]:
        raise ValueError("crop rectangle lies outside the rotated image")
    cropped = array[int(crop_y) : stop_y, int(crop_x) : stop_x, :]
    return cv2.resize(
        cropped,
        (int(output_w), int(output_h)),
        interpolation=cv2.INTER_AREA,
    ).astype(array.dtype, copy=False)
