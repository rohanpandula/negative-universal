# Noritsu installer pipeline recovery

This repository contains an evidence-labelled reconstruction of the Noritsu
LS-600 / RD5X film path. It separates three things that older versions of this
project mixed together:

1. exact integer or float32 kernels recovered from the shipped Windows DLLs;
2. stages whose selector or physical scanner inputs are unavailable for a given run;
3. an older, illustrative negative-conversion CLI that is useful creatively but
   is not a vendor reproduction.

The recovery audit uses Kuna 1.95 decompilation, instruction-level checks,
Noritsu `CorrectServ` tables, a captured EZ Controller correction file, and one
aligned LS-600 raw/machine-TIFF pair. It renders a full-resolution checkpoint
after every recovered normal-path color/geometry branch—even disabled and
blocked branches—and records
numeric, visual, and provenance evidence for each one.

## Run the fixed real-scan audit

Install Python 3.10+ dependencies:

```sh
python -m pip install -r requirements.txt
```

Then run:

```sh
python -m noritsu.audit \
  --preset portra400-frame1 \
  --output-dir ./stage-audit/portra400-frame1
```

The preset verifies SHA-256 hashes for the raw scan, machine/native references,
captured PRM, final PCB, COM tables, image descriptor, native gamma oracle, and
depth-capture log before processing. It refuses to overwrite a non-empty output
directory.

Important outputs:

| Artifact | Meaning |
| --- | --- |
| `report.md` | readable stage table and headline diagnostics |
| `manifest.json` | complete stage status, parameters, provenance, and metrics |
| `metrics.csv` | adjacent, cumulative-input, and machine-reference differences |
| `contact_sheet.png` | all stage previews in execution order |
| `checkpoints/NNN_*.tif` | full-resolution uint16 render after stage `NNN` |
| `diffs/NNN_*_vs_previous_*` | absolute, heatmap, and signed differences when an executed adjacent stage is comparable |
| `installer_color_boundary_native.tif` | last recovered fixed-job output before any reference fit |
| `recovered_boundary.tif` | same native-parameter recovery boundary, with machine/native comparisons |
| `final.tif` | hard link to `recovered_boundary.tif`, never to a fitted result |
| `boundary_comparison/` | recovered-boundary vs machine-target visual diagnostics |
| `native_engine_comparison/` | recovered Python stages vs shipped-engine oracle |
| `native_engine_vs_machine/` | shipped-engine oracle vs captured machine TIFF |
| `external_fit*.tif` | separately labelled in-sample reference-fit diagnostics |

Unchanged and carried-forward stages use hard-linked TIFF checkpoints, so every stage remains
independently inspectable without multiplying the storage cost of identical
full-resolution images. A blocked stage's adjacent effect is reported as N/A,
not as a fabricated zero; exact zero is reserved for proven identities and
disabled branches. Domain or shape transitions are logged as not
numerically comparable rather than being normalized into a misleading metric;
their before/after previews remain visible in the contact sheet.

## What is recovered

The normal film-output selector is `ExecPattern=5`. For 35 mm color negative,
the installer selects the following order:

```text
scanner acquisition
  -> CCD gain + FilmLUT
  -> CCD noise clear
  -> spatial corrections
  -> grain
  -> trim/scale/rotate
  -> RGB to YCC
  -> automatic contrast
  -> film sharp
  -> YCC correction
  -> hue/saturation
  -> YCC to RGB + gamma
  -> optional repair/filter/output branches
  -> CMS
```

Implemented exact or parameterized kernels include:

- LS-600 `FULL*.RAW` decoding as headerless little-endian uint16 BGR;
- CreateImage fused dark-offset/CCD-linear/per-position signed-Q12 4×4
  B/G/R/IR matrix, plus its separate alternative digital-gain operation;
- RD5X Q12 CCD-gain/FilmLUT composition and Nega FilmLUT application;
- full parameterized CCD-noise-clear S0 and Grain S4 array workers;
- full parameterized AutoContrast S0 luminance worker, including local-map
  downsampling, filtering, expansion, and tone application;
- full parameterized Hue/Saturation S0 worker, including its runtime
  `9x33x33x3` correction lattice and every saturation branch;
- the complete FilmSharp spatial graph, response/envelope/tone LUT builders,
  and final YCC compositor;
- trimming, the shared box prefilter, S0 cubic and S2 bilinear scaling,
  orthogonal and arbitrary-angle rotation/trim, and 8/16-bit blank canvas
  fill/placement with rotation-break halo and boundary smoothing;
- RGB-to-YCC S1 transfer-LUT construction and float32 pixel transform;
- YCC-correction control-table builders and the table-driven pixel worker;
- byte and 16-bit YCC-to-RGB S0 variant-1 gamut compression, fused-gamma LUT
  builders, and exact 8-to-16-bit container expansion, with the captured image
  descriptor selecting the byte path for this job;
- complete `pcb_makeNegaTone2`, including its embedded UO forward/inverse
  tables, three-region `pcb_MakeGammaCurve2`, and final anchor shift;
- observed 8-bit-code expansion and explicit external monotone-fit diagnostics.

For the fixed Portra 400 frame, the captured controller state serializes RGB
scanner gains of exactly `1.0, 1.0, 1.0` with `ScnCrg=1`. The installed 2015
SpecialPcb formula therefore yields unity Q12 gains `[4096, 4096, 4096]` with
high cross-version confidence. The result is also present in the native PCB
rebuilt by the shipped 2015 engine from the disposable compatibility-stamped
capture. The generic Nega FilmLUT is then applied with the installer’s
RGB-payload/BGR-raster mapping.

The captured `.prm` also proves that merge, lens aberration, distortion,
trapezoid, manual repair, frame red-eye, cross, soft, and spot-filter branches
are off. Hue/saturation remains selected: although its visible saturation and
balance controls are neutral, the native `COM_PARA_FOR_ALL+8` capture contains
a nonzero 9×33×33 chroma-correction lattice, which the audit applies exactly.
Media sharp, color sharp, color
grain, standalone gamma, and NxN are disabled by the effective selector or
captured switch. The captured `CmsInf.CmsSW` node at offsets `0xFF0/0xFF4`
has `count=1,value=0`; the recovered executor branch proves CMS is off, so no
profile payload is consumed.

## Native parameter reconstruction

The pair-specific `RD5X_PCB_PARA_DATA` is no longer missing. A disposable copy
of the captured correction file was given the installed loader's producer
stamp and a recomputed checksum; the captured payload itself was preserved.
The shipped `CreateOutputForFilm`/`RD5X_GetSpecialPcb` chain returned success
and was instrumented at its normal return boundary. Two preview runs produced
the same 84,286-byte PCB (`SHA-256 5384e7f8...4218`). Rebuilding with the
Trz-nominal full rectangle produced the canonical full-scan PCB
`SHA-256 a22aa8b6...28d3`: native descriptor `W=6391,H=4042`, trim
`(x=168,y=16,w=6048,h=4008)`, and exact S0 cubic scaling to 6048×4011
with sampling 2.  The two values are independent: main-procedure selector 2
dispatches `FmScalingS0_Do`, while scaling-block byte `+0x11=2` requests two
samples per output pixel.

This native output supersedes inference from the static pattern-5 selector
table. It proves that grain, automatic contrast, and blank/output fill are off
for the captured job; CCD noise clear, RGB-to-YCC, film sharp, YCC correction,
hue/saturation, and fused-gamma YCC-to-RGB remain selected. The core color
blocks are byte-identical between preview and full-size builds. The one
resolution-dependent FilmSharp word and every spatial block used by the audit
come from the canonical full-scan PCB, never the 399×252 preview block.

The actual owner FULL was also replayed end-to-end through the shipped engine
using the canonical disposable CF/PCB state. The native 6048×4011 RGB8 TIFF is
`SHA-256 ef673724...9c5b`. It is an independent execution oracle, not a fitted
replacement for any recovered stage. A packed image-descriptor capture records
`dstType=0`, and the propagated Unify depth flag is also zero, proving that the
selected fused-gamma worker returns RGB8 directly.

Two earlier scanner stages have genuine physical-data boundaries when starting
from a sensor-receive raster: the per-unit 5340-position offset/linear/color
calibration arrays, and the fourth IR/Kizu plane used for dust repair. The
owner’s `FULL*.RAW` is already downstream of scanner acquisition, so those
stages must not be reapplied to it.

## Fixed-pair diagnostics

The primary pair uses:

- raw storage array: `H=6391,W=4042`, three-channel BGR uint16 with nominal
  12-bit values; after the explicit repack, native source is `H=4042,W=6391`;
- explicit export orientation: `numpy.rot90(..., 1)`;
- native Trz-nominal crop: rotated `y=16, x=168, h=4008, w=6048`;
- native S0 cubic resize: 6048×4008 to 6048×4011, sampling 2;
- output array: `H=4011,W=6048` RGB;
- machine target: uint16 TIFF whose samples are all multiples of 257 and which
  has no embedded ICC profile.

The crop, output dimensions, sampling selector, and exact resampler come from
the native final PCB. Only the headerless FULL storage interpretation is
pair-specific: the portrait raster is repacked with `numpy.rot90(..., 1)`
before the landscape RD5X descriptor consumes it, and this cannot be inferred
from `ScnCrg` alone. The older `(y=17,x=171)` empirical registration remains a
diagnostic only and is not used by the recovered path.

At the recovered boundary, after native Trz trim/S0 cubic scaling and all
selected color stages, the audit reports three distinct comparisons: recovered Python
versus the machine TIFF, recovered Python versus the shipped-engine oracle,
and shipped engine versus machine TIFF. The generated report is authoritative
for the exact checked-out implementation.

| Diagnostic | MAE | RMSE | BT.709 luma NCC |
| --- | ---: | ---: | ---: |
| recovered vs machine | `0.06708299` | `0.09757489` | `0.94552705` |
| recovered vs native engine | `0.00000000` | `0.00000000` | `1.00000000` |
| native engine vs machine | `0.06708299` | `0.09757489` | `0.94552705` |

All 72,775,584 recovered values equal the shipped 2015 engine's RGB8 values
after the documented lossless `u8 * 257` uint16 representation expansion
(expanded-array SHA-256 `0f5babb0...791`). The remaining difference from the
separately captured 2017 machine TIFF is therefore a legacy-version/controller-
state attribution gap, not an unresolved local worker or scanner-hardware
explanation.

A separate per-channel monotone fit against the same target is emitted when
requested. That in-sample fit is deliberately not called a Noritsu stage.
Because the sole target is unprofiled and effectively 8-bit, all of these are
code-value and structural diagnostics, not colorimetric accuracy claims.

## Tables and film names

`CorrectServ` contains quality-group/type tables for negative/positive,
format, and color/B&W modes. They are not Kodak/Fuji stock profiles. Nega maps
to quality types 0 and 1; this project no longer silently defaults to the old
group-0/type-5 path.

The checked-in `vendor_nega_135.json` contains reproducible numeric extraction
for the supported Nega 135 selections. To read an installed table tree directly:

```sh
NORITSU_TABLES=/path/to/ImageDataProc/Data/CorrectServ \
  python -m noritsu.audit ...
```

Source file names, sizes, consumed struct sizes, and hashes are written into
the run manifest. When `NORITSU_TABLES` is used, resolved source paths are
recorded as well.

## Tests

Fast tests:

```sh
ruff check noritsu test_*.py tools
python -m pytest -q
```

The full real-pair regression is opt-in because it processes about 155 MB of
raw input and writes roughly 1.5–2 GB of trace artifacts:

```sh
NORITSU_RUN_REAL_PAIR=1 python -m unittest test_real_pair
```

It pins the input hashes, recovered-boundary pixel hash, both external-fit
pixel hashes, metrics, and the separation between `final.tif` and fitted output.

## Illustrative conversion CLI

The original convenience command remains available:

```sh
python -m noritsu.cli negative.tif -o positive.tif
```

It performs a generic density-domain C-41 inversion followed by synthetic tone,
white-balance, and saturation controls. `--stock` and `--calib` load external
reference-fit profiles; they are not installer film-stock tables.
`--real-curve` directly interpolates one recovered CommonCalc control bank but
does not reproduce the full SpecialPcb curve builder. `--lab` is an
illustrative gentler preset, not an exact mapping of EZ Controller DSA controls.

Use this CLI for creative conversion. Use `noritsu.audit` for claims about the
recovered installer path.

## Other documentation

- `docs/INSTALLER_PIPELINE_AUDIT.md` — evidence map, stage decisions, and exact
  blocker boundary;
- `docs/RD5X_INPUT_FORMATS.md` — recovered raster ABIs;
- `docs/RD5X_XML_SCHEMA.md` — recovered engine command schema;
- `docs/JOURNEY.md` and `docs/BLOG_POST.md` — historical notes that may preserve
  superseded hypotheses; this README and the audit manifest are authoritative.

## License

MIT. The reverse engineering was performed for research and interoperability.
No Noritsu binaries are distributed; checked-in data is derived numeric and
structural knowledge.
