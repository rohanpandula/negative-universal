# Installer pipeline audit

This is the static evidence map behind `python -m noritsu.audit`. The generated
`manifest.json` is authoritative for an individual run: it records every
checkpoint, native domain, selector, parameter source, status, numeric delta,
visual diff, and source hash.

## Evidence set

The recovery used the extracted Windows installer at
`<NORITSU_RE_ROOT>`, Kuna 1.95, instruction-level x86 checks, the
installed `CorrectServ` banks, and an external LS-600 capture corpus.

| Artifact | SHA-256 |
| --- | --- |
| owner `FULL000000010000.RAW` | `7d18317868cc7abf6b971d547b00e3fc21871efb384aa2a247ef6a87780a8247` |
| machine `000002250001.tif` | `715f937ab5097d372f197295da94709d306164de7d1ca7502a38dd93743a2dc7` |
| captured `Original/1.prm` | `098c33a4f11f4f338ffcb9c9c784543546fc13741b47eab1009543b757f288b0` |
| canonical final PCB | `a22aa8b6e1a6003d50cbffeb6bec732b63e5986c3ddae0270a561f29ffef28d3` |
| native end-to-end output | `ef67372413ede29625061d820414ffb32b2df194dddb196f130726aaa8fd9c5b` |
| native Hue/Saturation lattice | `d0a2f9a06e43fce3061d4afc34b1c25a74036f99980378309db342aa390aeb9b` |
| native Grain curve (disabled in this job) | `214ad9ae4f8bab62612e769773a1621d296f9699c0f531972cef7afbd6e491cd` |
| native image-descriptor prefix | `e3d727c920f757d112759da4ae780a184db5b24722ad7c3bd1cbee2fc371143a` |
| native fused-gamma uint8 LUT oracle | `b095760adb3c5485520eb0efc523f79c46f39215180a966b62ab1a74a88e410b` |

`CorrectServ` contains 808 files (55,626,521 bytes). Its axes are quality
group/type, negative/positive, format, and color/B&W—not Kodak, Fuji, or other
emulsion names. No film-stock spectral mapping was found.

## Native controller reconstruction

`ImgCorrectDLL!CreateOutputForFilm` sets `ExecPattern=5`. SpecialPcb
`FUN_100047f0 @ 0x100047f0` reads source row `r` at `30*r+5` and maps it to PCB
selectors `+0x10..+0x1f,+0x21..+0x31`. Those static bytes are only initial
choices; per-frame builders rewrite them.

The captured correction file has the expected 0x504e8 layout but a 2017
producer stamp. A disposable copy—not the source—was compatibility-stamped for
the installed 2015 loader and given a recomputed checksum. Native command 7
then returned success and `RD5X_GetSpecialPcb` emitted its exact 0x1493e-byte
output. Repeated preview captures were byte-identical. A full-size rebuild with
the Trz rectangle yielded the canonical PCB above:

```text
source        6391 x 4042
trim          x=168, y=16, w=6048, h=4008
S0 cubic      6048 x 4008 -> 6048 x 4011, prefilter=1, sampling=2
rotation      selected identity
```

The headerless owner file stores a portrait 6391×4042 BGR raster. RD5X expects
the landscape descriptor above, so the fixed pair is explicitly repacked with
`numpy.rot90(..., 1)`. The resulting BGR byte stream has SHA-256
`b89a8d2427f8fc5c4905dde773cb64f1035770a36233e372ce1599107ab28534`.
This is capture-storage interpretation, not the later RD5X rotation stage.

The same oriented owner image was executed end-to-end in the shipped engine.
The native RGB8 output is the independent oracle listed above. A COM hook also
captured the actual Hue/Saturation lattice and packed image descriptor. The
descriptor records `dstType=0`, and the propagated Unify depth flag is zero:
the selected fused-gamma path is the byte worker, followed by lossless `u8*257`
expansion only for comparison with the machine TIFF's uint16 container.
All 72,775,584 recovered values equal that expanded native oracle; both raw
expanded arrays have SHA-256
`0f5babb019761f0bfdab5af2ef66d9cbfd405f032cce5045fa4f077ee5add791`.

## Authoritative fixed-job selectors

The final PCB, not the base pattern table, decides this job:

| Selector | Stage | Final value | Decision |
| --- | --- | ---: | --- |
| `+0x10` | CCD gain | 1 | selected; derived Q12 unity |
| `+0x11` | Nega FilmLUT | 1 | selected |
| `+0x12` | CCD noise | 1 | S0 selected |
| `+0x14..+0x19` | SSU corrections and grain | 0 | disabled |
| `+0x1a/+0x1b` | trim/scaling | 1/2 | native trim + S0 cubic; selector 2 is in the S0 dispatch range |
| `+0x1c/+0x1d` | rotation/rotation trim | 1/0 | selected identity |
| `+0x1e` | RGB to YCC | 2 | S1 selected |
| `+0x1f` | automatic contrast | 0 | disabled |
| `+0x21` | film sharp | 1 | selected |
| `+0x22` | YCC correction | 1 | selected |
| `+0x23` | hue/saturation | 1 | selected |
| `+0x24/+0x25` | media/color sharp | 0/0 | disabled |
| `+0x26` | YCC to RGB + gamma | 1 | byte variant 1 selected |
| `+0x27..+0x31` | post-gamma optional branches | 0 | disabled |

The captured PRM independently proves merge, aberration, distortion,
trapezoid, manual repair, frame red-eye, cross, soft, and spot-filter switches
are zero. `CmsInf.CmsSW` at `0x0ff0/0x0ff4` is `count=1,value=0`; the recovered
executor therefore skips CMS. The no-ICE target deliberately has no dust
repair.

Neutral user Hue/Saturation controls do **not** make that selected stage an
identity. The native 9×33×33×3 signed lattice contains 19,339 nonzero values
with range -485..681. The audit applies this captured lattice.

## Recovered execution order

Primary order anchors are `RD5X_DoInputProcForFilm @ 0x10032150`,
`RD5X_DoMainProcForFilm @ 0x100475f0`, and post-main `RD5X_DoCms @ 0x100925d0`.

| Order | Fixed-pair stage | Result |
| ---: | --- | --- |
| 1 | owner FULL decode and storage orientation | observed/applied |
| 2 | CreateImage offset/linear/matrix, gain alternative, IR misregistration | upstream of FULL; physical unit state unavailable |
| 3 | CCD gain + Nega FilmLUT | applied; gain is exact unity |
| 4 | CCD noise S0 | applied from final PCB |
| 5 | merge/aberration/distortion/trapezoid | disabled |
| 6 | grain S4 | disabled by final PCB |
| 7 | native trim, S0 cubic scaling, rotation identity | applied |
| 8 | RGB-to-YCC S1 | applied from final PCB |
| 9 | automatic contrast S0 | disabled by final PCB |
| 10 | FilmSharp | applied from final PCB |
| 11 | YCC correction | applied from final PCB |
| 12 | Hue/Saturation S0 | applied from final PCB + native COM lattice |
| 13 | media sharp/color sharp | disabled |
| 14 | YCC-to-RGB + fused gamma | applied; native byte branch |
| 15 | repair/red-eye/cross/soft/blank/NxN | disabled |
| 16 | Digital ICE/Kizu | disabled by target; RGB FULL has no IR plane |
| 17 | CMS | disabled by captured PRM |
| 18 | optional external monotone residual fit | diagnostic only, never vendor-labelled |

Every row is rendered. Executed stages receive adjacent numeric and visual
diffs where domains match. Disabled identities record exact zero. Blocked or
cross-domain rows record N/A rather than pretending carried-forward pixels
estimate the missing effect. YCC checkpoints preserve lanes losslessly as
`[Y,C1,C2]` in the trace container and are explicitly labelled non-display RGB.

## Exact software implementations

The codebase contains parameterized ports for more paths than this fixed job
selects:

- CreateImage saturating offset, four CCD-linear LUTs, per-position signed-Q12
  4×4 B/G/R/IR matrix, separate digital gain, and BGR/IR misregistration;
- Q12 CCD-gain composition and the RGB-payload/BGR-raster Nega FilmLUT worker;
- complete CCD-noise S0, Grain S4, and AutoContrast S0 array workers;
- trim, shared prefilter, S0 cubic and S2 bilinear scaling, orthogonal and
  arbitrary rotation, and BlankOct/BlankHex with postbreak smoothing;
- RGB-to-YCC S1 control-LUT builder and float32 worker;
- complete FilmSharp spatial graph, LUT builders, and YCC compositor;
- YCC-correction table builders and pixel worker;
- complete Hue/Saturation S0 worker across every saturation branch;
- fused-gamma natural-cubic LUT builders and byte/16-bit YCC-to-RGB variant-1
  workers, plus exact byte-to-word expansion;
- complete `pcb_makeNegaTone2`, including UO forward/inverse helpers,
  three-region `pcb_MakeGammaCurve2`, seam handling, and anchoring.

The ports retain observable integer wrap/narrowing, signed division, SSE
float32 order, nearest-even packing or `_ftol2_sse` truncation as applicable,
native BGR/YCC lane order, and recovered border rules. Native LUT harnesses
match both byte and word fused-gamma tables byte-for-byte.

## Genuine stopping boundary

No selected fixed-job software stage is waiting for an uncaptured final PCB or
COM structure. Those artifacts were recovered and the selected chain executes.
That is distinct from numeric conformance: any nonzero recovered-versus-native
residual is a local software issue until isolated or eliminated, and the
generated `report.md` records it explicitly. Only after that local check closes
are the remaining unavailable inputs materially external:

1. **Physical scanner calibration.** CreateImage expects per-unit 5340-column
   dark offsets, CCD-linear curves, color matrices, shading/misregistration and
   defect maps. `AdjustScanner.dll` generates them from calibration scans; they
   are not hidden in the installer. The owner FULL is already downstream of
   this boundary and must not be calibrated twice.
2. **Physical IR/Kizu data.** The Kizu graph is local in `ScanTool.dll`, and
   CreateImage constructs its 0x6a parameter block in software. Exact dust
   repair still needs the physical fourth IR plane and matching scanner/job
   state. The fixed owner FULL is three-channel BGR and the target explicitly
   selects a no-ICE export, so this graph is inapplicable to the audited scan.
3. **Legacy-version attribution.** The installed 2015 engine is executable and
   provides a strong native oracle, but byte-identical attribution of its observed
   residual against the saved 2017 machine TIFF would require the original
   2017 binaries or same-version internal checkpoints.
4. **Named film-stock behavior.** CorrectServ has no Kodak/Fuji stock mapping.
   Stock-specific spectral behavior requires legacy reference captures or
   physical film/scanner measurements.

Optional branches disabled for this frame are represented as proven no-ops in
the same-scan trace. Measuring their legacy visual response on another job
would require selector-specific controller state or separable machine A/B
captures; it is not inferred from the final target.

## Validation cautions

- The machine TIFF has no ICC profile and stores 8-bit codes expanded to
  uint16. Reported normalized errors are code-value diagnostics, not
  colorimetric measurements.
- The external monotone fit is trained and scored on this same frame. It is a
  residual visualization, not an independent test and not a Noritsu stage.
- `recovered_boundary.tif`, `installer_color_boundary_native.tif`, and
  `final.tif` preserve the unfitted recovered output. External-fit files are
  separate.
- The empirical crop `(y=17,x=171)` is retained only as a fallback diagnostic.
  The fixed run uses the captured Trz nominal crop `(y=16,x=168)` and native
  S0 cubic parameters. `RD5X_DoMainProcForFilm` dispatches selector values
  1..4 to S0 and 5..8 to S2 (after subtracting four); the independent
  sampling byte is 2.
- Orientation is pair-specific; `ScnCrg` is not a universal rotation rule.
