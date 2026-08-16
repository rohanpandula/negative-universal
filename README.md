# Negative Universal

**A calibrated density-space pipeline for film scanners and camera-scanning
rigs.**

This repository connects four reverse-engineering efforts—Noritsu LS-600,
Nikon Coolscan, Hasselblad FlexColor, and Fujifilm Frontier SP-500—to answer a
practical question:

> How much of C-41 inversion can be shared while the scanner, film stock, and
> individual frame remain independently calibrated?

The current answer is encouraging but narrower than “one LUT fixes every
scanner.” Density-space inversion can be shared. Acquisition calibration,
stock response, and scene correction must remain explicit and testable.

## Working model

A C-41 negative is measured as transmittance and converted to optical density:

```text
density = -log10(transmittance)
```

The useful separation is:

| Layer | Corrects | Lifetime |
| --- | --- | --- |
| Device calibration | Camera offset, linearity, defects, light, optics, and carrier | Until the physical configuration drifts or changes |
| Film profile | Repeatable stock and process behavior | Selected per stock or roll |
| Frame correction | Scene cast and exposure variation | Recomputed per frame |

The complete target path is:

```text
sequential R/G/B illumination captures (mono plane or raw CFA mosaic)
  -> dark, defect, linearity, flat-field, and registration calibration
  -> calibrated scanner-relative transmittance
  -> measured clear-film base + film-stock profile
  -> density inversion
  -> optional per-frame density and gray-balance correction
  -> output tone and color rendering
```

Changing the output look must never change the preserved calibrated source
triplet.

## Target hardware

The calibration core accepts externally prepared PGM data from two sensor
classes behind the same capture boundary:

- a computer-controlled monochrome camera, where every R/G/B illumination
  exposure is a full monochrome sensor frame;
- a Bayer camera such as a Fujifilm GFX100 II or Nikon Zf, where every exposure
  is preserved as an undemosaiced raw CFA mosaic.

Native GFX100 II RAF and Nikon Zf NEF ingestion is implemented through rawpy's
undemosaiced `raw_image_visible` plane. The decoder copies rawpy's returned
integer mosaic without scaling or black subtraction and records the visible
crop and CFA phase; it never invokes white balance, demosaicing, rotation or
LibRaw post-processing. A first-party gPhoto/ScanLight `capture-one` adapter is
included, but has not yet been exercised on connected hardware.

The film remains stationary while narrow-band red, green, and blue lights are
exposed one at a time. `R.pgm`, `G.pgm`, and `B.pgm` name the illumination
exposure, not Bayer photosite classes. Light power and shutter time are
acquisition controls used for signal-to-noise and headroom; they are not
unrecorded white-balance controls.

Relative to an otherwise comparable monochrome sensor, a Bayer CFA reduces
spatial sampling and photon efficiency. Its red, two green, and blue site
classes must be calibrated independently, and weak off-band sites may not be
useful. Camera JPEG, white balance, demosaicing, auto exposure, rotation, and
lens correction must stay out of the measurement path.

See [RGB light scope](docs/RGB-LIGHT-SCOPE.md) for the calibration protocol,
the Frontier comparison, and the boundary expected from a hardware driver.

## Runnable scaffold

The initial Python scaffold captures and preserves `R -> G -> B` triplets
through one external rig command, then applies matched dark/flat normalization:

```text
T = clamp((film - dark) / (flat - dark), 0, 1)
```

It intentionally starts with binary PGM (`P5`) input and output. PGM preserves
one plane of 8- or 16-bit samples, including an undemosaiced Bayer mosaic,
without adding a native RAW/TIFF dependency to the calibration core. Samples
must be linear numeric codes, not display-ready grayscale; a generic image
viewer may apply an inappropriate display transfer curve. Native decoding sets
the PGM maximum to rawpy's global camera white level and retains the returned
black pedestal. When a RAW receipt supplies four equal per-site saturation
levels at or below that PGM maximum, clear-base calibration uses their scalar
value as the full-well ceiling; otherwise it uses the PGM maximum. The scalar
must match across the clear-base, dark, and flat captures. Unequal per-site
levels are not modeled and are rejected.

Install locally:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Install the optional RAW decoder and hardware adapter dependencies with:

```sh
python -m pip install -e '.[hardware]'
```

The adapter also needs the external `gphoto2` command; the Python extra does
not install that operating-system tool.

Create `rig.json`:

```json
{
  "_meta": {
    "configuration_id": "camera-light-lens-carrier-v1",
    "camera": "replace-with-camera-id",
    "sensor_layout": "mono",
    "gain": "replace-with-fixed-gain",
    "light_source": "replace-with-light-id",
    "lens_carrier": "replace-with-optical-configuration"
  },
  "R": {"power": 0.70, "exposure_us": 10000},
  "G": {"power": 0.55, "exposure_us": 12000},
  "B": {"power": 0.80, "exposure_us": 15000}
}
```

Use `mono`, `rggb`, `bggr`, `grbg`, or `gbrg` for `sensor_layout`. A Bayer
pattern is relative to the top-left sample in the stored PGM, after any RAW
crop or orientation operation. Film, dark, and flat captures must keep the
same dimensions, crop, orientation, gain, code range, and CFA phase. The
capture and normalized manifests record and cross-check this declared value.

Decode an existing camera RAW without demosaicing. The RAW, PGM and receipt
must be siblings with the same stem:

```sh
negative-universal decode-raw --input scan/R.RAF --output scan/R.pgm \
  --receipt scan/R.receipt.json
```

The receipt records the rawpy and bundled LibRaw versions, visible crop,
unapplied orientation, CFA phase, black levels and camera white level. Capture
manifest schema 2 hashes the original RAF/NEF and receipt, and rejects matched
film/dark/flat sets whose decoder, recorded camera state, connection endpoints,
or ScanLight timing differ. Camera model and serial are included when gPhoto
exposes them; the declared configuration ID remains authoritative when a device
does not expose a unique serial.

Capture matched datasets:

```sh
negative-universal capture --kind dark --settings rig.json \
  --rig .venv/bin/capture-one --output scan/dark
negative-universal capture --kind flat --settings rig.json \
  --rig .venv/bin/capture-one --output scan/flat
negative-universal capture --kind film --settings rig.json \
  --rig .venv/bin/capture-one --output scan/film
```

Normalize without overwriting the source captures:

```sh
negative-universal normalize --film scan/film --dark scan/dark \
  --flat scan/flat --output scan/normalized
```

For a clear processed-film ROI, recommend new shutter times while holding LED
power fixed. Both fractions are explicit experimental choices, not universal
defaults; any change requires recapturing the matched dark and flat:

```sh
negative-universal calibrate-clear-base \
  --clear-base calibration/clear-base --dark calibration/dark \
  --flat calibration/flat --roi 100,200,400,300 \
  --target-base-fraction 0.30 --flat-ceiling-fraction 0.80 \
  --output calibration/exposure-recommendation
```

The analysis meters the matching CFA sites for each illumination and caps the
recommendation with the highest per-site empty-gate-flat p99. This constrains
that recorded percentile, not every defect or outlier. Fractions use the
matched receipt saturation scalar rather than blindly using the PGM container
maximum; for example, four Nikon Zf values of `15311` are used even when the PGM
maximum is `16383`. The manifest separates commanded and applied exposure, the
ideal model multiplier, and the rounded command recommendation. It performs no
white balance or base removal.

The external `capture-one` command is the only hardware seam. It receives one
channel, normalized light power, exposure time, capture kind, and output path.
It must also implement an idempotent `capture-one off` command. The controller
calls `off` before the sequence and after every exposure, and terminates a
capture that exceeds the configurable timeout. The adapter must still turn the
light off in its own cleanup path; real hardware also needs a normally-off
output or watchdog because software cannot recover from power or process loss.

### Scanlight and tethered cameras

The reviewed [Scanlight firmware](https://github.com/jackw01/scanlight/tree/8f749d95cc3794c87452394a47f814afd2270adc/automation)
uses USB CDC serial at 115200 baud. Its current set-color packet is
`FE 00 06 R G B 00 00 00`; all-off is nine bytes ending in six zeros. A rig
adapter must allow only the selected RGB byte to be nonzero, map normalized
power to the nearest `0..255` command code, keep white/IR/save at zero, wait for
the light to settle, and send all-off in `finally`. Set-color has no
acknowledgement, and the command value is PWM—not measured optical power.

The included `capture-one` adapter uses an explicit serial port, sends OFF
before opening the shutter, probes firmware and hardware before enabling a
channel, verifies complete writes, and sends OFF in `finally`. Configure it
through inherited environment variables; no serial or camera auto-selection
is used while a light can be energized:

```sh
export NEGATIVE_UNIVERSAL_SCANLIGHT_PORT=/dev/cu.usbmodem-REPLACE
export NEGATIVE_UNIVERSAL_SCANLIGHT_HW_ID=3
export NEGATIVE_UNIVERSAL_GPHOTO_PORT=usb:001,002
export NEGATIVE_UNIVERSAL_SETTLE_MS=REPLACE_WITH_MEASURED_DELAY
capture-one probe
```

`NEGATIVE_UNIVERSAL_GPHOTO2` can select a non-default gPhoto executable and
`NEGATIVE_UNIVERSAL_GPHOTO_CAMERA` can select a named driver. Before enabling
the light, the adapter requires manual exposure mode, fixed numeric ISO and a
RAW-only image format, and records those values plus the camera model, available
serial, selected driver and settle delay. It downloads exactly one RAF or NEF
through a private staging directory, verifies or sets the requested shutter,
retains the local native file, and publishes a complete provenance receipt. It
also passes gPhoto `--keep`, preventing gPhoto from explicitly deleting the
camera-side object. When gPhoto exposes a capture-target control, the adapter
selects and re-verifies `Memory card` before light-on. Without that control,
camera-side recovery is best-effort; clear persistent card copies only after
verifying the published dataset. The
Scanlight firmware can retain its last output after host failure, so unattended
use still needs a normally-off hardware enable/watchdog and independent
thermal cutoff.

Neither named camera has been physically exercised by this repository yet.
Fujifilm documents
[GFX100 II tethering](https://fujifilm-dsc.com/en/manual/gfx100ii/connections/tethere/index.html),
and Nikon lists the Zf in
[NX Tether](https://downloadcenter.nikonimglib.com/en/download/sw/276.html).
The current
[libgphoto2 support list](https://gphoto.sourceforge.io/proj/libgphoto2/support.php)
also reports capture and configuration support for both. A published GFX100 II
snapshot exposes writable shutter choices. A published Zf snapshot exposes a
read-only shutter, so the adapter verifies the physically selected value and
fails before turning on the light if it differs. It never accepts a rendered
JPEG.

Manifests distinguish commanded power/exposure from the applied shutter choice
and emitted Scanlight PWM code. None of those values is measured optical dose.

Run the complete tests after installing the hardware extra:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## What the research projects contribute

| Project | Current result | Contribution |
| --- | --- | --- |
| **negative-universal** | capture/calibration scaffold | Device-independent experiment boundary |
| **noritsu-tool** | about 0.94 NCC on validated LS-600 pairs | Density inversion, tone curves, and stock experiments |
| **cool-colors** | 99.1% agreement in its validated Nikon path | Recovered Coolscan color pipeline |
| **flexcolor-tool** | engine math recovered | LUTs, gradation curves, and negative gamma |
| **frontier-sp500-research** | legacy behavior statically characterized | Device/stock/frame separation and learned gray balance |
| **digital-fauxice** | byte-for-byte validated path | Independent Nikon Digital ICE work |
| **ScanStudio** | working application | Modern Nikon scanning application |

These percentages describe narrow, project-specific validation sets. They are
not evidence that a new light/camera pair will reach the same result.

## Frontier lessons

The legacy SP-500 confirms three separate responsibilities:

1. **Scanner correction** runs with no film. It calibrates LED currents, dark
   response, defective pixels, bright/flat response, carrier, and optics.
2. **Film Channel** is a named stock preset selected manually or by DX code. It
   stores base C/M/Y balance and under/over/super-over slope adjustments.
3. **Color Correction** analyzes each frame. The negative-film path uses a
   learned gray-statistics model and produces red/blue offsets with green as
   reference; it is not Kelvin and does not retune the LEDs.

The independent evidence ledger is in
[frontier-sp500-research](https://git.roflix.club/rohan/frontier-sp500-research).
No Fujifilm binaries, tables, or decompiler output are copied here.

The MS11 installer also ships a first-party simulator harness: a scanner
simulator, dummy order/GUI clients, a simulated 1394 transport, and a
frame-detection telemetry override. It can run the genuine color pipeline with
no scanner attached, and its input contract is a small three-channel raw plus
INI files, so a VM golden-master comparison against this project's pipeline is
practical. See
[MS11 simulator harness](docs/MS11-SIMULATOR-HARNESS.md).

## Roadmap

- [x] Recover and validate narrow Noritsu, Nikon, and FlexColor paths.
- [x] Recover the SP-500 separation of scanner, film-channel, and frame
      correction.
- [x] Scaffold sequential PGM capture, manifests, integrity hashes, and
      dark/flat normalization.
- [x] Record and cross-check the declared mono/Bayer CFA layout through capture
      and normalization.
- [x] Map the current Scanlight firmware protocol to the `capture-one` seam.
- [x] Implement a fail-closed gPhoto/Scanlight `capture-one` adapter and test its
      protocol, shutter and cleanup behavior with simulated hardware.
- [x] Add fixed-scale native RAF/NEF mosaic extraction, receipts and manifest
      provenance; validate it on owned GFX100 II and Nikon Zf files.
- [x] Add a clear-base exposure/headroom recommender that preserves LED power
      and requires newly matched dark and flat captures.
- [ ] Hardware-test the adapter on the connected camera and Scanlight, including
      unplug, timeout and normally-off interlock behavior.
- [ ] Measure dark, flat, linearity, drift, LED spectrum/power, and R/G/B
      registration on the physical rig.
- [ ] Collect paired reference scans covering multiple frames, stocks, and
      exposure ranges, with hold-out validation.
- [ ] Run MS11 in a VM with its own scanner simulator; record the no-hardware
      module layout and fine-scan raw header; capture golden-master outputs
      for owned inputs.
- [ ] Calibrate the rig-to-SP-500 measurement delta from paired scans of the
      same frames, and test stock-independence with held-out stocks.
- [ ] Add versioned stock profiles and measured clear-roll base removal.
- [ ] Train an independent, auditable per-frame estimator from owned data.
- [ ] Compare Noritsu, Frontier, Nikon, FlexColor, and custom render stages.

One paired frame can bootstrap an experiment; it cannot establish a universal
profile. Narrow-band measurements can lose spectral information, and a LUT
cannot reconstruct information the camera/light combination never measured.

## Related repositories

- [noritsu-tool](noritsu-tool/) — vendored as a git subtree and now the
  canonical copy: includes the exact-replay pipeline (PCB/PRM decode,
  vendor tables, 38-stage audit) that reproduced the native 2015 RD5X
  engine byte-for-byte on the pinned fixed job. The GitHub mirror is
  the earlier historical state.
- [cool-colors](https://github.com/rohanpandula/cool-colors)
- [flexcolor-tool](https://github.com/rohanpandula/flexcolor-tool)
- [frontier-sp500-research](https://git.roflix.club/rohan/frontier-sp500-research)
- [digital-fauxice](https://github.com/rohanpandula/digital-fauxice)
- [ScanStudio](https://github.com/rohanpandula/ScanStudio)
- [TriRGB](https://github.com/rohanpandula/TriRGB)
- [NegPy](https://github.com/marcinz606/NegPy) (GPL-3.0 reference only; no code,
  profiles, matrices, or ICC assets are copied into this MIT project)
- [Scanlight](https://github.com/jackw01/scanlight)

## Latest validated stock findings

Tests with ICE disabled indicate that the largest Noritsu residual is not dust
removal. Portra 400 without ICE measured about the same NCC as Portra 160 with
ICE, while Superia 800 fit less closely. That supports explicit per-stock
profiles and validation across stock and exposure range rather than treating
one paired frame as a permanent device calibration.

The related Noritsu work has experimental per-stock profiles for Portra 160,
Portra 400, and Superia 800. Those results are inputs to this research, not
profiles shipped by this repository.

## License

MIT
