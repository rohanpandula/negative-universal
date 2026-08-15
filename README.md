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

Native RAF/NEF ingestion and the camera-specific `capture-one` adapter are not
implemented yet. “Bayer support” currently means recording and cross-checking
the declared CFA phase while normalizing an adapter-produced mosaic PGM.

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
viewer may apply an inappropriate display transfer curve.

Install locally:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

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

Capture matched datasets:

```sh
negative-universal capture --kind dark --settings rig.json \
  --rig ./capture-one --output scan/dark
negative-universal capture --kind flat --settings rig.json \
  --rig ./capture-one --output scan/flat
negative-universal capture --kind film --settings rig.json \
  --rig ./capture-one --output scan/film
```

Normalize without overwriting the source captures:

```sh
negative-universal normalize --film scan/film --dark scan/dark \
  --flat scan/flat --output scan/normalized
```

The external `capture-one` command is the only hardware seam. It receives one
channel, normalized light power, exposure time, capture kind, and output path.
It must also implement an idempotent `capture-one off` command. The controller
calls `off` before the sequence and after every exposure, and terminates a
capture that exceeds the configurable timeout. The adapter must still turn the
light off in its own cleanup path; real hardware also needs a normally-off
output or watchdog because software cannot recover from power or process loss.
Vendor-specific adapters come after the actual hardware APIs are known.

### Scanlight and tethered cameras

The reviewed [Scanlight firmware](https://github.com/jackw01/scanlight/tree/8f749d95cc3794c87452394a47f814afd2270adc/automation)
uses USB CDC serial at 115200 baud. Its current set-color packet is
`FE 00 06 R G B 00 00 00`; all-off is nine bytes ending in six zeros. A rig
adapter must allow only the selected RGB byte to be nonzero, map normalized
power to the nearest `0..255` command code, keep white/IR/save at zero, wait for
the light to settle, and send all-off in `finally`. Set-color has no
acknowledgement, and the command value is PWM—not measured optical power.

The existing [TriRGB `scanlightctl`](https://github.com/rohanpandula/TriRGB/tree/7ec5bdd8dfc5729f272601208bfc9b9b64e69ce2/phase1/scanlightctl)
is a source-reviewed protocol building block already installed on the
development Mac; it is not yet the fail-safe `capture-one` adapter. That
adapter must add firmware/hardware probing before enable, deterministic write
drain and settle timing, internal `finally: off`, and the parent cleanup. The
Scanlight firmware can retain its last output after host failure, so unattended
use still needs a normally-off hardware enable/watchdog and independent
thermal cutoff.

Both named cameras have candidate computer-capture paths, none integrated or
hardware-tested by this repository yet. Fujifilm documents
[GFX100 II tethering](https://fujifilm-dsc.com/en/manual/gfx100ii/connections/tethere/index.html),
and Nikon lists the Zf in
[NX Tether](https://downloadcenter.nikonimglib.com/en/download/sw/276.html).
The current
[libgphoto2 support list](https://gphoto.sourceforge.io/proj/libgphoto2/support.php)
also reports capture and configuration support for both. The first adapter
should use native RAF/NEF, lock manual camera state, and use a pinned decoder
validated to preserve linear mosaic codes, crop, CFA phase, and black/white
metadata. It must never feed the core a rendered JPEG.

Manifest power and exposure fields are explicitly **commanded** settings. They
must not be described as measured light dose or actual shutter timing until a
driver returns that feedback.

Run the tests with only the Python standard library:

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

## Roadmap

- [x] Recover and validate narrow Noritsu, Nikon, and FlexColor paths.
- [x] Recover the SP-500 separation of scanner, film-channel, and frame
      correction.
- [x] Scaffold sequential PGM capture, manifests, integrity hashes, and
      dark/flat normalization.
- [x] Record and cross-check the declared mono/Bayer CFA layout through capture
      and normalization.
- [x] Map the current Scanlight firmware protocol to the `capture-one` seam.
- [ ] Implement and hardware-test the first GFX100 II or Nikon Zf
      `capture-one` adapter.
- [ ] Measure dark, flat, linearity, drift, LED spectrum/power, and R/G/B
      registration on the physical rig.
- [ ] Add native RAF/NEF-to-mosaic interchange after real camera files define
      the decoder and metadata contract.
- [ ] Collect paired reference scans covering multiple frames, stocks, and
      exposure ranges, with hold-out validation.
- [ ] Add versioned stock profiles and measured clear-roll base removal.
- [ ] Train an independent, auditable per-frame estimator from owned data.
- [ ] Compare Noritsu, Frontier, Nikon, FlexColor, and custom render stages.

One paired frame can bootstrap an experiment; it cannot establish a universal
profile. Narrow-band measurements can lose spectral information, and a LUT
cannot reconstruct information the camera/light combination never measured.

## Related repositories

- [noritsu-tool](https://github.com/rohanpandula/noritsu-tool)
- [cool-colors](https://github.com/rohanpandula/cool-colors)
- [flexcolor-tool](https://github.com/rohanpandula/flexcolor-tool)
- [frontier-sp500-research](https://git.roflix.club/rohan/frontier-sp500-research)
- [digital-fauxice](https://github.com/rohanpandula/digital-fauxice)
- [ScanStudio](https://github.com/rohanpandula/ScanStudio)
- [TriRGB](https://github.com/rohanpandula/TriRGB)
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
