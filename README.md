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
sequential monochrome R/G/B captures
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

The primary experiment is deliberately not a Bayer-camera scan:

- one computer-controlled monochrome camera;
- stationary film;
- narrow-band red, green, and blue lights exposed one at a time;
- computer control of camera exposure and each light's power.

Each output channel is a full monochrome sensor frame measured under one light.
Light power and shutter time are acquisition controls used for signal-to-noise
and headroom. They are not unrecorded white-balance controls.

See [RGB light scope](docs/RGB-LIGHT-SCOPE.md) for the calibration protocol,
the Frontier comparison, and the boundary expected from a hardware driver.

## Runnable scaffold

The initial Python scaffold captures and preserves `R -> G -> B` triplets
through one external rig command, then applies matched dark/flat normalization:

```text
T = clamp((film - dark) / (flat - dark), 0, 1)
```

It intentionally starts with binary PGM (`P5`) input and output. PGM preserves
8- or 16-bit monochrome samples without adding a RAW/TIFF dependency before the
real camera contract is known. Samples are linear numeric codes, not
display-ready grayscale; a generic image viewer may apply an inappropriate
display transfer curve.

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
    "gain": "replace-with-fixed-gain",
    "light_source": "replace-with-light-id",
    "lens_carrier": "replace-with-optical-configuration"
  },
  "R": {"power": 0.70, "exposure_us": 10000},
  "G": {"power": 0.55, "exposure_us": 12000},
  "B": {"power": 0.80, "exposure_us": 15000}
}
```

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
- [ ] Implement the first real camera/light `capture-one` adapter.
- [ ] Measure dark, flat, linearity, drift, LED spectrum/power, and R/G/B
      registration on the physical rig.
- [ ] Add RAW/TIFF interchange after real camera files define the requirement.
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
