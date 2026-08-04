# Negative Universal

**One density-space inversion to rule every scanner. Calibrated per device.**

This repo connects three reverse-engineering projects — Nikon Coolscan, Noritsu
LS-600, and Hasselblad FlexColor — to answer one question:

> Can a single C-41 inversion pipeline, calibrated per device, reproduce the
> color of any scanner or DSLR scanning rig?

The answer so far: **yes, and at ~94–99% accuracy.**

## Why this exists

Every film scanner and DSLR scanning rig has a different spectral response —
different CCD/CMOS color filters, different light source, different optics. The
same negative looks different on every device. For decades the solution has been
to treat each scanner as a sealed box and build per-device color profiles.

But the math of C-41 inversion doesn't change. A negative is complementary dyes
under an orange mask, period. The physics is:
1. Measure transmittance per channel
2. Convert to density: D = -log10(T)
3. Remove the orange mask (Dmin)
4. Normalize by density range
5. Apply a tone curve

Steps 2–5 are scanner-agnostic. Only the mapping from your device's raw RGB
to linear transmittance (step 1) needs calibration. That's what this repo
provides: a single calibration framework that takes a paired scan (same frame
on your device + a reference Noritsu/Frontier lab scan) and fits a per-channel
LUT that makes your device render like the reference.

## What's already built

| Project | Accuracy | What it does |
|---|---|---|
| **noritsu-tool** | ~94% NCC on LS-600 raw | Density inversion + Noritsu tone curves from actual DLL tables |
| **cool-colors** | 99.1% of Nikon Coolscan color | Recovered Nikon Scan's full color pipeline (Nikon-specific) |
| **flexcolor-tool** | engine math recovered | Hasselblad FlexColor LUTs, gradation curves, neg-gamma |
| **digital-fauxice** | byte-for-byte exact | Nikon's Digital ICE reimplemented from scratch |
| **ScanStudio** | full app | Modern replacement for Nikon Scan that runs today |

## The big idea: calibrate once, render anything

```
Your device raw (DSLR / Coolscan / Plustek / Epson / whatever)
        │
        ▼
   Linearize (device-specific calibration — the one-time fit)
        │
        ▼
   Density inversion (universal — same math every time)
        │
        ▼
   Tone curve (Noritsu / Frontier / Nikon / FlexColor / custom)
        │
        ▼
   Positive output
```

The device-specific part (linearization + channel mapping) is a one-time
calibration. After that, the pipeline is universal — you pick which "look"
you want (Noritsu, Frontier, Nikon Scan, etc.) and the tone curve comes from
the actual recovered engine data.

## The RGB light question

DSLR scanning traditionally uses RGB LED panels (like the Raleno or Kaiser)
that flash red, green, and blue in sequence — capturing three monochrome frames
that combine into a color image. The spectral output of these lights is
different from a scanner's fluorescent/LED lamp, which means the R/G/B
channel ratios hitting the film are different.

The calibration framework handles this automatically: if you capture a
reference frame with both your RGB-light rig and a Noritsu scanner, the
calibration LUT absorbs the spectral difference. You get the Noritsu look from
your RGB light.

But there's a subtler question: **can a single white-LED or RGB-simultaneous
capture (one shot through a Bayer filter) produce the same color fidelity as a
scanner's tri-linear CCD?** The answer depends on:
- Whether the Bayer CFA's color separation is sufficient for C-41 dye clouds
- Whether the light source's spectrum covers the film dyes evenly
- Whether the calibration LUT can map the narrower color space

The early evidence from the Coolscan 5000 (which uses an RGB LED strip + tri-linear CCD) suggests yes. The LS-600 uses a different lamp and CCD, and we got NCC 0.937 after calibration. DSLR + RGB panel should be achievable with similar or better accuracy.

## Research roadmap

- [x] Recover Noritsu LS-600 pipeline (tables, tone curves, density inversion)
- [x] Validate against real LS-600 raw + TIFF pairs (NCC 0.937, 8 frames Portra 160)
- [x] Recover Nikon Coolscan color pipeline (cool-colors, 99.1%)
- [x] Recover FlexColor engine math (profiles, LUTs, gradation)
- [x] Build calibrate.py — fits per-channel LUT from any paired scan
- [ ] Cross-scanner calibration: Coolscan → Noritsu look (needs paired frame)
- [ ] DSLR + RGB light calibration rig protocol
- [ ] Unified CLI: `negative-universal calibrate --ref ref.tif --device device.tif -o profile.npz`
- [ ] Per-film-stock calibration profiles (Portra 400, Gold 200, Ektar 100, ECN-2)
- [ ] Web-based comparison gallery (upload a neg, see it rendered in Noritsu/Frontier/Nikon styles)

## Repos

- [github.com/rohanpandula/noritsu-tool](https://github.com/rohanpandula/noritsu-tool) — Noritsu LS-600 pipeline
- [github.com/rohanpandula/cool-colors](https://github.com/rohanpandula/cool-colors) — Nikon Coolscan color science
- [github.com/rohanpandula/flexcolor-tool](https://github.com/rohanpandula/flexcolor-tool) — Hasselblad FlexColor engine
- [github.com/rohanpandula/digital-fauxice](https://github.com/rohanpandula/digital-fauxice) — Digital ICE reimplementation
- [github.com/rohanpandula/ScanStudio](https://github.com/rohanpandula/ScanStudio) — Modern Coolscan software

## License

MIT
