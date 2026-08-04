# RGB light scanning + Negative Universal

## The question

Someone with an RGB panel (Raleno, Kaiser, Godot, etc.) and a DSLR wants to
know: can they get Noritsu colors from their home scanning rig?

Short answer: **almost certainly yes, with one calibration scan.**

## Why it should work

Film doesn't care what light you shine through it. It absorbs certain
wavelengths (that's the dye) and transmits the rest. A scanner measures that
transmitted light. A DSLR + RGB light does the same thing — it just does it in
one Bayer-filtered shot instead of three linear-CCD passes.

The pipeline is:

1. **Linearize** — convert camera raw to linear transmittance (device-specific)
2. **Density** — D = -log10(T) (universal)
3. **Mask removal** — subtract Dmin per channel (universal)
4. **Normalize** — scale by density range (universal)
5. **Tone curve** — apply recovered Noritsu tables (universal)
6. **Render** — output positive

Only step 1 needs calibration. And the calibration is a single paired scan:
one frame of film shot on your DSLR rig, and the same frame scanned on a
Noritsu (or Frontier, or any lab minilab).

## The RGB light variable

An RGB panel like the Raleno flashes R, G, B sequentially. Each flash
illuminates the film with a different spectral peak. A white-LED panel lights
everything at once with a broad spectrum. Both work — the calibration absorbs
the spectral difference.

The practical differences:

| Factor | RGB flash panel | White LED | Scanner tri-linear |
|---|---|---|---|
| Color separation | Sequential R/G/B | Bayer CFA | Tri-linear CCD |
| Spectral coverage | Narrow R/G/B peaks | Broad but spiky | Broad + IR channel |
| Noise per channel | Good (full sensor) | 1/3 sensor per color | 1 line per color |
| Calibration needed | Yes | Yes | Yes |
| Expected NCC after cal | ~0.90–0.95 | ~0.90–0.95 | ~0.94 |

The Bayer CFA is the weakest link — each pixel only sees one color, and the
demosaicing interpolates the other two. For C-41 film this matters less than
you'd think because the dye clouds are large and the color transitions are
gradual. The calibration LUT absorbs the demosaicing artifacts into the
channel mapping.

## The real variable: your camera's raw converter

The difference between "good" and "great" DSLR scanning is whether your raw
converter gives you true linear data. Most converters apply a tone curve even
when you set "linear." The calibration needs to see the camera's raw sensor
values — no demosaicing curve, no white balance baked in. Tools like
`dcraw -4 -T` or libraw give you this.

Once you have linear data, the pipeline handles the rest.

## Protocol to test this

If anyone wants to try:

1. Scan one frame of film on your Noritsu / Frontier / lab → get a TIFF
2. Scan the same frame on your DSLR rig → output linear 16-bit TIFF
3. Run: `negative-universal calibrate --ref noritsu_output.tif --device dslr_linear.tif -o profile.npz`
4. From then on: `negative-universal render dslr_linear.tif --profile profile.npz -o noritsu_look.tif`

That's it. One frame, one calibration, and your DSLR rig inherits the Noritsu
look.

## What we need to build this

- [ ] A reference target: a strip of mounted Portra 160 (or whatever stock)
      scanned on both a Noritsu and a DSLR rig. I can provide the mounted
      negatives and pay shipping both ways.
- [ ] The unified CLI: merge noritsu-tool's pipeline with calibrate.py into
      a single `negative-universal` command.
- [ ] Device profile registry: once someone does the calibration, their
      profile can be shared so others with the same camera + light can use it.
- [ ] Validation: compare DSLR+profile output against real Noritsu output
      on multiple film stocks and exposure levels.

## Next step

I'm looking for one person in LA (or willing to ship) who has both scanning
methods — ideally a Noritsu/Frontier plus a DSLR rig or another scanner.
One frame, ten minutes, and we prove (or disprove) the thesis.
