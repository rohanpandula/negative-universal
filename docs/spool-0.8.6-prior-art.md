# Prior art: Spool 0.8.6's inversion

Spool (`com.spool.app`, 0.8.6, Tauri/Rust) is a negative-inversion app with a
look engine and no calibration model. It was taken apart to see whether its
inversion contains anything worth adopting here. Summary: the renderer has two
ideas worth keeping, the model has none of the separation this project exists
for, and its domain is the wrong one.

Constants below are from the arm64 slice, offsets are `VA - 0x100000000`.
Everything lives in `spool_lib::film` and `spool_lib::invert`; the per-pixel
work is the closure at `0x10079f978`.

## What it does

Domain is **linear light**, sRGB primaries, Rec.601 luma (0.299 / 0.587 /
0.114 as literal f32s). Per pixel, with `B` = base luma and `S` = slope:

```
L    = .299r + .587g + .114b
T    = max(1 - B, 1e-4)
x    = (L - B) / T
y    = T*x / (x + 1)          # Reinhard shoulder anchored at the base
L'   = B + y
gain = L' / L
k    = S * clamp(x, 0, 1)
out  = rgb*gain + (L' - rgb*gain) * k
```

then, after the curve, if `max(out) > 1`:
`out = L' + (out - L') * (1 - L') / (max(out) - L')`.

Base is a per-channel linear value sampled from the border with a median
(`film::sample_base_at`), scored by chroma flatness, and collapsed to a scalar
luma. Per-channel slope planes (least squares, min 8 samples,
`film::fit_slope`) correct gate falloff. A per-channel 4096-entry print LUT is
indexed at `clamp(v * 0.25, 0, 1)` and a Catmull-Rom pass follows.

`invert_density` is a slider, not a domain: it is one `powf` exponent. The
renderer never takes a `log10`; `_log10f` resolves only in `scan_analysis`,
where it scores strip distance from a base profile while finding frames.

## Where it agrees with this project

* Base is measured from clear film, per channel, with a robust statistic.
* Falloff is fitted and removed rather than ignored.
* Hue is preserved: chroma is scaled about a remapped neutral, never inverted
  channel-independently.
* Stock and frame are separate rows (`film_presets` plus per-photo overrides).

## Where it breaks the contract

1. **Wrong domain.** `-log10` makes exposure additive and contrast a scale. A
   `x/(x+1)` shoulder is baked in *before* any control, so exposure and
   contrast interact and a base error is a colour cast instead of a density
   offset. The print LUT's input range of 0..4.0 is already density-shaped,
   which is the seam.
2. **Look fused into calibration.** Mask removal, neutral anchor, stock curve
   and the user curve are one chained render. Changing the look changes the
   numbers.
3. **No device model.** `camera_backs` is `(lens, back, frame size)` plus
   learned defaults, so the same frame from two backs needs different training.
4. **Coarse correction.** The "zone balance" table is 3 zones × C/M/Y offsets
   added on top of a global nudge; it cannot express a density offset without
   also moving contrast.

## What was adopted

* `probe.py` — border patch tiling, chroma-flatness cost, brightest-decile
  targeting, and best-patch-per-side selection so the falloff plane never
  degenerates. Spool ranks probes but fits its slope over the whole frame; we
  fit on the chosen bare-film patches instead, which keeps the fit on film.
* The hue-safe shadow anchor as an *optional* `--neutral-pull`: a chroma pull
  toward the remapped luma, weighted by density above the base.
* The 0..4 decade print domain, now as `DENSITY_CEILING = 4.0` on a LUT that
  actually takes `-log10` input.

## What it does not solve

Trial/licence is offline ed25519 over `spool-licence-v1\n{email}\n{device}\n
{product}\n{issued}\n`; the key string is unsigned. `license::recheck` deletes
the token when the server's device list does not name the machine — worth
knowing if the app is ever used as an oracle for A/B renders, because a forged
token survives only if that delete is stubbed.

## Reproducing

`spool-re/extract/inversion_reference.py` in the `trichroma` repo is a runnable
transcription of the core. This repo's `src/negative_universal/invert.py`
implements the density version of the same renderer, and
`src/negative_universal/probe.py` the base probe.
