# Sequential RGB light scanning with mono or Bayer cameras

## Target

The target rig holds film stationary and records three raw sensor frames:

1. red light on, capture `R`;
2. red off, green on, capture `G`;
3. green off, blue on, capture `B`;
4. blue off.

With a monochrome sensor, every file is a full monochrome plane. With a Bayer
sensor, every file is one undemosaiced CFA mosaic; `R`, `G`, and `B` identify
the illumination exposure, not the CFA site class. The computer controls
camera exposure and normalized power for each light. A hardware adapter must
wait for the selected light to settle, capture the frame, and guarantee that
every light is off before it returns—even after a camera or file error. The
parent controller also calls an idempotent `off` action before the sequence and
after every exposure, with a finite process timeout. Physical outputs must
default off or use a watchdog because neither cleanup path can cover computer
or power loss.

These three measurements are scanner-relative channels, not standard RGB. Each
measurement is approximately

```text
M_c,k = integral(light_c(wavelength)
                 * film_transmittance(wavelength)
                 * sensor_QE(wavelength)
                 * CFA_k(wavelength))
```

A monochrome sensor has `CFA_k = 1`. A Bayer sensor has distinct red, two green,
and blue site classes whose narrow-band response and signal-to-noise must be
calibrated separately. A profile must characterize the complete light, film,
optics, and sensor response. Narrow peaks can improve dye separation, but they
can also discard spectral information. Calibration can map measured
information; it cannot recover information that was never captured.

The calibration core accepts externally decoded sequential narrow-band Bayer
mosaics and preserves their declared phase. Native camera capture and RAF/NEF
decoding are not implemented here yet. Simultaneous white-light, one-shot Bayer
capture is a separate experiment and must not be mixed into the acceptance
criteria for the sequential rig.

## Three independent correction layers

| Layer | Input | Output | Rebuild when |
| --- | --- | --- | --- |
| Device calibration | No-film dark and flat captures | Stable scanner-relative R/G/B | Camera, light, optics, carrier, gain, exposure mode, or temperature behavior changes |
| Film profile | Calibrated film measurements | Stock/roll-relative density | Stock, process, or intended reference rendering changes |
| Frame correction | One calibrated frame | Scene density and gray-balance adjustment | Every frame |

Keeping these layers separate is the most reusable result from the Frontier
SP-500 research. It also prevents an attractive but fragile shortcut: changing
LED power until one image looks neutral and accidentally treating that physical
setting as a universal profile.

## 1. Device calibration

### Required captures

Build calibration data before trying to match a lab render:

- **Dark:** all lights off, matched exposure and camera gain. Repeat across the
  exposure times and temperature range the scanner will use.
- **Linearity:** sweep exposure time and light power independently. Identify the
  region where doubling measured dose doubles the dark-subtracted signal.
- **Flat:** no film, or a documented calibration diffuser, captured separately
  under R, G, and B. Rebuild for each lens, magnification, carrier, and mask that
  measurably changes shading.
- **CFA response:** for Bayer cameras, evaluate every CFA class under every LED.
  A positive `flat-dark` denominator can still be too small or noisy to use.
- **Defects:** identify pixels that remain invalid after dark and flat
  correction. Defect interpolation is intentionally not part of the first
  scaffold.
- **Registration:** measure translation, rotation, scale, and distortion among
  the three sequential frames.
- **Drift:** repeat flats through warm-up and ordinary operating time. Record
  camera and light temperature when available.

The SP-500 similarly separates source-current adjustment, CCD dark correction,
defective-pixel correction, pixel-sensitivity correction, and light/optical
bright correction. Its bright data are keyed to carrier and mask configuration,
and calibration waits for thermal stabilization.

### Initial normalization

For channel `c` and pixel `x`, use a matched empty-gate flat:

```text
T_c(x) = (film_c(x) - dark_c(x))
       / (flat_c(x) - dark_c(x))
```

The current scaffold rejects `flat <= dark`, clamps the result to `[0, 1]`, and
writes 16-bit PGM transmittance. It reports low and high clipping counts in the
output manifest. It never overwrites the film, dark, or flat source triplets.
PGM is used only as a one-plane numeric container: its samples must be linear
sensor codes and are treated as linear transmittance after normalization, not
as gamma-corrected, view-ready grayscale. The scaffold cannot verify the input
transfer function. For Bayer input the output remains a Bayer mosaic with the
same dimensions and declared CFA phase; no demosaicing occurs.

The flat is not the clear edge of the negative. An empty-gate flat corrects the
device. A clear edge measures the base of the actual processed roll and belongs
to the film/roll stage.

### Bayer/CFA capture contract

Set `_meta.sensor_layout` to `rggb`, `bggr`, `grbg`, or `gbrg`; omitted values
default to `mono` for old manifests. The pattern is relative to the top-left
sample in the stored PGM. Film, dark, and flat must use the same active crop,
orientation, dimensions, maximum code, ISO/gain, exposure regime, and CFA
phase. An odd raw-crop offset changes the phase.

The camera adapter must decode native RAW to a linear mosaic without camera
white balance, demosaicing, gamma/tone, auto-brightening, auto black-level
changes, rotation, lens correction, denoising, sharpening, pixel shift, or HDR.
Keep the native RAF/NEF alongside derived calibration data once the native-file
manifest contract is implemented. Matching CFA sites can be resampled later;
ordinary broadband demosaicing is not measurement-preserving under one
narrow-band LED.

### Power and exposure rules

- Prefer fixed, calibrated power/exposure pairs per channel for normal scans.
- Use exposure and power to avoid clipping and lift dense negatives above the
  noise floor, not to make a scene look neutral.
- If one exposure cannot cover the density range, define a fixed bracket
  schedule and validate its merge independently.
- Verify equivalent normalized results with multiple power/exposure pairs. Do
  not assume reciprocity.
- Check whether LED current or PWM changes peak wavelength, spectral width, or
  causes shutter banding.
- Preserve commanded settings, configuration identity, and file hashes with
  every capture. Record applied or measured values separately when the hardware
  can report them.

A monitor photodiode would make source dose and warm-up drift directly
observable. It is useful but not required for the first experiment if flats
bracket each scan session.

### Calibration keys

At minimum, calibration records must identify:

- camera, sensor mode, and `sensor_layout` relative to the stored raster;
- raw crop, orientation, black/white code metadata, and native RAW format;
- one immutable configuration ID shared by matched film/dark/flat captures;
- gain and exposure time;
- light source and R/G/B power setting;
- lens, aperture, focus, and magnification;
- carrier, mask, and film lane;
- temperature band and calibration time;
- hashes of every dark and flat source frame.

## 2. Film-channel profile

A film channel is a stock/polarity recipe, not a hardware calibration and not
per-frame white balance.

### Legacy Frontier behavior

The SP-500 stores named channels for negative or reversal film. A channel can be
selected manually or associated with DX code and film name. For color film, the
record contains signed C/M/Y triplets for:

- `BL-N`: base balance;
- `SL-U`: under-range slope;
- `SL-O`: over-range slope;
- `SL-Oo`: super-over-range slope.

A separate UI **Balance** triplet is enabled for sepia, B/W, and user-defined
monochrome tone modes rather than the normal color-film controls. DX selects a
record; DX is not itself a color transform.

In the rendered legacy profile, automatic `GRAY_BALANCE` is evaluated before
the later `BLSL` and `DCMY` stages. “Film prior,” “frame estimate,” and “profile
execution order” are therefore related but not identical concepts.

### Independent equivalent

Store a versioned digital profile rather than raw LED or shutter values. A
useful first profile contains:

- stock/polarity identity and optional DX aliases;
- the measured clear-base density of the calibration roll;
- per-channel density curves;
- a small cross-channel matrix only when held-out data proves it helps;
- reference target, processing lab, fitting data, and validation metrics.

Measure the clear edge of every roll even when a stock profile exists. Batch,
age, storage, and processing alter the base. Start with one generic negative
profile and add stock profiles only when repeated hold-out tests reveal a stable
stock-specific residual.

## 3. Per-frame correction

### Legacy Frontier behavior

The SP-500 UI calls this **Color Correction**, separate from **Density
Correction**. It is not a Kelvin estimate.

The negative-film path:

1. conditions scanner RGB through three 4096-entry LUTs;
2. calculates global, gray-subset, range, tail, and correlation features;
3. standardizes 25 selected features and projects them to five coordinates;
4. chooses the nearest of six learned prototypes;
5. applies two class-specific regressions to predict red and blue shifts;
6. keeps green at zero reference and attenuates implausible corrections.

Candidate gray pixels use the chroma-distance measure

```text
D = (R + G + B)^2 - 3(RG + RB + GB)
  = 1/2 ((R - G)^2 + (G - B)^2 + (B - R)^2)
```

The result is a scene-dependent `(delta_R, 0, delta_B)` profile offset. The
operator level is `0-100`, but the exact transfer from that level to output
offsets remains unproven. It must not be assumed to be `level / 100`.

Full evidence, hashes, confidence labels, and open questions are maintained in
[frontier-sp500-research](https://git.roflix.club/rohan/frontier-sp500-research).

### Independent equivalent

Do not copy Fujifilm's model or parameter payload. Begin with no automatic
correction and establish the device and stock stages first. Then compare:

1. a transparent gray-candidate estimator;
2. a robust global/gray-subset estimator;
3. an independently trained classifier or regressor using owned captures and
   reference renders.

Always store the estimated correction separately from the calibrated source
triplet so it can be disabled, changed, and audited.

## Runnable capture boundary

The initial package delegates one physical exposure to an external executable:

```text
capture-one capture
  --kind film|flat|dark
  --channel R|G|B
  --power 0..1
  --exposure-us INTEGER
  --output /absolute/path/channel.pgm
```

It also requires an idempotent lights-off action:

```text
capture-one off
```

The command is invoked without a shell in fixed `R -> G -> B` order. For dark
captures, power is forced to zero while each channel's matched exposure time is
preserved. The controller calls `off` before capture and in a post-capture
cleanup path; one exposure defaults to a 60-second timeout and can be changed
with `--timeout-seconds`. The adapter must:

1. switch all lights off;
2. configure camera exposure and the requested light;
3. wait for the physical system to settle;
4. capture one linear mono plane or raw CFA mosaic;
5. write binary PGM at the requested path;
6. switch every light off in a guaranteed cleanup path;
7. return zero only after the file is durable and complete.

This boundary keeps the camera and light implementations replaceable. The
manifest records **commanded** power and exposure, not measured radiance or
verified shutter timing.

### Scanlight mapping

The modern unified Scanlight firmware at commit
[`8f749d9`](https://github.com/jackw01/scanlight/tree/8f749d95cc3794c87452394a47f814afd2270adc/automation)
enumerates as USB CDC serial; the official app opens it at 115200 baud. Packets
are `FE`, header, payload length, then payload—there is no terminator or CRC.
For capture, send only these exact set-color packets:

```text
off:   FE 00 06 00 00 00 00 00 00
red:   FE 00 06 PP 00 00 00 00 00
green: FE 00 06 00 PP 00 00 00 00
blue:  FE 00 06 00 00 PP 00 00 00
```

`PP = floor(255 * power + 0.5)`. White, IR, and `save preset` stay zero; the
last byte must never be set during scanning because it writes flash. Set-color
has no acknowledgement or current-state readback. Do not pipeline packets; use
a configurable settle delay measured on the physical light, and describe `PP`
as a commanded PWM code rather than measured light power. Power supply mode,
hardware revision, and trim can change actual drive.

The existing
[TriRGB `scanlightctl`](https://github.com/rohanpandula/TriRGB/tree/7ec5bdd8dfc5729f272601208bfc9b9b64e69ce2/phase1/scanlightctl)
is a source-reviewed encoder/reader that can be reused after hardening; it is
not itself the `capture-one` adapter. It currently does not probe before
enable, drain and hold after writes, or send off when its context exits. The
adapter must add those behaviors, its own `finally: off`, and the parent
cleanup. Use an explicit serial port for unattended work and probe
firmware/hardware before enabling a channel.

Current firmware has no watchdog and retains the last commanded output after a
host failure. Static review also found that its over-temperature state can be
overwritten by the following VBUS branch and that the thermistor lookup derives
an array count from a pointer parameter. Treat telemetry as diagnostic only;
use a normally-off hardware enable/watchdog and an independent thermal cutoff.
Scanlight v4 also flashes W/R/G/B during startup, so wait for startup, then send
off before exposing film. Avoid the shutter jack in this camera-API rig: v4a's
jack is non-isolated while v4b's is isolated, and the light's trigger pulse does
not configure camera exposure.

### Candidate Bayer camera control paths

Neither path below is integrated or hardware-tested by this repository. Both
named cameras have vendor-supported tethering and current libgphoto2 capture
support:

- Fujifilm documents
  [GFX100 II USB/Wi-Fi/Ethernet tethering](https://fujifilm-dsc.com/en/manual/gfx100ii/connections/tethere/index.html)
  and provides a
  [Camera Control SDK](https://www.fujifilm-x.com/en-gb/camera-control-sdk/).
  Fujifilm warns that individual use of that SDK to connect to or control a
  camera voids its limited product warranty, and the SDK supplies no RAF
  conversion information. The generic libgphoto2 matrix reports capture,
  live-view, and configuration. A GFX100 II firmware 2.21
  [device snapshot](https://github.com/gphoto/libgphoto2/blob/6361aa9cead68ebfb21cb65c445980ea21f6f2a8/camlibs/ptp2/cameras/fuji-gfx100-ii.txt#L175-L187)
  exposed writable RAF mode, while its
  [shutter section](https://github.com/gphoto/libgphoto2/blob/6361aa9cead68ebfb21cb65c445980ea21f6f2a8/camlibs/ptp2/cameras/fuji-gfx100-ii.txt#L390-L456)
  exposed writable timing controls. Both must be verified on the actual camera
  firmware.
- Nikon explicitly supports the Zf in
  [NX Tether](https://downloadcenter.nikonimglib.com/en/download/sw/276.html)
  and its
  [Remote Module SDK](https://sdk.nikonimaging.com/information/en/). Current
  libgphoto2 reports capture, trigger, live-view, and configuration. A Zf
  firmware 1.10 snapshot exposed
  [NEF modes](https://github.com/gphoto/libgphoto2/blob/6361aa9cead68ebfb21cb65c445980ea21f6f2a8/camlibs/ptp2/cameras/nikon-zf.txt#L333-L350)
  but a
  [read-only shutter control](https://github.com/gphoto/libgphoto2/blob/6361aa9cead68ebfb21cb65c445980ea21f6f2a8/camlibs/ptp2/cameras/nikon-zf.txt#L404-L410).
  Test the actual firmware with the physical shutter dial at
  [`1/3 STEP`](https://onlinemanual.nikonimglib.com/zf/en/exposure_33.html)
  before relying on unattended gPhoto control. NX Tether itself documents
  [shutter controls](https://nikonimglib.com/nx_tether/onlinehelp/en/03_shooting_settings_pane_14.html)
  and [RAW transfer](https://nikonimglib.com/nx_tether/onlinehelp/en/02_image_settings_8.html),
  but it is a GUI, not evidence of a programmable `capture-one` API.

The generated
[libgphoto2 support list](https://gphoto.sourceforge.io/proj/libgphoto2/support.php)
is the shared capture/configuration reference for those statements.

Use direct USB rather than a hub, lock manual focus/aperture/ISO/shutter and
geometry, and disable auto ISO/exposure, stabilization, pixel shift,
long-exposure noise reduction, and multi-frame processing. Do not run two
camera-control applications simultaneously. NX Tether requires Nikon-ID sign
in and online reauthentication after more than 30 days offline, which makes it
less suitable as an unattended appliance dependency.

For initial Nikon decoder tests, use lossless-compressed NEF rather than HE or
HE★. [LibRaw 0.22.0](https://github.com/LibRaw/LibRaw/blob/0.22.0/Changelog.txt)
introduced Zf standard-compression and GFX100 II RAF support. That is only a
decoder compatibility claim: pin and record the actual decoder version, then
verify sample files preserve mosaic codes, active crop, CFA phase, and
black/white metadata before producing the core's PGM input.

## Validation protocol

### A. Prove acquisition calibration

1. Warm the camera and lights to a repeatable state.
2. Capture dark, flat, and a stable transmissive target.
3. Repeat at several calibrated power/exposure pairs.
4. Normalize each result and compare the same pixels.
5. Repeat after restart and on another day.
6. Measure clipping, noise, drift, and R/G/B registration before rendering.

### B. Prove stock and frame behavior

1. Capture multiple frames from each stock, including clear roll borders.
2. Include normal, underexposed, overexposed, neutral, saturated, and mixed-light
   scenes.
3. Scan the same frames on the chosen reference scanner without changing the
   negatives.
4. Fit only a training subset.
5. Report results on held-out frames, stocks, and exposure ranges.
6. Compare a generic profile against stock-specific and per-frame corrections
   stage by stage.

One paired frame is enough to test plumbing. It is not enough to validate color
generality. A per-channel LUT may fit one frame while hiding cross-channel or
density-dependent errors.

## First milestones

1. Implement a GFX100 II or Nikon Zf `capture-one` adapter and prove that
   failures always leave the lights off.
2. Produce repeatable dark/flat-normalized R/G/B captures from the physical rig.
3. Establish linearity, clipping, noise, drift, and registration limits.
4. Capture a multi-frame, multi-density paired reference dataset.
5. Add clear-roll base removal and the smallest stock profile that passes
   hold-out validation.
6. Evaluate per-frame correction only after the earlier stages are stable.

Deferred until real files and hardware require them: native RAF/NEF ingestion,
CFA resampling, defect repair, subpixel registration, HDR merging, temperature
compensation, profile fitting, and automatic gray balance.
