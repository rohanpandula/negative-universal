# MS11 first-party simulator harness (Frontier SP-500)

## Summary

The MS11W7-SP500 installer ships Fujifilm's own simulation layer at every
process boundary of the scanner stack: transport, device, order database, GUI,
and frame-detection telemetry. The image pipeline itself is never simulated.
A Windows 7 VM can therefore run the genuine MS11 negative-conversion pipeline
with no SP-500 attached, which makes a golden-master comparison against this
project's pipeline practical.

This is a supported vendor configuration, not a hack. The installer carries
explicit toggle scripts that switch each subsystem between real and simulated
components.

## What is simulated, layer by layer

| Layer | Real component | Simulator | Switch |
| --- | --- | --- | --- |
| 1394 transport | `ff1394Lab.dll` (system32) | `ff1394Lab_sim.dll` | `TOOL\1394sim.bat` / `1394real.bat` copy one over the other |
| Scanner + printer behavior | SP-500 hardware | `SsSim.exe` ("SsSim - Ix Simulation") | run standalone |
| Film magazine | physical magazine | `SimIxMag.dll`, reads `IxMag%03d.dat` | linked by SsSim |
| Order database client | `FDiaClient*.dll` | `FdiaDmy*.DLL` (four variants) | `SET_FDI_DUMMY.CMD` / `DEL_FDI_DUMMY.CMD` |
| GUI server | `VirGUI.dll` | `VirGUIDmy.dll` | `SET_VirGUI_DUMMY.CMD` / `DEL_VirGUI_DUMMY.CMD` |
| Other peripherals | OMA, PU800, PU8Ctrl, CDRAC, AC | `.dummy` stubs | `SET_*` / `DEL_*` pairs in `TOOL` |

Notable details:

- `SsSim.exe` is an MFC application containing both a `CSimScanner` and a
  `CSimPrinter` class, so it plays the whole lab bench. It speaks the same
  SCSI-shaped command set as the real device (`SCN_D_REQ_/SCN_D_RES_`
  `SEARCH_FRAME`, `DECIDE_FRAME`, `CANCEL_FRAME`, `START_REALSCAN`,
  `GET_REALSCAN`, `END_REALSCAN`, `SET_REALSCAN_DENSITY`).
- The dummy clients implement the same exported order API as the real clients
  (`CCreateOrder`, `CAddOrderItemFile`, `CGetDeviceInf`, ...) but service calls
  from flat files in `D:\Dummy\` instead of the server. They are subsets
  (observed 20 exports against the real client's 41), so a no-hardware
  configuration is a deliberate mix of real and stub modules, not "all stubs".
- `VirGUIDmy.dll` is near interface-identical to `VirGUI.dll` (145 vs 147
  exports).
- One DLL even ships with a `.sim` twin (`WFOMAIFOET.dll.sim`), confirming the
  copy-over swap pattern.

## Mode and environment switching

Beyond the file-swap dummies, the install ships a complete registry-driven mode
system in `Frontier\Setup` (113 `.reg` toggles) plus mode packages in `TOOL`:

- **Scanner presence**: `FilmScannerExist.REG` / `FilmScannerNotExist.REG`
  set `HKLM\SOFTWARE\FUJIFILM\Frontier\CurrentVersion\System\InputUnit`
  `FilmScanner_1/2 = 1/0`. The runtime trusts this key; software-level
  scanner presence does not require the USB device after install.
- **Printer presence and model**: `PrinterExist.reg` /
  `PrinterNotExist.reg`, model selectors (`Printer_LP1500.reg`,
  `Printer_LP3000.reg`, `Printer_LPKariba.reg`, `SysInfoPrinter*.reg`), and
  `DummyPrinterSet.reg`, which defines a `PrintOutputDeviceInfo` entry with
  `TargetName_1 = "Dummy@localhost"` and `HostName_1 = "localhost"` — a
  software printer target.
- **Legacy vs PIC master mode**: `B20_Legacy.REG` (Mode = 0) versus
  `B20_PIC.REG` (Mode = 1) under
  `HKLM\SOFTWARE\FUJIFILM\Frontier\B20\Config\CurrentVersion`. PIC mode
  re-points the system away from F-DIA toward a PIC server; the shipped
  `PIC_SETUP` README warns that after C4/C5, restoring does not bring
  F-DIA functions back.
- **PIC_SETUP package** (`TOOL\PIC_SETUP`): swaps
  `D:\Fujifilm\DnaApps\Q0\Profile\COMMON.INI` for `PIC_COMMON.ini`
  (retention deadlines to 0, `InspectionAuto = 1`) and hides media-service
  menus via `MenuList\UserLevel_*` registry values. The `COMMON.ini`
  template also documents the workflow-engine flags: `ServerHost`
  (settable to `localhost`), `IsEZ2`, and `Standalone` — an explicit
  "EZ2-alone mode" flag, which is the documented headless/no-server
  configuration.
- **Per-device dummy environments** (`SET_*` / `DEL_*` in `TOOL`): each
  script preserves the real component as `.ORG` and copies the dummy in
  place (FDIA clients, OMA, PU800/PU8Ctrl printers, VirGUI, CD-R AC,
  DummyAC/F25, MT, PTOP, DNA dummy environment rebuild).

Practical consequence for the harness: after a base install, the no-hardware
configuration is assembled from registry toggles (scanner/printer presence,
mode) plus dummy swaps — all scriptable, no hardware required at any point
after the installer's USB gate.

## How "film" is represented

Film is never physically modeled. Three separate mechanisms stand in for it:

1. **Geometry and state.** `test_film.ini` (read from the SsSim working
   directory) describes the strip physically: `FilmStartPos`, `FilmEndPos`,
   `FilmLength`, `FrameCount`, `minFrameNo`, `TotalFrameCnt`.
2. **Detection telemetry.** `InitData\FilmSim.ini` rewrites the output of the
   frame-detection module. Its header states the purpose: simulate situations
   that are difficult to reproduce on the real machine. It is enabled by
   `HKLM\SOFTWARE\FUJIFILM\DIPP\CurrentVersion\Sequence\FilmInfoSimulation = 1`
   (alternate file via `FilmInfoSimulateFile`) and is re-read on every
   prescan. Configurable fields:
   - `[Film] FrameCount`, `DX_Code` (decimal or `0x` hex).
   - `[Frame] Type_nn`: 135 mask types `FULL` / `HV` / `PANO` / `HALF`, IX240
     (APS) types `C` / `H` / `P`, and `FAIL` for "frame detection failed".
   - `[Frame] Frames`: 12 / 20 / 24 / 36 exposure rolls; `[Frame] No_nn`:
     frame-number edge codes (`XX`, `XXA`, `00`–`36A`, `E`, or raw `0xXXXX`).
   - `[Frame] Sunuke`: blank/undeveloped frames by position or `ALL`;
     `[Frame] Kaburi`: fogged frames by position or `ALL`.
   The shipped sample is a deliberately difficult six-frame roll: alternating
   `FULL`/`HV`/`PANO`, fogged frames 1–2, blank frames 3 and 5. This is a
   fault-injection rig for the prescan sequence logic.
3. **Pixels are replayed, not synthesized.** There is no exposure, density, or
   film-stock model anywhere in the simulator. Prescans replay one canned raw
   file; fine scans replay per-frame raw captures from a roll directory. The
   default replay directory name (`Luke135F-L`) refers to Fujifilm's Luke
   accessory scanner, whose driver components (`LukeCtrl.dll`, `LukeUsbApi.dll`,
   `LukeUsbDp.exe`, `LUKEON.reg`) ship in the same install.

## Input contract

The simulator's input format is a small calibrated raw plus text files. This is
the entire contract:

- **Prescan raw** (`Tool\SsSim_InitData\prescanimg.raw`, 477,392 bytes):
  - 32-byte header: magic `44 52 02 01` ("DR"), u32 header size = 32, u16
    width = 340, u16 height = 234, u32 roll frame count = 6, reserved zeros,
    `IM16` marker at offset 0x18.
  - Payload is exactly `width * height * 3 * 2` bytes: three interleaved
    channels, 16-bit little-endian, corresponding to the three sequential
    mono exposures. Observed sample range 0–1150, i.e. 12-bit sensor data in
    a 16-bit container.
- **Correction seed** (`prefine.dat`, 6,144 bytes): exactly three monotonic
  1,024-entry 16-bit curves, one per channel, with distinct starting offsets
  (observed ranges 60–1424, 250–1614, 347–1711). This matches the three
  1,024-entry frame-correction LUT stage recorded in the static findings, and
  the per-channel offsets are consistent with per-channel film-base
  compensation.
- **Fine-scan raws**: replayed per frame from
  `D:\ImageData\<roll>\FineScanImage%04d.raw`. No fine-scan raw ships in the
  installer, so its exact header layout is still to be determined on first
  runtime use. The SsSim file dialog also accepts arbitrary `*.raw` and
  `*.bmp` files, which may avoid the question entirely.
- **Roll description**: the two INI files above plus the registry flag.
- **Built-in observation**: SsSim writes `SsSimLog.txt` and `SsSimDebug.txt`,
  and can dump `real_image%d.bmp` files.

## Harness plan for this project

1. Run MS11 in a Windows 7 VM with its own simulator components. Record which
   modules the no-hardware configuration actually loads, and the fine-scan
   raw layout, on first run.
2. Feed owned sequential-RGB captures through the input contract: register the
   three exposures, resample to 340x234 for the prescan, full resolution for
   the fine scan, and write the roll INIs. MS11's genuine pipeline then
   processes owned film.
3. Capture MS11's output for those inputs as a golden master and diff against
   this project's pipeline stage by stage.
4. Cross-calibrate the measurement delta with paired scans: same physical
   frames through both this project's rig and the SP-500, raw measurements on
   both sides. For each channel fit `SP500 = f(rig)` with a monotonic
   (isotonic) fit; the curve absorbs exposure, light power, gain, and offset
   differences at once. Then check fit residuals against the other two
   channels: flat residuals mean per-channel curves suffice; correlation with
   another channel means spectral mismatch and calls for a 3x3 matrix before
   the curves. Validate stock-independence with leave-one-stock-out. The
   fitted maps, prepended to rig output, put the rig into the SP-500's
   measurement domain.

## Limits and open questions

- The simulator replays canned pixels, so acquisition-side calibration (dark,
  flat, LED current, defect correction at capture time) is not exercised by
  this harness. It validates pipeline logic, not sensor behavior.
- Even with the delta calibrated, film-stock transforms downstream are
  calibrated for the SP-500's illumination spectra and sensor response. Colors
  from a differently banded rig will be structurally correct but approximate;
  per-stock work remains this project's own.
- The dummy clients are export subsets; expect a working no-hardware
  configuration to mix real and stub modules.
- The no-scanner install layout and the fine-scan header are unverified until
  the VM run.

## Provenance

These are behavioral observations from listing and static strings of a lawfully
owned installer. No Fujifilm binaries, code, tables, calibration data, or
assets are copied into this repository. MS11 remains Fujifilm's software; the
harness is used only as a behavioral reference. The supporting evidence ledger
lives in
[frontier-sp500-research](https://git.roflix.club/rohan/frontier-sp500-research).
