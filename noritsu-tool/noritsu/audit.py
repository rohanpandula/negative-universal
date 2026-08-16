"""Evidence-labelled audit runner for the recovered Noritsu film pipeline.

This is deliberately separate from the older illustrative ``pipeline.py``.
Only a kernel and parameter set supported by installer/capture evidence may
alter pixels here.  Every other normal-path or optional stage still receives a
numbered, visually rendered checkpoint with an explicit reason for remaining
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tifffile

from . import __version__, ls600raw, pcb, prm, tables, vendor
from .trace import StageTrace, write_comparison_artifacts


# The portra400-frame1 preset reproduces the audited owner capture. Its
# source material (owner scans and native-capture evidence) is private and is
# not redistributed with this project. Set NORITSU_PORTRA400_DIR to the owner
# capture root (expected to contain "raw/1/...", "tif16-default-no-ice/", and
# "ezc-dsa-settings.jpg") and NORITSU_RE_ROOT to the private evidence root
# (expected to contain "work/final_pcb_capture/"), or pass explicit paths.
PORTRA400_DIR_ENV = "NORITSU_PORTRA400_DIR"
RE_ROOT_ENV = "NORITSU_RE_ROOT"

_UNSET_ROOT = Path("/nonexistent/noritsu-private-capture")
_owner_dir = os.environ.get(PORTRA400_DIR_ENV)
_re_root = os.environ.get(RE_ROOT_ENV)
_OWNER_DIR = Path(_owner_dir) if _owner_dir else _UNSET_ROOT
DEFAULT_RE_ROOT = Path(_re_root) if _re_root else _UNSET_ROOT
_EVIDENCE_DIR = DEFAULT_RE_ROOT / "work/final_pcb_capture"

PORTRA400_RAW = _OWNER_DIR / "raw/1/FullScan/FULL000000010000.RAW"
PORTRA400_REFERENCE = _OWNER_DIR / "tif16-default-no-ice/000002250001.tif"
PORTRA400_PRM = _OWNER_DIR / "raw/1/Original/1.prm"
PORTRA400_TRZ = _OWNER_DIR / "raw/1/Thumbnail/1.Trz"
PORTRA400_SETTINGS = _OWNER_DIR / "ezc-dsa-settings.jpg"
PORTRA400_FINAL_PCB = _EVIDENCE_DIR / "owner_full_trz_nominal_pcb.bin"
PORTRA400_NATIVE_REFERENCE = _EVIDENCE_DIR / "native_owner_full_trz_rot90ccw.tif"
PORTRA400_HUE_LATTICE = _EVIDENCE_DIR / "owner_full_com_para_hue_i16.bin"
PORTRA400_GRAIN_CURVE = _EVIDENCE_DIR / "owner_full_com_para_grain_u16.bin"
PORTRA400_IMAGE_DESCRIPTOR = _EVIDENCE_DIR / "owner_full_com_para_image_prefix.bin"
PORTRA400_GAMMA_U8_ORACLE = _EVIDENCE_DIR / "gamma_u8_bgr.bin"
PORTRA400_DEPTH_CAPTURE_LOG = _EVIDENCE_DIR / "com_para_dump_hook_log.txt"


def _portra400_preset_paths() -> dict[str, Path]:
    missing = [
        name
        for name, value in (
            (PORTRA400_DIR_ENV, _owner_dir),
            (RE_ROOT_ENV, _re_root),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"--preset portra400-frame1 needs the private capture roots: "
            f"set {', '.join(missing)}, or pass explicit "
            "--raw/--reference/--prm/--final-pcb/... paths."
        )
    return {
        "raw": PORTRA400_RAW,
        "reference": PORTRA400_REFERENCE,
        "prm": PORTRA400_PRM,
        "trz": PORTRA400_TRZ,
        "settings": PORTRA400_SETTINGS,
        "final_pcb": PORTRA400_FINAL_PCB,
        "native_reference": PORTRA400_NATIVE_REFERENCE,
        "hue_lattice": PORTRA400_HUE_LATTICE,
        "grain_curve": PORTRA400_GRAIN_CURVE,
        "image_descriptor": PORTRA400_IMAGE_DESCRIPTOR,
        "gamma_u8_oracle": PORTRA400_GAMMA_U8_ORACLE,
        "depth_capture_log": PORTRA400_DEPTH_CAPTURE_LOG,
        "re_root": DEFAULT_RE_ROOT,
    }
PORTRA400_RAW_SHA256 = (
    "7d18317868cc7abf6b971d547b00e3fc21871efb384aa2a247ef6a87780a8247"
)
PORTRA400_REFERENCE_SHA256 = (
    "715f937ab5097d372f197295da94709d306164de7d1ca7502a38dd93743a2dc7"
)
PORTRA400_PRM_SHA256 = (
    "098c33a4f11f4f338ffcb9c9c784543546fc13741b47eab1009543b757f288b0"
)
PORTRA400_FINAL_PCB_SHA256 = (
    "a22aa8b6e1a6003d50cbffeb6bec732b63e5986c3ddae0270a561f29ffef28d3"
)
PORTRA400_NATIVE_REFERENCE_SHA256 = (
    "ef67372413ede29625061d820414ffb32b2df194dddb196f130726aaa8fd9c5b"
)
PORTRA400_NATIVE_INPUT_SHA256 = (
    "b89a8d2427f8fc5c4905dde773cb64f1035770a36233e372ce1599107ab28534"
)
PORTRA400_HUE_LATTICE_SHA256 = (
    "d0a2f9a06e43fce3061d4afc34b1c25a74036f99980378309db342aa390aeb9b"
)
PORTRA400_GRAIN_CURVE_SHA256 = (
    "214ad9ae4f8bab62612e769773a1621d296f9699c0f531972cef7afbd6e491cd"
)
PORTRA400_IMAGE_DESCRIPTOR_SHA256 = (
    "e3d727c920f757d112759da4ae780a184db5b24722ad7c3bd1cbee2fc371143a"
)
PORTRA400_GAMMA_U8_ORACLE_SHA256 = (
    "b095760adb3c5485520eb0efc523f79c46f39215180a966b62ab1a74a88e410b"
)
PORTRA400_DEPTH_CAPTURE_LOG_SHA256 = (
    "b7cea29b8f4eda0a4b4916c2a80ef3084b67a4dcb9c7e3f801d4bb796a266d9e"
)


@dataclass(frozen=True)
class AuditConfig:
    raw: Path
    reference: Optional[Path]
    output_dir: Path
    raw_width: Optional[int] = None
    raw_height: Optional[int] = None
    group: int = 0
    quality_type: int = 1
    rotation_k: int = 1
    rotated_crop_y: int = 17
    rotated_crop_x: int = 171
    crop_height: int = 4008
    crop_width: int = 6048
    output_height: int = 4011
    output_width: int = 6048
    fit_external_reference: bool = True
    fit_sample_step: int = 2
    prm: Optional[Path] = None
    final_pcb: Optional[Path] = None
    native_reference: Optional[Path] = None
    hue_lattice: Optional[Path] = None
    grain_curve: Optional[Path] = None
    image_descriptor: Optional[Path] = None
    gamma_u8_oracle: Optional[Path] = None
    depth_capture_log: Optional[Path] = None
    trz: Optional[Path] = None
    settings_screenshot: Optional[Path] = None
    reverse_engineering_root: Optional[Path] = None
    expected_raw_sha256: Optional[str] = None
    expected_reference_sha256: Optional[str] = None
    expected_prm_sha256: Optional[str] = None
    expected_final_pcb_sha256: Optional[str] = None
    expected_native_reference_sha256: Optional[str] = None
    expected_hue_lattice_sha256: Optional[str] = None
    expected_grain_curve_sha256: Optional[str] = None
    expected_image_descriptor_sha256: Optional[str] = None
    expected_gamma_u8_oracle_sha256: Optional[str] = None
    expected_depth_capture_log_sha256: Optional[str] = None


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "status": "missing"}
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _git_record(repository: Path) -> dict[str, Any]:
    def run(*arguments: str, binary: bool = False) -> str | bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
        )
        return completed.stdout if binary else completed.stdout.strip()

    try:
        status_text = run("status", "--porcelain=v1")
        assert isinstance(status_text, str)
        status = status_text.splitlines()
        diff = run("diff", "--binary", "HEAD", "--", ".", binary=True)
        assert isinstance(diff, bytes)
        untracked_text = run("ls-files", "--others", "--exclude-standard")
        assert isinstance(untracked_text, str)
        untracked = []
        for relative in untracked_text.splitlines():
            path = repository / relative
            if path.is_file():
                untracked.append(
                    {
                        "path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status": status,
            "tracked_binary_diff": {
                "size_bytes": len(diff),
                "sha256": hashlib.sha256(diff).hexdigest(),
            },
            "untracked_files": untracked,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def _jsonable_config(config: AuditConfig) -> dict[str, Any]:
    result = asdict(config)
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in result.items()
    }


def _config_hash(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _binary_evidence(
    config: AuditConfig,
    binary: str,
    symbol: str,
    address: str,
    detail: str,
) -> dict[str, Any]:
    candidates = {
        "UniRdImgLib.dll": "imgproc/ImageDataProc/UniRdImgLib.dll",
        "SpecialPcb.dll": "imgproc/ImageDataProc/SpecialPcb.dll",
        "CreateImage.dll": "scansoft_w7/Twain_Module_Files/CreateImage.dll",
    }
    record: dict[str, Any] = {
        "method": "Kuna 1.95 x86 decompilation plus instruction-level verification",
        "binary": binary,
        "symbol": symbol,
        "virtual_address": address,
        "detail": detail,
    }
    if config.reverse_engineering_root is not None and binary in candidates:
        path = config.reverse_engineering_root / candidates[binary]
        record["source"] = _source_record(path)
    return record


def _pre_rotation_crop_and_scale(
    image: np.ndarray, config: AuditConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Express the measured rotated-frame crop before RD5X's rotation stage."""

    height, width = image.shape[:2]
    k = config.rotation_k % 4
    cy = config.rotated_crop_y
    cx = config.rotated_crop_x
    ch = config.crop_height
    cw = config.crop_width

    if k == 0:
        raw_y, raw_x, raw_h, raw_w = cy, cx, ch, cw
        resized_height, resized_width = config.output_height, config.output_width
    elif k == 1:
        raw_y = cx
        raw_x = width - (cy + ch)
        raw_h, raw_w = cw, ch
        resized_height, resized_width = config.output_width, config.output_height
    elif k == 2:
        raw_y = height - (cy + ch)
        raw_x = width - (cx + cw)
        raw_h, raw_w = ch, cw
        resized_height, resized_width = config.output_height, config.output_width
    else:
        raw_y = height - (cx + cw)
        raw_x = cy
        raw_h, raw_w = cw, ch
        resized_height, resized_width = config.output_width, config.output_height

    if min(raw_y, raw_x, raw_h, raw_w, resized_height, resized_width) < 0:
        raise ValueError("derived pre-rotation geometry contains a negative value")
    if raw_y + raw_h > height or raw_x + raw_w > width:
        raise ValueError("derived crop lies outside the raw raster")
    cropped = image[raw_y : raw_y + raw_h, raw_x : raw_x + raw_w]
    scaled = cv2.resize(
        cropped,
        (int(resized_width), int(resized_height)),
        interpolation=cv2.INTER_AREA,
    ).astype(image.dtype, copy=False)
    return scaled, {
        "pre_rotation_crop": {
            "y": raw_y,
            "x": raw_x,
            "height": raw_h,
            "width": raw_w,
        },
        "pre_rotation_output": {
            "height": resized_height,
            "width": resized_width,
        },
        "resampler": "OpenCV INTER_AREA (reference registration, not recovered vendor taps)",
    }


def _read_reference(path: Path) -> np.ndarray:
    try:
        image = tifffile.memmap(str(path), mode="r")
    except (TypeError, ValueError):
        image = tifffile.imread(str(path))
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("reference must be an RGB image")
    if image.dtype == np.uint8:
        image = image.astype(np.uint16) * np.uint16(257)
    if image.dtype != np.uint16:
        raise TypeError("reference must be uint8 or uint16")
    return image


def _ycc_checkpoint_view(image_c1c2y: np.ndarray) -> np.ndarray:
    """Expose RD5X YCC lanes as documented RGB preview slots ``[Y,C1,C2]``.

    This is a lossless lane permutation, not a color conversion.  It lets the
    RGB-only trace artifact format preserve exact per-lane changes without
    pretending that intermediate chroma codes are display RGB.
    """

    image = np.asarray(image_c1c2y)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint16:
        raise ValueError("RD5X YCC checkpoint input must be HxWx3 uint16")
    return np.ascontiguousarray(image[..., (2, 0, 1)])


def _reference_characterization(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    with tifffile.TiffFile(str(path)) as tiff:
        page = tiff.pages[0]
        result: dict[str, Any] = {
            "shape": list(page.shape),
            "dtype": str(page.dtype),
            "icc_profile_present": 34675 in page.tags,
        }
    if result["dtype"] == "uint16":
        pixels = tifffile.memmap(str(path), mode="r")
        expanded = True
        for start in range(0, pixels.shape[0], 256):
            if np.any(np.asarray(pixels[start : start + 256]) % 257):
                expanded = False
                break
        result["all_samples_multiples_of_257"] = expanded
        result["effective_precision"] = (
            "8-bit code values expanded to uint16" if expanded else "greater than 8-bit"
        )
    return result


def _write_json(path: Path, document: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _captured_ccd_gain_q12(
    capture: Optional[dict[str, Any]],
) -> Optional[list[int]]:
    """Derive SpecialPcb's three gain words from serialized controller state."""

    if capture is None:
        return None
    charge_node = capture["scanner_charge"]
    if not charge_node.get("count") or int(charge_node["value"]) <= 0:
        return None
    gains = np.asarray(capture["scanner_gain_rgb"], dtype=np.float64)
    if gains.shape != (3,) or not np.all(np.isfinite(gains)):
        return None
    charge = int(charge_node["value"]) & 0xFF
    if charge == 0:
        return None
    return (
        np.rint(np.clip(gains / charge * 4096.0, 0.0, 65535.0))
        .astype(np.uint16)
        .astype(int)
        .tolist()
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _stage_report(
    manifest: dict[str, Any], path: Path, recovered_boundary: dict[str, Any]
) -> None:
    external = next(
        (
            stage
            for stage in reversed(manifest["stages"])
            if stage["status"] == "applied_observed_output_encoding"
        ),
        None,
    )
    comparison = (
        external["difference_from_reference"]
        if external is not None
        else {"status": "not_available", "reason": "external fit was not requested"}
    )
    lines = [
        "# Noritsu installer pipeline audit",
        "",
        "Every row has a numbered full-resolution TIFF and preview. Unchanged rows are "
        "hard-linked checkpoints: they remain independently addressable without duplicating pixels.",
        "",
        "`installer_color_boundary_native.tif`, `final.tif`, and "
        "`recovered_boundary.tif` preserve the recovered fixed-job output before any "
        "reference fit. Fallback runs without a native PCB label their external geometry "
        "separately. External-reference fits never replace the recovery boundary.",
        "",
        "The paired machine TIFF has no embedded ICC profile and stores 8-bit code values "
        "expanded into a uint16 container. Metrics against it are code-value diagnostics, "
        "not colorimetric measurements. The external residual fit is evaluated in-sample.",
        "",
        "| # | Stage | Status | Applied | Previous MAE | Reference MAE |",
        "|---:|---|---|:---:|---:|---:|",
    ]
    for stage in manifest["stages"]:
        previous = stage["difference_from_previous"]
        reference = stage["difference_from_reference"]
        previous_mae = (
            f"{previous['overall']['mae']:.8f}"
            if previous.get("status") == "ok"
            else previous.get("status", "n/a")
        )
        reference_mae = (
            f"{reference['overall']['mae']:.8f}"
            if reference.get("status") == "ok"
            else reference.get("status", "n/a")
        )
        lines.append(
            f"| {stage['index']} | {stage['name']} | {stage['status']} | "
            f"{'yes' if stage['applied'] else 'no'} | {previous_mae} | {reference_mae} |"
        )
    lines.extend(["", "## Registered recovery-boundary diagnostic", ""])
    boundary_comparison = recovered_boundary.get("comparison", {})
    if boundary_comparison.get("status") == "ok":
        lines.extend(
            [
                "The code units are not proven to share a final display encoding; these values are "
                "therefore structural/photometric diagnostics, not a claim of final colorimetric equivalence.",
                "",
                f"- MAE: `{boundary_comparison['overall']['mae']:.8f}`",
                f"- RMSE: `{boundary_comparison['overall']['rmse']:.8f}`",
                f"- BT.709 luma NCC: `{boundary_comparison['luma_ncc']}`",
                "- Visuals: `boundary_comparison/` (current, target, absolute, heatmap, signed luma)",
            ]
        )
    else:
        lines.append(
            f"- Not available: {boundary_comparison.get('reason', boundary_comparison.get('status'))}"
        )
    lines.extend(["", "## Native-engine oracle diagnostic", ""])
    native_comparison = recovered_boundary.get("native_engine_comparison", {})
    if native_comparison.get("status") == "ok":
        lines.extend(
            [
                f"- Recovered vs native MAE: `{native_comparison['overall']['mae']:.8f}`",
                f"- Recovered vs native RMSE: `{native_comparison['overall']['rmse']:.8f}`",
                f"- Recovered vs native luma NCC: `{native_comparison['luma_ncc']}`",
                "- Visuals: `native_engine_comparison/`",
            ]
        )
    else:
        lines.append(
            f"- Not available: {native_comparison.get('reason', native_comparison.get('status'))}"
        )
    native_machine = recovered_boundary.get("native_engine_vs_machine", {})
    if native_machine.get("status") == "ok":
        lines.extend(
            [
                f"- Native vs machine MAE: `{native_machine['overall']['mae']:.8f}`",
                f"- Native vs machine RMSE: `{native_machine['overall']['rmse']:.8f}`",
                f"- Native vs machine luma NCC: `{native_machine['luma_ncc']}`",
                "- Visuals: `native_engine_vs_machine/`",
            ]
        )
    lines.extend(["", "## Recovery stopping boundary", ""])
    selected_blockers = [
        stage
        for stage in manifest["stages"]
        if stage["status"]
        in {
            "blocked_missing_native_runtime_parameters",
            "blocked_external_legacy_controller_state",
        }
    ]
    if selected_blockers:
        lines.append(
            "Selected software stages still awaiting runtime state: "
            + ", ".join(stage["name"] for stage in selected_blockers)
            + "."
        )
    else:
        lines.append(
            "No selected fixed-job software stage is awaiting a PCB, COM table, "
            "or controller selector."
        )
    if native_comparison.get("status") == "ok":
        native_mae = native_comparison["overall"]["mae"]
        if native_mae == 0.0:
            lines.append(
                "Every recovered pixel code equals the native RGB8 oracle after the "
                "documented lossless u8-to-uint16 multiplication by 257; the TIFF file "
                "representations themselves have different sample widths."
            )
        else:
            lines.append(
                "The nonzero recovered-versus-native MAE above is a same-installed-engine "
                "software-conformance diagnostic; it is not attributed to hardware."
            )
    lines.extend(
        [
            "Physical boundaries are the scanner's per-unit 5,340-column calibration "
            "arrays/maps and the fourth IR/Kizu capture. The owner FULL is already "
            "downstream of the former, and the paired target explicitly selects no ICE.",
            "Named film-stock spectral behavior is absent from CorrectServ and needs "
            "external legacy captures or physical film/scanner measurements.",
        ]
    )
    lines.extend(["", "## External-fit diagnostic", ""])
    if comparison.get("status") == "ok":
        lines.extend(
            [
                f"- MAE: `{comparison['overall']['mae']:.8f}`",
                f"- RMSE: `{comparison['overall']['rmse']:.8f}`",
                f"- p99 absolute error: `{comparison['overall']['p99_abs']:.8f}`",
                f"- changed pixels: `{comparison['changed_pixel_fraction']:.8f}`",
                f"- BT.709 luma NCC: `{comparison['luma_ncc']}`",
            ]
        )
    else:
        lines.append(
            f"- Not available: {comparison.get('reason', comparison.get('status'))}"
        )
    lines.extend(
        [
            "",
            "The external monotone fit is deliberately labelled as external-reference evidence. "
            "It is not represented as a recovered Noritsu stage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_audit(config: AuditConfig) -> dict[str, Any]:
    raw_path = config.raw.resolve()
    output_dir = config.output_dir.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if config.reference is not None and not config.reference.resolve().is_file():
        raise FileNotFoundError(config.reference)
    if config.fit_external_reference and config.reference is None:
        raise ValueError("fit_external_reference requires a reference image")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty audit directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_document = _jsonable_config(config)
    processing_document = {
        key: value for key, value in config_document.items() if key != "output_dir"
    }
    config_document["config_sha256"] = _config_hash(processing_document)
    config_document["config_hash_scope"] = (
        "processing inputs/options; output_dir excluded"
    )
    _write_json(output_dir / "config.json", config_document)

    repository = Path(__file__).resolve().parents[1]
    raw_record = _source_record(raw_path)
    reference_record = _source_record(config.reference)
    prm_record = _source_record(config.prm)
    final_pcb_record = _source_record(config.final_pcb)
    native_reference_record = _source_record(config.native_reference)
    hue_lattice_record = _source_record(config.hue_lattice)
    grain_curve_record = _source_record(config.grain_curve)
    image_descriptor_record = _source_record(config.image_descriptor)
    gamma_u8_oracle_record = _source_record(config.gamma_u8_oracle)
    depth_capture_log_record = _source_record(config.depth_capture_log)
    if (
        config.expected_raw_sha256 is not None
        and raw_record is not None
        and raw_record.get("sha256") != config.expected_raw_sha256
    ):
        raise ValueError("raw SHA-256 does not match the selected audit preset")
    if (
        config.expected_reference_sha256 is not None
        and reference_record is not None
        and reference_record.get("sha256") != config.expected_reference_sha256
    ):
        raise ValueError("reference SHA-256 does not match the selected audit preset")
    if (
        config.expected_prm_sha256 is not None
        and prm_record is not None
        and prm_record.get("sha256") != config.expected_prm_sha256
    ):
        raise ValueError("PRM SHA-256 does not match the selected audit preset")
    if (
        config.expected_final_pcb_sha256 is not None
        and final_pcb_record is not None
        and final_pcb_record.get("sha256") != config.expected_final_pcb_sha256
    ):
        raise ValueError("final PCB SHA-256 does not match the selected audit preset")
    if (
        config.expected_native_reference_sha256 is not None
        and native_reference_record is not None
        and native_reference_record.get("sha256")
        != config.expected_native_reference_sha256
    ):
        raise ValueError(
            "native reference SHA-256 does not match the selected audit preset"
        )
    if (
        config.expected_hue_lattice_sha256 is not None
        and hue_lattice_record is not None
        and hue_lattice_record.get("sha256") != config.expected_hue_lattice_sha256
    ):
        raise ValueError("Hue COM lattice SHA-256 does not match the audit preset")
    if (
        config.expected_grain_curve_sha256 is not None
        and grain_curve_record is not None
        and grain_curve_record.get("sha256") != config.expected_grain_curve_sha256
    ):
        raise ValueError("Grain COM curve SHA-256 does not match the audit preset")
    if (
        config.expected_image_descriptor_sha256 is not None
        and image_descriptor_record is not None
        and image_descriptor_record.get("sha256")
        != config.expected_image_descriptor_sha256
    ):
        raise ValueError("image descriptor SHA-256 does not match the audit preset")
    if (
        config.expected_gamma_u8_oracle_sha256 is not None
        and gamma_u8_oracle_record is not None
        and gamma_u8_oracle_record.get("sha256")
        != config.expected_gamma_u8_oracle_sha256
    ):
        raise ValueError("gamma uint8 oracle SHA-256 does not match the audit preset")
    if (
        config.expected_depth_capture_log_sha256 is not None
        and depth_capture_log_record is not None
        and depth_capture_log_record.get("sha256")
        != config.expected_depth_capture_log_sha256
    ):
        raise ValueError("depth-capture log SHA-256 does not match the audit preset")
    prm_capture = (
        prm.decode_prm(config.prm)
        if config.prm is not None and config.prm.resolve().is_file()
        else None
    )
    final_pcb_data = (
        pcb.read_final_pcb(config.final_pcb)
        if config.final_pcb is not None and config.final_pcb.resolve().is_file()
        else None
    )
    final_pcb_summary = (
        pcb.decode_final_pcb_summary(final_pcb_data)
        if final_pcb_data is not None
        else None
    )
    hue_lattice_data = (
        pcb.read_hue_saturation_lattice(config.hue_lattice)
        if config.hue_lattice is not None and config.hue_lattice.resolve().is_file()
        else None
    )
    grain_curve_data = (
        pcb.read_grain_curve(config.grain_curve)
        if config.grain_curve is not None and config.grain_curve.resolve().is_file()
        else None
    )
    decoded_image_descriptor = (
        pcb.read_image_descriptor_prefix(config.image_descriptor)
        if config.image_descriptor is not None
        and config.image_descriptor.resolve().is_file()
        else None
    )
    base_selectors = tables.load_normal_output_selectors(
        config.group, config.quality_type
    )
    effective_selectors = (
        final_pcb_summary["effective_selectors"]
        if final_pcb_summary is not None
        else None
    )

    def captured_value(section: str, key: str) -> Optional[int | float]:
        if prm_capture is None:
            return None
        node = prm_capture[section][key]
        return node["value"] if node.get("count") else None

    captured_mode = (
        0
        if captured_value("correction_switches", "standard_density") == 1
        and captured_value("correction_switches", "standard_color") == 0
        else None
    )
    captured_cms_switch = captured_value("cms", "switch")
    captured_ccd_gain_q12 = _captured_ccd_gain_q12(prm_capture)

    run_provenance = {
        "config_sha256": config_document["config_sha256"],
        "input": {
            **(raw_record or {}),
            "layout": "headerless little-endian uint16, HxWx3 BGR, nominal 12-bit",
        },
        "machine_reference": reference_record,
        "native_engine_reference": native_reference_record,
        "machine_reference_characterization": _reference_characterization(
            config.reference.resolve() if config.reference is not None else None
        ),
        "captured_job": {
            "prm": prm_record,
            "decoded_prm": prm_capture,
            "native_final_pcb": final_pcb_record,
            "decoded_native_final_pcb": final_pcb_summary,
            "native_com_para": {
                "hue_saturation_lattice": hue_lattice_record,
                "grain_variance_curve": grain_curve_record,
                "image_descriptor": image_descriptor_record,
                "decoded_image_descriptor": decoded_image_descriptor,
                "gamma_u8_lut_oracle": gamma_u8_oracle_record,
                "depth_capture_log": depth_capture_log_record,
                "hue_lattice_statistics": (
                    {
                        "shape": list(hue_lattice_data.shape),
                        "minimum": int(hue_lattice_data.min()),
                        "maximum": int(hue_lattice_data.max()),
                        "nonzero_values": int(np.count_nonzero(hue_lattice_data)),
                    }
                    if hue_lattice_data is not None
                    else None
                ),
            },
            "trz": _source_record(config.trz),
            "settings_screenshot": _source_record(config.settings_screenshot),
        },
        "geometry": {
            "rotation_k": config.rotation_k,
            "rotated_empirical_crop": {
                "y": config.rotated_crop_y,
                "x": config.rotated_crop_x,
                "height": config.crop_height,
                "width": config.crop_width,
                "role": (
                    "diagnostic only; not used when a native final PCB is supplied"
                    if final_pcb_data is not None
                    else "fallback registration geometry"
                ),
            },
            "trz_nominal_crop_for_primary_pair": (
                {"y": 16, "x": 168, "height": 4008, "width": 6048}
                if config.expected_raw_sha256 == PORTRA400_RAW_SHA256
                else None
            ),
            "output": {
                "height": config.output_height,
                "width": config.output_width,
            },
            "native_final_pcb": (
                final_pcb_summary["scaling_trimming"]
                if final_pcb_summary is not None
                else None
            ),
            "qualification": (
                "native SpecialPcb trim/S0-cubic parameters reconstructed from the "
                "captured Trz nominal rectangle; storage orientation remains an "
                "explicit pair-level interpretation"
                if final_pcb_data is not None
                else "empirical external-reference fallback; not exact vendor crop"
            ),
        },
        "tables": {
            "selection": {"group": config.group, "quality_type": config.quality_type},
            "source": tables.source_provenance(config.group, config.quality_type),
            "normal_output_exec_pattern": 5,
            "captured_mode": captured_mode,
            "base_selectors_before_frame_rewrite": base_selectors,
        },
        "reverse_engineering": {
            "primary_tool": "Kuna 1.95",
            "tool_binary": (
                _source_record(
                    config.reverse_engineering_root
                    / "kuna/kuna-v1.95-macos-arm64/kuna"
                )
                if config.reverse_engineering_root is not None
                else None
            ),
            "verification": "instruction-level x86 disassembly and focused Ghidra cross-checks",
        },
        "repository": _git_record(repository),
        "runtime": {
            "noritsu_tool": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tifffile": tifffile.__version__,
        },
    }

    trace = StageTrace(
        output_dir,
        reference=config.reference.resolve() if config.reference is not None else None,
        reference_name="EZ Controller machine TIFF",
        reference_domain="display_rgb",
        run_provenance=run_provenance,
    )

    raw_bgr = ls600raw.load_ls600_raw_u16(
        raw_path, width=config.raw_width, height=config.raw_height
    )
    trace.add(
        "Owner FULL raster (post scanner acquisition)",
        raw_bgr[..., ::-1],
        status="observed_input",
        provenance={
            "source": raw_record,
            "channel_reorder_for_checkpoint": "stored BGR -> rendered RGB",
        },
        confidence="exact file decoding; acquisition position is provenance-dependent",
        params={"stored_channel_order": "BGR", "nominal_range": [0, 4095]},
        domain="scanner_u12",
        encoding="u12_code_values_in_uint16",
        expected_range=(0, 4095),
        reference_comparable=False,
    )

    native_geometry = (
        pcb.decode_geometry(final_pcb_data) if final_pcb_data is not None else None
    )
    native_storage_orientation_applied = False
    if native_geometry is not None:
        oriented_bgr = np.ascontiguousarray(
            np.rot90(raw_bgr, config.rotation_k % 4)
        )
        expected_shape = (
            native_geometry["source_height"],
            native_geometry["source_width"],
        )
        if oriented_bgr.shape[:2] != expected_shape:
            raise ValueError(
                "configured FULL storage orientation does not match the native "
                f"PCB source raster: {oriented_bgr.shape[:2]} != {expected_shape}"
            )
        raw_bgr = oriented_bgr
        native_storage_orientation_applied = True
        native_bgr_sha256 = hashlib.sha256(
            raw_bgr.astype("<u2", copy=False).tobytes(order="C")
        ).hexdigest()
        if (
            config.expected_raw_sha256 == PORTRA400_RAW_SHA256
            and native_bgr_sha256 != PORTRA400_NATIVE_INPUT_SHA256
        ):
            raise ValueError("oriented FULL bytes do not match the native replay input")
        trace.add(
            "FULL storage orientation to RD5X source raster",
            raw_bgr[..., ::-1],
            status="applied_explicit_capture_geometry",
            provenance={
                "native_final_pcb": final_pcb_record,
                "validation": (
                    "pair-specific RAW/TIFF structural correlation and native PCB "
                    "source dimensions"
                ),
                "qualification": (
                    "capture-storage interpretation before RD5X; not the later "
                    "image-filter rotation stage"
                ),
            },
            confidence=(
                "explicit pair orientation validated against the PCB source dimensions; "
                "ScnCrg alone is not an orientation rule"
            ),
            params={
                "numpy_rot90_k": config.rotation_k % 4,
                "native_source_width": native_geometry["source_width"],
                "native_source_height": native_geometry["source_height"],
                "native_bgr_byte_stream_sha256": native_bgr_sha256,
            },
            domain="scanner_u12",
            encoding="u12_code_values_in_uint16",
            expected_range=(0, 4095),
            reference_comparable=False,
        )

    trace.unchanged(
        "CreateImage fused dark-offset / CCD-linear / I-color matrix",
        status="not_executed_upstream_of_input_boundary",
        reason=(
            "The exact software kernel is implemented, but FULL is downstream of scanner "
            "acquisition and the required per-unit 5340-position calibration arrays are absent."
        ),
        confidence="exact kernel; physical per-unit parameters unavailable",
        provenance=_binary_evidence(
            config,
            "CreateImage.dll",
            "sub_10053e60",
            "0x10053e60",
            "u12 saturating offset -> four 4096 LUTs -> signed-Q12 4x4 matrix",
        ),
        params={
            "required": {
                "d_offset": [5340, 4],
                "ccd_linear": [4, 4096],
                "i_color_gain_q12": [5340, 4, 4],
            },
            "implementation": "noritsu.vendor.apply_scanner_preprocess",
        },
    )
    trace.unchanged(
        "CreateImage digital gain (alternative dispatcher operation)",
        status="not_selected_alternative",
        reason=(
            "Dispatcher op4 is an alternative to fused op12, not an additional normal-path "
            "stage; no capture/configuration evidence selects it for this FULL raster."
        ),
        confidence="exact dispatcher relationship",
        provenance=_binary_evidence(
            config,
            "CreateImage.dll",
            "sub_100541d0 / sub_1004fa50",
            "0x100541d0 / 0x1004fa50",
            "per-position signed-Q12 d_gain alternative",
        ),
        params={"implementation": "noritsu.vendor.apply_digital_gain"},
    )
    trace.unchanged(
        "CreateImage BGR/IR misregistration",
        status="blocked_physical_unit_calibration_and_missing_ir_plane",
        reason=(
            "The exact four-lane worker is implemented, but this FULL raster is already "
            "downstream of CreateImage, contains only BGR, and does not carry the scanner's "
            "per-column IR and color displacement maps."
        ),
        confidence="exact parameterized worker; physical per-unit inputs are absent",
        provenance=_binary_evidence(
            config,
            "CreateImage.dll",
            "CIpmBeforeImageConvs::misregMMF2",
            "0x100514b0 (core 0x10051ad3-0x10051cd3)",
            (
                "BGR pass-through plus IR relocation using signed per-column X/Y maps "
                "with clamped coordinates and row-major last-writer collisions"
            ),
        ),
        params={
            "implementation": "noritsu.vendor.apply_ir_misregistration_u16",
            "required_input": "interleaved uint16 B,G,R,IR",
            "required_unit_state": ["pMapIR[x]", "pMap[x]"],
        },
    )

    gain_is_identity = captured_ccd_gain_q12 == [4096, 4096, 4096]
    trace.unchanged(
        "RD5X CCD-gain LUT composition",
        status=(
            "applied_installer_formula_captured_identity"
            if gain_is_identity
            else "blocked_captured_job_parameter"
        ),
        applied=gain_is_identity,
        reason=(
            "Serialized RGB gains are exactly 1.0 and ScnCrg is 1; the recovered "
            "SpecialPcb formula produces three unity Q12 words, so this selected stage "
            "is an exact pixel identity."
            if gain_is_identity
            else "The captured controller values needed to derive all three runtime Q12 gain words are unavailable."
        ),
        confidence=(
            "exact 2015 integer formula and unambiguous captured values; high cross-version confidence"
            if gain_is_identity
            else "exact kernel; captured runtime parameter unavailable"
        ),
        provenance=_binary_evidence(
            config,
            "UniRdImgLib.dll",
            "sub_1005eb10 / FmCCDGain_FilmLUT_Do",
            "0x1005eb10 / 0x1005ff50",
            "compose source FilmLUT by Q12 gain and clamp index to 4095",
        ),
        params={
            "serialized_gain_rgb": (
                prm_capture["scanner_gain_rgb"] if prm_capture is not None else None
            ),
            "scanner_charge": (
                prm_capture["scanner_charge"]["value"]
                if prm_capture is not None
                and prm_capture["scanner_charge"].get("count")
                else None
            ),
            "derived_gain_q12_rgb": captured_ccd_gain_q12,
            "formula": "round(clamp(gain / uint8(ScnCrg) * 4096, 0, 65535))",
            "implementation": "noritsu.vendor.compose_ccd_gain_lut",
        },
    )

    film_lut = tables.load_film_lut(config.group, config.quality_type)
    if captured_ccd_gain_q12 is not None:
        film_lut = np.stack(
            [
                vendor.compose_ccd_gain_lut(film_lut[channel], gain)
                for channel, gain in enumerate(captured_ccd_gain_q12)
            ]
        )
    current = vendor.apply_film_lut(raw_bgr, film_lut)
    del raw_bgr, film_lut
    film_lut_stage = trace.add(
        "RD5X Nega FilmLUT base-table sub-kernel",
        current,
        status=(
            "applied_exact_installer_formula_and_table"
            if captured_ccd_gain_q12 is not None
            else "applied_exact_table_unresolved_gain"
        ),
        provenance={
            "binary": _binary_evidence(
                config,
                "UniRdImgLib.dll",
                "sub_1005ebe0 / FmCCDGain_FilmLUT_Do",
                "0x1005ebe0 / 0x1005ff50",
                "BGR raster lanes consume B/G/R pointers built from RGB payload curves",
            ),
            "table": tables.source_provenance(config.group, config.quality_type),
        },
        confidence=(
            "exact integer table application and derived captured Q12 gains"
            if captured_ccd_gain_q12 is not None
            else "exact table sub-kernel; captured CCD gain unavailable"
        ),
        params={
            "group": config.group,
            "quality_type": config.quality_type,
            "film": "NEGA",
            "format": "135_COL",
            "gain_q12_rgb": captured_ccd_gain_q12,
            "implementation": "noritsu.vendor.apply_film_lut",
        },
        domain="rd5x_working_rgb",
        encoding="uint16_working_rgb",
        expected_range=(0, 65535),
        reference_comparable=False,
    )
    installer_native_stage = film_lut_stage

    ccd_selector = (
        effective_selectors.get("0x12") if effective_selectors is not None else None
    )
    if ccd_selector == 1 and final_pcb_data is not None:
        ccd_parameters = pcb.decode_ccd_noise_s0_parameters(final_pcb_data)
        ccd_response = vendor.build_ccd_noise_response_luts_rgb(
            ccd_parameters["levels_rgb_u8"],
            ccd_parameters["breakpoints_rgb_u16"],
            ccd_parameters["gains_rgb_u8"],
        )
        current = vendor.apply_ccd_noise_clear_s0(
            current,
            ccd_response,
            ccd_parameters["window_sizes_rgb"],
            ccd_parameters["edge_sensitivity_rgb_u8"],
            ccd_parameters["edge_limits_rgb_u8"],
        )
        ccd_stage = trace.add(
            "RD5X CCD noise clear S0",
            current,
            status="applied_exact_native_final_pcb",
            confidence=(
                "complete instruction-matched worker and native SpecialPcb output; "
                "parameter block is byte-identical at preview and full declared sizes"
            ),
            provenance={
                "binary": _binary_evidence(
                    config,
                    "UniRdImgLib.dll",
                    "FmCCDNoiseClearS0_Do",
                    "0x1007f780",
                    "input-processing noise-clear selector +0x12=1",
                ),
                "native_final_pcb": final_pcb_record,
            },
            params={
                "base_selector": base_selectors["0x12"],
                "effective_selector": ccd_selector,
                "decoded_parameters": {
                    key: value.tolist() for key, value in ccd_parameters.items()
                },
                "implementation": [
                    "noritsu.vendor.build_ccd_noise_response_luts_rgb",
                    "noritsu.vendor.apply_ccd_noise_clear_s0",
                ],
            },
            domain="rd5x_working_rgb",
            encoding="uint16_working_rgb",
            expected_range=(0, 65535),
            reference_comparable=False,
        )
        installer_native_stage = ccd_stage
        del ccd_response
    else:
        trace.unchanged(
            "RD5X CCD noise clear S0",
            status=(
                "not_selected_by_effective_selector"
                if ccd_selector == 0
                else "blocked_missing_native_final_pcb"
            ),
            reason=(
                "Native final selector +0x12 is zero."
                if ccd_selector == 0
                else "An exact native final PCB is required to decode the selected S0 controls."
            ),
            confidence="native final selector when available",
            provenance=_binary_evidence(
                config,
                "UniRdImgLib.dll",
                "FmCCDNoiseClearS0_Do",
                "0x1007f780",
                "input-processing noise-clear selector +0x12",
            ),
            params={
                "base_selector": base_selectors["0x12"],
                "effective_selector": ccd_selector,
                "implementation": [
                    "noritsu.vendor.build_ccd_noise_response_luts_rgb",
                    "noritsu.vendor.apply_ccd_noise_clear_s0",
                ],
            },
        )
    trace.unchanged(
        "RD5X CCD noise clear S1",
        status=(
            "not_selected_by_effective_selector"
            if ccd_selector == 1
            else "blocked_dynamic_frame_rewrite"
        ),
        reason=(
            "Effective CCD-noise selector is 1, which chooses S0 rather than S1."
            if ccd_selector == 1
            else "Captured correction mode is unavailable, so the S0/S1 choice is unresolved."
        ),
        confidence="exact selector dispatch",
        provenance=_binary_evidence(
            config,
            "UniRdImgLib.dll",
            "FmCCDNoiseClearS1_Do",
            "0x10080660",
            "unselected alternative input-processing branch",
        ),
    )

    spatial_stages = (
        ("SSU merge / misregistration", "0x14", "merge"),
        ("SSU lens aberration", "0x15", "lens_aberration"),
        ("SSU distortion", "0x16", "distortion"),
        ("SSU trapezoid", "0x17", "trapezoid"),
    )
    for name, selector, prm_key in spatial_stages:
        switch = captured_value("frame_switches", prm_key)
        if switch == 0:
            status = "disabled_by_captured_prm"
            reason = (
                f"Captured FrameInf {prm_key}=0; the output builder clears base "
                f"selector +{selector}."
            )
            confidence = "exact captured switch and builder disable rule"
        else:
            status = "blocked_dynamic_frame_rewrite"
            reason = (
                f"Base selector +{selector}={base_selectors[selector]}, but no decoded "
                "captured FrameInf switch proves its effective state."
            )
            confidence = "base selector exact; captured rewrite unavailable"
        trace.unchanged(
            name,
            status=status,
            reason=reason,
            confidence=confidence,
            provenance=_binary_evidence(
                config,
                "UniRdImgLib.dll",
                "RD5X_DoMainProcForFilm",
                "0x100475f0",
                f"selector +{selector}; captured FrameInf {prm_key}={switch}",
            ),
            params={
                "base_selector": base_selectors[selector],
                "captured_frame_switch": switch,
            },
        )

    grain_selector = (
        effective_selectors.get("0x18") if effective_selectors is not None else None
    )
    trace.unchanged(
        "Film grain / grain reduction S4",
        status=(
            "disabled_by_native_final_pcb"
            if grain_selector == 0
            else "blocked_missing_native_final_pcb_or_com_parameters"
        ),
        reason=(
            "The native post-rewrite selector +0x18 is zero, so the recovered S4 worker is not run."
            if grain_selector == 0
            else "No native final selector/COM variance curve proves a callable S4 job."
        ),
        confidence="authoritative post-rewrite native PCB selector when available",
        provenance=_binary_evidence(
            config,
            "UniRdImgLib.dll",
            "FmGrainS4_Do / RD5X_DoMainProcForFilm",
            "0x100475f0 selector +0x18",
            f"pattern5 base3 -> native effective selector {grain_selector}",
        ),
        params={
            "base_selector": base_selectors["0x18"],
            "captured_mode": captured_mode,
            "effective_selector": grain_selector,
            "captured_grain_adjustment": captured_value("adjustments", "grain"),
            "parameterized_kernel": {
                "implementation": [
                    "noritsu.vendor.build_grain_s4_variance_lut",
                    "noritsu.vendor.apply_grain_s4_pixel_kernel",
                    "noritsu.vendor.apply_grain_s4",
                ],
                "scope": "complete parameterized S4 array worker",
                "fixed_job_status": "not selected" if grain_selector == 0 else "unresolved",
                "captured_com_variance_curve": {
                    "source": grain_curve_record,
                    "shape": (
                        list(grain_curve_data.shape)
                        if grain_curve_data is not None
                        else None
                    ),
                    "minimum": (
                        int(grain_curve_data.min())
                        if grain_curve_data is not None
                        else None
                    ),
                    "maximum": (
                        int(grain_curve_data.max())
                        if grain_curve_data is not None
                        else None
                    ),
                },
            },
        },
    )

    if final_pcb_data is not None and native_storage_orientation_applied:
        scaling_params = pcb.decode_scaling_trimming_parameters(final_pcb_data)
        expected_source = (
            native_geometry["source_height"],
            native_geometry["source_width"],
        )
        if current.shape[:2] != expected_source:
            raise ValueError(
                "working raster no longer matches the native PCB source dimensions"
            )
        if (
            scaling_params["input_width"],
            scaling_params["input_height"],
        ) != tuple(scaling_params["trim"][2:]):
            raise ValueError(
                "native PCB scaling input dimensions do not match its trim rectangle"
            )
        scaling_selector = scaling_params["scaling_selector"]
        # RD5X_DoMainProcForFilm dispatches selector values 1..4 to the S0
        # family unchanged.  Values 5..8 dispatch to S2 with selector-4.
        # The selector is not the family number: the canonical value 2 is
        # therefore the S0 cubic worker with sampling byte 2.
        if 1 <= scaling_selector <= 4:
            scaling_worker = vendor.apply_scaling_s0_u16
            scaling_symbol = "FmScalingS0_Do"
            scaling_address = "0x100d50d0"
            scaling_family = "S0"
        elif 5 <= scaling_selector <= 8:
            scaling_worker = vendor.apply_scaling_s2_u16
            scaling_symbol = "FmScalingS2_Do"
            scaling_address = "0x100d6e30"
            scaling_family = "S2"
        else:
            raise ValueError(
                f"unsupported selected native scaling worker {scaling_selector}"
            )
        current = scaling_worker(
            current,
            scaling_params["output_width"],
            scaling_params["output_height"],
            pre_scale=scaling_params["pre_scale"],
            sampling=scaling_params["sampling"],
            trim=scaling_params["trim"],
        )
        scaling_stage = trace.add(
            "RD5X trimming and Scaling %s" % scaling_family,
            current,
            status="applied_exact_native_final_pcb",
            provenance={
                "binary": _binary_evidence(
                    config,
                    "UniRdImgLib.dll",
                    scaling_symbol,
                    scaling_address,
                    "native +0x178 trim and +0x188 scaling blocks",
                ),
                "native_final_pcb": final_pcb_record,
                "trz": _source_record(config.trz),
            },
            confidence=(
                "complete instruction-matched worker with native builder output at the "
                "Trz-declared nominal full-scan rectangle"
            ),
            params={
                **scaling_params,
                "implementation": (
                    "noritsu.vendor.apply_scaling_s0_u16"
                    if scaling_symbol == "FmScalingS0_Do"
                    else "noritsu.vendor.apply_scaling_s2_u16"
                ),
                "native_dispatch": (
                    "selector 1..4 -> FmScalingS0_Do; "
                    "selector 5..8 -> FmScalingS2_Do(selector-4)"
                ),
                "native_source_geometry": native_geometry,
            },
            domain="rd5x_working_rgb",
            encoding="uint16_working_rgb",
            expected_range=(0, 65535),
            reference_comparable=False,
        )
        installer_native_stage = scaling_stage
    else:
        current, scaling_params = _pre_rotation_crop_and_scale(current, config)
        trace.add(
            "Scaling / trimming to reference raster",
            current,
            status="applied_external_registration",
            provenance={
                "installer_order": _binary_evidence(
                    config,
                    "UniRdImgLib.dll",
                    "FmScalingS0/S2_Do",
                    "RD5X main selector +0x1a/+0x1b",
                    "scaling precedes rotation",
                ),
                "registration_source": _source_record(config.trz),
            },
            confidence=(
                "explicit fallback registration; no native final PCB was supplied"
            ),
            params={
                **scaling_params,
                "available_parameterized_workers": [
                    "noritsu.vendor.apply_trimming_u16",
                    "noritsu.vendor.apply_scaling_s0_u16",
                    "noritsu.vendor.apply_scaling_s2_u16",
                ],
            },
            domain="rd5x_working_rgb",
            encoding="uint16_working_rgb",
            expected_range=(0, 65535),
            reference_comparable=False,
        )

    rotation_node = (
        prm_capture["final_image_filter_rotation_degrees"]
        if prm_capture is not None
        else None
    )
    captured_rotation = (
        rotation_node["value"]
        if rotation_node is not None and rotation_node.get("count")
        else None
    )
    if captured_rotation == 0.0:
        trace.unchanged(
            "RD5X image-filter rotation",
            status=(
                "applied_exact_identity_native_final_pcb"
                if effective_selectors is not None
                else "disabled_by_captured_prm"
            ),
            applied=effective_selectors is not None,
            reason=(
                "Captured ImgFltInf final rotation is exactly 0.0 degrees. The full-size "
                "native PCB keeps the rotation/trim dispatch selected but its transform is "
                "an exact pixel identity."
            ),
            confidence="exact decoded CF field and native post-rewrite selectors",
            provenance=_binary_evidence(
                config,
                "UniRdImgLib.dll",
                "FmRotation_Do",
                "RD5X main selector +0x1c/+0x1d",
                "captured ImgFltInf Deg=0.0",
            ),
            params={
                "base_rotation_selector": base_selectors["0x1c"],
                "base_rotation_trim_selector": base_selectors["0x1d"],
                "effective_rotation_selector": (
                    effective_selectors.get("0x1c")
                    if effective_selectors is not None
                    else None
                ),
                "effective_rotation_trim_selector": (
                    effective_selectors.get("0x1d")
                    if effective_selectors is not None
                    else None
                ),
                "captured_degrees": captured_rotation,
                "available_parameterized_workers": [
                    "noritsu.vendor.apply_rotation_orthogonal_u16",
                    "noritsu.vendor.apply_rotation_arbitrary_u16",
                ],
            },
        )
    else:
        trace.unchanged(
            "RD5X image-filter rotation",
            status="blocked_dynamic_frame_rewrite",
            reason="Captured final image-filter rotation is unavailable or nonzero.",
            confidence="base selector exact; effective transform unresolved",
            params={"captured_degrees": captured_rotation},
        )

    if native_storage_orientation_applied:
        trace.unchanged(
            "External raster orientation for paired export",
            status="not_required_after_storage_orientation",
            reason=(
                "The explicit FULL storage-to-RD5X orientation was applied before "
                "spatial processing; no second rotation is performed."
            ),
            confidence="native PCB dimensions and explicit capture geometry",
            params={"numpy_rot90_k_already_applied": config.rotation_k % 4},
        )
    elif config.rotation_k % 4:
        current = np.ascontiguousarray(np.rot90(current, config.rotation_k))
        trace.add(
            "External raster orientation for paired export",
            current,
            status="applied_explicit_capture_geometry",
            provenance={
                "method": "pair-specific raw/TIFF edge-correlation registration",
                "qualification": "storage/export orientation, not RD5X image-filter rotation",
            },
            confidence="pair-specific orientation independently validated",
            params={
                "numpy_rot90_k": config.rotation_k,
                "warning": "ScnCrg alone does not determine export orientation",
            },
            domain="rd5x_working_rgb",
            encoding="uint16_working_rgb",
            expected_range=(0, 65535),
            reference_comparable=False,
        )
    else:
        trace.unchanged(
            "External raster orientation for paired export",
            status="not_required_by_pair_geometry",
            reason="The explicit pair geometry requests rotation_k=0.",
            confidence="explicit configuration",
        )

    main_sequence = (
        (
            "RGB to YCC S1",
            "0x1e",
            "selected",
            "Effective selector 2 chooses S1. Exact replay requires the final-PCB matrix, control arrays/counts, and signed RGB anchor offsets.",
        ),
        (
            "Automatic contrast S0",
            "0x1f",
            "selected",
            "Selector 1 runs automatic scene analysis; user controls do not supply its generated LUT.",
        ),
        (
            "Film sharp",
            "0x21",
            "selected",
            "Selector 1 runs the base film-sharp algorithm and requires its final 0x154-byte runtime parameter block.",
        ),
        ("Media sharp", "0x24", "disabled", "Effective selector is zero."),
        (
            "YCC correction",
            "0x22",
            "selected",
            "Selector 1 requires the final-PCB piecewise controls from which its Y and chroma tables are built.",
        ),
        (
            "Hue / saturation correction S0",
            "0x23",
            "selected",
            "Selector 1 consumes the generated COM correction lattice even when visible controls are neutral.",
        ),
        ("Color sharp", "0x25", "disabled", "Effective selector is zero."),
        (
            "YCC to RGB with fused gamma",
            "0x26",
            "selected",
            "Selector 1 is active while standalone gamma selector +0x27 is zero; the captured destination type selects its byte-output variant-1 worker.",
        ),
        (
            "Standalone gamma LUT octet/hex",
            "0x27",
            "disabled",
            "Effective selector is zero because gamma is fused into YCC-to-RGB.",
        ),
    )
    parameterized_kernels = {
        "0x1e": {
            "implementation": [
                "noritsu.vendor.build_rgb_to_ycc_luts_u16",
                "noritsu.vendor.apply_rgb_to_ycc_s1",
            ],
            "static_matrix_q10000": list(
                vendor.LS600_G0_TYPE1_RGB_TO_YCC_Q10000
            ),
            "required_runtime_fields": "control arrays/counts and signed RGB anchor offsets",
            "fixed_capture_source": "native final PCB +0x1ac",
        },
        "0x22": {
            "implementation": "noritsu.vendor.apply_ycc_correction",
            "required_runtime_fields": "Y/chroma piecewise controls used to build four runtime tables",
            "fixed_capture_source": "native final PCB +0x506",
        },
        "0x23": {
            "implementation": "noritsu.vendor.apply_hue_saturation_s0",
            "scope": "complete parameterized S0 array worker",
            "fixed_capture_behavior": (
                "captured saturation=100 and zero balance with a nonzero native "
                "9x33x33x3 COM correction lattice"
            ),
            "captured_lattice": hue_lattice_record,
        },
        "0x1f": {
            "implementation": [
                "noritsu.vendor.apply_auto_contrast_s0",
                "noritsu.vendor.expand_auto_contrast_s0_kernel",
                "noritsu.vendor.downsample_auto_contrast_s0_luminance",
                "noritsu.vendor.filter_auto_contrast_s0_local_map",
                "noritsu.vendor.resize_auto_contrast_s0_local_index",
                "noritsu.vendor.build_auto_contrast_s0_luts",
                "noritsu.vendor.apply_auto_contrast_s0_luts",
            ],
            "scope": "complete parameterized S0 luminance array worker",
            "missing_target_fields": "final PCB +0x2ba analysis/filter controls",
        },
        "0x21": {
            "implementation": "noritsu.vendor.apply_film_sharp_full_ycc",
            "scope": "complete parameterized spatial graph, LUT builders, and YCC compositor",
            "target_parameters": "native final PCB +0x3b2, 0x154 bytes",
        },
        "0x26": {
            "implementation": [
                "noritsu.vendor.build_fused_gamma_luts_bgr",
                "noritsu.vendor.apply_ycc_to_rgb_s0_variant1_u8",
                "noritsu.vendor.expand_u8_to_u16",
            ],
            "static_matrix_q10000": list(
                vendor.LS600_G0_TYPE1_YCC_TO_RGB_Q10000
            ),
            "static_variant": vendor.LS600_G0_TYPE1_YCC_TO_RGB_VARIANT,
            "fixed_capture_branch": (
                "byte worker from captured packed descriptor dstType=0 and propagated "
                "Unify depth flag zero"
            ),
        },
    }
    for name, selector, outcome, reason in main_sequence:
        effective_selector = (
            effective_selectors.get(selector)
            if effective_selectors is not None
            else None
        )
        common_provenance = _binary_evidence(
            config,
            "UniRdImgLib.dll",
            (
                "FmHueSatCorrectionS0_Do"
                if selector == "0x23"
                else "RD5X_DoMainProcForFilm"
            ),
            "0x10055c10" if selector == "0x23" else "0x100475f0",
            f"normal-output selector +{selector}={base_selectors[selector]}",
        )
        if selector == "0x1e" and effective_selector == 2 and final_pcb_data is not None:
            rgb_to_ycc_parameters = pcb.decode_rgb_to_ycc_s1_parameters(
                final_pcb_data
            )
            rgb_to_ycc_luts = vendor.build_rgb_to_ycc_luts_u16(
                rgb_to_ycc_parameters["x_controls_rgb"],
                rgb_to_ycc_parameters["y_controls_rgb"],
                rgb_to_ycc_parameters["counts_rgb"],
            )
            current = vendor.apply_rgb_to_ycc_s1(
                current,
                rgb_to_ycc_parameters["matrix_q10000"],
                rgb_to_ycc_parameters["input_offsets_rgb"],
                rgb_to_ycc_luts,
            )
            rgb_to_ycc_stage = trace.add(
                name,
                _ycc_checkpoint_view(current),
                status="applied_exact_native_final_pcb",
                confidence=(
                    "complete instruction-matched S1 worker and native SpecialPcb controls; "
                    "block is byte-identical at preview and full declared sizes"
                ),
                provenance={
                    "binary": _binary_evidence(
                        config,
                        "UniRdImgLib.dll",
                        "FmRgbToYccS1_Do",
                        "0x100872e0",
                        "S1 transfer-LUT setup and RGB-to-C1/C2/Y pixel worker",
                    ),
                    "native_final_pcb": final_pcb_record,
                },
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "decoded_parameters": {
                        key: value.tolist()
                        for key, value in rgb_to_ycc_parameters.items()
                    },
                    "implementation": [
                        "noritsu.vendor.build_rgb_to_ycc_luts_u16",
                        "noritsu.vendor.apply_rgb_to_ycc_s1",
                    ],
                    "checkpoint_lane_mapping": "RGB artifact slots = [Y,C1,C2]",
                },
                domain="rd5x_ycc_visualization",
                encoding="lossless_rgb_slots_y_c1_c2",
                expected_range=(0, 65535),
                reference_comparable=False,
            )
            installer_native_stage = rgb_to_ycc_stage
            del rgb_to_ycc_luts
            continue
        if selector == "0x21" and effective_selector == 1 and final_pcb_data is not None:
            film_sharp_parameters = vendor.decode_film_sharp_parameter_block(
                final_pcb_data,
                offset=vendor.FILM_SHARP_PARAMETER_PCB_OFFSET,
            )
            ycc_to_rgb_parameters = pcb.decode_ycc_to_rgb_gamma_controls(
                final_pcb_data
            )
            current = vendor.apply_film_sharp_full_ycc(
                current,
                final_pcb_data,
                ycc_to_rgb_parameters["matrix_q10000"],
                parameter_offset=vendor.FILM_SHARP_PARAMETER_PCB_OFFSET,
            )
            film_sharp_stage = trace.add(
                name,
                _ycc_checkpoint_view(current),
                status="applied_exact_native_final_pcb",
                confidence=(
                    "complete instruction-matched spatial graph and compositor with the "
                    "canonical full-resolution +0x3b2 parameter block"
                ),
                provenance={
                    "binary": _binary_evidence(
                        config,
                        "UniRdImgLib.dll",
                        "FmFilmSharp_Do",
                        "0x100b9a30",
                        "five spatial filters, directional response, local means, LUT compositor",
                    ),
                    "native_final_pcb": final_pcb_record,
                },
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "parameter_offset": hex(
                        vendor.FILM_SHARP_PARAMETER_PCB_OFFSET
                    ),
                    "parameter_size": vendor.FILM_SHARP_PARAMETER_BLOCK_SIZE,
                    "filter_sizes": list(film_sharp_parameters["filter_sizes"]),
                    "filter_scales": list(film_sharp_parameters["filter_scales"]),
                    "blend_u16": film_sharp_parameters["blend_u16"],
                    "mean_radius_u8": film_sharp_parameters["mean_radius_u8"],
                    "feature_radius_u8": film_sharp_parameters[
                        "feature_radius_u8"
                    ],
                    "implementation": "noritsu.vendor.apply_film_sharp_full_ycc",
                    "checkpoint_lane_mapping": "RGB artifact slots = [Y,C1,C2]",
                },
                domain="rd5x_ycc_visualization",
                encoding="lossless_rgb_slots_y_c1_c2",
                expected_range=(0, 65535),
                reference_comparable=False,
            )
            installer_native_stage = film_sharp_stage
            continue
        if selector == "0x22" and effective_selector == 1 and final_pcb_data is not None:
            ycc_parameters = pcb.decode_ycc_correction_controls(final_pcb_data)
            y_lut, chroma_scale, chroma_offsets = vendor.build_ycc_correction_tables(
                ycc_parameters["y_x_u16"],
                ycc_parameters["y_values_q4_u16"],
                ycc_parameters["y_count"],
                ycc_parameters["chroma_x_u16"],
                ycc_parameters["chroma_offsets_i16"],
                ycc_parameters["chroma_scale_u16"],
                ycc_parameters["chroma_count"],
            )
            current = vendor.apply_ycc_correction(
                current, y_lut, chroma_scale, chroma_offsets
            )
            ycc_correction_stage = trace.add(
                name,
                _ycc_checkpoint_view(current),
                status="applied_exact_native_final_pcb",
                confidence=(
                    "complete instruction-matched control expansion and pixel worker; "
                    "native block is byte-identical at preview and full target sizes"
                ),
                provenance={
                    "binary": _binary_evidence(
                        config,
                        "UniRdImgLib.dll",
                        "FmYccCorrection_Do",
                        "0x100890a0",
                        "Y and chroma table application after native control expansion",
                    ),
                    "native_final_pcb": final_pcb_record,
                },
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "decoded_controls": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in ycc_parameters.items()
                    },
                    "implementation": [
                        "noritsu.vendor.build_ycc_correction_tables",
                        "noritsu.vendor.apply_ycc_correction",
                    ],
                    "checkpoint_lane_mapping": "RGB artifact slots = [Y,C1,C2]",
                },
                domain="rd5x_ycc_visualization",
                encoding="lossless_rgb_slots_y_c1_c2",
                expected_range=(0, 65535),
                reference_comparable=False,
            )
            installer_native_stage = ycc_correction_stage
            del y_lut, chroma_scale, chroma_offsets
            continue
        if (
            selector == "0x23"
            and effective_selector == 1
            and final_pcb_data is not None
            and hue_lattice_data is not None
        ):
            hue_controls = pcb.decode_hue_saturation_s0_controls(final_pcb_data)
            current = vendor.apply_hue_saturation_s0(
                current,
                hue_lattice_data,
                int(hue_controls["saturation_u8"]),
                hue_controls["total_balance_i16"],
            )
            hue_stage = trace.add(
                name,
                _ycc_checkpoint_view(current),
                status="applied_exact_native_com_para",
                confidence=(
                    "complete instruction-matched S0 worker, native final-PCB controls, "
                    "and post-warm COM lattice captured at the selected Unify call"
                ),
                provenance={
                    "binary": common_provenance,
                    "native_final_pcb": final_pcb_record,
                    "native_com_lattice": hue_lattice_record,
                    "capture_boundary": (
                        "ImgCorrectDLL DoUnifyProcForFilm/file IAT RVA 0x2925c4, "
                        "COM_PARA_FOR_ALL+8"
                    ),
                },
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "saturation_u8": int(hue_controls["saturation_u8"]),
                    "total_balance_i16": hue_controls[
                        "total_balance_i16"
                    ].tolist(),
                    "lattice_shape": list(hue_lattice_data.shape),
                    "lattice_minimum": int(hue_lattice_data.min()),
                    "lattice_maximum": int(hue_lattice_data.max()),
                    "lattice_nonzero_values": int(
                        np.count_nonzero(hue_lattice_data)
                    ),
                    "implementation": "noritsu.vendor.apply_hue_saturation_s0",
                    "checkpoint_lane_mapping": "RGB artifact slots = [Y,C1,C2]",
                },
                domain="rd5x_ycc_visualization",
                encoding="lossless_rgb_slots_y_c1_c2",
                expected_range=(0, 65535),
                reference_comparable=False,
            )
            installer_native_stage = hue_stage
            continue
        if selector == "0x26" and effective_selector == 1 and final_pcb_data is not None:
            if decoded_image_descriptor is None:
                raise ValueError(
                    "selected fused-gamma depth requires the captured Unify image descriptor"
                )
            if decoded_image_descriptor["dst_type"] != 0:
                raise ValueError(
                    "fixed audit expects byte fused-gamma branch (captured dstType=0)"
                )
            gamma_parameters = pcb.decode_ycc_to_rgb_gamma_controls(
                final_pcb_data
            )
            gamma_luts_bgr = vendor.build_fused_gamma_luts_bgr(
                gamma_parameters["x_controls_rgb"],
                gamma_parameters["y_controls_rgb_u8"],
                gamma_parameters["counts_rgb"],
            )
            gamma_lut_sha256 = hashlib.sha256(
                gamma_luts_bgr.tobytes(order="C")
            ).hexdigest()
            if config.expected_final_pcb_sha256 == PORTRA400_FINAL_PCB_SHA256 and gamma_lut_sha256 != (
                "b095760adb3c5485520eb0efc523f79c46f39215180a966b62ab1a74a88e410b"
            ):
                raise ValueError("rebuilt fused-gamma LUTs do not match the native oracle")
            rgb_u8 = vendor.apply_ycc_to_rgb_s0_variant1_u8(
                current,
                gamma_parameters["matrix_q10000"],
                gamma_luts_bgr,
            )
            gamma_stage = trace.add(
                name,
                rgb_u8,
                status="applied_exact_native_final_pcb",
                confidence=(
                    "native-conformant natural-cubic LUT builder, exact variant-1 pixel "
                    "worker, and independently captured byte-depth dispatch"
                ),
                provenance={
                    "binary": _binary_evidence(
                        config,
                        "UniRdImgLib.dll",
                        "FmYccToRgbS0_Do / sub_100810c0 / variant1",
                        "0x1008d2a0 / 0x100810c0 / 0x100894e0",
                        "natural-cubic RGB controls -> BGR byte LUTs -> RGB8 output",
                    ),
                    "native_final_pcb": final_pcb_record,
                    "captured_image_descriptor": image_descriptor_record,
                    "decoded_image_descriptor": decoded_image_descriptor,
                    "native_lut_oracle": {
                        "source": gamma_u8_oracle_record,
                        "qualification": (
                            "private DLL builders over packed RGB controls; captured "
                            "channels are identical, while BGR reversal is proven "
                            "separately by wrapper disassembly and synthetic tests"
                        ),
                    },
                    "depth_dispatch_capture": depth_capture_log_record,
                },
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "variant": vendor.LS600_G0_TYPE1_YCC_TO_RGB_VARIANT,
                    "matrix_q10000": gamma_parameters[
                        "matrix_q10000"
                    ].tolist(),
                    "counts_rgb": gamma_parameters["counts_rgb"].tolist(),
                    "gamma_luts_bgr_sha256": gamma_lut_sha256,
                    "dst_type": decoded_image_descriptor["dst_type"],
                    "native_unify_depth_flag": 0,
                    "implementation": [
                        "noritsu.vendor.build_fused_gamma_luts_bgr",
                        "noritsu.vendor.apply_ycc_to_rgb_s0_variant1_u8",
                    ],
                },
                domain="display_rgb",
                encoding="rgb8_codes_expanded_for_trace",
                expected_range=(0, 255),
                reference_comparable=True,
            )
            current = vendor.expand_u8_to_u16(rgb_u8)
            del rgb_u8, gamma_luts_bgr
            expansion_stage = trace.add(
                "RGB8 code expansion to uint16 audit/machine-TIFF container",
                current,
                status="applied_exact_storage_representation",
                confidence="exact integer identity in 8-bit code space",
                provenance={
                    "binary": _binary_evidence(
                        config,
                        "UniRdImgLib.dll",
                        "MdConv8To16_Do",
                        "0x10136ad0 (inner 0x10135770)",
                        "(u8 * 0xffff + 0x80) / 0xff = u8 * 257",
                    ),
                    "qualification": (
                        "lossless storage expansion seen in the machine TIFF; the fixed "
                        "native color worker itself returns RGB8"
                    ),
                },
                params={
                    "formula": "u16 = u8 * 257",
                    "implementation": "noritsu.vendor.expand_u8_to_u16",
                    "gamma_stage_index": gamma_stage["index"],
                },
                domain="display_rgb",
                encoding="rgb8_codes_expanded_for_trace",
                expected_range=(0, 65535),
                reference_comparable=True,
            )
            installer_native_stage = expansion_stage
            continue
        if effective_selector == 0:
            trace.unchanged(
                name,
                status="disabled_by_native_final_pcb",
                reason=(
                    f"The authoritative post-rewrite selector +{selector} is zero; "
                    "the base-table choice is not executed for this job."
                ),
                confidence="native SpecialPcb output",
                provenance=common_provenance,
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "parameterized_kernel": parameterized_kernels.get(selector),
                },
            )
            continue
        if outcome == "selected":
            trace.unchanged(
                name,
                status="blocked_missing_native_runtime_parameters",
                reason=(
                    reason
                    + " The effective native selector or its complete runtime parameter "
                    "artifact was not supplied for this run."
                ),
                confidence="recovered parameterized worker; fixed-job runtime input absent",
                provenance=common_provenance,
                params={
                    "base_selector": base_selectors[selector],
                    "effective_selector": effective_selector,
                    "parameterized_kernel": parameterized_kernels.get(selector),
                    "captured_adjustments": (
                        prm_capture["adjustments"] if prm_capture is not None else None
                    ),
                },
            )
        else:
            trace.unchanged(
                name,
                status=(
                    "disabled_by_effective_selector"
                    if captured_mode == 0
                    else "blocked_dynamic_frame_rewrite"
                ),
                reason=(
                    reason
                    if captured_mode == 0
                    else "Base selector is known, but captured correction mode/frame rewrite is unavailable."
                ),
                confidence=(
                    "exact pattern5/mode0 selector"
                    if captured_mode == 0
                    else "base selector exact; effective selector unresolved"
                ),
                provenance=common_provenance,
                params={
                    "base_selector": base_selectors[selector],
                    "available_gamma_builder": (
                        "noritsu.vendor.make_nega_tone2 (including UO forward/inverse)"
                        if selector == "0x27"
                        else None
                    ),
                },
            )

    optional_after_gamma = (
        (
            "Color grain / color noise",
            "0x28",
            None,
            "disabled_by_mode0_rewrite",
            "Base selector 2 rewrites to effective mode0 selector 0.",
        ),
        (
            "Manual repair / spot correction",
            "0x2e",
            "spot_filter",
            "disabled_by_captured_prm",
            "Captured FrameInf spot-filter switch is zero; builder clears the base selector.",
        ),
        (
            "Manual red-eye / face path",
            "0x2b",
            "red_eye",
            "disabled_by_captured_prm",
            "Captured frame red-eye switch is zero; FaceFind/AtRedEye correction capability remains enabled globally but has no frame region.",
        ),
        (
            "Cross filter",
            "0x2c",
            "cross_filter",
            "disabled_by_captured_prm",
            "Captured FrameInf cross-filter switch is zero.",
        ),
        (
            "Soft filter",
            "0x2d",
            "soft_filter",
            "disabled_by_captured_prm",
            "Captured FrameInf soft-filter switch is zero.",
        ),
        (
            "Blank color / output fill",
            "0x29",
            None,
            "blocked_missing_native_runtime_parameters",
            "The parameterized Blank workers are recovered, but a run without a final PCB does not supply the +0x836 fill rectangle/color block.",
        ),
        (
            "MediaSharp NN branch",
            None,
            None,
            "disabled_by_effective_selector",
            "Separate EgSW argument is zero in the normal film-output path.",
        ),
    )
    for name, selector, prm_key, status, reason in optional_after_gamma:
        native_selector = (
            effective_selectors.get(selector)
            if effective_selectors is not None and selector is not None
            else None
        )
        switch = (
            captured_value("frame_switches", prm_key) if prm_key is not None else None
        )
        if native_selector == 0:
            status = "disabled_by_native_final_pcb"
            reason = (
                f"The authoritative post-rewrite selector +{selector} is zero."
            )
        if name == "Color grain / color noise" and captured_mode != 0:
            status = "blocked_dynamic_frame_rewrite"
            reason = "Captured correction mode is unavailable; base selector 2 rewrite is unresolved."
        if status == "disabled_by_captured_prm" and switch is None:
            status = "blocked_dynamic_frame_rewrite"
            reason = "Captured FrameInf switch is unavailable; base selector rewrite is unresolved."
        trace.unchanged(
            name,
            status=status,
            reason=reason,
            confidence="branch order, base selector, and decoded frame switch where present",
            provenance=_binary_evidence(
                config,
                "UniRdImgLib.dll",
                "RD5X_DoMainProcForFilm",
                "0x100475f0",
                f"post-gamma branch selector {selector}; captured switch={switch}",
            ),
            params={
                "base_selector": base_selectors.get(selector) if selector else 0,
                "effective_selector": native_selector,
                "captured_frame_switch": switch,
                "parameterized_kernel": (
                    {
                        "implementation": [
                            "noritsu.vendor.apply_blank_oct_color_u8",
                            "noritsu.vendor.apply_blank_hex_color_u16",
                        ],
                        "scope": (
                            "complete canvas fill/placement workers, including the exact "
                            "rotation-break halo and 3x3 boundary-smoothing branch"
                        ),
                        "required_runtime_fields": "final PCB +0x836 canvas/placement/color/postbreak block",
                    }
                    if name == "Blank color / output fill"
                    else None
                ),
                "global_face_find": (
                    captured_value("correction_switches", "face_find")
                    if "red-eye" in name
                    else None
                ),
                "global_automatic_red_eye": (
                    captured_value("correction_switches", "automatic_red_eye")
                    if "red-eye" in name
                    else None
                ),
            },
        )

    no_ice_reference = (
        config.reference is not None and "no-ice" in str(config.reference).lower()
    )
    trace.unchanged(
        "Digital ICE / IR dust masking",
        status=(
            "disabled_by_reference_capture"
            if no_ice_reference
            else "blocked_hardware_ir_plane"
        ),
        reason=(
            "The paired target path explicitly identifies the no-ICE export."
            if no_ice_reference
            else "A three-channel FULL raster has no fourth IR plane or Kizu mask with which to replay dust repair."
        ),
        confidence=(
            "explicit reference-corpus designation"
            if no_ice_reference
            else "scanner data requirement exact; capture selection unknown"
        ),
        provenance={"reference": reference_record},
    )
    trace.unchanged(
        "CMS / NKC-ICCS",
        status=(
            "disabled_by_captured_prm"
            if captured_cms_switch == 0
            else "blocked_external_legacy_controller_state"
        ),
        reason=(
            "Captured CmsInf.CmsSW has count=1,value=0, so RD5X skips CMS and no "
            "profile payload is consumed."
            if captured_cms_switch == 0
            else "The CMS engine and its post-main order are recovered, but this "
            "capture does not provide an effective CmsSW selection."
        ),
        confidence=(
            "captured switch and skip branch exact"
            if captured_cms_switch == 0
            else "engine/order exact; effective selection unresolved"
        ),
        provenance={
            **_binary_evidence(
                config,
                "UniRdImgLib.dll",
                "RD5X_DoCms",
                "0x100925d0",
                "CMS runs after film main processing",
            ),
            "captured_prm_offset": "CmsInf.CmsSW count/value at 0x0ff0/0x0ff4",
            "consumer_chain": (
                "ImgCorrectDLL CreateOutputForFilm 0x100fcc64 -> operation field; "
                "RD5X_DoUnifyProcForFilm 0x10145052"
            ),
        },
        params={
            "captured_cms_switch": captured_cms_switch,
            "captured_prm_root": (
                prm_capture["validation"]["root_name"]
                if prm_capture is not None
                else None
            ),
            "installed_loader_compatible": (
                prm_capture["installed_loader_compatible"]
                if prm_capture is not None
                else None
            ),
        },
    )

    boundary_stage = trace.stages[-1]
    boundary_checkpoint = output_dir / boundary_stage["artifacts"]["checkpoint_tiff"]
    _link_or_copy(boundary_checkpoint, output_dir / "recovered_boundary.tif")
    _link_or_copy(boundary_checkpoint, output_dir / "final.tif")
    target = (
        _read_reference(config.reference.resolve())
        if config.reference is not None
        else None
    )
    if target is None:
        boundary_comparison = {
            "status": "not_available",
            "reason": "no machine reference supplied",
        }
    elif current.shape != target.shape:
        boundary_comparison = {
            "status": "shape_mismatch",
            "current_shape": list(current.shape),
            "reference_shape": list(target.shape),
            "reason": "reference diagnostic requires aligned equal-size frames",
        }
    else:
        comparison_output = write_comparison_artifacts(
            current,
            target,
            output_dir / "boundary_comparison",
            prefix="recovered_boundary_vs_machine",
        )
        boundary_comparison = comparison_output["metrics"]
    native_engine_target = (
        _read_reference(config.native_reference.resolve())
        if config.native_reference is not None
        and config.native_reference.resolve().is_file()
        else None
    )
    native_comparison_output = None
    native_machine_output = None
    if native_engine_target is None:
        native_engine_comparison = {
            "status": "not_available",
            "reason": "no native installer-engine output supplied",
        }
    elif current.shape != native_engine_target.shape:
        native_engine_comparison = {
            "status": "shape_mismatch",
            "current_shape": list(current.shape),
            "reference_shape": list(native_engine_target.shape),
            "reason": "native-engine diagnostic requires equal-size frames",
        }
    else:
        native_comparison_output = write_comparison_artifacts(
            current,
            native_engine_target,
            output_dir / "native_engine_comparison",
            prefix="recovered_boundary_vs_native_engine",
        )
        native_engine_comparison = native_comparison_output["metrics"]
    if (
        native_engine_target is not None
        and target is not None
        and native_engine_target.shape == target.shape
    ):
        native_machine_output = write_comparison_artifacts(
            native_engine_target,
            target,
            output_dir / "native_engine_vs_machine",
            prefix="native_engine_vs_machine",
        )
    del native_engine_target
    if final_pcb_data is not None:
        native_boundary_stage = boundary_stage
        native_boundary_comparison = boundary_comparison
        native_qualification = (
            "Last recovered fixed-job output after native SpecialPcb selectors, "
            "parameters, and Trz-nominal geometry. The initial FULL storage orientation "
            "is an explicit pair interpretation, not an RD5X color operation."
        )
    else:
        native_boundary_stage = installer_native_stage
        native_boundary_comparison = {
            "status": "not_available",
            "reason": "native raster precedes fallback external registration",
        }
        native_qualification = (
            "Last executed installer-derived stage before fallback external registration."
        )
    native_boundary_checkpoint = (
        output_dir / native_boundary_stage["artifacts"]["checkpoint_tiff"]
    )
    _link_or_copy(
        native_boundary_checkpoint,
        output_dir / "installer_color_boundary_native.tif",
    )
    installer_native_boundary = {
        "stage_index": native_boundary_stage["index"],
        "stage_name": native_boundary_stage["name"],
        "checkpoint": "installer_color_boundary_native.tif",
        "shape": native_boundary_stage["dimensions"]["shape"],
        "pixel_hash": native_boundary_stage["pixel_hash"],
        "qualification": native_qualification,
        "comparison": native_boundary_comparison,
    }
    _write_json(
        output_dir / "installer_color_boundary_native_metrics.json",
        installer_native_boundary,
    )
    recovered_boundary = {
        "stage_index": boundary_stage["index"],
        "stage_name": boundary_stage["name"],
        "checkpoint": "recovered_boundary.tif",
        "qualification": (
            "Recovered installer pipeline at the native Trz-nominal crop and S0 cubic scaling; "
            "the only non-vendor spatial interpretation is the explicit FULL storage "
            "orientation. Direct code-unit comparison assumes the working RGB and machine "
            "TIFF ranges are numerically comparable; use luma NCC primarily for structural "
            "validation."
            if final_pcb_data is not None
            else "Fallback comparison boundary after external crop, OpenCV resampling, and orientation."
        ),
        "native_installer_boundary": installer_native_boundary,
        "comparison": boundary_comparison,
        "native_engine_comparison": native_engine_comparison,
        "native_engine_vs_machine": (
            native_machine_output["metrics"]
            if native_machine_output is not None
            else {"status": "not_available"}
        ),
        "native_engine_visual_artifacts": (
            {
                key: f"native_engine_comparison/{value}"
                for key, value in native_comparison_output["artifacts"].items()
            }
            if native_comparison_output is not None
            else None
        ),
        "native_engine_vs_machine_visual_artifacts": (
            {
                key: f"native_engine_vs_machine/{value}"
                for key, value in native_machine_output["artifacts"].items()
            }
            if native_machine_output is not None
            else None
        ),
        "visual_artifacts": (
            {
                key: f"boundary_comparison/{value}"
                for key, value in comparison_output["artifacts"].items()
            }
            if boundary_comparison.get("status") == "ok"
            else None
        ),
    }
    _write_json(output_dir / "recovered_boundary_metrics.json", recovered_boundary)

    if config.fit_external_reference:
        assert target is not None
        if current.shape != target.shape:
            raise ValueError(
                f"aligned working frame {current.shape} does not match reference {target.shape}"
            )
        fitted_lut = vendor.fit_monotone_reference_lut(
            current, target, sample_step=config.fit_sample_step
        )
        lut_path = output_dir / "external_reference_monotone_lut.npy"
        np.save(lut_path, fitted_lut, allow_pickle=False)
        lut_record = _source_record(lut_path)
        current = vendor.apply_rgb_lut_u16(current, fitted_lut)
        del fitted_lut
        trace.add(
            "External legacy-reference residual fit (not a Noritsu stage)",
            current,
            status="applied_external_reference_measurement",
            provenance={
                "reference": reference_record,
                "method": "per-channel binned isotonic regression on aligned pixels",
                "lut": lut_record,
            },
            confidence="exactly reproducible in-sample measurement; not vendor attribution",
            params={
                "sample_step": config.fit_sample_step,
                "implementation": "noritsu.vendor.fit_monotone_reference_lut",
                "limitation": "absorbs unresolved composite tone/color response only; no spatial model",
            },
            domain="display_rgb",
            encoding="display_rgb_u16",
            expected_range=(0, 65535),
            reference_comparable=True,
        )
        unquantized_checkpoint = (
            output_dir / trace.stages[-1]["artifacts"]["checkpoint_tiff"]
        )
        _link_or_copy(
            unquantized_checkpoint, output_dir / "external_fit_unquantized.tif"
        )

        current = vendor.quantize_oct_to_hex(current)
        trace.add(
            "Observed machine 8-bit precision expanded to 16-bit",
            current,
            status="applied_observed_output_encoding",
            provenance={
                "observation": "all paired machine TIFF samples are exact multiples of 257",
                "reference": reference_record,
                "qualification": (
                    "This second quantization follows only the external diagnostic fit. "
                    "The recovered fixed-job path already used the captured dstType=0 "
                    "byte YCC-to-RGB worker and exact x257 container expansion."
                ),
            },
            confidence="exact observation and deterministic conversion; output boundary not symbol-mapped",
            params={"formula": "round(u16 * 255 / 65535) * 257"},
            domain="display_rgb",
            encoding="display_rgb_u16",
            expected_range=(0, 65535),
            reference_comparable=True,
        )
        quantized_checkpoint = (
            output_dir / trace.stages[-1]["artifacts"]["checkpoint_tiff"]
        )
        _link_or_copy(quantized_checkpoint, output_dir / "external_fit.tif")
    del target

    manifest = trace.finalize()

    recovery = {
        "installer_color_boundary_native": installer_native_boundary,
        "recovered_boundary": recovered_boundary,
        "implemented_kernels": [
            "CreateImage op12 dark offset / CCD linear / 4x4 matrix and separate op4 digital gain",
            "RD5X Q12 CCD-gain LUT composition",
            "RD5X BGR-to-RGB Nega FilmLUT application",
            "RD5X complete parameterized CCD-noise S0 array worker",
            "RD5X complete parameterized Grain S4 array worker",
            "RD5X complete parameterized AutoContrast S0 luminance array worker",
            "RD5X complete parameterized Hue/Saturation S0 array worker",
            "RD5X complete FilmSharp spatial graph, LUT builders, and final YCC compositor",
            "RD5X trimming, S0 cubic/S2 bilinear scaling, and orthogonal/arbitrary rotation workers",
            "RD5X BlankOct/BlankHex canvas fill, placement, rotation-break halo, and boundary smoothing workers",
            "RD5X RGB-to-YCC S1 transfer-LUT builder and float32 pixel kernel",
            "RD5X YCC-correction control-table builders and pixel worker",
            "RD5X fused-gamma LUT builders plus YCC-to-RGB S0 variant-1 byte/16-bit workers",
            "SpecialPcb complete makeNegaTone2 including UO forward/inverse tables and MakeGammaCurve2",
            "RGB uint16 LUT application",
            "observed 8-bit-to-16-bit output quantization",
        ],
        "stage_outcomes": [
            {
                "index": stage["index"],
                "name": stage["name"],
                "status": stage["status"],
                "applied": stage["applied"],
                "reason": stage["skipped_reason"],
            }
            for stage in manifest["stages"]
        ],
        "remaining_gap_boundary": {
            "local_software_conformance": (
                []
                if native_engine_comparison.get("status") == "ok"
                and native_engine_comparison["overall"]["mae"] == 0.0
                else [
                    {
                        "stage_scope": "recovered fixed-job chain versus installed native oracle",
                        "classification": "not hardware; must be isolated before declaring recovery complete",
                        "metrics": native_engine_comparison,
                    }
                ]
            ),
            "physical_hardware": [
                "per-unit 5340-position offset/linear/color calibration",
                "scanner shading/misregistration/defect maps",
                "fourth-channel IR capture or Kizu mask for dust repair",
            ],
            "external_legacy_reference_or_controller_state": [
                "the original 2017 controller binaries or same-version internal checkpoints for byte-identical attribution of the observed 2015/native-versus-2017-machine residual",
                "named film-stock spectral mappings, which are not present in CorrectServ and require an external legacy reference or physical film/scanner measurements",
                "separable legacy-machine A/B captures for optional controls that are disabled in this fixed job",
            ],
        },
        "selected_stages_blocked_by_external_legacy_controller_state": [
            stage["name"]
            for stage in manifest["stages"]
            if stage["status"]
            in {
                "blocked_missing_native_runtime_parameters",
                "blocked_external_legacy_controller_state",
            }
        ],
    }
    _write_json(output_dir / "recovery_report.json", recovery)
    _stage_report(manifest, output_dir / "report.md", recovered_boundary)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m noritsu.audit",
        description="Run an evidence-labelled, stage-by-stage Noritsu installer audit.",
    )
    parser.add_argument("raw", nargs="?", type=Path)
    parser.add_argument("--preset", choices=("portra400-frame1",))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-width", type=int)
    parser.add_argument("--raw-height", type=int)
    parser.add_argument("--group", type=int, default=0, choices=(0, 1))
    parser.add_argument("--quality-type", type=int, default=1, choices=(0, 1))
    parser.add_argument("--rotation-k", type=int, default=1)
    parser.add_argument("--crop-y", type=int, default=17)
    parser.add_argument("--crop-x", type=int, default=171)
    parser.add_argument("--crop-height", type=int, default=4008)
    parser.add_argument("--crop-width", type=int, default=6048)
    parser.add_argument("--output-height", type=int, default=4011)
    parser.add_argument("--output-width", type=int, default=6048)
    parser.add_argument("--fit-sample-step", type=int, default=2)
    parser.add_argument("--no-reference-fit", action="store_true")
    parser.add_argument("--prm", type=Path)
    parser.add_argument("--final-pcb", type=Path)
    parser.add_argument("--native-reference", type=Path)
    parser.add_argument("--hue-lattice", type=Path)
    parser.add_argument("--grain-curve", type=Path)
    parser.add_argument("--image-descriptor", type=Path)
    parser.add_argument("--gamma-u8-oracle", type=Path)
    parser.add_argument("--depth-capture-log", type=Path)
    parser.add_argument("--trz", type=Path)
    parser.add_argument("--settings-screenshot", type=Path)
    parser.add_argument("--re-root", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    raw = args.raw
    reference = args.reference
    prm = args.prm
    final_pcb = args.final_pcb
    native_reference = args.native_reference
    hue_lattice = args.hue_lattice
    grain_curve = args.grain_curve
    image_descriptor = args.image_descriptor
    gamma_u8_oracle = args.gamma_u8_oracle
    depth_capture_log = args.depth_capture_log
    trz = args.trz
    settings = args.settings_screenshot
    re_root = args.re_root
    expected_raw_sha256 = None
    expected_reference_sha256 = None
    expected_prm_sha256 = None
    expected_final_pcb_sha256 = None
    expected_native_reference_sha256 = None
    expected_hue_lattice_sha256 = None
    expected_grain_curve_sha256 = None
    expected_image_descriptor_sha256 = None
    expected_gamma_u8_oracle_sha256 = None
    expected_depth_capture_log_sha256 = None
    if args.preset == "portra400-frame1":
        preset_needed = (
            raw is None
            or reference is None
            or prm is None
            or final_pcb is None
            or native_reference is None
            or hue_lattice is None
            or grain_curve is None
            or image_descriptor is None
            or gamma_u8_oracle is None
            or depth_capture_log is None
            or trz is None
            or settings is None
            or re_root is None
        )
        preset = _portra400_preset_paths() if preset_needed else {}
        raw = raw or preset.get("raw")
        reference = reference or preset.get("reference")
        prm = prm or preset.get("prm")
        final_pcb = final_pcb or preset.get("final_pcb")
        native_reference = native_reference or preset.get("native_reference")
        hue_lattice = hue_lattice or preset.get("hue_lattice")
        grain_curve = grain_curve or preset.get("grain_curve")
        image_descriptor = image_descriptor or preset.get("image_descriptor")
        gamma_u8_oracle = gamma_u8_oracle or preset.get("gamma_u8_oracle")
        depth_capture_log = depth_capture_log or preset.get("depth_capture_log")
        trz = trz or preset.get("trz")
        settings = settings or preset.get("settings")
        re_root = re_root or preset.get("re_root")
        expected_raw_sha256 = PORTRA400_RAW_SHA256
        expected_reference_sha256 = PORTRA400_REFERENCE_SHA256
        expected_prm_sha256 = PORTRA400_PRM_SHA256
        expected_final_pcb_sha256 = PORTRA400_FINAL_PCB_SHA256
        expected_native_reference_sha256 = PORTRA400_NATIVE_REFERENCE_SHA256
        expected_hue_lattice_sha256 = PORTRA400_HUE_LATTICE_SHA256
        expected_grain_curve_sha256 = PORTRA400_GRAIN_CURVE_SHA256
        expected_image_descriptor_sha256 = PORTRA400_IMAGE_DESCRIPTOR_SHA256
        expected_gamma_u8_oracle_sha256 = PORTRA400_GAMMA_U8_ORACLE_SHA256
        expected_depth_capture_log_sha256 = PORTRA400_DEPTH_CAPTURE_LOG_SHA256
    if raw is None:
        raise SystemExit("raw path is required unless --preset supplies it")

    config = AuditConfig(
        raw=raw,
        reference=reference,
        output_dir=args.output_dir,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        group=args.group,
        quality_type=args.quality_type,
        rotation_k=args.rotation_k,
        rotated_crop_y=args.crop_y,
        rotated_crop_x=args.crop_x,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        output_height=args.output_height,
        output_width=args.output_width,
        fit_external_reference=not args.no_reference_fit,
        fit_sample_step=args.fit_sample_step,
        prm=prm,
        final_pcb=final_pcb,
        native_reference=native_reference,
        hue_lattice=hue_lattice,
        grain_curve=grain_curve,
        image_descriptor=image_descriptor,
        gamma_u8_oracle=gamma_u8_oracle,
        depth_capture_log=depth_capture_log,
        trz=trz,
        settings_screenshot=settings,
        reverse_engineering_root=re_root,
        expected_raw_sha256=expected_raw_sha256,
        expected_reference_sha256=expected_reference_sha256,
        expected_prm_sha256=expected_prm_sha256,
        expected_final_pcb_sha256=expected_final_pcb_sha256,
        expected_native_reference_sha256=expected_native_reference_sha256,
        expected_hue_lattice_sha256=expected_hue_lattice_sha256,
        expected_grain_curve_sha256=expected_grain_curve_sha256,
        expected_image_descriptor_sha256=expected_image_descriptor_sha256,
        expected_gamma_u8_oracle_sha256=expected_gamma_u8_oracle_sha256,
        expected_depth_capture_log_sha256=expected_depth_capture_log_sha256,
    )
    manifest = run_audit(config)
    print(
        f"wrote {len(manifest['stages'])} stage checkpoints to {config.output_dir.resolve()}"
    )
    with (config.output_dir.resolve() / "recovered_boundary_metrics.json").open(
        encoding="utf-8"
    ) as handle:
        boundary = json.load(handle)
    comparison = boundary["comparison"]
    if comparison.get("status") == "ok":
        print(
            "recovered-boundary diagnostic MAE %.8f, RMSE %.8f, luma NCC %s"
            % (
                comparison["overall"]["mae"],
                comparison["overall"]["rmse"],
                comparison["luma_ncc"],
            )
        )
    external_stage = next(
        (
            stage
            for stage in reversed(manifest["stages"])
            if stage["status"] == "applied_observed_output_encoding"
        ),
        None,
    )
    external = (
        external_stage["difference_from_reference"]
        if external_stage is not None
        else None
    )
    if external is not None and external.get("status") == "ok":
        print(
            "separate external-fit diagnostic MAE %.8f, RMSE %.8f, luma NCC %s"
            % (
                external["overall"]["mae"],
                external["overall"]["rmse"],
                external["luma_ncc"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
