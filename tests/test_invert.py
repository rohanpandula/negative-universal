from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from array import array
from pathlib import Path

from negative_universal import invert, probe
from negative_universal.cli import (
    NegativeUniversalError,
    PgmImage,
    invert_triplets,
    main,
    read_pgm,
    sha256_file,
    write_pgm,
)

WIDTH = 64
HEIGHT = 48


def write_capture(
    root: Path,
    name: str,
    *,
    fill: tuple[int, int, int] = (40000, 42000, 30000),
    border: tuple[int, int, int] | None = None,
    band: int = 5,
    gradient: float = 0.0,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Path:
    """Write a synthetic `normalized_transmittance` capture.

    `gradient` scales a left-to-right brightness ramp (0 = flat), `border`
    paints the outer `band` pixels, so a flat border plus a busy centre can be
    built without any real film.
    """
    directory = root / name
    directory.mkdir()
    frames = {}
    for index, channel in enumerate(("R", "G", "B")):
        pixels = array("H")
        for y in range(height):
            for x in range(width):
                value = fill[index]
                if gradient:
                    value = int(value * (1.0 + gradient * (x / (width - 1) - 0.5)))
                if border is not None and (
                    x < band or y < band or x >= width - band or y >= height - band
                ):
                    value = border[index]
                pixels.append(max(0, min(65535, value)))
        path = directory / f"{channel}.pgm"
        write_pgm(path, PgmImage(width, height, 65535, pixels))
        frames[channel] = {
            "file": path.name,
            "width": width,
            "height": height,
            "max_value": 65535,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": 1,
        "kind": "normalized_transmittance",
        "created_at": "2026-01-01T00:00:00+00:00",
        "channel_order": ["R", "G", "B"],
        "rig_metadata": {"sensor_layout": "mono"},
        "frames": frames,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def load(directory: Path) -> dict[str, PgmImage]:
    return {
        channel: read_pgm(directory / f"{channel}.pgm") for channel in ("R", "G", "B")
    }


class DensityTableTests(unittest.TestCase):
    def test_endpoints_and_decades(self) -> None:
        table = invert.build_density_table()
        self.assertAlmostEqual(table[65535], 0.0, places=6)
        self.assertAlmostEqual(table[6553], 1.0, places=2)
        self.assertAlmostEqual(table[0], 4.0, places=6)

    def test_table_is_monotone_falling(self) -> None:
        table = invert.build_density_table()
        for index in (1, 100, 1000, 30000, 65535):
            self.assertGreaterEqual(table[index - 1], table[index])


class LookTests(unittest.TestCase):
    def test_endpoints_exact_and_monotone(self) -> None:
        look = invert.LookParams(white_density=0.2, black_density=2.5)
        lut = invert.build_display_lut(look, max_value=1000)
        self.assertEqual(lut[0], 0.0)
        self.assertEqual(lut[-1], 1000.0)
        for index in range(1, len(lut)):
            self.assertGreaterEqual(lut[index], lut[index - 1] - 1e-9)

    def test_white_is_black_and_black_is_white(self) -> None:
        look = invert.LookParams(white_density=0.3, black_density=2.0)
        lut = invert.build_display_lut(look)
        self.assertEqual(lut[0], 0.0)
        index = int(invert.LUT_SIZE * (0.2 - invert.DENSITY_FLOOR) / invert.DENSITY_SPAN)
        self.assertEqual(lut[index], 0.0)
        self.assertEqual(lut[invert.LUT_SIZE], 65535.0)

    def test_auto_range_orders_endpoints(self) -> None:
        samples = {"R": [-0.4] * 5 + [0.3] * 40 + [1.4] * 5}
        white, black = invert.auto_range(samples)
        self.assertLess(white, black)
        self.assertGreaterEqual(white, -0.5)
        self.assertLessEqual(black, invert.DENSITY_CEILING)

    def test_auto_range_keeps_a_minimum_span(self) -> None:
        white, black = invert.auto_range({"R": [0.4] * 100, "G": [0.4] * 100})
        self.assertGreaterEqual(black - white, 0.8)

    def test_auto_range_needs_samples(self) -> None:
        with self.assertRaises(invert.InvertError):
            invert.auto_range({"R": []})

    def test_rejects_inverted_densities(self) -> None:
        with self.assertRaises(invert.InvertError):
            invert.LookParams(white_density=2.0, black_density=1.0).validate()


class ProbeTests(unittest.TestCase):
    def test_prefers_neutral_border_over_tinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = write_capture(
                root, "frame", fill=(12000, 18000, 9000), border=(42000, 42000, 42000)
            )
            images = load(capture)
            # tint the top band so it is bright but not neutral
            for channel, value in zip(("R", "G", "B"), (42000, 60000, 30000)):
                image = images[channel]
                for y in range(5):
                    for x in range(WIDTH):
                        image.pixels[y * WIDTH + x] = value
                write_pgm(capture / f"{channel}.pgm", image)
            estimate = probe.probe_base(images)
            neutral = [patch for patch in estimate.chosen if patch.chroma < 0.02]
            self.assertGreaterEqual(len(neutral), 3)
            # a tinted top band still cannot drag the median off neutral
            for base in estimate.base:
                self.assertAlmostEqual(base, 42000 / 65535, delta=0.001)

    def test_roi_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = write_capture(Path(temporary_directory), "frame")
            estimate = probe.probe_base(load(capture), roi=(10, 8, 20, 12))
            self.assertEqual(estimate.source, "roi")
            self.assertEqual(len(estimate.chosen), 1)
            self.assertAlmostEqual(estimate.base[0], 40000 / 65535, delta=0.005)

    def test_roi_outside_frame_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = write_capture(Path(temporary_directory), "frame")
            with self.assertRaises(probe.ProbeError):
                probe.probe_base(load(capture), roi=(60, 40, 20, 20))

    def test_black_frame_has_no_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = write_capture(Path(temporary_directory), "frame", fill=(0, 0, 0))
            with self.assertRaises(probe.ProbeError):
                probe.probe_base(load(capture))

    def test_plane_follows_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = write_capture(
                Path(temporary_directory), "frame", gradient=0.6
            )
            estimate = probe.probe_base(load(capture))
            for _, x_slope, y_slope in estimate.plane:
                self.assertAlmostEqual(x_slope, 0.3, delta=0.1)
                self.assertAlmostEqual(y_slope, 0.0, delta=0.1)

    def test_flat_frame_has_flat_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture = write_capture(Path(temporary_directory), "frame")
            estimate = probe.probe_base(load(capture))
            for offset, x_slope, y_slope in estimate.plane:
                self.assertAlmostEqual(offset, 1.0, delta=0.005)
                self.assertAlmostEqual(x_slope, 0.0, delta=0.005)
                self.assertAlmostEqual(y_slope, 0.0, delta=0.005)


class RenderTests(unittest.TestCase):
    def _estimate(self) -> probe.BaseEstimate:
        patch = probe.Patch(0, 0, 4, 4, (0.5, 0.5, 0.5), 0.0, 0.5)
        return probe.BaseEstimate(
            base=(0.5, 0.5, 0.5),
            plane=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            patches=(patch,),
            chosen=(patch,),
            source="roi",
        )

    def _images(self, values: tuple[int, int, int]) -> dict[str, PgmImage]:
        return {
            channel: PgmImage(2, 2, 65535, array("H", [value] * 4))
            for channel, value in zip(("R", "G", "B"), values)
        }

    def test_base_prints_black(self) -> None:
        outputs, _ = invert.render(self._images((32768,) * 3), self._estimate(), invert.RenderParams())
        self.assertEqual(list(outputs["R"]), [0, 0, 0, 0])

    def test_density_offset_brightens(self) -> None:
        images = self._images((12000,) * 3)
        dark, _ = invert.render(images, self._estimate(), invert.RenderParams())
        lifted, _ = invert.render(
            images, self._estimate(), invert.RenderParams(density_offset=0.4)
        )
        self.assertGreater(sum(lifted["R"]), sum(dark["R"]))

    def test_gray_offsets_split_channels(self) -> None:
        outputs, _ = invert.render(
            self._images((12000,) * 3),
            self._estimate(),
            invert.RenderParams(gray=(0.4, 0.0, -0.4)),
        )
        self.assertGreater(sum(outputs["R"]), sum(outputs["G"]))
        self.assertGreater(sum(outputs["G"]), sum(outputs["B"]))

    def test_neutral_pull_desaturates(self) -> None:
        images = self._images((3000, 30000, 12000))
        raw, _ = invert.render(images, self._estimate(), invert.RenderParams())
        pulled, _ = invert.render(
            images, self._estimate(), invert.RenderParams(neutral_pull=1.0)
        )
        spread = lambda out: max(out["R"][0], out["G"][0], out["B"][0]) - min(
            out["R"][0], out["G"][0], out["B"][0]
        )
        self.assertLess(spread(pulled), spread(raw))

    def test_statistics_count_clipped_transmittance(self) -> None:
        _, stats = invert.render(
            self._images((65535, 65535, 65535)), self._estimate(), invert.RenderParams()
        )
        self.assertEqual(stats.pixels, 4)
        self.assertLessEqual(stats.density_floor, 0.0)

    def test_rejects_mismatched_geometry(self) -> None:
        images = self._images((12000,) * 3)
        images["R"] = PgmImage(3, 1, 65535, array("H", [1, 2, 3]))
        with self.assertRaises(invert.InvertError):
            invert.render(images, self._estimate(), invert.RenderParams())


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_invert_writes_manifest_and_render(self) -> None:
        capture = write_capture(self.root, "frame")
        output = self.root / "positive"
        invert_triplets(normalized=capture, output=output)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["kind"], "positive_render")
        self.assertEqual(manifest["output_encoding"], "srgb")
        self.assertEqual(manifest["base"]["source"], "border")
        self.assertIn("log10", manifest["formula"]["density"])
        for channel in ("R", "G", "B"):
            record = manifest["frames"][channel]
            path = output / record["file"]
            self.assertEqual(sha256_file(path), record["sha256"])
            self.assertEqual(read_pgm(path).max_value, 65535)
        ppm = (output / "positive.ppm").read_bytes()
        self.assertTrue(ppm.startswith(b"P6\n64 48\n65535\n"))
        self.assertEqual(len(ppm), len(b"P6\n64 48\n65535\n") + WIDTH * HEIGHT * 6)

    def test_auto_range_is_recorded(self) -> None:
        capture = write_capture(self.root, "frame")
        output = self.root / "positive"
        invert_triplets(
            normalized=capture,
            output=output,
            white_density=0.0,
            black_density=0.5,
            auto_range=True,
        )
        look = json.loads((output / "manifest.json").read_text())["look"]
        self.assertTrue(look["auto_range"])
        self.assertGreaterEqual(look["black_density"] - look["white_density"], 0.79)

    def test_refuses_to_overwrite(self) -> None:
        capture = write_capture(self.root, "frame")
        output = self.root / "positive"
        output.mkdir()
        with self.assertRaises(NegativeUniversalError):
            invert_triplets(normalized=capture, output=output)

    def test_clear_base_and_roi_are_exclusive(self) -> None:
        capture = write_capture(self.root, "frame")
        with self.assertRaises(NegativeUniversalError):
            invert_triplets(
                normalized=capture,
                output=self.root / "positive",
                clear_base=capture,
                base_roi=(0, 0, 4, 4),
            )

    def test_clear_base_geometry_must_match(self) -> None:
        capture = write_capture(self.root, "frame")
        base = write_capture(self.root, "base", width=32, height=24)
        with self.assertRaises(NegativeUniversalError):
            invert_triplets(
                normalized=capture, output=self.root / "positive", clear_base=base
            )

    def test_detects_tampered_frame(self) -> None:
        capture = write_capture(self.root, "frame")
        (capture / "G.pgm").write_bytes(
            b"P5\n64 48\n65535\n" + b"\x00" * (WIDTH * HEIGHT * 2)
        )
        with self.assertRaises(NegativeUniversalError):
            invert_triplets(normalized=capture, output=self.root / "positive")

    def test_rejects_wrong_kind(self) -> None:
        directory = self.root / "raw"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"kind": "film"}))
        with self.assertRaises(NegativeUniversalError):
            invert_triplets(normalized=directory, output=self.root / "positive")

    def test_probe_reports_base_without_writing(self) -> None:
        capture = write_capture(self.root, "frame")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(["probe", "--normalized", str(capture)])
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["base"]["source"], "border")
        self.assertAlmostEqual(report["base"]["transmittance"][0], 40000 / 65535, delta=0.01)
        self.assertGreater(report["candidates"], len(report["chosen"]))
        self.assertIn("spread_density", report["base"])

    def test_probe_accepts_roi(self) -> None:
        capture = write_capture(self.root, "frame")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(["probe", "--normalized", str(capture), "--base-roi", "8,8,10,10"])
        self.assertEqual(json.loads(stdout.getvalue())["base"]["source"], "roi")

    def test_cli_exit_code_on_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["invert", "--normalized", str(self.root / "missing"), "--output", "x"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
