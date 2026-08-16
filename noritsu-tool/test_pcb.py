#!/usr/bin/env python3
"""Tests for the captured RD5X final-PCB ABI decoder."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from noritsu import pcb, vendor


class FinalPcbDecoderTests(unittest.TestCase):
    def setUp(self):
        self.data = bytearray(pcb.FINAL_PCB_SIZE)

    def test_exact_size_and_path_validation(self):
        with self.assertRaisesRegex(ValueError, "exactly 0x1493e"):
            pcb.decode_geometry(b"short")
        with self.assertRaises(TypeError):
            pcb.decode_geometry(object())

        struct.pack_into("<4I", self.data, 0, 399, 252, 160, 120)
        with tempfile.TemporaryDirectory(prefix="final_pcb_") as directory:
            path = Path(directory) / "capture.bin"
            path.write_bytes(self.data)
            self.assertEqual(
                pcb.decode_geometry(path),
                {
                    "source_width": 399,
                    "source_height": 252,
                    "destination_width": 160,
                    "destination_height": 120,
                },
            )

    def test_runtime_com_pointee_decoders_preserve_native_shapes(self):
        grain = np.arange(256, dtype="<u2") * 13
        decoded_grain = pcb.decode_com_grain_curve_u16(grain.tobytes())
        self.assertEqual(decoded_grain.shape, (256,))
        self.assertEqual(decoded_grain.dtype, np.dtype("<u2"))
        np.testing.assert_array_equal(decoded_grain, grain)

        hue = (
            np.arange(np.prod(pcb.COM_HUE_LATTICE_SHAPE), dtype=np.int32)
            % 2001
            - 1000
        ).astype("<i2").reshape(pcb.COM_HUE_LATTICE_SHAPE)
        decoded_hue = pcb.decode_com_hue_lattice_i16(hue.tobytes())
        self.assertEqual(decoded_hue.shape, (9, 33, 33, 3))
        self.assertEqual(decoded_hue.dtype, np.dtype("<i2"))
        np.testing.assert_array_equal(decoded_hue, hue)

        decoded_hue[0, 0, 0, 0] = 30000
        self.assertEqual(
            pcb.decode_com_hue_lattice_i16(hue.tobytes())[0, 0, 0, 0],
            hue[0, 0, 0, 0],
        )
        with self.assertRaisesRegex(ValueError, "exactly 58806 bytes"):
            pcb.decode_com_hue_lattice_i16(b"short")

        with tempfile.TemporaryDirectory(prefix="com_tables_") as directory:
            root = Path(directory)
            grain_path = root / "grain.bin"
            hue_path = root / "hue.bin"
            header_path = root / "header.bin"
            grain_path.write_bytes(grain.tobytes())
            hue_path.write_bytes(hue.tobytes())
            header_path.write_bytes(bytes(range(pcb.COM_HEADER_SIZE)))
            np.testing.assert_array_equal(pcb.read_grain_curve(grain_path), grain)
            np.testing.assert_array_equal(
                pcb.read_hue_saturation_lattice(hue_path), hue
            )
            self.assertEqual(
                pcb.read_com_header(header_path), bytes(range(pcb.COM_HEADER_SIZE))
            )

    def test_unify_image_descriptor_prefix_decodes_packed_dst_type(self):
        prefix = bytearray(pcb.IMAGE_DESCRIPTOR_PREFIX_SIZE)
        struct.pack_into("<i", prefix, 0x0A, 6391)
        struct.pack_into("<i", prefix, 0x0E, 4042)
        struct.pack_into("<H", prefix, pcb.IMAGE_DESCRIPTOR_DST_TYPE_OFFSET, 0)
        struct.pack_into("<i", prefix, 0x1E, 6048)
        struct.pack_into("<i", prefix, 0x22, 4011)
        self.assertEqual(
            pcb.decode_image_descriptor_prefix(prefix),
            {
                "size_bytes": 64,
                "source_width": 6391,
                "source_height": 4042,
                "dst_type": 0,
                "destination_width": 6048,
                "destination_height": 4011,
            },
        )
        with tempfile.TemporaryDirectory(prefix="image_descriptor_") as directory:
            path = Path(directory) / "prefix.bin"
            path.write_bytes(prefix)
            self.assertEqual(
                pcb.read_image_descriptor_prefix(path)["dst_type"], 0
            )
        with self.assertRaisesRegex(ValueError, "exactly 64 bytes"):
            pcb.decode_image_descriptor_prefix(prefix[:-1])

    def test_effective_selectors_skip_nonselector_0x20(self):
        for offset in pcb.SELECTOR_OFFSETS:
            self.data[offset] = (offset * 3) & 0xFF
        self.data[0x20] = 0xEE
        selectors = pcb.decode_effective_selectors(self.data)
        self.assertEqual(len(selectors), 33)
        self.assertNotIn("0x20", selectors)
        self.assertEqual(selectors["0x10"], 0x30)
        self.assertEqual(selectors["0x31"], 0x93)

    def test_scaling_trimming_layout_matches_s2_worker_api(self):
        struct.pack_into(
            "<4i", self.data, pcb.TRIMMING_OFFSET, 168, 16, 6048, 4008
        )
        struct.pack_into(
            "<4I", self.data, pcb.SCALING_OFFSET, 6048, 4008, 6048, 4011
        )
        self.data[pcb.SCALING_OFFSET + 0x10 : pcb.SCALING_OFFSET + 0x12] = bytes(
            [1, 2]
        )
        self.data[0x1A : 0x1C] = bytes([1, 2])

        decoded = pcb.decode_scaling_trimming_parameters(self.data)
        self.assertEqual(
            decoded,
            {
                "trim": (168, 16, 6048, 4008),
                "input_width": 6048,
                "input_height": 4008,
                "output_width": 6048,
                "output_height": 4011,
                "pre_scale": 1,
                "sampling": 2,
                "trimming_selector": 1,
                "scaling_selector": 2,
            },
        )

        # Selector 2 belongs to the S0 dispatch range (1..4).  The independent
        # sampling byte is also 2; it does not select the S2 family.
        image = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3)
        direct = vendor.apply_scaling_s0_u16(
            image,
            output_width=4,
            output_height=3,
            pre_scale=1,
            sampling=2,
            trim=(1, 0, 4, 4),
        )
        forwarded_parameters = {
            key: value
            for key, value in {
                **decoded,
                "trim": (1, 0, 4, 4),
                "output_width": 4,
                "output_height": 3,
            }.items()
            if key
            in {"trim", "output_width", "output_height", "pre_scale", "sampling"}
        }
        forwarded = vendor.apply_scaling_s0_u16(image, **forwarded_parameters)
        np.testing.assert_array_equal(forwarded, direct)

    def test_ccd_noise_layout_and_rgb_triplets(self):
        base = pcb.CCD_NOISE_S0_OFFSET
        self.data[base : base + 6] = bytes([3, 5, 7, 11, 12, 13])
        levels_by_point = np.array(
            [[1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33]],
            dtype=np.uint8,
        )
        self.data[base + 6 : base + 0x12] = levels_by_point.tobytes()
        breakpoints_by_point = np.array(
            [[100, 200, 300], [1100, 1200, 1300], [2100, 2200, 2300], [3100, 3200, 3300]],
            dtype="<u2",
        )
        self.data[base + 0x12 : base + 0x2A] = breakpoints_by_point.tobytes()
        self.data[base + 0x2A : base + 0x30] = bytes([41, 42, 43, 51, 52, 53])

        decoded = pcb.decode_ccd_noise_s0_parameters(self.data)
        np.testing.assert_array_equal(decoded["window_sizes_rgb"], [3, 5, 7])
        np.testing.assert_array_equal(decoded["gains_rgb_u8"], [11, 12, 13])
        np.testing.assert_array_equal(
            decoded["levels_rgb_u8"], levels_by_point.T
        )
        np.testing.assert_array_equal(
            decoded["breakpoints_rgb_u16"], breakpoints_by_point.T
        )
        np.testing.assert_array_equal(
            decoded["edge_sensitivity_rgb_u8"], [41, 42, 43]
        )
        np.testing.assert_array_equal(
            decoded["edge_limits_rgb_u8"], [51, 52, 53]
        )

    def test_rgb_to_ycc_s1_layout(self):
        base = pcb.RGB_TO_YCC_OFFSET
        matrix = np.arange(-4, 5, dtype="<i2")
        offsets = np.array([-101, 202, -303], dtype="<i2")
        counts = np.array([18, 19, 20], dtype=np.uint8)
        x = np.arange(60, dtype="<u2").reshape(3, 20) * 101
        y = np.arange(60, dtype="<u2").reshape(3, 20) * 103
        self.data[base : base + 0x12] = matrix.tobytes()
        self.data[base + 0x12 : base + 0x18] = offsets.tobytes()
        self.data[base + 0x18 : base + 0x1B] = counts.tobytes()
        self.data[base + 0x1C : base + 0x94] = x.tobytes()
        self.data[base + 0x94 : base + 0x10C] = y.tobytes()

        decoded = pcb.decode_rgb_to_ycc_s1_parameters(self.data)
        np.testing.assert_array_equal(decoded["matrix_q10000"], matrix)
        np.testing.assert_array_equal(decoded["input_offsets_rgb"], offsets)
        np.testing.assert_array_equal(decoded["counts_rgb"], counts)
        np.testing.assert_array_equal(decoded["x_controls_rgb"], x)
        np.testing.assert_array_equal(decoded["y_controls_rgb"], y)

        # Every call returns owned arrays, never a mutable view into caller bytes.
        decoded["matrix_q10000"][0] = 30000
        self.assertEqual(
            pcb.decode_rgb_to_ycc_s1_parameters(self.data)["matrix_q10000"][0],
            -4,
        )

    def test_auto_contrast_rejects_inactive_zero_block(self):
        with self.assertRaisesRegex(ValueError, "active AutoContrast"):
            pcb.decode_auto_contrast_s0_parameters(self.data)

    def test_grain_and_hue_controls_keep_external_com_boundary_explicit(self):
        grain = pcb.GRAIN_OFFSET
        struct.pack_into("<II", self.data, grain, 1000, 2000)
        self.data[grain + 8 : grain + 0x0B] = bytes([7, 8, 9])
        self.assertEqual(
            pcb.decode_grain_s4_controls(self.data),
            {
                "base_divisor_u32": 1000,
                "range_divisor_u32": 2000,
                "color_offset_u8": 7,
                "luminance_offset_u8": 8,
                "window_size": 9,
            },
        )

        hue = pcb.HUE_SATURATION_OFFSET
        self.data[hue] = 100
        struct.pack_into("<2h", self.data, hue + 2, -11, 22)
        decoded = pcb.decode_hue_saturation_s0_controls(self.data)
        self.assertEqual(decoded["saturation_u8"], 100)
        np.testing.assert_array_equal(decoded["total_balance_i16"], [-11, 22])
        self.assertEqual(
            decoded["correction_lattice_source"], "COM_PARA_FOR_ALL+8"
        )

    def test_ycc_correction_control_layout_and_exact_builders(self):
        base = pcb.YCC_CORRECTION_OFFSET
        y_x = np.zeros(20, dtype="<u2")
        y_q4 = np.zeros(20, dtype="<u2")
        chroma_x = np.zeros(20, dtype="<u2")
        offsets = np.zeros((2, 20), dtype="<i2")
        scale = np.zeros(20, dtype="<u2")
        y_x[:3] = [0, 32768, 65535]
        y_q4[:3] = [0, 32768, 65520]
        chroma_x[:3] = [0, 32768, 65535]
        offsets[0, :3] = [-100, 0, 100]
        offsets[1, :3] = [100, 0, -100]
        scale[:3] = [16000, 8000, 16000]
        self.data[base : base + 0x28] = y_x.tobytes()
        self.data[base + 0x28 : base + 0x50] = y_q4.tobytes()
        self.data[base + 0x50 : base + 0x52] = bytes([3, 3])
        self.data[base + 0x52 : base + 0x7A] = chroma_x.tobytes()
        self.data[base + 0x7A : base + 0xA2] = offsets[0].tobytes()
        self.data[base + 0xA2 : base + 0xCA] = offsets[1].tobytes()
        self.data[base + 0xCA : base + 0xF2] = scale.tobytes()

        controls = pcb.decode_ycc_correction_controls(self.data)
        np.testing.assert_array_equal(controls["y_x_u16"], y_x)
        np.testing.assert_array_equal(controls["y_values_q4_u16"], y_q4)
        np.testing.assert_array_equal(controls["chroma_x_u16"], chroma_x)
        np.testing.assert_array_equal(controls["chroma_offsets_i16"], offsets)
        np.testing.assert_array_equal(controls["chroma_scale_u16"], scale)
        self.assertEqual(controls["y_count"], 3)
        self.assertEqual(controls["chroma_count"], 3)

        y_lut, scale_lut, offset_lut = vendor.build_ycc_correction_tables(
            **controls
        )
        self.assertEqual(y_lut.dtype, np.uint16)
        self.assertEqual(scale_lut.dtype, np.float32)
        self.assertEqual(offset_lut.dtype, np.int32)
        self.assertEqual(y_lut[[0, 32768, 65535]].tolist(), [0, 2048, 4095])
        np.testing.assert_allclose(
            scale_lut[[0, 32768, 65535]], [1.0, 0.5, 1.0], rtol=1e-7
        )
        np.testing.assert_array_equal(
            offset_lut[[0, 32768, 65535]],
            [[-100, 100], [0, 0], [100, -100]],
        )

    def test_ycc_to_rgb_gamma_layout_preserves_rgb_abi_order(self):
        matrix = np.array(
            [10000, -3334, 12441, 10000, 12439, -3332, 10000, -9109, -9107],
            dtype="<i2",
        )
        counts = np.array([18, 19, 20], dtype=np.uint8)
        x = (np.arange(60, dtype=np.uint16).reshape(3, 20) * 101).astype(
            "<u2"
        )
        y = np.arange(60, dtype=np.uint8).reshape(3, 20)
        self.data[
            pcb.YCC_TO_RGB_OFFSET : pcb.YCC_TO_RGB_OFFSET + matrix.nbytes
        ] = matrix.tobytes()
        base = pcb.FUSED_GAMMA_OFFSET
        self.data[base : base + 3] = counts.tobytes()
        self.data[base + 4 : base + 4 + x.nbytes] = x.tobytes()
        self.data[base + 0x7C : base + 0x7C + y.nbytes] = y.tobytes()

        decoded = pcb.decode_ycc_to_rgb_gamma_controls(self.data)
        np.testing.assert_array_equal(decoded["matrix_q10000"], matrix)
        np.testing.assert_array_equal(decoded["counts_rgb"], counts)
        np.testing.assert_array_equal(decoded["x_controls_rgb"], x)
        np.testing.assert_array_equal(decoded["y_controls_rgb_u8"], y)

        # The decoder exposes the on-disk RGB ABI.  It must not silently
        # reverse rows to the BGR worker order used by FmYccToRgbS0_Do.
        self.assertEqual(decoded["x_controls_rgb"][0, 0], 0)
        self.assertEqual(decoded["x_controls_rgb"][2, 0], 4040)
        decoded["x_controls_rgb"][0, 0] = 65535
        self.assertEqual(
            pcb.decode_ycc_to_rgb_gamma_controls(self.data)[
                "x_controls_rgb"
            ][0, 0],
            0,
        )

        summary = pcb.decode_final_pcb_summary(self.data)
        self.assertEqual(summary["ycc_to_rgb_gamma"]["counts_rgb"], [18, 19, 20])
        self.assertEqual(
            summary["ycc_to_rgb_gamma"]["matrix_q10000"], matrix.tolist()
        )


if __name__ == "__main__":
    unittest.main()
