#!/usr/bin/env python3
"""Tests for installer-derived table parsing and exact integer kernels."""

import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from noritsu import tables, vendor


class VendorKernelTests(unittest.TestCase):
    @staticmethod
    def _film_sharp_identity_block() -> bytes:
        block = bytearray(vendor.FILM_SHARP_PARAMETER_BLOCK_SIZE)
        layouts = (
            (0x000, 0x002, 0x014),
            (0x016, 0x018, 0x02A),
            (0x02C, 0x02E, 0x090),
            (0x092, 0x094, 0x0DC),
            (0x0DE, 0x0E0, 0x112),
        )
        for size_offset, kernel_offset, scale_offset in layouts:
            block[size_offset] = 1
            struct.pack_into("<hH", block, kernel_offset, 1, 0)
            struct.pack_into("<H", block, scale_offset, 1)
        struct.pack_into("<H", block, 0x114, 256)
        block[0x116] = 1
        block[0x117] = 1
        struct.pack_into("<4H", block, 0x118, 0, 65535, 1, 65535)
        struct.pack_into("<4H", block, 0x120, 2, 1, 1, 1)
        struct.pack_into("<10H", block, 0x128, *range(10))
        struct.pack_into("<10H", block, 0x13C, *([65535] * 10))
        struct.pack_into("<Hh", block, 0x150, 100, 0)
        return bytes(block)

    def test_table_selection_and_layout(self):
        lut = tables.load_film_lut(0, 1)
        x, y = tables.load_common_control_points(0, 1)
        self.assertEqual(lut.shape, (3, 4096))
        self.assertEqual(lut.dtype, np.uint16)
        self.assertEqual(tuple(lut[:, 0]), (64879, 62201, 61660))
        self.assertEqual(tuple(lut[:, -1]), (0, 0, 0))
        self.assertEqual(x.shape, (3, 20))
        self.assertEqual(y.shape, (3, 20))
        self.assertEqual(x[0, :4].tolist(), [0, 76, 203, 445])
        self.assertEqual(y[0, :4].tolist(), [29855, 30615, 31885, 34305])
        selectors = tables.load_normal_output_selectors(0, 1)
        self.assertEqual(len(selectors), 33)
        self.assertEqual(selectors["0x10"], 1)
        self.assertEqual(selectors["0x18"], 3)
        self.assertEqual(selectors["0x1e"], 2)
        self.assertEqual(selectors["0x20"] if "0x20" in selectors else None, None)
        self.assertEqual(selectors["0x21"], 1)
        self.assertEqual(selectors["0x31"], 0)

    def test_table_loaders_do_not_expose_mutable_cache_state(self):
        lut = tables.load_film_lut(0, 1)
        lut[0, 0] = 0
        self.assertEqual(tables.load_film_lut(0, 1)[0, 0], 64879)

        x, y = tables.load_common_control_points(0, 1)
        x[0, 0] = 123
        y[0, 0] = 123
        fresh_x, fresh_y = tables.load_common_control_points(0, 1)
        self.assertEqual(fresh_x[0, 0], 0)
        self.assertEqual(fresh_y[0, 0], 29855)

        provenance = tables.source_provenance(0, 1)
        provenance["NEGA_para.dat"]["sha256"] = "corrupted"
        self.assertNotEqual(
            tables.source_provenance(0, 1)["NEGA_para.dat"]["sha256"],
            "corrupted",
        )
        tone = tables.load_base_tone_curve()
        tone[0] = 1.0
        self.assertNotEqual(tables.load_base_tone_curve()[0], 1.0)

    def test_table_cache_is_scoped_to_configured_source_root(self):
        with tempfile.TemporaryDirectory(prefix="noritsu_tables_a_") as first:
            with tempfile.TemporaryDirectory(prefix="noritsu_tables_b_") as second:
                for root, first_value in ((first, 111), (second, 222)):
                    directory = (
                        Path(root) / "IQGrp000" / "00000001" / "PCB_PARA"
                    )
                    directory.mkdir(parents=True)
                    lut = np.zeros((3, 4096), dtype="<u2")
                    lut[0, 0] = first_value
                    (directory / "NEGA_para.dat").write_bytes(
                        b"\x01\x01\x00\x60\x00\x00" + lut.tobytes()
                    )

                with mock.patch.dict(
                    os.environ, {"NORITSU_TABLES": first}, clear=False
                ):
                    self.assertEqual(tables.load_film_lut(0, 1)[0, 0], 111)
                with mock.patch.dict(
                    os.environ, {"NORITSU_TABLES": second}, clear=False
                ):
                    self.assertEqual(tables.load_film_lut(0, 1)[0, 0], 222)

    def test_film_lut_reorders_bgr_to_rgb(self):
        raw = np.array([[[3, 2, 1]]], dtype=np.uint16)
        lut = np.tile(np.arange(4096, dtype=np.uint16), (3, 1))
        lut[0] += 100
        lut[1] += 200
        lut[2] += 300
        output = vendor.apply_film_lut(raw, lut)
        self.assertEqual(output.tolist(), [[[101, 202, 303]]])

    def test_rgb_to_ycc_lut_builder_uses_integer_interpolation_and_bgr_order(self):
        x = np.zeros((3, 20), dtype=np.uint16)
        y = np.zeros((3, 20), dtype=np.uint16)
        x[:, :2] = (10, 20)
        y[0, :2] = (100, 200)  # R
        y[1, :2] = (300, 500)  # G
        y[2, :2] = (700, 900)  # B
        luts = vendor.build_rgb_to_ycc_luts_u16(x, y, (2, 2, 2))
        self.assertEqual(luts.shape, (3, 65536))
        self.assertEqual(luts.dtype, np.uint16)
        self.assertEqual(luts[:, 0].tolist(), [700, 300, 100])
        self.assertEqual(luts[:, 15].tolist(), [800, 400, 150])
        self.assertEqual(luts[:, 30].tolist(), [900, 500, 200])

    def test_rgb_to_ycc_s1_lane_order_anchor_and_signed_pack(self):
        image = np.array([[[10000, 20000, 30000]]], dtype=np.uint16)
        identity_luts = np.tile(np.arange(65536, dtype=np.uint16), (3, 1))
        matrix = np.array(
            [10000, 0, 0, 10000, 0, 0, 10000, 0, 0], dtype=np.int16
        )
        output = vendor.apply_rgb_to_ycc_s1(
            image,
            matrix,
            np.array([32767, 0, 0], dtype=np.int16),
            identity_luts,
            chunk_rows=1,
        )
        # C1 and C2 select transferred R.  Y selects raw R and receives the
        # setup routine's -1 anchor before the signed-pack bias is applied.
        self.assertEqual(output.tolist(), [[[42768, 42768, 42767]]])

    def test_ycc_correction_identity_and_both_gamut_limits(self):
        image = np.array(
            [
                [[1234, 54321, 7], [65535, 32768, 8]],
                [[0, 32768, 9], [32768, 32768, 10]],
            ],
            dtype=np.uint16,
        )
        y_lut = np.arange(65536, dtype=np.uint16)
        offsets = np.zeros((65536, 2), dtype=np.int32)

        identity = vendor.apply_ycc_correction(
            image,
            y_lut,
            np.ones(65536, dtype=np.float32),
            offsets,
            chunk_rows=1,
        )
        np.testing.assert_array_equal(identity, image)

        doubled = vendor.apply_ycc_correction(
            image,
            y_lut,
            np.full(65536, 2.0, dtype=np.float32),
            offsets,
            chunk_rows=1,
        )
        self.assertEqual(doubled[0, 1].tolist(), [65535, 32768, 8])
        self.assertEqual(doubled[1, 0].tolist(), [0, 32768, 9])

    def test_ycc_to_rgb_variant1_matrix_gamma_order_and_gamut_compression(self):
        # Flat vendor matrix: R=C1, G=C2, B=Y after chroma centering.
        matrix = np.array(
            [0, 10000, 0, 0, 0, 10000, 10000, 0, 0], dtype=np.int16
        )
        gamma = np.tile(np.arange(65536, dtype=np.uint16), (3, 1))
        image = np.array(
            [
                [
                    [32768 + 1000, 32768 + 2000, 3000],
                    [32768 + 5000, 32768 + 1000, 1000],
                    [32768 - 1000, 32768 + 1000, 3000],
                ]
            ],
            dtype=np.uint16,
        )
        output = vendor.apply_ycc_to_rgb_s0_variant1_u16(
            image, matrix, gamma, chunk_rows=1
        )
        self.assertEqual(output[0, 0].tolist(), [1000, 2000, 3000])
        self.assertEqual(output[0, 1].tolist(), [4095, 1000, 1000])
        self.assertEqual(output[0, 2].tolist(), [0, 1500, 3000])

        channel_gamma = gamma.copy()
        channel_gamma[0, :4000] += 10  # native B
        channel_gamma[1, :4000] += 20  # native G
        channel_gamma[2, :4000] += 30  # native R
        reordered = vendor.apply_ycc_to_rgb_s0_variant1_u16(
            image[:, :1], matrix, channel_gamma
        )
        self.assertEqual(reordered.tolist(), [[[1030, 2020, 3010]]])

    def test_ycc_to_rgb_variant1_byte_wrapper_uses_byte_luts_and_rgb_order(self):
        matrix = np.array(
            [0, 10000, 0, 0, 0, 10000, 10000, 0, 0], dtype=np.int16
        )
        image = np.array(
            [
                [
                    [32768 + 1000, 32768 + 2000, 3000],
                    [32768 + 5000, 32768 + 1000, 1000],
                    [32768 - 1000, 32768 + 1000, 3000],
                ]
            ],
            dtype=np.uint16,
        )
        domain = np.arange(65536, dtype=np.uint32)
        gamma = np.empty((3, 65536), dtype=np.uint8)
        gamma[0] = domain % 251  # native B
        gamma[1] = (domain + 17) % 251  # native G
        gamma[2] = (domain + 43) % 251  # native R

        output = vendor.apply_ycc_to_rgb_s0_variant1_u8(
            image, matrix, gamma, chunk_rows=1
        )

        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(
            output.tolist(),
            [[[39, 9, 239], [122, 13, 247], [43, 11, 239]]],
        )
        with self.assertRaises(ValueError):
            vendor.apply_ycc_to_rgb_s0_variant1_u8(
                image, matrix, gamma.astype(np.uint16)
            )

    def test_fused_gamma_builders_match_native_captured_lut_hashes(self):
        control_x = np.array(
            [
                0, 16, 241, 466, 691, 915, 1140, 1365, 1590, 1815,
                2056, 2280, 2505, 2730, 2955, 3180, 3404, 3629, 3854, 4095,
            ],
            dtype=np.uint16,
        )
        control_y = np.array(
            [
                0, 1, 15, 29, 43, 57, 71, 85, 99, 113,
                128, 142, 156, 170, 184, 198, 212, 226, 240, 255,
            ],
            dtype=np.uint8,
        )
        x_rgb = np.tile(control_x, (3, 1))
        y_rgb = np.tile(control_y, (3, 1))
        counts = np.full(3, 20, dtype=np.uint8)

        byte_luts = vendor.build_fused_gamma_luts_bgr(x_rgb, y_rgb, counts)
        word_luts = vendor.build_fused_gamma_luts_bgr_u16(
            x_rgb, y_rgb, counts
        )
        self.assertEqual(byte_luts.shape, (3, 65536))
        self.assertEqual(byte_luts.dtype, np.uint8)
        self.assertEqual(word_luts.shape, (3, 65536))
        self.assertEqual(word_luts.dtype, np.uint16)
        self.assertEqual(
            hashlib.sha256(byte_luts.tobytes()).hexdigest(),
            "b095760adb3c5485520eb0efc523f79c46f39215180a966b62ab1a74a88e410b",
        )
        self.assertEqual(
            hashlib.sha256(word_luts.astype("<u2", copy=False).tobytes()).hexdigest(),
            "c5ffcf674feb6f87c2db9d8f98950e5b1739f15a635fd0a0cfde67efc7ea54b4",
        )

    def test_fused_gamma_rgb_controls_reverse_explicitly_to_native_bgr(self):
        x = np.zeros((3, 20), dtype=np.uint16)
        y = np.zeros((3, 20), dtype=np.uint8)
        x[:, :2] = [0, 65535]
        y[0, :2] = [0, 10]
        y[1, :2] = [20, 30]
        y[2, :2] = [40, 50]
        luts = vendor.build_fused_gamma_luts_bgr(
            x, y, np.full(3, 2, dtype=np.uint8)
        )
        np.testing.assert_array_equal(luts[:, 0], [40, 20, 0])
        np.testing.assert_array_equal(luts[:, -1], [50, 30, 10])

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            vendor.build_fused_gamma_lut_u8(
                np.zeros(20, dtype=np.uint16),
                np.zeros(20, dtype=np.uint8),
                2,
            )

    def test_scanner_preprocess_identity_calibration(self):
        raw = (np.arange(2 * 3 * 4).reshape(2, 3, 4) * 7).astype(np.uint16)
        linear = np.tile(np.arange(4096, dtype=np.uint16), (4, 1))
        matrix = np.tile(np.eye(4, dtype=np.int16) * 4096, (3, 1, 1))
        output = vendor.apply_scanner_preprocess(
            raw,
            dark_offset=np.zeros((3, 4), dtype=np.uint16),
            ccd_linear=linear,
            color_matrix_q12=matrix,
        )
        np.testing.assert_array_equal(output, raw)

        with self.assertRaises(TypeError):
            vendor.apply_scanner_preprocess(
                raw,
                dark_offset=np.zeros((3, 4), dtype=np.uint16),
                ccd_linear=linear,
                color_matrix_q12=matrix,
                digital_gain_q12=np.full((3, 4), 4096, dtype=np.int16),
            )

    def test_scanner_integer_order_and_clipping(self):
        raw = np.array([[[100, 200, 300, 400]]], dtype=np.uint16)
        offset = np.array([[150, 50, 0, 500]], dtype=np.uint16)
        gain = np.array([[4096, 2048, 8192, 4096]], dtype=np.int16)
        after_offset = vendor.apply_dark_offset(raw, offset)
        self.assertEqual(after_offset.tolist(), [[[0, 150, 300, 0]]])
        after_gain = vendor.apply_digital_gain(after_offset, gain)
        self.assertEqual(after_gain.tolist(), [[[0, 75, 600, 0]]])

    def test_scanner_q12_words_are_signed(self):
        raw = np.array([[[4095, 4095, 4095, 4095]]], dtype=np.uint16)
        encoded_negative_one = np.full((1, 4), 0xFFFF, dtype=np.uint16)
        gained = vendor.apply_digital_gain(raw, encoded_negative_one)
        self.assertEqual(gained.tolist(), [[[0, 0, 0, 0]]])

        matrix = np.zeros((1, 4, 4), dtype=np.uint16)
        np.fill_diagonal(matrix[0], 4096)
        matrix[0, 0, 0] = 0xFFFF
        transformed = vendor.apply_color_matrix_q12(raw, matrix)
        self.assertEqual(transformed[0, 0, 0], 0)
        self.assertEqual(transformed[0, 0, 1:].tolist(), [4095, 4095, 4095])

    def test_ir_misregistration_identity_and_bgri_lane_contract(self):
        image = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
        output = vendor.apply_ir_misregistration_u16(
            image,
            np.zeros(4, dtype=np.int32),
            np.zeros(4, dtype=np.int32),
        )
        np.testing.assert_array_equal(output, image)
        self.assertIsNot(output, image)

    def test_ir_misregistration_clips_and_uses_last_row_major_source(self):
        image = np.zeros((3, 3, 4), dtype=np.uint16)
        image[..., :3] = np.arange(27, dtype=np.uint16).reshape(3, 3, 3)
        image[..., 3] = np.array(
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=np.uint16
        )

        output = vendor.apply_ir_misregistration_u16(
            image,
            np.array([0, 1, -1], dtype=np.int32),
            np.array([1, 0, -1], dtype=np.int32),
        )

        np.testing.assert_array_equal(output[..., :3], image[..., :3])
        np.testing.assert_array_equal(
            output[..., 3],
            np.array([[10, 1, 1], [20, 2, 11], [20, 22, 21]], dtype=np.uint16),
        )

        encoded_negative_one = np.array([0, 0xFFFFFFFF, 0], dtype=np.uint32)
        decoded = vendor.apply_ir_misregistration_u16(
            image,
            np.zeros(3, dtype=np.int32),
            encoded_negative_one,
        )
        self.assertEqual(decoded[1, 1, 3], image[0, 1, 3])
        with self.assertRaises(ValueError):
            vendor.apply_ir_misregistration_u16(
                image,
                np.zeros(3, dtype=np.int32),
                np.array([0, 4, 0], dtype=np.int32),
            )

    def test_ccd_noise_response_lut_preserves_idiv_segments_and_float_scale(self):
        levels = np.array([10, 20, 5, 25], dtype=np.uint8)
        breakpoints = np.array([0x0200, 0x0400, 0x0700, 0x0900], dtype=np.uint16)
        output = vendor.build_ccd_noise_response_lut(levels, breakpoints, 255)

        integer_response = np.array(
            [10, 10, 10, 15, 20, 15, 10, 5, 15, 25], dtype=np.int64
        )
        expected = np.asarray(
            integer_response
            * 255
            * float(vendor.CCD_NOISE_RESPONSE_SCALE_F32),
            dtype=np.float32,
        )
        self.assertEqual(output.shape, (256,))
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(output[:10], expected)
        self.assertTrue(np.all(output[10:] == output[9]))

    def test_ccd_noise_s0_full_worker_uses_reflected_box_mean_and_chunking(self):
        image = np.zeros((5, 5, 3), dtype=np.uint16)
        image[0, 0] = (900, 1800, 2700)
        response = np.ones((3, 256), dtype=np.float32)

        output = vendor.apply_ccd_noise_clear_s0(
            image,
            response,
            (3, 3, 3),
            (0, 0, 0),
            (255, 255, 255),
            chunk_rows=2,
        )
        expected = np.zeros_like(image)
        expected[:2, :2] = (100, 200, 300)
        np.testing.assert_array_equal(output, expected)

        unchunked = vendor.apply_ccd_noise_clear_s0(
            image,
            response,
            (3, 3, 3),
            (0, 0, 0),
            (255, 255, 255),
            chunk_rows=5,
        )
        np.testing.assert_array_equal(output, unchunked)

        unchanged = vendor.apply_ccd_noise_clear_s0(
            image,
            np.zeros((3, 256), dtype=np.float32),
            (3, 5, 3),
            (255, 255, 255),
            (1, 1, 1),
        )
        np.testing.assert_array_equal(unchanged, image)

    def test_ccd_noise_s0_nonzero_edge_estimator_matches_scalar_fixture(self):
        plane = np.array(
            [
                [1000, 1400, 1900, 2500, 3200, 4000],
                [1200, 1800, 2600, 3500, 4500, 5600],
                [1500, 2300, 3400, 4800, 6500, 8500],
                [1900, 3000, 4600, 6800, 9300, 12300],
                [2400, 3900, 6200, 9300, 13200, 18000],
                [3000, 5000, 8100, 12400, 18100, 25000],
            ],
            dtype=np.uint16,
        )
        image = np.repeat(plane[..., None], 3, axis=2)
        response = np.full((3, 256), np.float32(0.5), dtype=np.float32)

        output = vendor.apply_ccd_noise_clear_s0(
            image,
            response,
            (3, 3, 3),
            (47, 47, 47),
            (9, 9, 9),
            chunk_rows=2,
        )

        # Frozen from a direct scalar neighborhood oracle.  Unlike the
        # impulse fixture above, these controls drive edge from 1.0 through
        # the unclamped range and into its 9.0 upper clamp.
        expected_plane = np.array(
            [
                [1228, 1488, 2032, 2685, 3438, 4108],
                [1331, 1826, 2624, 3541, 4576, 5485],
                [1656, 2347, 3465, 4882, 6597, 8221],
                [2109, 3065, 4685, 6885, 9447, 11891],
                [2679, 3988, 6303, 9426, 13365, 17250],
                [3189, 4885, 7847, 11941, 17324, 20833],
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(
            output, np.repeat(expected_plane[..., None], 3, axis=2)
        )

    def test_ccd_noise_s0_preserves_asymmetric_rgb_controls(self):
        y, x = np.mgrid[:9, :10]
        image = np.empty((9, 10, 3), dtype=np.uint16)
        image[..., 0] = 3000 + 731 * x + 1201 * y + ((x * y + 3 * x) % 7) * 499
        image[..., 1] = 11000 + 421 * x + 877 * y + ((2 * x * y + y) % 9) * 613
        image[..., 2] = 23000 + 311 * x + 557 * y + ((3 * x * y + x + y) % 11) * 827
        image[2, 7, 0] = 50000
        image[6, 2, 1] = 1000
        image[4, 5, 2] = 61000

        response = np.empty((3, 256), dtype=np.float32)
        indexes = np.arange(256, dtype=np.float32)
        for channel in range(3):
            response[channel] = np.asarray(
                np.float32((channel + 1) * 0.015)
                + indexes * np.float32((channel + 1) * 0.00007),
                dtype=np.float32,
            )

        output = vendor.apply_ccd_noise_clear_s0(
            image,
            response,
            (3, 5, 7),
            (31, 97, 173),
            (5, 9, 13),
            chunk_rows=3,
        )

        # The digest and samples were generated by an independent scalar
        # oracle that evaluates every reflected box neighborhood directly.
        # They catch either an RGB/BGR swap or shared-control regression.
        self.assertEqual(
            hashlib.sha256(output.astype("<u2", copy=False).tobytes()).hexdigest(),
            "9c96c2c8954e474be534e7e478c883799a314b5be6a451c75ff84b20c9e7015a",
        )
        np.testing.assert_array_equal(output[2, 7], (49546, 17553, 32036))
        np.testing.assert_array_equal(output[4, 5], (11474, 21430, 59342))
        np.testing.assert_array_equal(output[6, 2], (13659, 1536, 27254))

    def test_ccd_noise_s0_second_differences_use_sequential_subtractions(self):
        plane = np.array(
            [
                [5249, 5297, 5693, 4495, 4538, 5554, 5566],
                [5821, 6130, 6256, 5127, 4449, 5431, 6113],
                [6754, 6869, 8092, 6675, 5274, 5483, 7232],
                [6421, 6983, 8133, 6780, 5158, 5352, 7569],
                [5742, 5940, 6357, 6183, 5431, 5774, 8337],
                [5184, 4977, 5120, 6194, 6851, 7097, 9545],
                [6040, 5151, 4939, 6463, 7962, 7946, 9313],
            ],
            dtype=np.uint16,
        )
        image = np.repeat(plane[..., None], 3, axis=2)
        lane_response = vendor.build_ccd_noise_response_lut(
            (100, 0, 0, 100),
            (0, 10923, 21845, 36460),
            10,
        )
        response = np.repeat(lane_response[None, :], 3, axis=0)

        output = vendor.apply_ccd_noise_clear_s0(
            image,
            response,
            (5, 5, 5),
            (115, 115, 115),
            (255, 255, 255),
            chunk_rows=2,
        )

        # Native core 0x1007f644..0x1007f684 evaluates both second
        # differences as ``((a + b) - c) - d``.  Grouping the negative
        # terms as ``(a + b) - (c + d)`` rounds this oracle pixel to 6772.
        np.testing.assert_array_equal(output[3, 3], (6773, 6773, 6773))

    def test_grain_s4_setup_and_pixel_tail_use_q15_weights(self):
        curve = np.zeros(256, dtype=np.uint16)
        curve[1] = 4095
        curve[2] = 65535
        response = vendor.build_grain_s4_variance_lut(curve, 2, 4)
        self.assertEqual(response.dtype, np.float64)
        self.assertEqual(response[0], 8192.0)
        self.assertEqual(response[1], 16384.0)
        self.assertEqual(response[2], 16384.0)

        image = np.array([[[1000, 2000, 4000]]], dtype=np.uint16)
        count = 9
        means = np.array([900, 1800, 2700, 1800], dtype=np.int64)
        sums = (means * count).reshape(1, 1, 4)
        square_sums = (means * means * count).reshape(1, 1, 4)
        output = vendor.apply_grain_s4_pixel_kernel(
            image,
            sums,
            square_sums,
            np.zeros(256, dtype=np.float64),
            window_size=3,
            color_offset_u8=255,
            luminance_offset_u8=0,
        )
        # Color weight 32768 keeps the source RGB.  Luminance weight zero
        # targets the local mean, so the common -533 correction is applied.
        self.assertEqual(output.tolist(), [[[467, 1467, 3467]]])

    def test_grain_s4_full_worker_uses_reflected_statistics_and_chunking(self):
        image = np.zeros((5, 5, 3), dtype=np.uint16)
        image[0, 0] = (900, 1800, 2700)
        zero_curve = np.zeros(256, dtype=np.uint16)

        output = vendor.apply_grain_s4(
            image,
            zero_curve,
            1,
            1,
            window_size=3,
            color_offset_u8=0,
            luminance_offset_u8=0,
            chunk_rows=2,
        )
        expected = np.zeros_like(image)
        expected[:2, :2] = (100, 200, 300)
        np.testing.assert_array_equal(output, expected)

        unchunked = vendor.apply_grain_s4(
            image,
            zero_curve,
            1,
            1,
            window_size=3,
            color_offset_u8=0,
            luminance_offset_u8=0,
            chunk_rows=5,
        )
        np.testing.assert_array_equal(output, unchunked)

    def test_auto_contrast_s0_lut_builder_and_luminance_worker(self):
        intercept, blend = vendor.build_auto_contrast_s0_luts(
            [0, 100, 200, 400, 65535],
            [10, 110, 60, 260, 500],
            np.array([512, 256, -512, 0], dtype=np.int16),
        )
        self.assertEqual(intercept.dtype, np.int32)
        self.assertEqual(blend.dtype, np.int16)
        self.assertEqual(intercept[[0, 50, 100, 150, 200, 300]].tolist(),
                         [10, 60, 110, 85, 60, 160])
        self.assertEqual(blend[[50, 150, 300, 65535]].tolist(),
                         [512, 256, -512, 0])

        luminance = np.array([[80, 170, 100, 65535]], dtype=np.uint16)
        local_index = np.array([[50, 150, 300, 65535]], dtype=np.uint16)
        output = vendor.apply_auto_contrast_s0_luts(
            luminance, local_index, intercept, blend
        )
        self.assertEqual(output.tolist(), [[90, 95, 360, 500]])

    def test_auto_contrast_s0_downsample_and_kernel_expansion(self):
        x_curve, y_curve = vendor.prepare_auto_contrast_s0_curve(
            [100, 200, 65000],
            [1000, 2000, 3000],
            np.array([513, 0, 0, -513], dtype=np.int16),
        )
        self.assertEqual(x_curve.tolist(), [0, 100, 200, 65000, 65535])
        # Both signed products divide toward zero, not Python's negative
        # floor: 51300/512 -> 100 and -274455/512 -> -536.
        self.assertEqual(y_curve.tolist(), [900, 1000, 2000, 3000, 2464])

        quadrant = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8
        )
        weights = vendor.expand_auto_contrast_s0_kernel(quadrant, 5)
        np.testing.assert_array_equal(
            weights,
            np.array(
                [
                    [1, 2, 3, 2, 1],
                    [4, 5, 6, 5, 4],
                    [7, 8, 9, 8, 7],
                    [4, 5, 6, 5, 4],
                    [1, 2, 3, 2, 1],
                ],
                dtype=np.int16,
            ),
        )

        divisible = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100
        np.testing.assert_array_equal(
            vendor.downsample_auto_contrast_s0_luminance(divisible, 2),
            np.array([[250, 450], [1050, 1250]], dtype=np.uint16),
        )

        # The non-divisible path stretches 5x7 to the aligned 6x8 canvas
        # before taking 2x2 averages; it is not edge padding or partial-box
        # normalization.
        nondivisible = np.arange(35, dtype=np.uint16).reshape(5, 7) * 100
        np.testing.assert_array_equal(
            vendor.downsample_auto_contrast_s0_luminance(nondivisible, 2),
            np.array(
                [
                    [322, 494, 665, 837],
                    [1442, 1614, 1785, 1957],
                    [2562, 2734, 2905, 3077],
                ],
                dtype=np.uint16,
            ),
        )

    def test_auto_contrast_s0_two_pass_filter_and_map_expansion(self):
        local_map = np.zeros((7, 7), dtype=np.uint16)
        local_map[0, 0] = 2500
        weights = np.ones((5, 5), dtype=np.int16)
        filtered = vendor.filter_auto_contrast_s0_local_map(
            local_map, weights, 128, 65535
        )
        np.testing.assert_array_equal(
            filtered,
            np.array(
                [
                    [100, 80, 60, 40, 20, 0, 0],
                    [80, 64, 48, 32, 16, 0, 0],
                    [60, 48, 36, 24, 12, 0, 0],
                    [40, 32, 24, 16, 8, 0, 0],
                    [20, 16, 12, 8, 4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
        )
        np.testing.assert_array_equal(
            vendor.filter_auto_contrast_s0_local_map(
                local_map, weights, 255, 0
            ),
            local_map,
        )

        expanded = vendor.resize_auto_contrast_s0_local_index(
            np.array([[0, 100], [200, 300]], dtype=np.uint16), (3, 3)
        )
        np.testing.assert_array_equal(
            expanded,
            np.array(
                [[0, 50, 100], [100, 150, 200], [200, 250, 300]],
                dtype=np.uint16,
            ),
        )

    def test_auto_contrast_s0_full_worker_runs_nondivisible_neutral_curve(self):
        luminance = (
            np.arange(11 * 13, dtype=np.uint16).reshape(11, 13) * 137
        )
        output = vendor.apply_auto_contrast_s0(
            luminance,
            np.ones((3, 3), dtype=np.int8),
            [0, 10000, 30000, 50000, 65535],
            [0, 10000, 30000, 50000, 65535],
            np.full(4, 512, dtype=np.int16),
            downsample_factor=4,
            kernel_size=5,
            spatial_strength_u8=128,
            clamp_radius_u16=65535,
        )
        np.testing.assert_array_equal(output, luminance)

    def test_hue_saturation_s0_zero_lattice_normalizes_and_chunks(self):
        table = np.zeros((9, 33, 33, 3), dtype=np.int16)
        image = np.array(
            [
                [[32768, 32768, 0], [30720, 34815, 4095]],
                [[36862, 34815, 2048], [33768, 31768, 1000]],
            ],
            dtype=np.uint16,
        )
        output = vendor.apply_hue_saturation_s0(
            image, table, 100, chunk_rows=1
        )
        expected = image.copy()
        # 4094,2047 projects onto 2047,1023 using signed IDIV.
        expected[1, 0, :2] = (34815, 33791)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(
            output,
            vendor.apply_hue_saturation_s0(
                image, table, 100, chunk_rows=20
            ),
        )

    def test_hue_saturation_s0_lattice_axes_channels_and_terminal_cells(self):
        table = np.zeros((9, 33, 33, 3), dtype=np.int16)
        table[2, 16, 16] = (10, 20, -30)
        table[8, 32, 32] = (7, 8, 9)
        table[1, 16, 16] = (160, -160, 320)
        image = np.array(
            [
                [
                    [32768, 32768, 1024],
                    [34815, 34815, 4095],
                    [32832, 32800, 256],
                ]
            ],
            dtype=np.uint16,
        )
        output = vendor.apply_hue_saturation_s0(image, table, 100)
        self.assertEqual(
            output.tolist(),
            [[[32788, 32738, 1034], [34823, 34824, 4102], [32802, 32860, 286]]],
        )

    def test_hue_saturation_s0_uses_native_pre_and_post_scaling(self):
        table = np.zeros((9, 33, 33, 3), dtype=np.int16)
        image = np.array([[[33769, 31767, 1000]]], dtype=np.uint16)
        reduced = vendor.apply_hue_saturation_s0(image, table, 50)
        # Q10 scale is 512.  SAR floors the negative half to -501.
        self.assertEqual(reduced.tolist(), [[[33268, 32267, 1000]]])
        expanded = vendor.apply_hue_saturation_s0(image, table, 200)
        self.assertEqual(expanded.tolist(), [[[34770, 30766, 1000]]])

    def test_hue_saturation_s0_zero_uses_centerline_and_balances(self):
        table = np.zeros((9, 33, 33, 3), dtype=np.int16)
        table[2, 16, 16] = (10, 20, -30)
        table[7, 16, 16] = (-1, 0, 0)
        image = np.array(
            [[[1000, 65000, 1024], [50000, 20000, 3585]]],
            dtype=np.uint16,
        )
        output = vendor.apply_hue_saturation_s0(
            image, table, 0, np.array([5, -7], dtype=np.int16)
        )
        # At Y=3585 the terminal numerator -510 divides by 511 toward zero.
        self.assertEqual(
            output.tolist(),
            [[[32793, 32731, 1034], [32773, 32761, 3585]]],
        )
        with self.assertRaises(ValueError):
            vendor.apply_hue_saturation_s0(
                np.array([[[32768, 32768, 4096]]], dtype=np.uint16),
                table,
                100,
            )

    def test_film_sharp_envelope_and_tone_lut_builders_preserve_integer_order(self):
        envelope = vendor.build_film_sharp_envelope_lut_u16(5, 1000, 2, 0)
        self.assertEqual(envelope.dtype, np.uint16)
        self.assertEqual(
            envelope[:9].tolist(), [0, 0, 0, 0, 250, 500, 750, 1000, 1000]
        )
        zero_width = vendor.build_film_sharp_envelope_lut_u16(3, 9, 0, 4)
        self.assertEqual(zero_width[:5].tolist(), [4, 4, 4, 9, 9])

        x = np.arange(0, 40, 4, dtype=np.uint16)
        y = np.arange(100, 1100, 100, dtype=np.uint16)
        tone = vendor.build_film_sharp_tone_lut_u16(x, y)
        self.assertEqual(tone.dtype, np.uint16)
        self.assertEqual(tone[:6].tolist(), [100, 125, 150, 175, 200, 225])
        self.assertEqual(tone[35:38].tolist(), [975, 1000, 1000])

        # 0xffff * 0xffff wraps to -131071 before signed IDIV in the DLL.
        wrapped_x = np.full(10, 65535, dtype=np.uint16)
        wrapped_x[0] = 0
        wrapped_y = np.full(10, 65535, dtype=np.uint16)
        wrapped = vendor.build_film_sharp_tone_lut_u16(wrapped_x, wrapped_y)
        self.assertEqual(int(wrapped[0]), 65534)
        self.assertEqual(int(wrapped[-1]), 65535)

    def test_film_sharp_response_lut_uses_signed_domain_and_clamping(self):
        response = vendor.build_film_sharp_response_lut_i16(
            2, 65535, 1, 0, 1000, 2047
        )
        self.assertEqual(response.shape, (65536,))
        self.assertEqual(response.dtype, np.int16)
        self.assertEqual(
            response[32768 + np.array([0, 1, 2, 3, 4, 32767])].tolist(),
            [0, 0, 0, 3, 4, 1000],
        )
        self.assertEqual(
            response[32768 - np.array([1, 2, 3, 4, 32768])].tolist(),
            [0, 0, -3, -4, -1000],
        )

    def test_film_sharp_final_compositor_uses_explicit_auxiliary_luminance(self):
        image = np.array(
            [[[11, 21, 1], [12, 22, 2], [13, 23, 3], [14, 24, 4]]],
            dtype=np.uint16,
        )
        auxiliary = np.array([[1000, 1000, 65530, 1000]], dtype=np.uint16)
        features = np.zeros((1, 4, 2), dtype=np.uint16)
        features[0, 3] = (1, 1)
        detail = np.array([[100, -100, -100, 3]], dtype=np.int16)
        envelope = np.full(65536, 65535, dtype=np.uint16)
        tone = np.full(65536, 65535, dtype=np.uint16)
        envelope[1] = 40000
        tone[1] = 40000
        response = np.zeros(65536, dtype=np.int16)
        response[32768] = 10
        response[32768 + 100] = 200
        response[32768 - 100] = -200

        output = vendor.apply_film_sharp_final_ycc(
            image,
            auxiliary,
            features,
            detail,
            envelope,
            response,
            tone,
            chunk_rows=1,
        )
        self.assertEqual(
            output.tolist(),
            [[[11, 21, 800], [12, 22, 1200], [13, 23, 65535], [14, 24, 990]]],
        )
        with self.assertRaises(ValueError):
            vendor.apply_film_sharp_final_ycc(
                image,
                image[..., 2],
                features,
                detail.astype(np.uint16),
                envelope,
                response,
                tone,
            )

    def test_film_sharp_parameter_decoder_maps_pcb_offset_and_quadrants(self):
        block = bytearray(self._film_sharp_identity_block())
        block[0x000] = 5
        struct.pack_into("<9h", block, 0x002, *range(1, 10))
        struct.pack_into("<H", block, 0x014, 1234)
        struct.pack_into("<H", block, 0x114, 259)
        block[0x116] = 4
        block[0x117] = 2
        struct.pack_into("<4H", block, 0x118, 6560, 65535, 4100, 65535)
        struct.pack_into("<4H", block, 0x120, 480, 65535, 480, 65535)
        tone_x = (0, 9600, 65528, 65529, 65530, 65531, 65532, 65533, 65534, 65535)
        struct.pack_into("<10H", block, 0x128, *tone_x)
        struct.pack_into("<Hh", block, 0x150, 7000, -700)
        prefixed = b"prefix!" + block + b"trailer"

        decoded = vendor.decode_film_sharp_parameter_block(
            prefixed, offset=7
        )
        self.assertEqual(decoded["filter_sizes"], (5, 1, 1, 1, 1))
        self.assertEqual(decoded["filter_scales"], (1234, 1, 1, 1, 1))
        np.testing.assert_array_equal(
            decoded["kernels_i16"][0],
            np.array(
                [
                    [1, 2, 3, 2, 1],
                    [4, 5, 6, 5, 4],
                    [7, 8, 9, 8, 7],
                    [4, 5, 6, 5, 4],
                    [1, 2, 3, 2, 1],
                ],
                dtype=np.int16,
            ),
        )
        self.assertEqual(decoded["blend_u16"], 259)
        self.assertEqual(decoded["mean_radius_u8"], 4)
        self.assertEqual(decoded["feature_radius_u8"], 2)
        self.assertEqual(
            decoded["envelope_controls_u16"], (6560, 65535, 4100, 65535)
        )
        self.assertEqual(
            decoded["response_controls_u16"], (480, 65535, 480, 65535)
        )
        self.assertEqual(decoded["tone_x_u16"].tolist(), list(tone_x))
        self.assertEqual(decoded["response_clamp_u16"], 7000)
        self.assertEqual(decoded["response_gain_i16"], -700)
        with self.assertRaises(ValueError):
            vendor.decode_film_sharp_parameter_block(block[:-1])
        block[0] = 4
        with self.assertRaises(ValueError):
            vendor.decode_film_sharp_parameter_block(block)

    def test_film_sharp_full_spatial_graph_constant_identity_fixture(self):
        block = self._film_sharp_identity_block()
        image = np.empty((7, 8, 3), dtype=np.uint16)
        image[..., 0] = 32768
        image[..., 1] = 32768
        image[..., 2] = 1000
        y_only_matrix = np.array(
            [10000, 0, 0, 10000, 0, 0, 10000, 0, 0], dtype=np.int16
        )

        auxiliary, features, detail = vendor.build_film_sharp_spatial_features(
            image,
            block,
            y_only_matrix,
            chunk_rows=3,
        )
        np.testing.assert_array_equal(auxiliary, image[..., 2])
        np.testing.assert_array_equal(
            features[..., 0], np.full(image.shape[:2], 32767, dtype=np.uint16)
        )
        np.testing.assert_array_equal(
            features[..., 1], np.full(image.shape[:2], 1000, dtype=np.uint16)
        )
        np.testing.assert_array_equal(
            detail, np.full(image.shape[:2], 1000, dtype=np.int16)
        )
        output = vendor.apply_film_sharp_full_ycc(
            image,
            b"ignored" + block,
            y_only_matrix,
            parameter_offset=7,
            chunk_rows=2,
        )
        np.testing.assert_array_equal(output, image)

    def test_film_sharp_spatial_graph_is_chunk_invariant(self):
        block = bytearray(self._film_sharp_identity_block())
        # Exercise both general convolution dispatches with reflected halos.
        block[0x000] = 3
        struct.pack_into("<4h", block, 0x002, 1, 2, 3, 20)
        struct.pack_into("<H", block, 0x014, 32)
        block[0x016] = 3
        struct.pack_into("<4h", block, 0x018, 0, 1, 1, 4)
        struct.pack_into("<H", block, 0x02A, 8)
        rng = np.random.default_rng(20260815)
        image = rng.integers(0, 65536, (13, 15, 3), dtype=np.uint16)

        whole = vendor.build_film_sharp_spatial_features(
            image, block, chunk_rows=64
        )
        chunked = vendor.build_film_sharp_spatial_features(
            image, block, chunk_rows=2
        )
        for expected, actual in zip(whole, chunked):
            np.testing.assert_array_equal(actual, expected)

    def test_gamma_curve2_neutral_and_region_ratios(self):
        x, y = tables.load_common_control_points(0, 1)
        neutral_y, neutral_x = vendor.make_gamma_curve2(y[0], x[0], 100, 100, 100)
        np.testing.assert_array_equal(neutral_x, x[0])
        # The two anchor interpolations are each stored as binary64 before
        # x87 subtraction.  Their mathematically integral difference lands
        # just below 29855, and _ftol2_sse truncates it to 29854.
        np.testing.assert_array_equal(neutral_y, y[0].astype(np.int64) - 1)

        changed_y, changed_x = vendor.make_gamma_curve2(y[0], x[0], 80, 100, 120)
        np.testing.assert_array_equal(changed_x, x[0])
        self.assertFalse(np.array_equal(changed_y, y[0]))
        # MakeGammaCurve2 preserves the curve's reference level at 0x5f00.
        original_anchor = vendor._interp_control(y[0], x[0], vendor.GAMMA_ANCHOR_X)
        changed_anchor = vendor._interp_control(
            changed_y, changed_x, vendor.GAMMA_ANCHOR_X
        )
        # The same binary64-store/_ftol2_sse sequence can leave the adjusted
        # curve one code value below the real-valued source anchor.
        self.assertLessEqual(abs(original_anchor - changed_anchor), 1.0)

    def test_uo_forward_inverse_tables_and_full_nega_tone(self):
        for mode in range(4):
            for index in (0, 4, 10, 23, 46):
                self.assertEqual(
                    vendor.make_uo_raw_sub2(mode, vendor.UO_RAW_X[index]),
                    vendor.UO_RAW_Y[mode][index],
                )

        selected = np.linspace(0, 46, 20, dtype=np.int64)
        for mode in (1, 2, 3):
            forward = np.asarray(vendor.UO_RAW_Y[mode], dtype=np.int64)[selected]
            inverse = vendor.apply_uo_raw_sub1(forward, mode)
            np.testing.assert_array_equal(
                inverse, np.asarray(vendor.UO_RAW_X, dtype=np.int64)[selected]
            )

        # Both helpers end in MSVC _ftol2_sse, so exact half values truncate
        # rather than using the x87 control word's nearest-even mode.
        self.assertEqual(vendor.make_uo_raw_sub2(1, 150), 1)
        inverse_tie = np.full(20, 16000, dtype=np.int64)
        inverse_tie[10] = 2079  # Mode-1 inverse is exactly 5637.5 here.
        self.assertEqual(int(vendor.apply_uo_raw_sub1(inverse_tie, 1)[10]), 5637)

        # The below-range branch walks backward using the original
        # current-to-next slope and the already converted next ordinate.
        continuation = np.full(20, 16000, dtype=np.int64)
        continuation[10] = -100
        continuation[11] = 2079
        continued = vendor.apply_uo_raw_sub1(continuation, 1)
        self.assertEqual(int(continued[10]), 3458)

        x, y = tables.load_common_control_points(0, 1)
        original_anchor = vendor._interp_control_trunc(
            y[0], x[0], vendor.GAMMA_ANCHOR_X
        )
        changed_y, changed_x = vendor.make_nega_tone2(
            y[0], x[0], original_anchor, original_anchor
        )
        np.testing.assert_array_equal(changed_x, x[0])
        np.testing.assert_array_equal(changed_y, y[0])

        # Channel two's anchor is 66294.899..., making nearest-even observably
        # wrong by one throughout a neutral full makeNegaTone2 pass.
        third_anchor = vendor._interp_control_trunc(
            y[2], x[2], vendor.GAMMA_ANCHOR_X
        )
        third_y, third_x = vendor.make_nega_tone2(
            y[2], x[2], third_anchor, third_anchor
        )
        np.testing.assert_array_equal(third_x, x[2])
        np.testing.assert_array_equal(third_y, y[2])

    def test_octet_expansion_matches_machine_tiff_precision(self):
        values = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        output = vendor.quantize_oct_to_hex(values)
        self.assertEqual(output.tolist(), [[[0, 32896, 65535]]])
        self.assertTrue(np.all(output % 257 == 0))

        native_octets = np.array([[[0, 1, 255], [127, 128, 254]]], dtype=np.uint8)
        expanded = vendor.expand_u8_to_u16(native_octets)
        np.testing.assert_array_equal(
            expanded,
            native_octets.astype(np.uint16) * np.uint16(257),
        )
        with self.assertRaises(TypeError):
            vendor.expand_u8_to_u16(native_octets.astype(np.uint16))

    def test_scaling_axis_plans_preserve_vendor_centers_and_cubic_table(self):
        indexes, weights = vendor._scaling_axis_plan(4, 2, 1, 2)
        self.assertEqual(indexes.tolist(), [[[0, 1]], [[2, 3]]])
        np.testing.assert_array_equal(
            weights, np.full((2, 1, 2), np.float32(0.5))
        )

        cubic_indexes, cubic_weights = vendor._scaling_axis_plan(4, 2, 1, 4)
        self.assertEqual(
            cubic_indexes.tolist(), [[[0, 0, 1, 2]], [[1, 2, 3, 3]]]
        )
        np.testing.assert_array_equal(
            cubic_weights,
            np.tile(
                np.array([-0.125, 0.625, 0.625, -0.125], dtype=np.float32),
                (2, 1, 1),
            ),
        )

    def test_scaling_s0_s2_two_sample_workers_and_box_prefilter(self):
        plane = np.arange(16, dtype=np.uint16).reshape(4, 4) * 10
        image = np.stack((plane, plane + 1, plane + 2), axis=2)
        expected_plane = np.array([[25, 45], [105, 125]], dtype=np.uint16)
        expected = np.stack(
            (expected_plane, expected_plane + 1, expected_plane + 2), axis=2
        )
        for worker in (vendor.apply_scaling_s0_u16, vendor.apply_scaling_s2_u16):
            output = worker(image, 2, 2, sampling=2)
            np.testing.assert_array_equal(output, expected)

        edge_plane = np.array(
            [[0, 10, 20], [30, 40, 50], [60, 70, 80]], dtype=np.uint16
        )
        edge_image = np.repeat(edge_plane[..., None], 3, axis=2)
        expected_prefilter = np.array([[20, 35], [65, 80]], dtype=np.uint16)
        output = vendor.apply_scaling_s2_u16(
            edge_image, 2, 2, pre_scale=2, sampling=1
        )
        np.testing.assert_array_equal(output[..., 0], expected_prefilter)
        np.testing.assert_array_equal(output[..., 1], expected_prefilter)
        np.testing.assert_array_equal(output[..., 2], expected_prefilter)

    def test_scaling_s0_groups_each_cubic_sample_before_accumulating(self):
        image = np.array(
            [
                [
                    [15461, 16079, 38923],
                    [15804, 64594, 7801],
                    [35577, 5616, 43173],
                    [58657, 9475, 13520],
                ],
                [
                    [35357, 35690, 31787],
                    [31824, 1100, 18778],
                    [31542, 18614, 61418],
                    [43900, 30762, 45508],
                ],
                [
                    [57478, 56884, 20303],
                    [62985, 61574, 54333],
                    [19410, 11101, 48352],
                    [52085, 18548, 561],
                ],
                [
                    [7125, 12716, 48388],
                    [41002, 29645, 64117],
                    [32582, 9753, 54195],
                    [22435, 24548, 55303],
                ],
            ],
            dtype=np.uint16,
        )

        output = vendor.apply_scaling_s0_u16(image, 3, 3, sampling=2)

        # Native FmScalingS0 groups the four cubic taps for each sample,
        # then adds the two sample totals.  Flattening all eight products
        # into one float32 accumulator gives 33003 instead of this oracle.
        self.assertEqual(output[1, 2, 2], 33004)

    def test_trimming_and_orthogonal_rotation_use_clockwise_tenths(self):
        image = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
        trimmed = vendor.apply_trimming_u16(image, 1, 1, 2, 2)
        np.testing.assert_array_equal(trimmed, image[1:3, 1:3])

        clockwise = vendor.apply_rotation_orthogonal_u16(image, 900)
        np.testing.assert_array_equal(clockwise, np.rot90(image, k=-1))
        cropped_rotation = vendor.apply_rotation_orthogonal_u16(
            image, 2700, trim=(1, 0, 2, 2)
        )
        np.testing.assert_array_equal(
            cropped_rotation, np.rot90(image, k=1)[0:2, 1:3]
        )
        with self.assertRaises(ValueError):
            vendor.apply_rotation_orthogonal_u16(image, 450)

    def test_arbitrary_rotation_fixed_coordinates_and_break_table(self):
        sine, cosine = vendor._rotation_fixed_coordinates(
            4, 45.0 * np.pi / 180.0
        )
        self.assertEqual(sine.tolist(), [1448, 724, 0, -724])
        self.assertEqual(cosine.tolist(), [-1448, -724, 0, 724])

        image = np.full((7, 7, 3), 1234, dtype=np.uint16)
        rotated, breaks = vendor.apply_rotation_arbitrary_u16(
            image, 450, chunk_rows=2
        )
        self.assertEqual(rotated.shape, (10, 10, 3))
        self.assertEqual(breaks.shape, (10, 4))
        self.assertEqual(breaks.dtype, np.int32)
        self.assertTrue(np.all(np.diff(breaks, axis=1) >= 0))
        self.assertTrue(np.all((breaks >= 0) & (breaks <= 10)))
        valid_rows = np.flatnonzero(breaks[:, 1] < breaks[:, 2])
        self.assertGreater(valid_rows.size, 0)
        center_row = int(valid_rows[valid_rows.size // 2])
        center_x = int((breaks[center_row, 1] + breaks[center_row, 2]) // 2)
        self.assertEqual(rotated[center_row, center_x].tolist(), [1234] * 3)

        cropped, cropped_breaks = vendor.apply_rotation_arbitrary_u16(
            image, 450, trim=(2, 1, 5, 6), chunk_rows=3
        )
        self.assertEqual(cropped.shape, (6, 5, 3))
        self.assertEqual(cropped_breaks.shape, (6, 4))

    def test_blank_oct_and_hex_color_placement_and_expansion(self):
        oct_image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        oct_output = vendor.apply_blank_oct_color_u8(
            oct_image, 4, 3, 1, 1, (0, 128, 255)
        )
        self.assertEqual(oct_output[0, 0].tolist(), [0, 128, 255])
        np.testing.assert_array_equal(oct_output[1, 1:3], oct_image[0])

        hex_image = (oct_image.astype(np.uint16) * 1000)
        hex_output = vendor.apply_blank_hex_color_u16(
            hex_image, 4, 3, 1, 1, (0, 128, 255)
        )
        self.assertEqual(hex_output[0, 0].tolist(), [255, 33023, 65535])
        np.testing.assert_array_equal(hex_output[1, 1:3], hex_image[0])
        with self.assertRaises(ValueError):
            vendor.apply_blank_hex_color_u16(
                hex_image, 2, 1, 1, 0, (0, 0, 0)
            )

    def test_blank_postbreak_prefill_and_integer_boundary_mean(self):
        image = np.full((3, 5, 3), 90, dtype=np.uint8)
        breaks = np.tile(np.array([1, 2, 3, 4], dtype=np.int32), (3, 1))
        output = vendor.apply_blank_oct_color_u8(
            image,
            7,
            5,
            1,
            1,
            (0, 0, 0),
            rotation_breaks=breaks,
        )
        expected = np.array(
            [[20, 40, 60, 40, 20], [30, 60, 90, 60, 30], [20, 40, 60, 40, 20]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(output[1:4, 1:6, 0], expected)
        np.testing.assert_array_equal(output[..., 0], output[..., 1])
        np.testing.assert_array_equal(output[..., 1], output[..., 2])

        image_u16 = image.astype(np.uint16) * 100
        output_u16 = vendor.apply_blank_hex_color_u16(
            image_u16,
            7,
            5,
            1,
            1,
            (0, 0, 0),
            rotation_breaks=breaks,
        )
        # Hex black is 0x00ff, so its integer mean deliberately differs from
        # simply multiplying the Oct result by 100.
        self.assertEqual(output_u16.dtype, np.uint16)
        self.assertEqual(output_u16[2, 3, 0], 9000)
        self.assertEqual(output_u16[2, 1, 0], (3 * 9000 + 6 * 255) // 9)


if __name__ == "__main__":
    unittest.main()
