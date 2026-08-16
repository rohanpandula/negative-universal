#!/usr/bin/env python3
"""Regression tests for the legacy illustrative convenience pipeline."""

import unittest

import numpy as np

from noritsu import pipeline


class IllustrativePipelineTests(unittest.TestCase):
    def test_calibration_reuses_saved_density_bounds(self):
        image = np.linspace(0.08, 0.92, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        identity_lut = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (3, 1))
        calibration = {
            "lut": identity_lut,
            "dmin": np.array([0.05, 0.10, 0.15], dtype=np.float32),
            "dmax": np.array([0.80, 0.90, 1.00], dtype=np.float32),
        }

        _, metadata, _ = pipeline.process_negative(
            image,
            calib=calibration,
            crop=False,
            use_border=False,
        )
        np.testing.assert_allclose(metadata["dmin"], calibration["dmin"])
        np.testing.assert_allclose(metadata["dmax"], calibration["dmax"])


if __name__ == "__main__":
    unittest.main()
