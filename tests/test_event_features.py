#!/usr/bin/env python3
"""Dataset-free sanity tests for canonical di-Higgs utilities."""

import os
import sys
import unittest

import awkward as ak
import numpy as np
import vector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.dihiggs import R_hh_func, compute_significance, pair_from_4jets


class TestEventLevelHelpers(unittest.TestCase):
    def setUp(self):
        ak.behavior.update(vector.backends.awkward.behavior)
        pt = [[100.0, 80.0, 60.0, 50.0]]
        eta = [[0.5, -0.3, 1.2, -0.8]]
        phi = [[0.1, 1.5, -0.5, 2.0]]
        mass = [[15.0, 12.0, 18.0, 10.0]]
        vectors = ak.zip(
            {"pt": pt, "eta": eta, "phi": phi, "mass": mass},
            with_name="Momentum4D",
        )
        self.jets4 = ak.zip(
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "mass": mass,
                "vector": vectors,
            },
            with_name="Momentum4D",
        )

    def test_pair_from_4jets(self):
        lead, sub, hh = pair_from_4jets(self.jets4)
        self.assertEqual(len(lead), 1)
        self.assertEqual(len(sub), 1)
        self.assertEqual(len(hh), 1)
        self.assertGreater(float(lead.mass[0]), 0.0)
        self.assertGreater(float(sub.mass[0]), 0.0)
        self.assertGreater(float(hh.mass[0]), 0.0)
        self.assertGreaterEqual(float(lead.pt[0]), float(sub.pt[0]))

    def test_r_hh_func(self):
        r0 = R_hh_func(np.array([125.0]), np.array([120.0]))
        self.assertAlmostEqual(float(r0[0]), 0.0, places=7)

    def test_compute_significance(self):
        sig_mh1 = np.array([124.0, 126.0, 180.0])
        sig_mh2 = np.array([119.0, 121.0, 170.0])
        bkg_mh1 = np.array([90.0, 130.0, 200.0])
        bkg_mh2 = np.array([80.0, 118.0, 210.0])

        res = compute_significance(
            sig_mh1,
            sig_mh2,
            bkg_mh1,
            bkg_mh2,
            region="circular",
            r_hh_cut=25.0,
        )

        self.assertIn("S", res)
        self.assertIn("B", res)
        self.assertIn("significance", res)
        self.assertGreaterEqual(res["S"], 0.0)
        self.assertGreaterEqual(res["B"], 0.0)
        self.assertGreaterEqual(res["significance"], 0.0)


if __name__ == "__main__":
    unittest.main()
