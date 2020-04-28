"""
This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

import numpy as np
import unittest

from mpctools.extensions import cvext

W = 256
H = 64


class TestSWAHE(unittest.TestCase):
    @staticmethod
    def brute_force_equalisation(img, pad_r, pad_c, hist):

        valid_r = [pad_r, img.shape[0] - pad_r - 1]
        valid_c = [pad_c, img.shape[1] - pad_c - 1]
        # Iterate over cells.
        for r in range(*valid_r):
            for c in range(*valid_c):
                # Iterate over neighbourhood
                for n_r in range(r - pad_r, r + pad_r + 1):
                    for n_c in range(c - pad_c, c + pad_c + 1):
                        hist[r - pad_r, c - pad_c, img[n_r, n_c]] += 1

    @staticmethod
    def brute_force_clipping(hist, limit, scaler):
        hist = hist.copy()
        higher = hist > limit
        to_clip = (np.ma.masked_array(hist, np.logical_not(higher)) - limit).sum(
            axis=-1, keepdims=True
        )
        hist[higher] = limit
        hist += to_clip / 256.0
        return np.around(hist.cumsum(axis=-1) * scaler)

    def test_histogram_update(self):
        """
        Test based on consistency with brute-force way of doing this (i.e. going over all cells)
        """
        # Seed and create objects
        np.random.seed(100)

        # Create First Matrix and compute Histogram Equalisation based on CLAHE and Brute-Force
        a = np.pad(
            np.random.randint(0, 255, [H, W], dtype=np.uint8),
            pad_width=8,
            mode="symmetric",
        )
        hist_s = np.zeros([H, W, 256], dtype=np.uint16)
        cvext.SwCLAHE._SwCLAHE__update_hist(a, 8, 8, hist_s)
        hist_b = np.zeros([H, W, 256], dtype=np.uint16)
        self.brute_force_equalisation(a, 8, 8, hist_b)
        self.assertTrue(np.array_equal(hist_s, hist_b))

        # Do Second One
        b = np.pad(
            np.random.randint(0, 255, [H, W], dtype=np.uint8),
            pad_width=[[3], [7]],
            mode="symmetric",
        )
        hist_s = np.zeros([H, W, 256], dtype=np.uint16)
        cvext.SwCLAHE._SwCLAHE__update_hist(b, 3, 7, hist_s)
        hist_b = np.zeros([H, W, 256], dtype=np.uint16)
        self.brute_force_equalisation(b, 3, 7, hist_b)
        self.assertTrue(np.array_equal(hist_s, hist_b))

    def test_clipping(self):
        """
        Method to test the clipping. This is based on an alternative way of computing it...
        """
        # Seed and create objects
        np.random.seed(100)

        # Create First Histogram, and compute Clipped LUT based on brute-force and own method.
        # The Scaler computation is based as follows. With a randint of 20, the mean is 10. This means that across 256
        #  channels, the total count is on average 2560. This would amount to having 2560 cells for histogram. Hence,
        #  256/2560 is 0.1. However, to be sure, use a slightly smaller value.
        h = np.random.randint(0, 20, [H, W, 256]).astype(float)
        h_cpy = h.copy()

        # Perform the SwClahe version
        lut_s = np.zeros([H, W, 256], dtype=np.uint8)
        cvext.SwCLAHE._SwCLAHE__clip_limit(h, 10.0, lut_s, 0.08)
        # Ensure that nothing changed...
        self.assertTrue(np.array_equal(h_cpy, h))
        # Perform Brute-Force version
        lut_b = self.brute_force_clipping(h, 10, 0.08).astype(np.uint8)
        self.assertTrue(np.array_equal(lut_s, lut_b))

        # Do Second One: this time, scaler < 256/(256*15) = 0.06667
        h = np.random.randint(0, 30, [H, W, 256]).astype(float)
        h_cpy = h.copy()
        # Perform the SwClahe version: however, do not initialise lut_s....
        cvext.SwCLAHE._SwCLAHE__clip_limit(h, 10.0, lut_s, 0.05)
        # Ensure that nothing changed...
        self.assertTrue(np.array_equal(h_cpy, h))
        lut_b = self.brute_force_clipping(h, 10, 0.05).astype(np.uint8)
        self.assertTrue(np.array_equal(lut_s, lut_b))

    def test_side_effects(self):
        """
        Test for various side-effects of the methods...
        :return:
        """
        # Seed and create objects
        np.random.seed(100)

        # Create SwCLAHE object
        swclahe = cvext.SwCLAHE([W, H])

        # Create 'Image'
        a = np.random.randint(0, 255, [H, W], dtype=np.uint8)
        a_cpy = a.copy()
        b = np.random.randint(0, 255, [H, W], dtype=np.uint8)

        # ----- Test that Histogram is currently 0 ------- #
        self.assertTrue((swclahe.transform(a_cpy) == 0).all())
        # Check that no modification so far...
        self.assertTrue(np.array_equal(a, a_cpy))

        # ------- Now Test histogram update -------- #
        swclahe.update_model(a_cpy)
        # Check that no modification so far...
        self.assertTrue(np.array_equal(a, a_cpy))
        # Now Histogram is not 0!
        tr_1 = swclahe.transform(a_cpy)
        self.assertFalse((tr_1 == 0).all())
        # However, applying same transform again, should not have changed anything...
        self.assertTrue(np.array_equal(tr_1, swclahe.transform(a_cpy)))

        # ------- Do another update -------- #
        swclahe.update_model(a_cpy)
        # However, applying same transform again, should not have changed anything...
        self.assertTrue(np.array_equal(tr_1, swclahe.transform(a_cpy)))
        swclahe.transform(b)
        self.assertTrue(np.array_equal(tr_1, swclahe.transform(a_cpy)))
        # However, applying something else does...
        swclahe.update_model(b)
        self.assertFalse(np.array_equal(tr_1, swclahe.transform(a_cpy)))

        # ------- Do final update with clear histogram -------- #
        swclahe.clear_histogram()
        swclahe.update_model(a_cpy)
        # However, applying same transform again, should not have changed anything...
        self.assertTrue(np.array_equal(tr_1, swclahe.transform(a_cpy)))

        # finally check that apply works as expected
        self.assertTrue(np.array_equal(tr_1, swclahe.apply(a_cpy)))
        self.assertTrue(np.array_equal(tr_1, swclahe.apply(a_cpy)))
        self.assertTrue(np.array_equal(tr_1, swclahe.apply(a_cpy, clear=False)))
        self.assertFalse(np.array_equal(tr_1, swclahe.transform(b)))
        swclahe.apply(b, clear=False)
        self.assertFalse(np.array_equal(tr_1, swclahe.transform(a_cpy)))
        self.assertTrue(np.array_equal(a, a_cpy))


class TestIntersectionOverUnion(unittest.TestCase):
    def test_equal(self):
        # A bunch of equal rectangles
        self.assertEqual(
            cvext.intersection_over_union([0, 0, 100, 20], [0, 0, 100, 20]), 1.0
        )
        self.assertEqual(
            cvext.intersection_over_union([10, 20, 40, 20], [10, 20, 40, 20]), 1.0
        )
        self.assertEqual(
            cvext.intersection_over_union([-5, -1, 40, 30], [-5, -1, 40, 30]), 1.0
        )

    def test_pred_within(self):
        # A bunch of predictions fully contained within the ground-truth
        self.assertEqual(
            cvext.intersection_over_union([0, 0, 10, 10], [0, 0, 5, 10]), 0.5
        )
        self.assertEqual(
            cvext.intersection_over_union([10, 10, 10, 10], [12, 11, 5, 5]), 0.25
        )
        self.assertEqual(
            cvext.intersection_over_union([-1, -1, 20, 20], [0, 0, 10, 10]), 0.25
        )

    def test_gt_within(self):
        # A bunch of predictions fully encompassing the ground-truth
        self.assertEqual(
            cvext.intersection_over_union([0, 0, 5, 10], [0, 0, 10, 10]), 0.5
        )
        self.assertEqual(
            cvext.intersection_over_union([12, 11, 5, 5], [10, 10, 10, 10]), 0.25
        )
        self.assertEqual(
            cvext.intersection_over_union([0, 0, 5, 5], [-1, -1, 10, 10]), 0.25
        )

    def test_outwith(self):
        # A bunch of predictions entirely disjoint from the ground-truth
        self.assertEqual(
            cvext.intersection_over_union([0, 0, 5, 5], [5, 5, 6, 10]), 0.0
        )
        self.assertEqual(
            cvext.intersection_over_union([5, 5, 6, 10], [0, 0, 5, 5]), 0.0
        )
        self.assertEqual(
            cvext.intersection_over_union([-10, -10, 10, 11], [5, 5, 6, 10]), 0.0
        )

    def test_partial(self):
        # A bunch of predictions with partial overlap
        self.assertEqual(
            cvext.intersection_over_union([2, 3, 10, 10], [7, 8, 10, 10]), 25 / 175
        )
        self.assertEqual(
            cvext.intersection_over_union([2, 3, 10, 10], [7, 8, 20, 20]), 25 / 475
        )
        self.assertEqual(
            cvext.intersection_over_union([7, 8, 10, 10], [2, 3, 10, 10]), 25 / 175
        )
        self.assertEqual(
            cvext.intersection_over_union([7, 8, 20, 20], [2, 3, 10, 10]), 25 / 475
        )
