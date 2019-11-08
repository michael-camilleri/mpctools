import unittest
import numpy as np

from mpctools.extensions import cvext

W = 256
H = 64


class TestSWAHE(unittest.TestCase):

    def brute_force_equalisation(self, img, pad_r, pad_c, hist):

        valid_r = [pad_r, img.shape[0] - pad_r - 1]
        valid_c = [pad_c, img.shape[1] - pad_c - 1]
        # Iterate over cells.
        for r in range(*valid_r):
            for c in range(*valid_c):
                # Iterate over neighbourhood
                for n_r in range(r - pad_r, r + pad_r + 1):
                    for n_c in range(c - pad_c, c + pad_c + 1):
                        hist[r - pad_r, c - pad_c, img[n_r, n_c]] += 1

    def test_consistency_raw(self):
        """
        Test based on consistency with brute-force way of doing this.
        :return:
        """
        # Seed and create objects
        np.random.seed(100)

        # Create First Matrix and compute Histogram Equalisation based on CLAHE and Brute-Force
        a = np.pad(np.random.randint(0, 255, [H, W], dtype=np.uint8), pad_width=8, mode='symmetric')
        hist_s = np.zeros([H, W, 256], dtype=np.uint16)
        cvext.SWAHE._SWAHE__update_hist(a, 8, 8, hist_s)
        hist_b = np.zeros([H, W, 256], dtype=np.uint16)
        self.brute_force_equalisation(a, 8, 8, hist_b)
        self.assertTrue(np.array_equal(hist_s, hist_b))

        # Do Second One
        b = np.pad(np.random.randint(0, 255, [H, W], dtype=np.uint8), pad_width=[[3], [7]], mode='symmetric')
        hist_s = np.zeros([H, W, 256], dtype=np.uint16)
        cvext.SWAHE._SWAHE__update_hist(b, 3, 7, hist_s)
        hist_b = np.zeros([H, W, 256], dtype=np.uint16)
        self.brute_force_equalisation(b, 3, 7, hist_b)
        self.assertTrue(np.array_equal(hist_s, hist_b))
