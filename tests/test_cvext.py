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
        higher = hist > limit
        to_clip = (hist[higher] - limit).sum(axis=-1, keepdims=True)
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
        a = np.pad(np.random.randint(0, 255, [H, W], dtype=np.uint8), pad_width=8, mode='symmetric')
        hist_s = np.zeros([H, W, 256], dtype=np.uint16)
        cvext.SwCLAHE._SwCLAHE__update_hist(a, 8, 8, hist_s)
        hist_b = np.zeros([H, W, 256], dtype=np.uint16)
        self.brute_force_equalisation(a, 8, 8, hist_b)
        self.assertTrue(np.array_equal(hist_s, hist_b))

        # Do Second One
        b = np.pad(np.random.randint(0, 255, [H, W], dtype=np.uint8), pad_width=[[3], [7]], mode='symmetric')
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
        lut_s = np.zeros([H, W, 256], dtype=np.uint8)
        cvext.SwCLAHE._SwCLAHE__clip_limit(h, 10.0, lut_s, 0.08)
        lut_b = self.brute_force_clipping(h, 10, 0.08).astype(np.uint8)
        self.assertTrue(np.array_equal(lut_s, lut_b))

        # Do Second One: this time, scaler < 256/(256*15) = 0.06667
        h = np.random.randint(0, 30, [H, W, 256]).astype(float)
        lut_s = np.zeros([H, W, 256], dtype=np.uint8)
        cvext.SwCLAHE._SwCLAHE__clip_limit(h, 10.0, lut_s, 0.05)
        lut_b = self.brute_force_clipping(h, 10, 0.05).astype(np.uint8)
        self.assertTrue(np.array_equal(lut_s, lut_b))