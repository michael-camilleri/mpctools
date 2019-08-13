import unittest
import numpy as np
from scipy.special import softmax

from mpctools.extensions import npext


class TestInvertSoftmax(unittest.TestCase):

    def test_sum_to_0(self):
        np.random.seed(100)
        for _ in range(50):
            sample = np.random.dirichlet(np.ones(np.random.randint(10, 20)))
            inverse = npext.invert_softmax(sample)
            self.assertAlmostEqual(inverse.sum(), 0)

    def test_correct_1D(self):
        np.random.seed(100)
        for _ in range(50):
            sample = np.random.dirichlet(np.ones(np.random.randint(10, 20)))
            inverse = npext.invert_softmax(sample)
            self.assertTrue(np.allclose(sample, softmax(inverse)))

    def test_correct_2D(self):
        np.random.seed(100)
        for _ in range(50):
            sample = np.random.dirichlet(np.ones(15), np.random.choice(np.random.randint(2, 5))+1)
            inverse = npext.invert_softmax(sample)
            self.assertTrue(np.allclose(sample, softmax(inverse, axis=-1)))

    def test_index_0(self):
        np.random.seed(100)
        for _ in range(20):
            sample = np.random.dirichlet(np.ones(np.random.randint(10, 15)))
            for index0 in range(len(sample)):
                inverse = npext.invert_softmax(sample, enforce_unique=int(index0))
                self.assertAlmostEqual(inverse[int(index0)], 0)

    def test_correct_1D_indexed(self):
        np.random.seed(100)
        for _ in range(20):
            sample = np.random.dirichlet(np.ones(np.random.randint(10, 15)))
            for index0 in range(len(sample)):
                inverse = npext.invert_softmax(sample, enforce_unique=int(index0))
                self.assertTrue(np.allclose(sample, softmax(inverse)))

    def test_correct_2D_indexed(self):
        np.random.seed(100)
        for _ in range(20):
            sample = np.random.dirichlet(np.ones(15), np.random.choice(np.random.randint(2, 5))+1)
            for index0 in range(15):
                inverse = npext.invert_softmax(sample, enforce_unique=int(index0))
                self.assertTrue(np.allclose(sample, softmax(inverse, axis=-1)))
                self.assertTrue(np.allclose(inverse[:, int(index0)], 0))

    def test_correct_KD_indexed(self):
        np.random.seed(100)
        for _ in range(50):
            sample = np.random.dirichlet(np.ones(np.random.randint(1, 15)),
                                         np.random.randint(1, 6, size=np.random.randint(1, 3)))
            for index0 in range(sample.shape[-1]):
                inverse = npext.invert_softmax(sample, enforce_unique=int(index0))
                self.assertTrue(np.allclose(sample, softmax(inverse, axis=-1)))
                self.assertTrue(np.allclose(inverse[..., int(index0)], 0))


class TestNaNRunLengths(unittest.TestCase):

    def test_correct_values(self):
        # Define Array
        a = np.array([0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2])
        self.assertTrue((npext.null_run_lengths(a) == [1, 2, 1]).all())

    def test_handles_edges(self):
        # Define Array
        a = np.array([0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2, np.NaN])
        b = np.array([np.NaN, np.NaN, 0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2])
        # Test
        self.assertTrue((npext.null_run_lengths(a) == [1, 2, 1, 1]).all())
        self.assertTrue((npext.null_run_lengths(b) == [2, 1, 2, 1]).all())

    def test_handle_reshaping(self):
        # Define array
        a = np.array([np.NaN, np.NaN, 0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN]).reshape([4, 3])
        self.assertTrue((npext.null_run_lengths(a) == [2, 1, 2, 1]).all())