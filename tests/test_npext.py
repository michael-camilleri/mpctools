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


class TestRunLengths(unittest.TestCase):

    def test_standard(self):
        a = np.array([0, 0, 0, 1, 1, np.NaN, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 2, 3, 0.56, 0.56, 0.56])
        self.assertTrue((npext.run_lengths(a, how='i') == [3, 2, 5, 1, 4, 1, 1, 3]).all())
        self.assertTrue((npext.run_lengths(a, how='a') == [3, 2, 1, 5, 1, 4, 1, 1, 3]).all())

    def test_NaN_handling(self):
        # Test 1
        a = np.array([0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2])
        self.assertTrue((npext.run_lengths(a, how='o') == [1, 2, 1]).all())
        self.assertTrue((npext.run_lengths(a, how='a') == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]).all())
        self.assertTrue((npext.run_lengths(a, how='i') == [1, 1, 1, 1, 1, 1, 1]).all())
        # Test with None
        a = np.arange(100)
        self.assertEqual(len(npext.run_lengths(a, how='o')), 0)

    def test_NaN_handles_edges(self):
        # Define Array
        a = np.array([0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2, np.NaN])
        b = np.array([np.NaN, np.NaN, 0, 5, np.NaN, 1., np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2])
        # Test
        c = npext.run_lengths(a, how='o')
        self.assertTrue(np.array_equal(c, np.array([1, 2, 1, 1])))
        self.assertTrue((npext.run_lengths(b, how='o') == [2, 1, 2, 1]).all())

    def test_handle_reshaping(self):
        # Define array
        a = np.array([np.NaN, np.NaN, 1, 1, np.NaN, 2, np.NaN, np.NaN, 3, 3, 3, -np.Inf]).reshape([4, 3])
        self.assertTrue((npext.run_lengths(a, how='o') == [2, 1, 2]).all())
        self.assertTrue((npext.run_lengths(a, how='a') == [2, 2, 1, 1, 2, 3, 1]).all())