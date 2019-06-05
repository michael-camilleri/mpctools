import unittest
import numpy as np
from scipy.special import softmax

from mpctools.extensions import npext


class TestInvertSoftmax(unittest.TestCase):

    def test_correct_1D(self):
        np.random.seed(100)
        for _ in range(100):
            sample = np.random.dirichlet(np.ones(np.random.randint(10, 20)))
            inverse = npext.invert_softmax(sample)
            self.assertTrue(np.allclose(sample, softmax(inverse)))

    def test_correct_kD(self):
        np.random.seed(100)
        for _ in range(100):
            sample = np.random.dirichlet(np.ones(15), np.random.choice(np.random.randint(2, 5)))
            inverse = npext.invert_softmax(sample)
            self.assertTrue(np.allclose(sample, softmax(inverse, axis=-1)))
