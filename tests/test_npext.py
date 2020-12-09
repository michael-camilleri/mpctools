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

import unittest
import numpy as np
import itertools as it
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
            sample = np.random.dirichlet(np.ones(15), np.random.choice(np.random.randint(2, 5)) + 1)
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
            sample = np.random.dirichlet(np.ones(15), np.random.choice(np.random.randint(2, 5)) + 1)
            for index0 in range(15):
                inverse = npext.invert_softmax(sample, enforce_unique=int(index0))
                self.assertTrue(np.allclose(sample, softmax(inverse, axis=-1)))
                self.assertTrue(np.allclose(inverse[:, int(index0)], 0))

    def test_correct_KD_indexed(self):
        np.random.seed(100)
        for _ in range(50):
            sample = np.random.dirichlet(
                np.ones(np.random.randint(1, 15)),
                np.random.randint(1, 6, size=np.random.randint(1, 3)),
            )
            for index0 in range(sample.shape[-1]):
                inverse = npext.invert_softmax(sample, enforce_unique=int(index0))
                self.assertTrue(np.allclose(sample, softmax(inverse, axis=-1)))
                self.assertTrue(np.allclose(inverse[..., int(index0)], 0))


class TestRunLengths(unittest.TestCase):
    def test_standard(self):
        a = np.array([0, 0, 0, 1, 1, np.NaN, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 2, 3, 0.56, 0.56, 0.56])
        self.assertTrue(np.array_equal(npext.run_lengths(a, how="i"), [3, 2, 5, 1, 4, 1, 1, 3]))
        self.assertTrue((npext.run_lengths(a, how="a") == [3, 2, 1, 5, 1, 4, 1, 1, 3]).all())

    def test_NaN_handling(self):
        # Test 1
        a = np.array([0, 5, np.NaN, 1.0, np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2])
        self.assertTrue((npext.run_lengths(a, how="o") == [1, 2, 1]).all())
        self.assertTrue((npext.run_lengths(a, how="a") == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1]).all())
        self.assertTrue((npext.run_lengths(a, how="i") == [1, 1, 1, 1, 1, 1, 1]).all())
        # Test with None
        a = np.arange(100)
        self.assertEqual(len(npext.run_lengths(a, how="o")), 0)

    def test_NaN_handles_edges(self):
        # Define Array
        a = np.array([0, 5, np.NaN, 1.0, np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2, np.NaN])
        b = np.array(
            [np.NaN, np.NaN, 0, 5, np.NaN, 1.0, np.NaN, np.NaN, 4.3, -5, -np.Inf, np.NaN, 5.2]
        )
        # Test
        c = npext.run_lengths(a, how="o")
        self.assertTrue(np.array_equal(c, np.array([1, 2, 1, 1])))
        self.assertTrue((npext.run_lengths(b, how="o") == [2, 1, 2, 1]).all())

    def test_handle_reshaping(self):
        # Define array
        a = np.array([np.NaN, np.NaN, 1, 1, np.NaN, 2, np.NaN, np.NaN, 3, 3, 3, -np.Inf]).reshape(
            [4, 3]
        )
        self.assertTrue((npext.run_lengths(a, how="o") == [2, 1, 2]).all())
        self.assertTrue((npext.run_lengths(a, how="a") == [2, 2, 1, 1, 2, 3, 1]).all())

    def test_position_return(self):
        a = np.array([0, 0, 0, 1, 1, np.NaN, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 2, 3, 0.56, 0.56, 0.56])
        self.assertTrue(
            (
                npext.run_lengths(a, how="a", return_positions=True)[1]
                == [0, 3, 5, 6, 11, 12, 16, 17, 18]
            ).all()
        )
        self.assertTrue(
            (
                npext.run_lengths(a, how="i", return_positions=True)[1]
                == [0, 3, 6, 11, 12, 16, 17, 18]
            ).all()
        )
        self.assertTrue((npext.run_lengths(a, how="o", return_positions=True)[1] == [5]).all())

    def test_value_return(self):
        a = np.array([0, 0, 0, 1, 1, np.NaN, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 2, 3, 0.56, 0.56, 0.56])
        self.assertTrue(
            npext.array_nan_equal(
                npext.run_lengths(a, how="a", return_values=True)[1],
                [0, 1, np.NaN, 1, -1, 1, 2, 3, 0.56],
            )
        )
        self.assertTrue(
            npext.array_nan_equal(
                npext.run_lengths(a, how="i", return_values=True)[1], [0, 1, 1, -1, 1, 2, 3, 0.56],
            )
        )
        self.assertTrue(
            npext.array_nan_equal(npext.run_lengths(a, how="o", return_values=True)[1], [np.NaN])
        )


class TestHungarian(unittest.TestCase):
    def setUp(self):
        np.random.seed(100)

    def test_basecases(self):
        r, c = npext.hungarian(1 - np.eye(7), maximise=False)
        self.assertTrue(np.array_equal(r, np.arange(7)))
        self.assertTrue(np.array_equal(c, np.arange(7)))

        r, c = npext.hungarian(np.eye(7), maximise=True)
        self.assertTrue(np.array_equal(r, np.arange(7)))
        self.assertTrue(np.array_equal(c, np.arange(7)))

        r, c = npext.hungarian(1 - np.eye(5, 7), maximise=False)
        self.assertTrue(np.array_equal(r, np.arange(5)))
        self.assertTrue(np.array_equal(c, np.arange(5)))

        r, c = npext.hungarian(np.eye(5, 7), maximise=True)
        self.assertTrue(np.array_equal(r, np.arange(5)))
        self.assertTrue(np.array_equal(c, np.arange(5)))

    def test_general_case(self):
        _cost = np.asarray(
            [
                [5.9, 0.0, 2.3, 3.1, 3.5],
                [1.2, 10, 0.8, 6.3, 8.1],
                [11, 5.4, 20, 0.4, 9.1],
                [5.0, 3.8, 0.2, 6.1, 3.1],
            ]
        )
        for _order in it.permutations([0, 1, 2, 3]):
            r, c = npext.hungarian(_cost[_order, :], maximise=True)
            self.assertTrue(np.array_equal(r, [0, 1, 2, 3]))
            self.assertTrue(np.array_equal(c, _order))

    def test_inadmissables(self):
        _cost = 1 - np.eye(7)
        _cost[5, :] = np.NaN
        r, c = npext.hungarian(_cost)
        self.assertTrue(np.array_equal(r, [0, 1, 2, 3, 4, 6]))
        self.assertTrue(np.array_equal(c, [0, 1, 2, 3, 4, 6]))
        r, c = npext.hungarian(_cost, row_labels=["A", "B", "C", "D", "E", "F", "G"])
        self.assertTrue(np.array_equal(r, ["A", "B", "C", "D", "E", "G"]))
        self.assertTrue(np.array_equal(c, [0, 1, 2, 3, 4, 6]))
        r, c = npext.hungarian(_cost, col_labels=["A", "B", "C", "D", "E", "F", "G"])
        self.assertTrue(np.array_equal(r, [0, 1, 2, 3, 4, 6]))
        self.assertTrue(np.array_equal(c, ["A", "B", "C", "D", "E", "G"]))

        _cost = np.eye(7)
        _cost[5, :] = np.NaN
        r, c = npext.hungarian(_cost, maximise=True)
        self.assertTrue(np.array_equal(r, [0, 1, 2, 3, 4, 6]))
        self.assertTrue(np.array_equal(c, [0, 1, 2, 3, 4, 6]))

    def test_symmetry(self):
        for _ in range(10):
            _sz = np.random.randint(3, 10, size=2)
            _cost = np.random.randint(2, 100, _sz)
            # Minimum
            r1, c1 = npext.hungarian(_cost)
            c2, r2 = npext.hungarian(_cost.T)
            self.assertTrue(np.array_equal(r1, np.sort(r2)))
            self.assertTrue(np.array_equal(c1, c2[np.argsort(r2)]))
            # Maximum
            r1, c1 = npext.hungarian(_cost, maximise=True)
            c2, r2 = npext.hungarian(_cost.T, maximise=True)
            self.assertTrue(np.array_equal(r1, np.sort(r2)))
            self.assertTrue(np.array_equal(c1, c2[np.argsort(r2)]))

    def test_degenerates(self):
        for _ in range(10):
            _sz = np.random.randint(3, 10, size=2)
            self.assertEqual(npext.hungarian(np.full(_sz, np.NaN)), ([], []))
            self.assertEqual(npext.hungarian(np.full(_sz, np.NaN), False), ([], []))
            self.assertEqual(npext.hungarian(np.ones(_sz), cutoff=0.5), ([], []))
            self.assertEqual(npext.hungarian(np.ones(_sz), True, cutoff=2), ([], []))

    def test_hard_cases(self):
        _cost = np.asarray([[np.NaN, 1.0, np.NaN], [np.NaN, 2.0, np.NaN], [1.0, 3.1, 2.1]])
        # Minimise
        r, c = npext.hungarian(_cost, False)
        self.assertTrue(np.array_equal(r, [0, 2]))
        self.assertTrue(np.array_equal(c, [1, 0]))
        r, c = npext.hungarian(_cost, False, cutoff=2.0)
        self.assertTrue(np.array_equal(r, [0, 2]))
        self.assertTrue(np.array_equal(c, [1, 0]))
        # Maximise
        r, c = npext.hungarian(_cost, True)
        self.assertTrue(np.array_equal(r, [1, 2]))
        self.assertTrue(np.array_equal(c, [1, 2]))
        r, c = npext.hungarian(_cost, True, cutoff=2.05)
        self.assertTrue(np.array_equal(r, [2]))
        self.assertTrue(np.array_equal(c, [1]))

