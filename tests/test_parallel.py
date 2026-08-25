import unittest

import joblib as jl

from mpctools.parallel import parallel_progress


def _identity(value):
    return value


class _RecordingProgressBar:
    def __init__(self):
        self.values = []

    def reset(self):
        self.update(value=0)
        return self

    def update(self, *, value):
        self.values.append(value)


class TestParallelProgress(unittest.TestCase):
    def test_sequential_completion_is_reported_once(self):
        progress = _RecordingProgressBar()

        with parallel_progress(progress):
            result = jl.Parallel(n_jobs=1)(
                jl.delayed(_identity)(value)
                for value in range(2)
            )

        self.assertEqual(result, [0, 1])
        self.assertEqual(progress.values, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
