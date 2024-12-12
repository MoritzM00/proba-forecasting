"""Tests for the interval_score module."""

import numpy as np

from probafcst.metrics.interval_score import interval_score_vectorized


def test_interval_score_vectorized():
    """Test the interval_score_vectorized function."""
    alpha = 0.1
    a = np.array([1, 1, 1])
    b = np.array([3, 3, 3])
    y = np.array([0, 2, 4])
    expected = np.array([22.0, 2.0, 22.0])
    np.testing.assert_array_equal(interval_score_vectorized(alpha, a, b, y), expected)
    print("All vectorized tests passed!")
