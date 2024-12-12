"""Interval score for prediction intervals."""

import numpy as np


def interval_score_vectorized(alpha, a, b, y):
    r"""
    Compute the interval score for given prediction intervals and observations in a vectorized manner.

    The interval score is defined by:

    $$\operatorname{IS}_\alpha([a, b], y):=(b-a)+\frac{2}{\alpha}(a-y) \mathbb{1}(y<a)+\frac{2}{\alpha}(y-b) \mathbb{1}(y>b)$$

    The interval score is a proper scoring rule for evaluating the accuracy of prediction intervals.
    It penalizes both the width of the prediction interval and the distance of observations that fall
    outside the interval.

    Paper:
    https://arxiv.org/pdf/2007.05709

    Parameters
    ----------
    alpha : float
        The significance level of the prediction interval (e.g., 0.05 for a 95% prediction interval).
    a : np.ndarray
        The lower bounds of the prediction intervals.
    b : np.ndarray
        The upper bounds of the prediction intervals.
    y : np.ndarray
        The observed values.

    Returns
    -------
    np.ndarray
        The interval scores for each prediction interval and observation.
    """
    iscore = (b - a) + (2 / alpha) * (
        np.where(y < a, a - y, 0) + np.where(y > b, y - b, 0)
    )
    return iscore
