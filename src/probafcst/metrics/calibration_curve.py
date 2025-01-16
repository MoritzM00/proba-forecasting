"""Provide Function to Compute Calibration Curve on Time Series Crossvalidation."""

from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_calibration_curve(
    predictions: pd.DataFrame,
    quantile_levels: list[float],
    std_divisor: Literal["folds", "samples"] = "folds",
) -> tuple:
    """Plot calibration curve for probabilistic forecasts.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing the predictions made during folds. This is the output
        of the backtest function in `probafcst.backtest`.
        The DataFrame should have the following columns:
        - 'y_train': Training values for the time series (ignored)
        - 'y_test': Actual values for the time series
        - 'y_pred_quantiles': Predicted quantiles for the time series
    quantile_levels : list[float]
        List of quantile levels to compute the calibration curve for.
    std_divisor: {'folds', 'samples'}, default='folds'
        The divisor to use for the standard deviation of the empirical coverage. If set to 'folds' (the default),
        then divide the variance by the number of folds. If set to 'samples', divide the variance by the total number
        of samples across all folds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # compute empirical coverage for each quantile level
    # i.e. actual #obs < predicted quantile / #obs
    calibration_values = {q: np.zeros(shape=len(predictions)) for q in quantile_levels}
    sample_size = 0
    n_folds = len(predictions)
    for i, (_, y_test, y_pred_quantiles) in predictions.iterrows():
        name = y_test.name
        sample_size += len(y_test)

        for q in quantile_levels:
            coverage = np.mean(y_test <= y_pred_quantiles[(name, q)])
            calibration_values[q][i] = coverage

    # Compute mean and standard deviation of coverage across folds
    empirical_coverage_mean = {
        q: np.mean(calibration_values[q]) for q in quantile_levels
    }

    empirical_coverage_std = {}
    for q in quantile_levels:
        var = np.var(calibration_values[q])
        if std_divisor == "samples":
            # correct the division
            var = var * n_folds / sample_size
        elif std_divisor != "folds":
            raise ValueError("std_divisor should be either 'folds' or 'samples'")

        empirical_coverage_std[q] = np.sqrt(var)

    # Plot calibration curve with error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.errorbar(
        quantile_levels,
        list(empirical_coverage_mean.values()),
        yerr=list(empirical_coverage_std.values()),
        fmt="o-",
        capsize=4,
        label="Model Calibration",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Nominal Quantile Level")
    plt.ylabel("Empirical Coverage")
    plt.title(f"Calibration Plot with Standard Deviation Across {std_divisor}")
    plt.legend()
    plt.grid(True)
    return fig, ax
