"""Provide Function to Compute Calibration Curve on Time Series Crossvalidation."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_calibration_curve(predictions: pd.DataFrame, quantile_levels: list[float]):
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
    for i, (_, y_test, y_pred_quantiles) in predictions.iterrows():
        name = y_test.name

        for q in quantile_levels:
            coverage = np.mean(y_test <= y_pred_quantiles[(name, q)])
            calibration_values[q][i] = coverage

    # Compute mean and standard deviation of coverage across folds
    empirical_coverage_mean = {
        q: np.mean(calibration_values[q]) for q in quantile_levels
    }
    empirical_coverage_std = {q: np.std(calibration_values[q]) for q in quantile_levels}

    # Plot calibration curve with error bars
    _, ax = plt.subplots(figsize=(8, 6))
    plt.errorbar(
        quantile_levels,
        list(empirical_coverage_mean.values()),
        yerr=list(empirical_coverage_std.values()),
        fmt="o-",
        capsize=4,
        label="Empirical Coverage",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Nominal Quantile Level")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Plot with Standard Deviation Across Folds")
    plt.legend()
    plt.grid(True)
    return ax
