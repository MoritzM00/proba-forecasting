"""Plotting functions for probabilistic forecasting."""

import seaborn as sns
from matplotlib import pyplot as plt
from sktime.utils.plotting import plot_series

from probafcst.utils.sktime import quantiles_to_interval_df


def plot_quantiles(actual_series, pred_quantiles, ax=None):
    """Plot actual series and forecasted quantiles.

    Parameters
    ----------
    actual_series : pd.Series
        Actual time series.
    pred_quantiles : pd.DataFrame
        Forecasted quantiles. Columns are quantiles, rows are time points.
        Must have exactly five quantiles, including median.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot.
    """
    assert (
        pred_quantiles.shape[1] == 5
    ), "Pred quantiles must have exactly five quantiles, including median."
    columns = [pred_quantiles[i] for i in pred_quantiles.columns]
    quantiles = [rf"$\alpha={q}$" for _, q in pred_quantiles.columns]
    colors = sns.color_palette("icefire", n_colors=len(quantiles))
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    plot_series(
        actual_series,
        *columns,
        labels=["actual", *quantiles],
        colors=["black"] + colors,
        markers=["o", "", "", "x", "", ""],
        ax=ax,
    )
    plt.xticks(rotation=45, ha="center")

    linestyles = ["dashdot", "dashed", "solid", "dashed", "dashdot"]
    for line, linestyle in zip(ax.lines[1:], linestyles):
        line.set_linestyle(linestyle)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    return fig, ax


def plot_interval(actual, median, pred_interval, ax=None):
    """Plot actual series and forecasted interval."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))

    fig, ax = plot_series(
        actual,
        median,
        pred_interval=pred_interval,
        labels=["actual", "median"],
    )
    # set legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig, ax


def plot_interval_from_quantiles(actual, pred_quantiles, ax=None):
    """Plot actual series and forecasted interval from quantiles."""
    pred_interval, median = quantiles_to_interval_df(pred_quantiles)
    return plot_interval(actual, median, pred_interval, ax=ax)
