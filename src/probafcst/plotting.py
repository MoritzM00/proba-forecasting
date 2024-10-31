"""Plotting functions for probabilistic forecasting."""

import seaborn as sns
from matplotlib import pyplot as plt
from sktime.utils.plotting import plot_series


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
    # rotate x labels
    plt.xticks(rotation=45, ha="center")

    # change marker size
    for line in ax.lines:
        line.set_markersize(4)

    linestyles = ["dashdot", "dashed", "solid", "dashed", "dashdot"]
    for line, linestyle in zip(ax.lines[1:], linestyles):
        line.set_linestyle(linestyle)

    # make actual line thicker
    # ax.lines[0].set_linewidth(3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    return fig, ax
