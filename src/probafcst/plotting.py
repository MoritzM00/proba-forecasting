"""Plotting functions for probabilistic forecasting."""

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
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    interval_df, median = quantiles_to_interval_df(pred_quantiles)
    ax = plot_series(
        actual_series,
        median,
        labels=["actual", "median"],
        ax=ax,
        colors=["black", "#1f78b4"],
        markers=["o", ""],
    )
    _plot_interval(ax, interval_df)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha="center")

    return fig, ax


def _plot_interval(ax, interval_df):
    var_name = interval_df.columns.levels[0][0]

    colors = ["#1f78b4", "#a6cee3"]

    for i, cov in enumerate(interval_df.columns.levels[1]):
        ax.fill_between(
            interval_df.index,
            interval_df[var_name][cov]["lower"].astype("float64").to_numpy(),
            interval_df[var_name][cov]["upper"].astype("float64").to_numpy(),
            alpha=0.4,
            color=colors[i],
            label=f"{int(cov * 100)}% PI",
        )
    ax.legend()
    return ax
