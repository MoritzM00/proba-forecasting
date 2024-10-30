"""Plotting functions for probabilistic forecasting."""

import seaborn as sns
from sktime.utils.plotting import plot_series


def plot_quantiles(actual_series, pred_quantiles):
    """Plot actual series and forecasted quantiles."""
    columns = [pred_quantiles[i] for i in pred_quantiles.columns]
    quantiles = [rf"$\alpha={q}$" for _, q in pred_quantiles.columns]
    colors = sns.color_palette("icefire", n_colors=len(quantiles))
    fig, ax = plot_series(
        actual_series,
        *columns,
        labels=["actual", *quantiles],
        colors=["black"] + colors,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.autofmt_xdate(ha="center")
    return fig, ax
