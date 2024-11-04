"""Utility functions for sktime models."""

import numpy as np
import pandas as pd


def quantiles_to_interval_df(
    pred_quantiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert quantile predictions frame to interval frame and median series.

    Parameters
    ----------
    pred_quantiles : pd.DataFrame
        Forecasted quantiles. Columns are quantiles, rows are time points.
        Must have exactly five quantiles, including median.
    """
    quantiles = pred_quantiles.columns.get_level_values(1)
    expected_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    if not all(quantiles == expected_quantiles):
        raise ValueError(
            f"Pred quantiles must have these quantiles: {expected_quantiles}"
        )
    name = pred_quantiles.columns.get_level_values(0)[0]
    data = np.array(
        [
            pred_quantiles[(name, 0.25)],
            pred_quantiles[(name, 0.75)],
            pred_quantiles[(name, 0.025)],
            pred_quantiles[(name, 0.975)],
        ]
    ).T
    y_pred_interval = pd.DataFrame(
        index=pred_quantiles.index,
        columns=pd.MultiIndex.from_product([[name], [0.5, 0.95], ["lower", "upper"]]),
        data=data,
    )
    return y_pred_interval, pred_quantiles[(name, 0.5)]