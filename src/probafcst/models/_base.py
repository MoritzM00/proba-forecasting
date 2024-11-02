"""Base implementations for probabilistic forecasting models."""

from typing import Literal

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_y


def get_model(target: str, params: dict | None = None) -> BaseForecaster:
    """Return the model for the given target."""
    if params is None:
        params = {}
    match target:
        case "energy" | "no2":
            return get_energy_model(**params)
        case "bikes":
            return get_bikes_model(**params)
        case _:
            raise ValueError(f"Unknown target: {target}")


def get_energy_model(**params) -> BaseForecaster:
    """Return the energy model."""
    return BenchmarkForecaster()


def get_bikes_model(**params) -> BaseForecaster:
    """Return the bikes model."""
    return BenchmarkForecaster()


class BenchmarkForecaster(BaseForecaster):
    """Benchmark forecaster for probabilistic forecasting.

    This forecaster computes quantiles based on the `n` recent weeks of
    the same weekday (and hour, if hourly data is given).

    Parameters
    ----------
    n_weeks : int, default=100
        Number of recent weeks to consider for computing quantiles.
    freq : Literal {"D", "h"}, default=None
        Frequency of the time series index. If None, the frequency is inferred
        from the time series index.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": True,
        "capability:insample": True,
        "capability:pred_int": True,
    }

    def __init__(self, n_weeks=5, freq: Literal["D", "h"] | None = None):
        # n_weeks: number of recent weeks to consider for computing quantiles
        self.n_weeks = n_weeks
        """The number of recent weeks to consider for computing quantiles."""
        self.freq = freq
        """Frequency of the time series index."""
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self._y = check_y(y).copy()

        if self.freq is None:
            freq = pd.infer_freq(self._y.index)
            if not freq:
                raise ValueError(
                    "Frequency of the time series index could not be inferred."
                    " Pass the frequency as a string to the `freq` parameter."
                )
            self.freq = freq
        elif self.freq not in ["D", "h"]:
            raise ValueError(
                f"Unsupported frequency: {freq}. Supported frequencies are 'D' and 'h'."
            )
        return self

    def _predict(self, fh, X=None):
        # Use _predict_quantiles() to get the median for point prediction
        median_alpha = [0.5]
        quantile_predictions = self._predict_quantiles(fh, median_alpha)

        return quantile_predictions[(self._y.name, 0.5)]

    def _predict_quantiles(self, fh, X, alpha):
        quantile_preds = []
        y_index = self._y.index

        for time_point in fh.to_absolute(self.cutoff):
            condition = y_index.weekday == time_point.weekday()
            if self.freq == "h":
                condition &= y_index.hour == time_point.hour

            relevant_data = self._y[condition][-self.n_weeks :]
            if len(relevant_data) >= self.n_weeks:
                quantiles = np.nanquantile(relevant_data, alpha)
                quantile_preds.append(quantiles)
            else:
                quantile_preds.append([np.nan] * len(alpha))

        var_names = self._get_varnames()
        columns = pd.MultiIndex.from_product([var_names, alpha])
        return pd.DataFrame(
            quantile_preds, columns=columns, index=fh.to_absolute_index(self.cutoff)
        )
