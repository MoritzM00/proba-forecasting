"""Probabilistic XGBoost Forecasting Model."""

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from xgboost import XGBRegressor

from probafcst.utils.tabularization import create_lagged_features


class XGBQuantileForecaster(BaseForecaster):
    """XGBoost forecaster for probabilistic forecasting."""

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "enforce_index_type": pd.DatetimeIndex,
        "X-y-must-have-same-index": True,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        lags: list[int],
        quantiles: list[int],
        include_seasonal_dummies=True,
        cyclical_encodings=True,
        X_lag_cols=None,
        xgb_kwargs=None,
    ):
        self.lags = lags
        self.quantiles = quantiles
        self.include_seasonal_dummies = include_seasonal_dummies
        self.cyclical_encodings = cyclical_encodings
        self.xgb_kwargs = xgb_kwargs or {}
        self.X_lag_cols = X_lag_cols

        self.model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles,
            **self.xgb_kwargs,
        )
        super().__init__()

    def _fit(self, y, X, fh=None):
        self.freq_ = pd.infer_freq(y.index)
        self._target_name = self._y_metadata["feature_names"][0]
        self.max_lag_ = max(self.lags)

        y = y.copy()
        y.name = self._target_name

        X_lagged, y_lagged = create_lagged_features(
            X=X,
            y=y,
            lags=self.lags,
            include_seasonal_dummies=self.include_seasonal_dummies,
            cyclical_encodings=self.cyclical_encodings,
            X_lag_cols=self.X_lag_cols,
            is_training=True,
            freq=self.freq_,
        )
        self.model.fit(X_lagged, y_lagged)
        self.feature_names_in_ = X_lagged.columns
        return self

    def _predict(self, fh, X=None):
        # Use _predict_quantiles() to get the median for point prediction
        quantile_predictions = self._predict_quantiles(fh, X=X, alpha=self.quantiles)
        return quantile_predictions[(self._target_name, 0.5)]

    def _predict_quantiles(self, fh, X, alpha):
        if X is None:
            X = pd.DataFrame(index=fh.to_absolute_index(self.cutoff))
            X_train = pd.DataFrame(index=self._y.index)
        else:
            X_train = self._X.copy()

        if X.shape[0] < len(fh):
            raise ValueError(f"X must contain at least {self.max_lag_} rows")
        elif X.shape[0] > len(fh):
            max_needed_timestamp = fh.to_absolute_index(self.cutoff).max()
            X = X.loc[:max_needed_timestamp]

        assert alpha == self.quantiles, "alpha must be equal to quantiles used in fit"

        y_train = self._y.copy()

        X_test = X.copy()
        X_full = pd.concat([X_train, X_test])

        y_pred = pd.Series(np.nan, index=X_test.index)
        y_full = pd.concat([y_train, y_pred])
        y_full.name = self._target_name

        forecast_index = fh.to_absolute_index(self.cutoff)

        # Initialize results DataFrame
        results = pd.DataFrame(index=forecast_index, columns=[q for q in alpha])

        for timestamp in forecast_index:
            X_lagged, _ = create_lagged_features(
                X=X_full,
                y=y_full,
                lags=self.lags,
                include_seasonal_dummies=self.include_seasonal_dummies,
                cyclical_encodings=self.cyclical_encodings,
                X_lag_cols=self.X_lag_cols,
                is_training=False,
                freq=self.freq_,
            )

            X_step = X_lagged.loc[[timestamp]]
            pred_quantiles_step = self.model.predict(X_step)
            results.loc[timestamp] = pred_quantiles_step

            # append the median prediction to use for lag creation in the next step
            median_pred = results.loc[timestamp, 0.5]
            y_full.loc[timestamp] = median_pred

        columns = pd.MultiIndex.from_product(
            [[self._target_name], alpha], names=["variable", "quantiles"]
        )
        results.columns = columns
        results = results.astype(float)

        # sort each row in dataframe y_pred ascending over the columns
        predictions = results.to_numpy()
        predictions.sort(axis=1)
        results.iloc[:, :] = predictions
        return results
