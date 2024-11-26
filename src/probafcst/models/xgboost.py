"""Probabilistic XGBoost Forecasting Model."""

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from xgboost import XGBRegressor


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
        "capability:insample": True,
        "capability:pred_int": True,
    }

    def __init__(self, lags: list[int], quantiles: list[int], cyclical_encodings=True):
        self.lags = lags
        self.quantiles = quantiles
        self.cyclical_encodings = cyclical_encodings
        self.model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles,
            n_estimators=100,
            learning_rate=0.1,
        )
        super().__init__()

    def _fit(self, y, X, fh=None):
        self.freq_ = pd.infer_freq(y.index)
        self._target_name = self._y_metadata["feature_names"][0]

        X_lagged, y_lagged = self._create_lagged_features(y, X, self.lags)
        self.model.fit(X_lagged, y_lagged)
        self.feature_names_in_ = X_lagged.columns
        return self

    def _create_lagged_features(self, y, X, lags, is_training=True):
        varname = self._target_name
        y = y.copy()
        X = pd.DataFrame(index=y.index) if X is None else X.copy()

        seasonal_features = self._create_seasonal_features(
            X, cyclical=self.cyclical_encodings
        )
        X = pd.concat([X, seasonal_features], axis=1)

        X_lags = X.shift(lags, suffix="_lag")
        for lag in lags:
            X_lags[f"{varname}_lag_{lag}"] = y.shift(lag)
        X = pd.concat([X, X_lags], axis=1)
        X[varname] = y
        X = X.dropna()
        y = X.pop(varname)
        return X, y

    def _predict(self, fh, X=None):
        # Use _predict_quantiles() to get the median for point prediction
        median_alpha = [0.5]
        quantile_predictions = self._predict_quantiles(fh, median_alpha)

        return quantile_predictions[(self._y.name, 0.5)]

    def _predict_quantiles(self, fh, X, alpha):
        # if len(X) < self.max_lag + len(fh):
        #     raise ValueError(f"X must contain at least {self.max_lag} rows")

        past_X = self._X.copy()
        past_y = self._y.copy()

        # Create future timestamps
        assert len(fh) <= len(X), "fh must be less than or equal to the length of X"

        # Create future DataFrame
        forecast_index = fh.to_absolute_index(self.cutoff)

        # append future dates to past_X and past_y
        future_X = pd.DataFrame(index=forecast_index)
        future_X = pd.concat([past_X, future_X], axis=0)
        future_y = pd.Series(index=forecast_index)
        future_y = pd.concat([past_y, future_y], axis=0)

        # Initialize results DataFrame
        results = pd.DataFrame(
            index=forecast_index, columns=[q for q in self.quantiles]
        )
        # Generate predictions one step at a time
        forecast_steps = len(fh)

        for i in range(forecast_steps):
            # todo: create lagged features for next prediction
            # need to append the last prediction to y, and a new row for X

            next_predictions = self.predict_quantiles(X.iloc[[-1]])
            results.iloc[i] = next_predictions.iloc[0]

            # Update the forecast DataFrame with the median prediction for the next iteration
            median_prediction = next_predictions[0.5].iloc[0]
            past_y.iloc[-forecast_steps + i][self._target_name] = median_prediction

        columns = pd.MultiIndex.from_product([self._target_name, alpha])
        results.columns = columns
        return results

    def _create_seasonal_features(self, X, cyclical=True) -> pd.DataFrame:
        assert self.freq_ in ["h", "D"], "Only daily and hourly data supported"

        encodings = [("dayofweek", 7), ("month", 12)]
        datetime_features = pd.DataFrame(index=X.index)
        datetime_features["dayofweek"] = X.index.dayofweek
        datetime_features["month"] = X.index.month
        if self.freq_ == "h":
            encodings.append(("hour", 24))
            datetime_features["hour"] = X.index.hour

        if cyclical:
            # Create cyclical features for periodic patterns
            seasonal_features = pd.DataFrame(index=X.index)

            for col, period in encodings:
                seasonal_features[f"{col}_sin"] = np.sin(
                    2 * np.pi * datetime_features[col] / period
                )
                seasonal_features[f"{col}_cos"] = np.cos(
                    2 * np.pi * datetime_features[col] / period
                )
        else:
            seasonal_features = datetime_features
        return seasonal_features
