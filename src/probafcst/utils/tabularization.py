"""Tabularization utilities for scikit-learn regressors."""

from typing import Literal

import numpy as np
import pandas as pd


def create_seasonal_features(
    X: pd.DataFrame,
    freq: Literal["h", "D"] | None = None,
    cyclical=True,
    is_weekend=True,
):
    """Create seasonal features for daily or hourly data.

    Parameters
    ----------
    X : pd.DataFrame
        Input data with datetime index.
    freq : Literal["h", "D"], optional
        Frequency of the data, by default it is inferred from the index.
    cyclical : bool, optional
        Whether to create cyclical features for periodic patterns, by default True.
        Creates sin and cos features for dayofweek, month and hour.
    is_weekend : bool, optional
        Whether to create a binary feature for weekend, by default True.
    """
    if freq is None:
        freq = pd.infer_freq(X.index)
    assert freq in ["h", "D"], "Only daily and hourly data supported"

    encodings = [("dayofweek", 7), ("month", 12)]
    datetime_features = pd.DataFrame(index=X.index)
    datetime_features["dayofweek"] = X.index.dayofweek
    datetime_features["month"] = X.index.month

    if freq == "h":
        encodings.append(("hour", 24))
        datetime_features["hour"] = X.index.hour

    if cyclical:
        # Create cyclical features for periodic patterns
        # using sin and cos functions
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

    if is_weekend:
        seasonal_features["is_weekend"] = X.index.dayofweek > 4

    return seasonal_features


def create_lagged_features(
    X: pd.DataFrame,
    y: pd.Series,
    lags: list[int],
    include_seasonal_dummies=True,
    cyclical_encodings=True,
    is_training=True,
):
    """Create lagged features X, y for time series forecasting.

    Parameters
    ----------
    X : pd.DataFrame
        Exogeneous features (like rain) with datetime index.
    y : pd.Series
        Target variable with datetime index. Must have a name, otherwise
        the lagged column names make no sense.
    lags : list[int]
        List of lags to create features for.
    include_seasonal_dummies : bool, default=True
        Whether to include seasonal features, by default True.
    cyclical_encodings : bool, default=True
        Whether to create cyclical features for periodic patterns.
    is_training : bool, default=True
        Whether to return y in the output, by default True.
    """
    if np.max(lags) > len(y):
        raise ValueError(
            "Max Lag cannot be greater than the length of the time series."
        )

    varname = y.name
    y = y.copy()
    X = pd.DataFrame(index=y.index) if X is None else X.copy()

    if include_seasonal_dummies:
        seasonal_features = create_seasonal_features(X, cyclical=cyclical_encodings)
        X = pd.concat([X, seasonal_features], axis=1)

    X_lags = X.shift(lags, suffix="_lag")

    for lag in lags:
        X_lags[f"{varname}_lag_{lag}"] = y.shift(lag)

    X = pd.concat([X, X_lags], axis=1)
    if is_training:
        X[varname] = y
        X = X.dropna()
        y = X.pop(varname)
        return X, y
    else:
        X = X.dropna()
        return X, None
