"""Tabularization utilities for scikit-learn regressors."""

from typing import Literal

import numpy as np
import pandas as pd


def create_seasonal_features(
    X: pd.DataFrame,
    freq: Literal["h", "D"] = "D",
    cyclical=True,
    is_weekend=True,
):
    """Create seasonal features for daily or hourly data.

    Parameters
    ----------
    X : pd.DataFrame
        Input data with datetime index.
    freq : Literal["h", "D"], optional
        Frequency of the data.
    cyclical : bool, optional
        Whether to create cyclical features for periodic patterns, by default True.
        Creates sin and cos features for dayofweek, month and hour.
    is_weekend : bool, optional
        Whether to create a binary feature for weekend, by default True.
    """
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
        seasonal_features["is_weekend"] = (X.index.dayofweek > 4).astype(int)

    return seasonal_features


def create_lagged_features(
    X: pd.DataFrame | None,
    y: pd.Series,
    lags: list[int],
    X_lag_cols: list[str] | None = None,
    include_seasonal_dummies=True,
    cyclical_encodings=True,
    include_rolling_stats=False,
    is_training=True,
    freq: Literal["h", "D"] | None = None,
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
    X_lag_cols : list[str], optional
        List of columns to create lags for, by default uses all columns in X for lags,
        including seasonal dummies.
        Explicitly specify columns to avoid creating lags for all columns or pass
        an empty list to only lag the target values.
    include_seasonal_dummies : bool, default=True
        Whether to include seasonal features, by default True.
    cyclical_encodings : bool, default=True
        Whether to create cyclical features for periodic patterns.
    include_rolling_stats : bool, default=False
        Whether to include rolling statistics like median and quantiles.
    is_training : bool, default=True
        Whether to return y in the output, by default True.
    freq: Literal["h", "D"], optional
        Frequency of the data, by default it is inferred from the index.

    Returns
    -------
    features : pd.DataFrame
        Lagged features for training or prediction.
    labels : pd.Series or None
        Labels for training, None if is_training=False.

    Raises
    ------
    ValueError
        If max lag is greater than the length of the time series.
    """
    if np.max(lags) > len(y):
        raise ValueError(
            "Max Lag cannot be greater than the length of the time series."
        )

    varname = y.name
    y = y.copy()
    X = pd.DataFrame(index=y.index) if X is None else X.copy()

    if freq is None:
        freq = pd.infer_freq(X.index)
    assert freq is not None, "Could not infer frequency from the index."
    assert freq in ["h", "D"], f"Only daily and hourly data supported, but got {freq}"

    if include_seasonal_dummies:
        seasonal_features = create_seasonal_features(
            X, cyclical=cyclical_encodings, freq=freq
        )
        X = pd.concat([X, seasonal_features], axis=1)

    if X_lag_cols is None:
        X_lag_cols = X.columns

    X_lags = X[X_lag_cols].shift(lags, suffix="_lag")

    for lag in lags:
        X_lags[f"{varname}_lag_{lag}"] = y.shift(lag)

    if include_rolling_stats:
        freq = pd.infer_freq(y.index)
        groupby = [y.index.hour, y.index.weekday] if freq == "h" else y.index.weekday
        grouped_median = (
            y.groupby(groupby).rolling(window="100D", closed="left").median()
        )
        level = [0, 1] if freq == "h" else 0
        grouped_median = grouped_median.reset_index(level=level, drop=True).sort_index()
        X_lags[f"{varname}_grouped_median"] = grouped_median

        # and just x day rolling median and quantiles
        window = "30D"
        X_lags[f"{varname}_rolling_median"] = y.rolling(
            window=window, closed="left"
        ).median()
        X_lags[f"{varname}_rolling_q025"] = y.rolling(
            window=window, closed="left"
        ).quantile(0.025)
        X_lags[f"{varname}_rolling_q25"] = y.rolling(
            window=window, closed="left"
        ).quantile(0.25)
        X_lags[f"{varname}_rolling_q75"] = y.rolling(
            window=window, closed="left"
        ).quantile(0.75)
        X_lags[f"{varname}_rolling_q975"] = y.rolling(
            window=window, closed="left"
        ).quantile(0.975)

    X = pd.concat([X, X_lags], axis=1)
    if is_training:
        X[varname] = y
        X = X.dropna()
        y = X.pop(varname)
        return X, y
    else:
        X = X.dropna()
        return X, None
