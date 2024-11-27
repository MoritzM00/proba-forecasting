"""Test the tabularization utility functions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from probafcst.utils.tabularization import create_lagged_features


@pytest.fixture(scope="module")
def test_data():
    """Return test data for tabularization utility functions."""
    index = pd.date_range("2021-01-01", periods=14, freq="h")
    X = pd.DataFrame(
        index=index,
        data={
            "rain": range(
                -10,
                4,
            )
        },
    )
    y = pd.Series(
        index=index,
        data=range(
            -10,
            4,
        ),
        name="load",
    )
    y *= 10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=3, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def test_create_lagged_features_train(test_data):
    """Test create_lagged_features for creation during training."""
    X_train, _, y_train, _ = test_data

    X_lagged, y_lagged = create_lagged_features(
        X_train,
        y_train,
        lags=[1, 2],
        is_training=True,
        cyclical_encodings=False,
        include_seasonal_dummies=False,
    )
    assert X_lagged.shape[0] == y_lagged.shape[0]
    # the rain feature, plus 2 lags for target and rain
    assert X_lagged.shape[1] == 1 + 2 * 2

    X_expected = pd.DataFrame(
        index=X_train.index,
        data={
            "rain": X_train["rain"],
            "rain_lag_1": X_train["rain"].shift(1),
            "rain_lag_2": X_train["rain"].shift(2),
            "load_lag_1": y_train.shift(1),
            "load_lag_2": y_train.shift(2),
        },
    ).dropna()
    pd.testing.assert_frame_equal(X_lagged, X_expected)

    y_expected = y_train.loc[X_lagged.index]
    pd.testing.assert_series_equal(y_lagged, y_expected)


def test_create_lagged_features_predict(test_data):
    """Test create_lagged_features for creation during prediction."""
    X_train, X_test, y_train, y_test = test_data

    # this will be filled in with 1-step ahead predictions
    y_pred = pd.Series(np.nan, index=X_test.index, name="load")

    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_pred])

    lags = [1, 2]
    max_lag = max(lags)

    # if we would know the true values of y_test, we would use them here
    # we assume that we have predictions for y_hat at train cutoff
    y_hat = y_test
    X_lagged_full, _ = create_lagged_features(
        X_full,
        pd.concat([y_train, y_hat]),
        lags=lags,
        include_seasonal_dummies=False,
        is_training=True,
    )

    # assume fixed predictions y_hat
    for i, (timestamp, y_hat) in enumerate(zip(X_test.index, y_hat)):
        X_lagged_test, _ = create_lagged_features(
            X_full,
            y_full,
            lags=lags,
            include_seasonal_dummies=False,
            is_training=False,
        )
        # one row more each timestamp we predict ahead
        # need to subtract the max_lag to account for dropped NA's
        assert X_lagged_test.shape[0] == X_train.shape[0] - max_lag + i + 1

        # X_step should match the i+len(X_train)-th row in X_lagged_full
        expected = X_lagged_full.iloc[i + len(X_train) - max_lag]
        actual = X_lagged_test.loc[timestamp]
        pd.testing.assert_series_equal(actual, expected)

        # in practice, we would predict y_hat using X_step
        y_full.loc[timestamp] = y_hat
