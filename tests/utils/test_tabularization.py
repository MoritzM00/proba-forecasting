"""Test the tabularization utility functions."""

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


def test_create_lagged_features_predict():
    """Test create_lagged_features for creation during prediction."""
    pass
