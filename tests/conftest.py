"""Configuration and fixtures for testing."""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture()
def test_data():
    """Return test data."""
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
