"""Test the _regression module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import QuantileRegressor
from sktime.forecasting.base import ForecastingHorizon

from probafcst.models.regression import (
    MultipleQuantileRegressor,
    QuantileRegressionForecaster,
)


def test_MultipleQuantileRegressor():
    """Test the MultipleQuantileRegressor class."""
    regressor = QuantileRegressor()
    model = MultipleQuantileRegressor(
        quantiles=[0.1, 0.9], regressor=regressor, alpha_name="quantile", n_jobs=2
    )

    # Test fit
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    model.fit(X, y)
    assert len(model.regressors_) == 2
    for q, regressor in model.regressors_.items():
        assert regressor.quantile == q

    # Test predict
    X = pd.DataFrame({"feature": [4, 5, 6]})
    predictions = model.predict(X)
    assert predictions.shape == (3, 2)
    assert np.all(predictions[:, 0] <= predictions[:, 1])


def test_insample_prediction(test_data):
    """Test that insample predictions are rejected."""
    X_train, _, y_train, _ = test_data

    regressor = QuantileRegressor()
    model = QuantileRegressionForecaster(
        regressor, lags=[1, 2], quantiles=[0.1, 0.5, 0.9]
    )
    model.fit(X_train, y_train)

    # Test insample prediction
    fh = ForecastingHorizon(y_train.index, is_relative=False)

    with pytest.raises(NotImplementedError):
        model.predict(fh, X=X_train)


def test_X_too_short(test_data):
    """Test that X too short is rejected."""
    X_train, _, y_train, _ = test_data

    regressor = QuantileRegressor()
    model = QuantileRegressionForecaster(
        regressor, lags=[1, 2], quantiles=[0.1, 0.5, 0.9]
    )
    model.fit(X_train, y_train)

    fh = np.arange(1, 4)

    with pytest.raises(ValueError):
        model.predict(fh, X=X_train)
