"""Test the _regression module."""

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from probafcst.models.regression import MultipleQuantileRegressor


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
