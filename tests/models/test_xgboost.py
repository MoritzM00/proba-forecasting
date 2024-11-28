"""Test XGBoostQuantileForecaster."""

import numpy as np

from probafcst.models.xgboost import XGBQuantileForecaster


def test_xgboost_fit_predict_simple(test_data):
    """Test fit method of XGBoost Model."""
    quantiles = [0.1, 0.5, 0.9]
    model = XGBQuantileForecaster(
        lags=[1, 2],
        quantiles=quantiles,
        include_seasonal_dummies=False,
    )
    X_train, X_test, y_train, _ = test_data
    model.fit(y_train, X_train)
    assert model.is_fitted

    y_pred_quantiles = model.predict_quantiles(
        fh=np.arange(1, len(X_test) + 1), X=X_test, alpha=quantiles
    )
    assert y_pred_quantiles.shape == (len(X_test), len(quantiles))
