"""Test XGBoostQuantileForecaster."""

import numpy as np
import pandas as pd
import pytest

from probafcst.backtest import backtest
from probafcst.models.xgboost import XGBQuantileForecaster

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]


def test_xgboost_fit_predict_simple(test_data):
    """Test fit method of XGBoost Model."""
    model = XGBQuantileForecaster(
        lags=[1, 2],
        quantiles=QUANTILES,
        include_seasonal_dummies=False,
    )
    X_train, X_test, y_train, _ = test_data
    model.fit(y_train, X_train)
    assert model.is_fitted

    # The X passed to predict must contain training and test data!
    X = pd.concat([X_train, X_test])
    y_pred_quantiles = model.predict_quantiles(
        fh=np.arange(1, len(X_test) + 1), X=X, alpha=QUANTILES
    )
    assert y_pred_quantiles.shape == (len(X_test), len(QUANTILES))


@pytest.mark.parametrize("include_seasonal_dummies", [True, False])
def test_xgboost_evaluate_no_X(energy_sample, include_seasonal_dummies):
    """Test backtest method for XGBoost Model."""
    # only use the last 8 weeks for testing
    y = energy_sample.iloc[24 * 7 * 44 :]
    model = XGBQuantileForecaster(
        lags=[24, 168],
        quantiles=QUANTILES,
        include_seasonal_dummies=include_seasonal_dummies,
        cyclical_encodings=False,
        xgb_kwargs={"n_jobs": 1, "n_estimators": 10},
    )
    # generate three splits, using 6 weeks for training
    backtest_results = backtest(
        forecaster=model,
        y=y,
        forecast_steps=24,
        initial_window=24 * 7 * 6,
        step_length=24 * 7,
        quantiles=QUANTILES,
        X=None,
        backend=None,
        splitter_type="sliding",
    )
    assert backtest_results.eval_results.shape[0] == 3


def test_xgboost_evaluate_with_X(energy_sample):
    """Test backtest method for XGBoost Model with X."""
    # only use the last 8 weeks for testing
    y = energy_sample.iloc[24 * 7 * 44 :]
    X = pd.DataFrame(index=y.index)
    X["hour"] = X.index.hour
    model = XGBQuantileForecaster(
        lags=[24, 168],
        quantiles=QUANTILES,
        include_seasonal_dummies=False,
        cyclical_encodings=False,
        xgb_kwargs={"n_jobs": 1, "n_estimators": 10},
    )
    # generate three splits, using 6 weeks for training
    backtest_results = backtest(
        forecaster=model,
        y=y,
        forecast_steps=24,
        initial_window=24 * 7 * 6,
        step_length=24 * 7,
        quantiles=QUANTILES,
        X=X,
        backend=None,
        splitter_type="sliding",
    )
    assert backtest_results.eval_results.shape[0] == 3
