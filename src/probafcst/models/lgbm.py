"""LGBM Quantile Regression Model."""

import lightgbm as lgb

from probafcst.models.regression import (
    MultipleQuantileRegressor,
    QuantileRegressionForecaster,
)


class LGBMQuantileForecaster(QuantileRegressionForecaster):
    """Quantile regression forecaster using LightGBM.

    A forecaster model that uses LightGBM for quantile regression forecasting. This class inherits from
    QuantileRegressionForecaster and is designed to predict quantiles of the target variable distribution.
    """

    def __init__(
        self,
        lags: list[int],
        quantiles: list[int],
        include_seasonal_dummies=True,
        cyclical_encodings=True,
        include_rolling_stats=False,
        X_lag_cols: list[str] | None = None,
        kwargs: dict | None = None,
    ):
        self.kwargs = kwargs or {}

        lgb_model = lgb.LGBMRegressor(
            objective="quantile",
            **self.kwargs,
        )
        # lgbm does not support native multiple quantile regression
        model = MultipleQuantileRegressor(
            quantiles=quantiles, regressor=lgb_model, alpha_name="alpha"
        )
        super().__init__(
            model=model,
            lags=lags,
            quantiles=quantiles,
            include_rolling_stats=include_rolling_stats,
            include_seasonal_dummies=include_seasonal_dummies,
            cyclical_encodings=cyclical_encodings,
            X_lag_cols=X_lag_cols,
        )
