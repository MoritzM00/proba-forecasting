"""Catboost Quantile Forecasting Model."""

from catboost import CatBoostRegressor

from probafcst.models.regression import QuantileRegressionForecaster


class CatBoostQuantileForecaster(QuantileRegressionForecaster):
    """Quantile regression forecaster using CatBoost."""

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

        quantile_str = ",".join(map(str, quantiles))
        model = CatBoostRegressor(
            objective=f"MultiQuantile:alpha={quantile_str}",
            **self.kwargs,
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
