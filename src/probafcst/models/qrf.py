"""Quantile regression forecaster using a random forest model."""

from omegaconf import ListConfig, OmegaConf
from quantile_forest import RandomForestQuantileRegressor

from probafcst.models.regression import QuantileRegressionForecaster


class RandomForestQuantileForecaster(QuantileRegressionForecaster):
    """Quantile regression forecaster using a random forest model."""

    def __init__(
        self,
        lags: list[int],
        quantiles: list[float],
        include_seasonal_dummies=True,
        cyclical_encodings=True,
        include_rolling_stats=False,
        X_lag_cols: list[str] | None = None,
        kwargs: dict | None = None,
    ):
        self.kwargs = kwargs or {}
        if isinstance(quantiles, ListConfig):
            # quantile-forests needs a list object
            quantiles = OmegaConf.to_object(quantiles)

        model = RandomForestQuantileRegressor(
            default_quantiles=quantiles,
            **self.kwargs,
        )
        super().__init__(
            model=model,
            lags=lags,
            quantiles=quantiles,
            include_seasonal_dummies=include_seasonal_dummies,
            include_rolling_stats=include_rolling_stats,
            cyclical_encodings=cyclical_encodings,
            X_lag_cols=X_lag_cols,
        )
