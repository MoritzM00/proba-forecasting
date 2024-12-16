"""Linear Quantile Regression model."""

from sklearn.linear_model import QuantileRegressor

from .regression import MultipleQuantileRegressor, QuantileRegressionForecaster


class LinearQuantileForecaster(QuantileRegressionForecaster):
    """Linear Quantile Regression model.

    Fits separate quantile regressors for each quantile.
    """

    def __init__(
        self,
        lags: list[int],
        quantiles: list[int],
        include_seasonal_dummies=True,
        cyclical_encodings=True,
        X_lag_cols: list[str] | None = None,
        kwargs: dict | None = None,
    ):
        self.kwargs = kwargs or {}

        # use different default solver
        if "solver" not in self.kwargs:
            self.kwargs["solver"] = "highs-ipm"

        regressor = QuantileRegressor(**self.kwargs)
        # lgbm does not support native multiple quantile regression
        model = MultipleQuantileRegressor(
            quantiles=quantiles, regressor=regressor, alpha_name="quantile"
        )
        super().__init__(
            model=model,
            lags=lags,
            quantiles=quantiles,
            include_seasonal_dummies=include_seasonal_dummies,
            cyclical_encodings=cyclical_encodings,
            X_lag_cols=X_lag_cols,
        )
