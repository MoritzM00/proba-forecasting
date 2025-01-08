"""Linear Quantile Regression model."""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        include_rolling_stats=False,
        X_lag_cols: list[str] | None = None,
        kwargs: dict | None = None,
    ):
        self.kwargs = kwargs or {}

        # use different default solver
        if "solver" not in self.kwargs:
            self.kwargs["solver"] = "highs-ipm"

        model = self._build_model(quantiles=quantiles, qr_kwargs=self.kwargs)

        super().__init__(
            model=model,
            lags=lags,
            quantiles=quantiles,
            include_rolling_stats=include_rolling_stats,
            include_seasonal_dummies=include_seasonal_dummies,
            cyclical_encodings=cyclical_encodings,
            X_lag_cols=X_lag_cols,
        )

    def _build_model(self, quantiles, qr_kwargs):
        """Build the model."""
        qr = QuantileRegressor(**qr_kwargs)
        regressor = MultipleQuantileRegressor(
            quantiles=quantiles, regressor=qr, alpha_name="quantile"
        )
        preprocessor = self._get_preprocessor()
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )
        return model

    def _get_preprocessor(self):
        """Get preprocessor for the model."""
        return ColumnTransformer(
            transformers=[
                (
                    "one-hot",
                    OneHotEncoder(drop="first"),
                    make_column_selector(dtype_include="category"),
                ),
                (
                    "target-passthrough",
                    "passthrough",
                    make_column_selector(pattern="bike_count|load"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                    make_column_selector(dtype_include="float"),
                ),
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )
