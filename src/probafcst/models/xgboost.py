"""Probabilistic XGBoost Regression Forecaster."""

import pandas as pd
import xgboost as xgb
from sktime.split import temporal_train_test_split

from probafcst.models._regression import QuantileRegressionForecaster
from probafcst.utils.tabularization import create_lagged_features


class XGBQuantileForecaster(QuantileRegressionForecaster):
    """Quantile regression forecaster using XGBoost."""

    def __init__(
        self,
        lags: list[int],
        quantiles: list[int],
        include_seasonal_dummies=True,
        cyclical_encodings=True,
        X_lag_cols: list[str] | None = None,
        xgb_kwargs: dict | None = None,
    ):
        self.xgb_kwargs = xgb_kwargs or {}
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantiles,
            **(xgb_kwargs or {}),
        )
        super().__init__(
            model=model,
            lags=lags,
            quantiles=quantiles,
            include_seasonal_dummies=include_seasonal_dummies,
            cyclical_encodings=cyclical_encodings,
            X_lag_cols=X_lag_cols,
        )

    # override _fit method to add early stopping capability
    def _fit(self, y, X, fh=None):
        self.freq_ = pd.infer_freq(y.index)
        self._target_name = self._y_metadata["feature_names"][0]
        self.max_lag_ = max(self.lags)

        y = y.copy()
        y.name = self._target_name

        features, labels = create_lagged_features(
            X=X,
            y=y,
            lags=self.lags,
            include_seasonal_dummies=self.include_seasonal_dummies,
            cyclical_encodings=self.cyclical_encodings,
            X_lag_cols=self.X_lag_cols,
            is_training=True,
            freq=self.freq_,
        )

        if "early_stopping_rounds" in self.xgb_kwargs:
            train_features, val_features, train_labels, val_labels = (
                temporal_train_test_split(
                    features, labels, test_size=30 if self.freq_ == "D" else 24 * 7
                )
            )
            self.model.fit(
                train_features,
                train_labels,
                eval_set=[(val_features, val_labels)],
                verbose=False,
            )
        else:
            self.model.fit(features, labels)
        self.feature_names_in_ = features.columns
        return self
