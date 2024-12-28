"""Base Implementations for Regression Forecasting Models.

This module contains base implementations for regression forecasting models, i.e. traditional
tabular regression models like QuantileRegressor of sklearn, adapted for forecasting tasks.

If a regressor does not support multi-quantile regression natively, the MultipleQuantileRegressor
can be used to predict multiple quantiles using a single-quantile regressor, like it is the case
for the sklearn QuantileRegressor.

The QuantileRegressionForecaster is a probabilistic regression forecaster for quantile forecasting, that
creates lagged features using timestamp information and exogeneous variables,
and predicts quantiles using the given regression model.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sktime.forecasting.base import BaseForecaster

from probafcst.utils.tabularization import create_lagged_features


class MultipleQuantileRegressor(BaseEstimator, RegressorMixin):
    """Predict multiple quantiles using single-quantile optimized regressors.

    Parameters
    ----------
    quantiles : list of float
        List of quantiles to predict.
    regressor : BaseEstimator
        Regressor to use for each quantile. Must have capability to predict quantiles.
    alpha_name : str, default="alpha"
        Name of the parameter in the regressor that sets the quantile to predict.
    n_jobs : int, default=1
        Number of jobs to run fit in parallel.

    Attributes
    ----------
    regressors_ : dict
        Fitted regressors for each quantile.
    """

    def __init__(
        self,
        quantiles: list[float],
        regressor: BaseEstimator,
        alpha_name="alpha",
        n_jobs=1,
    ):
        self.quantiles = quantiles
        self.regressor = regressor
        self.alpha_name = alpha_name
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        """Fit regressors for each quantile.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs
            Additional keyword arguments passed to the regressor during fit.

        Returns
        -------
        self
            The fitted instance.
        """
        regressors = []
        for q in self.quantiles:
            reg_q = clone(self.regressor)
            reg_q.set_params(**{self.alpha_name: q})
            params = reg_q.get_params()
            if params.get(self.alpha_name, -1) != q:
                raise ValueError(
                    f"Regressor {reg_q} does not support setting quantile {q} with parameter {self.alpha_name}"
                )
            regressors.append(reg_q)

        # fit regressors in parallel
        fitted_regressors = Parallel(n_jobs=self.n_jobs)(
            delayed(regressor.fit)(X, y, **kwargs) for regressor in regressors
        )
        # put them into dict
        regressors_ = {}
        for i, q in enumerate(self.quantiles):
            regressors_[q] = fitted_regressors[i]
        self.regressors_ = regressors_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict quantiles for each regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples, n_quantiles)
            Predicted quantiles for each quantile level.
        """
        predictions = {}
        for q, regressor in self.regressors_.items():
            predictions[q] = regressor.predict(X)
        predictions = np.vstack([predictions[q] for q in self.quantiles]).T
        return predictions

    def __sklearn_tags__(self):
        """Return sklearn tags for the regressor."""
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        return tags


class QuantileRegressionForecaster(BaseForecaster):
    """Probabilistic regression forecaster for quantile forecasting.

    Parameters
    ----------
    model : BaseEstimator
        Model that supports quantile regression.
    lags : list of int
        List of lag values to use for creating lagged features.
    quantiles : list of float
        List of quantiles to predict.
    include_seasonal_dummies : bool, default=True
        If True, include seasonal dummy variables in the lagged features.
    cyclical_encodings : bool, default=True
        If True, use cyclical (cos and sin functions) encoding for time features.
    X_lag_cols : list of str, default=None
        List of column names in X to include as lagged features. If None, all columns are used.

    Attributes
    ----------
    max_lag_ : int
        Maximum lag value used for creating lagged features.
    freq_ : str
        Inferred frequency of the time series index.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "enforce_index_type": pd.DatetimeIndex,
        "X-y-must-have-same-index": True,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        model: BaseEstimator,
        lags: list[int],
        quantiles: list[int],
        include_seasonal_dummies=True,
        include_rolling_stats=True,
        cyclical_encodings=True,
        X_lag_cols: list[str] | None = None,
    ):
        self.lags = lags
        self.quantiles = quantiles
        self.include_seasonal_dummies = include_seasonal_dummies
        self.cyclical_encodings = cyclical_encodings
        self.include_rolling_stats = include_rolling_stats
        self.X_lag_cols = X_lag_cols

        self.model = model

        super().__init__()

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
            include_rolling_stats=self.include_rolling_stats,
            X_lag_cols=self.X_lag_cols,
            is_training=True,
            freq=self.freq_,
        )

        self.model.fit(features, labels)
        self.feature_names_in_ = features.columns
        return self

    def _predict(self, fh, X=None):
        # Use _predict_quantiles() to get the median for point prediction
        quantile_predictions = self._predict_quantiles(fh, X=X, alpha=self.quantiles)
        return quantile_predictions[(self._target_name, 0.5)]

    def _predict_quantiles(self, fh, X, alpha):
        if X is None:
            # index must be the one in _y plus the forecast horizon
            index = self._y.index.union(fh.to_absolute_index(self.cutoff))
            X = pd.DataFrame(index=index)

        if X.shape[0] < len(fh):
            raise ValueError(f"X must contain at least {self.max_lag_} rows")
        elif X.shape[0] > len(fh):
            max_needed_timestamp = fh.to_absolute_index(self.cutoff).max()
            X = X.loc[:max_needed_timestamp]
            logger.debug(f"X truncated to {X.index[0]} - {X.index[-1]}")

        logger.debug(f"Predicting {len(fh)} steps ahead.")
        logger.debug(f"Future X shape: {X.shape}")

        assert alpha == self.quantiles, "alpha must be equal to quantiles used in fit"

        y_train = self._y.copy()

        X_full = X.copy()
        logger.debug(f"X values available from {X_full.index[0]} to {X_full.index[-1]}")

        forecast_index = fh.to_absolute_index(self.cutoff)
        y_pred = pd.Series(np.nan, index=forecast_index)
        y_full = pd.concat([y_train, y_pred])
        y_full.name = self._target_name

        logger.debug(f"Forecast index: {forecast_index[0]} - {forecast_index[-1]}")

        # Initialize results DataFrame
        results = pd.DataFrame(index=forecast_index, columns=[q for q in alpha])

        for timestamp in forecast_index:
            X_lagged, _ = create_lagged_features(
                X=X_full,
                y=y_full,
                lags=self.lags,
                include_seasonal_dummies=self.include_seasonal_dummies,
                cyclical_encodings=self.cyclical_encodings,
                include_rolling_stats=self.include_rolling_stats,
                X_lag_cols=self.X_lag_cols,
                is_training=False,
                freq=self.freq_,
            )

            X_step = X_lagged.loc[[timestamp]]
            pred_quantiles_step = self.model.predict(X_step)
            results.loc[timestamp] = pred_quantiles_step

            # append the median prediction to use for lag creation in the next step
            median_pred = results.loc[timestamp, 0.5]
            y_full.loc[timestamp] = median_pred

        columns = pd.MultiIndex.from_product(
            [[self._target_name], alpha], names=["variable", "quantiles"]
        )
        results.columns = columns
        results = results.astype(float)

        # sort each row in dataframe y_pred ascending over the columns
        predictions = results.to_numpy()
        predictions.sort(axis=1)
        results.iloc[:, :] = predictions
        return results
