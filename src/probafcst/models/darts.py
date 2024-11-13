"""XGBoost Quantile Regression Model."""

from typing import Literal

from sktime.forecasting.darts import DartsLinearRegressionModel, DartsXGBModel


def get_darts_config(freq: Literal["D", "h"]):
    """Return the lag and encoder configuration for Darts Models."""
    match freq:
        case "D":
            lags = list(range(-7, 0)) + [-14 - (7 * i) for i in range(11)]
            lags_future_covariates = [lag * -1 for lag in lags] + [0] + lags
            additional_encoders = []
        case "h":
            lags = list(range(-24, 0)) + [-48 - (24 * i) for i in range(7)]
            lags_future_covariates = [lag * -1 for lag in lags] + [0] + lags
            additional_encoders = ["hour"]
        case _:
            raise ValueError(
                f"Invalid frequency: {freq}. Only 'D' and 'h' are supported."
            )
    add_encoders = {
        "cyclic": {
            "future": ["day", "month", "day_of_week", "quarter", *additional_encoders]
        },
    }
    return lags, lags_future_covariates, add_encoders


def get_xgboost_model(
    quantiles: list[float],
    freq: Literal["D", "h"] = "h",
    xgb_kwargs: dict | None = None,
    random_state: int = 0,
    output_chunk_length: int = 1,
) -> DartsXGBModel:
    """Create and configure a DartsXGBModel for time series forecasting.

    Parameters
    ----------
    quantiles : array_like
        Array of quantiles to predict.
    freq : {'D', 'h'}, optional
        Frequency of the time series, default: 'h'
        - 'D': daily data
        - 'h': hourly data
    xgb_kwargs : dict, optional
        Additional XGBoost parameters, default: None
    random_state : int, optional
        Random seed for reproducibility, default: 0
    output_chunk_length : int, optional
        Number of time steps to predict at once, default: 1

    Returns
    -------
    DartsXGBModel
        Configured XGBoost model for time series forecasting
    """
    lags, lags_future_covariates, add_encoders = get_darts_config(freq)
    model = DartsXGBModel(
        lags=lags,
        lags_future_covariates=lags_future_covariates,
        add_encoders=add_encoders,
        random_state=random_state,
        likelihood="quantile",
        quantiles=quantiles,
        output_chunk_length=output_chunk_length,
        multi_models=False,
        kwargs=xgb_kwargs,
    )
    # disable user warnings
    model.set_config(warnings="off")
    return model


def get_quantile_regressor(
    quantiles: list[int],
    freq: Literal["D", "h"] = "h",
    output_chunk_length: int = 1,
    random_state: int = 0,
) -> DartsLinearRegressionModel:
    """Get a quantile regressor model for time series forecasting.

    Parameters
    ----------
    freq : {'D', 'h'}, optional
        Frequency of the time series, default: 'h'
        - 'D': daily data
        - 'h': hourly data

    Returns
    -------
    DartsLinearRegressionModel
    """
    lags, lags_future_covariates, add_encoders = get_darts_config(freq)
    model = DartsLinearRegressionModel(
        lags=lags,
        lags_future_covariates=lags_future_covariates,
        add_encoders=add_encoders,
        output_chunk_length=output_chunk_length,
        likelihood="quantile",
        quantiles=quantiles,
        multi_models=False,
        random_state=random_state,
    )
    # disable user warnings
    model.set_config(warnings="off")
    return model
