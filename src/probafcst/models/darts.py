"""XGBoost Quantile Regression Model."""

from typing import Literal

from sktime.forecasting.darts import DartsLinearRegressionModel, DartsXGBModel


def get_darts_config(freq: Literal["D", "h"]):
    """Return the lag and encoder configuration for Darts Models."""
    match freq:
        case "D":
            lags = 30
            lags_future_covariates = [-7, -14, -21, -28, 0, 7, 14, 21, 28]
            lags_past_covariates = 14
            additional_encoders = []
        case "h":
            lags = 24 * 7
            lags_future_covariates = [
                -24 * 14,
                -24 * 7,
                -24,
                0,
                24,
                24 * 7,
                24 * 14,
            ]
            lags_past_covariates = [-24 * 14, -24 * 7, -24]
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
    return lags, lags_future_covariates, lags_past_covariates, add_encoders


def get_xgboost_model(
    freq: Literal["D", "h"],
    quantiles: list[float],
    xgb_kwargs: dict | None = None,
    random_state: int = 0,
    output_chunk_length: int = 1,
    multi_models: bool = False,
) -> DartsXGBModel:
    """Create and configure a DartsXGBModel for time series forecasting.

    Parameters
    ----------
    quantiles : array_like
        Array of quantiles to predict.
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
    lags, lags_future_covariates, lags_past_covariates, add_encoders = get_darts_config(
        freq
    )
    model = DartsXGBModel(
        lags=lags,
        lags_future_covariates=lags_future_covariates,
        # lags_past_covariates=lags_past_covariates,
        add_encoders=add_encoders,
        random_state=random_state,
        likelihood="quantile",
        quantiles=quantiles,
        output_chunk_length=output_chunk_length,
        multi_models=multi_models,
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
    match freq:
        case "h":
            DAY_DURATION = 24
        case "D":
            DAY_DURATION = 1
        case _:
            raise ValueError(
                f"Invalid frequency: {freq}. Only 'D' and 'h' are supported."
            )

    lags = [-DAY_DURATION * i for i in [1, 2, 3, 7]]
    lags_future_covariates = [-DAY_DURATION * i for i in [-7, -2, -1, 0, 1, 2, 7]]
    add_encoders = {
        "cyclic": {
            "future": ["day_of_week", "month"] + (["hour"] if freq == "h" else [])
        },
    }
    model = DartsLinearRegressionModel(
        lags=lags,
        lags_future_covariates=lags_future_covariates,
        add_encoders=add_encoders,
        output_chunk_length=output_chunk_length,
        likelihood="quantile",
        quantiles=quantiles,
        multi_models=False,
        random_state=random_state,
        # solver 'highs' leads to 'high-ds' which takes 5x longer
        kwargs=dict(solver="highs-ipm"),
    )
    # disable user warnings
    model.set_config(warnings="off")
    return model
