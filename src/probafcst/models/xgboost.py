"""XGBoost Quantile Regression Model."""

from typing import Literal

from sktime.forecasting.darts import DartsXGBModel


def get_xgboost_model(
    quantiles: list[float] = None,
    freq: Literal["D", "h"] = "h",
    xgb_kwargs: dict | None = None,
    random_state: int = 0,
    output_chunk_length: int = 1,
) -> DartsXGBModel:
    """Create and configure a DartsXGBModel for time series forecasting.

    Parameters
    ----------
    quantiles : array_like, optional
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
    match freq:
        case "D":
            lags = 60
            lags_future_covariates = list(range(-60, 61))
            additional_encoders = []
        case "h":
            lags = 24 * 7
            lags_future_covariates = [-24 * 30, -24 * 7, -24, 0, 24, 24 * 7, 24 * 30]
            additional_encoders = ["hour"]
        case _:
            raise ValueError(
                f"Invalid frequency: {freq}. Only 'D' and 'h' are supported."
            )
    if quantiles is None:
        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    add_encoders = {
        "cyclic": {"future": ["day", "month", "day_of_year", *additional_encoders]},
    }

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
