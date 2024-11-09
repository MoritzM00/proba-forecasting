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
    """Get the DartsXGBModel.

    TODO: docstring
    """
    match freq:
        case "D":
            lags = [-1, -2, -3, -7, -14, -30]
            lags_future_covariates = [-30, -14, -7, 0, 7, 14, 30]
        case "h":
            lags = [-1, -2, -3, -24, -24 * 7, -24 * 30]
            lags_future_covariates = [-24 * 30, -24 * 7, -24, 0, 24, 24 * 7, 24 * 30]
        case _:
            raise ValueError(
                f"Invalid frequency: {freq}. Only 'D' and 'h' are supported."
            )
    if quantiles is None:
        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    add_encoders = {
        "cyclic": {"future": ["day", "month", "day_of_year"]},
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
    return model
