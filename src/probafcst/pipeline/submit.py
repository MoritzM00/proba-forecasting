"""Submission stage."""

import pickle
from datetime import date
from pathlib import Path

import dvc.api
from omegaconf import OmegaConf
from sktime.forecasting.base import ForecastingHorizon

from probafcst.utils import check_submission, create_submission
from probafcst.utils.paths import get_model_path
from probafcst.utils.time import get_forecast_dates


def submit():
    """Submit the forecasts."""
    params = dvc.api.params_show()
    params = OmegaConf.create(params)

    _, forecast_hours = get_forecast_dates()

    model_path = get_model_path(params.model_dir, target="energy")

    with open(model_path, "rb") as f:
        forecaster = pickle.load(f)

    fh = ForecastingHorizon(forecast_hours, is_relative=True)
    y_pred = forecaster.predict_quantiles(fh, alpha=params.quantiles)

    submission = create_submission(
        forecast_date=date.today(), energy_preds=y_pred.values
    )
    check_submission(submission)

    submission_path = Path(params.output_dir) / "submission.csv"
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    submit()
