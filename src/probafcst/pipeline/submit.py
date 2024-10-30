"""Submission stage."""

import pickle
from datetime import date
from pathlib import Path

import dvc.api
import numpy as np
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

    pred_quantiles = {}
    for target in ["energy", "bikes"]:
        model_path = get_model_path(params.model_dir, target=target)

        with open(model_path, "rb") as f:
            forecaster = pickle.load(f)

        # TODO: put that into the params.yaml file
        if target == "bikes":
            fh = ForecastingHorizon(np.arange(1, 7), is_relative=True)
        else:
            fh = ForecastingHorizon(forecast_hours, is_relative=True)

        y_pred = forecaster.predict_quantiles(fh, alpha=params.quantiles)
        pred_quantiles[target] = y_pred.to_numpy()

    submission = create_submission(
        forecast_date=date.today(),
        energy_preds=pred_quantiles["energy"],
        bikes_preds=pred_quantiles["bikes"],
        no2_preds=None,
    )
    check_submission(submission)

    submission_path = Path(params.output_dir) / "submission.csv"
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    submit()
