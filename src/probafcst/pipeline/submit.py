"""Submission stage."""

import pickle
from datetime import date
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from sktime.forecasting.base import ForecastingHorizon

from probafcst.plotting import plot_quantiles
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
            lead_times = np.arange(2, 8)
            fh = ForecastingHorizon(lead_times, is_relative=True)
        else:
            fh = ForecastingHorizon(forecast_hours, is_relative=True)

        y_pred = forecaster.predict_quantiles(fh, alpha=params.quantiles)

        # TODO: clean this up
        sns.set_theme(style="ticks")
        last_date = forecaster._y.index[-1]
        to_plot = forecaster._y.loc[last_date - pd.Timedelta(days=14) :]
        fig, _ = plot_quantiles(to_plot, y_pred)
        fig.savefig(
            Path(params.output_dir) / f"{target}_forecast.png", bbox_inches="tight"
        )

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
