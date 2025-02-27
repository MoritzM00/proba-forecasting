"""Submission stage."""

import pickle
from datetime import date
from pathlib import Path

import pandas as pd
import seaborn as sns
from sktime.forecasting.base import ForecastingHorizon

from probafcst.pipeline._base import pipeline_setup
from probafcst.plotting import plot_quantiles
from probafcst.utils import check_submission, create_submission
from probafcst.utils.paths import get_data_path, get_model_path
from probafcst.utils.time import get_forecast_dates, get_next_wednesday


def submit():
    """Submit the forecasts."""
    params = pipeline_setup()

    pred_quantiles = {}
    for target in ["energy", "bikes"]:
        model_path = get_model_path(params.model_dir, target=target)
        data_path = get_data_path(params.data_dir, target=target)
        target_col = params.data[target].target_col
        X = pd.read_parquet(data_path).drop(columns=target_col)

        with open(model_path, "rb") as f:
            forecaster = pickle.load(f)

        start_date = get_next_wednesday()
        if target == "bikes":
            start_date = start_date.replace(hour=0, minute=0, second=0)
            horizon = pd.date_range(start_date, periods=7, freq="D", inclusive="right")
            fh = ForecastingHorizon(horizon, is_relative=False)
        else:
            start_date = get_next_wednesday()
            forecast_dates = get_forecast_dates(start_date=start_date)
            fh = ForecastingHorizon(forecast_dates, is_relative=False)

        y_pred = forecaster.predict_quantiles(fh, X=X, alpha=params.quantiles)

        sns.set_theme(style="ticks")
        # plot last 14 days of actual series, and Out-of-Sample forecast
        last_date = forecaster.cutoff[0]
        to_plot = forecaster._y.loc[last_date - pd.Timedelta(days=14) :]
        fig, _ = plot_quantiles(to_plot, y_pred)
        fig.savefig(
            Path(params.output_dir) / f"{target}_forecast.svg", bbox_inches="tight"
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
