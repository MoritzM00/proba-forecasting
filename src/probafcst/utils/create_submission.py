"""Helpers for submission creation."""

import numpy as np
import pandas as pd


def create_submission_frame(
    forecast_date: str | None = None,
    bikes_preds=None,
    energy_preds=None,
    no2_preds=None,
):
    """Create the submission dataframe.

    Parameters
    ----------
    forecast_date : str
        Forecast date in the format "YYYY-MM-DD".
    bikes_preds : array-like or None
        Predictions for bikes. Must have shape (6, 5).
        If None, it will be filled with NaNs.
    energy_preds : array-like or None
        Predictions for energy. Must have shape (6, 5).
        If None, it will be filled with NaNs.
    no2_preds : array-like or None
        Predictions for no2. Must have shape (6, 5).
        If None, it will be filled with NaNs.
    """
    if forecast_date is None:
        # use today
        forecast_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Define horizons and targets
    bike_horizons = ["1 day", "2 day", "3 day", "4 day", "5 day", "6 day"]
    energy_horizons = ["36 hour", "40 hour", "44 hour", "60 hour", "64 hour", "68 hour"]
    no2_horizons = ["36 hour", "40 hour", "44 hour", "60 hour", "64 hour", "68 hour"]

    # Define quantile column names
    quantiles = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]

    # Ensure predictions are valid or filled with NaNs
    n_rows = len(bike_horizons)
    n_cols = len(quantiles)
    bikes_preds = check_predictions(bikes_preds, n_rows=n_rows, n_cols=n_cols)
    energy_preds = check_predictions(energy_preds, n_rows=n_rows, n_cols=n_cols)
    no2_preds = check_predictions(no2_preds, n_rows=n_rows, n_cols=n_cols)

    bikes_df = pd.DataFrame(
        {
            "forecast_date": forecast_date,
            "target": "bikes",
            "horizon": bike_horizons,
        }
    )

    energy_df = pd.DataFrame(
        {
            "forecast_date": forecast_date,
            "target": "energy",
            "horizon": energy_horizons,
        }
    )

    no2_df = pd.DataFrame(
        {
            "forecast_date": forecast_date,
            "target": "no2",
            "horizon": no2_horizons,
        }
    )

    submission = pd.concat([bikes_df, energy_df, no2_df], ignore_index=True)

    submission.loc[submission["target"] == "bikes", quantiles] = bikes_preds
    submission.loc[submission["target"] == "energy", quantiles] = energy_preds
    submission.loc[submission["target"] == "no2", quantiles] = no2_preds

    return submission


def check_predictions(preds, n_rows, n_cols):
    """Check predictions are valid or fill with NaNs.

    Return the provided predictions,
    or a NumPy array of NaNs if preds is None.

    Parameters
    ----------
    preds : array-like or None
        Predictions to check.
    n_rows : int
        Number of rows for the predictions.
    n_cols : int
        Number of columns for the predictions.

    Returns
    -------
        2D NumPy array of predictions or NaNs.
    """
    if preds is None:
        return np.full((n_rows, n_cols), np.nan)
    assert preds.shape == (n_rows, n_cols), "Invalid shape for predictions."
    return preds
