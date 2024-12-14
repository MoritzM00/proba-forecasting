"""Utility functions for working with time."""

import numpy as np
import pandas as pd


def get_hours_until_midnight(start: pd.Timestamp | None = None) -> int:
    """Calculate the number of hours until the next midnight.

    Parameters
    ----------
    start : pd.Timestamp, optional
        The starting time. If None, the current time is used.

    Returns
    -------
    hours_until_midnight : int
        The number of hours until the next midnight.
    """
    if start is None:
        start = pd.Timestamp.now()
    midnight = (start + pd.Timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    hours_until_midnight = (midnight - start).total_seconds() / 3600
    return int(hours_until_midnight)


def get_current_wednesday(start: pd.Timestamp | None = None) -> pd.Timestamp:
    """Get the current Wednesday.

    Parameters
    ----------
    start : pd.Timestamp, optional
        The starting time. If None, the current time is used.

    Returns
    -------
    pd.Timestamp
        The current Wednesday.
    """
    if start is None:
        start = pd.Timestamp.now()
        # Get the current week's Wednesday
    # Week starts on Monday (0), Tuesday (1), Wednesday (2), ..., Sunday (6)
    weekday = start.weekday()
    wednesday = start + pd.Timedelta(days=(2 - weekday))
    return wednesday.floor("h")


def get_forecast_dates(
    start_date: pd.Timestamp | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Get the forecast dates for a submission.

    This function generates the forecast dates for the submission dataframe.

    Parameters
    ----------
    start_date : pd.Timestamp
        The starting date, i.e. the last available timestamp in the data.

    Returns
    -------
    tuple of pd.DateTimeIndex, np.ndarray
        The forecast dates and the relative forecast hours.
    """
    if start_date is None:
        start_date = pd.Timestamp.now().floor("h")

    idx = pd.date_range(start_date, start_date + pd.Timedelta(days=5), freq="h")

    # Hours from 0-23 (not 1-24!)
    CONSIDERED_HOURS = [12, 16, 20]
    CONSIDERED_DAYS = [4, 5]  # Friday and Saturday

    forecast_dates = idx[
        idx.hour.isin(CONSIDERED_HOURS) & idx.dayofweek.isin(CONSIDERED_DAYS)
    ]
    return forecast_dates
