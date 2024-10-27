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
        start_date = pd.Timestamp.now()
    relative_hours = np.array([36, 40, 44, 60, 64, 68])

    # relative hours are based on midnight of the start date
    offset = get_hours_until_midnight(start=start_date)

    relative_hours += offset
    timestamps = start_date + pd.to_timedelta(relative_hours, unit="h")
    return timestamps, relative_hours
