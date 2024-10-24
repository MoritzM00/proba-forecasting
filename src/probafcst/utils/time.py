"""Utility functions for working with time."""

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
    midnight = (start + pd.Timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    hours_until_midnight = (midnight - start).total_seconds() / 3600
    # round down
    hours_until_midnight = int(hours_until_midnight)
    return hours_until_midnight
