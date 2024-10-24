"""Test cases for the time module."""

import pandas as pd
import pytest

from probafcst.utils.time import get_hours_until_midnight


@pytest.mark.parametrize(
    "start, expected_hours",
    [
        (pd.Timestamp("2021-01-01 12:00:00"), 12),
        (pd.Timestamp("2021-01-01 23:00:00"), 1),
        (pd.Timestamp("2021-01-01 01:00:00"), 23),
        (pd.Timestamp("2021-01-01 11:00:00", tz="Europe/Berlin"), 13),
    ],
)
def test_get_hours_until_midnight(start, expected_hours):
    """Test get_hours_until_midnight computation."""
    hours_until_midnight = get_hours_until_midnight(start=start)
    assert hours_until_midnight == expected_hours
