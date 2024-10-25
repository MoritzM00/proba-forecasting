"""Data module for probafcst package."""

from datetime import date

import pandas as pd


def get_no2_data(start_date: str = "2020-01-01") -> pd.DataFrame:
    """Get NO2 data."""
    end_date = date.today()

    url = (
        f"https://www.umweltbundesamt.de/api/air_data/v3/measures/csv?date_from={start_date}"
        f"&time_from=24&date_to={end_date}"
        f"&time_to=23&data%5B0%5D%5Bco%5D=5&data%5B0%5D%5Bsc%5D=2&data%5B0%5D%5Bda%5D={end_date}"
        "&data%5B0%5D%5Bti%5D=12&data%5B0%5D%5Bst%5D=282&data%5B0%5D%5Bva%5D=27&lang=en"
    )

    no2 = pd.read_csv(url, sep=";", na_values="-")
    no2 = no2.drop(no2.index[-1])
    no2["Measure value"] = no2["Measure value"].astype(float)

    # Extract hour from (string) Time column and get it in correct format

    no2["hour"] = no2["Time"].str[1:3].astype(int)
    no2["hour"] = no2["hour"].replace(24, 0)

    # Create datetime column and set it as index
    no2["datetime"] = pd.to_datetime(
        no2["Date"] + " " + no2["hour"].astype(str) + ":00"
    )
    no2 = no2.set_index("datetime")

    # no2 = no2.dropna(axis=0)

    return no2
