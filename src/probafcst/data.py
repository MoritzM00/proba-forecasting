"""Data module for probafcst package."""

from datetime import date, datetime

import pandas as pd
import requests
from tqdm import tqdm


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
    no2 = no2.drop_duplicates(subset=["datetime"])
    no2 = no2.set_index("datetime")

    no2 = no2.resample("h").bfill()

    return no2["Measure value"]


def get_bikes_data(start_date: str = "01/01/2019") -> pd.DataFrame:
    """Get bikes data.

    Parameters
    ----------
    start_date : str
        Start date in format "dd/mm/yyyy".

    Returns
    -------
    bikes_df : pd.DataFrame
        DataFrame with bike data.
    """
    dataurl = (
        "https://www.eco-visio.net/api/aladdin/1.0.0/pbl/publicwebpageplus/data/"
        f"100126474?idOrganisme=4586&idPdc=100126474&interval=4&flowIds=100126474&debut={start_date}"
    )
    response = requests.get(dataurl)
    rawdata = response.json()

    bikes_df = pd.DataFrame(rawdata, columns=["date", "bike_count"])
    bikes_df["date"] = pd.to_datetime(bikes_df["date"])
    bikes_df = bikes_df.set_index("date")
    bikes_df = bikes_df.astype({"bike_count": float})
    return bikes_df


def get_energy_data(ignore_years: int = 6) -> pd.DataFrame:
    """Get energy data.

    Parameters
    ----------
    ignore_years : int
        Number of years to ignore.

    Returns
    -------
    energy_data : pd.DataFrame
        DataFrame with bike data.
    """
    # get all available time stamps
    stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
    response = requests.get(stampsurl)

    # ignore first x years
    timestamps = list(response.json()["timestamps"])[ignore_years * 52 :]

    col_names = ["date_time", "Netzlast_Gesamt"]
    energy_data = pd.DataFrame(columns=col_names)

    # loop over all available timestamps
    for stamp in tqdm(timestamps):
        dataurl = (
            "https://www.smard.de/app/chart_data/410/DE/410_DE_hour_"
            + str(stamp)
            + ".json"
        )
        response = requests.get(dataurl)
        raw_data = response.json()["series"]
        for i in range(len(raw_data)):
            raw_data[i][0] = datetime.fromtimestamp(
                int(str(raw_data[i][0])[:10])
            ).strftime("%Y-%m-%d %H:%M:%S")

        energy_data = pd.concat(
            [energy_data, pd.DataFrame(raw_data, columns=col_names)]
        )

    energy_data = energy_data.dropna()

    # adjust label
    energy_data["date_time"] = pd.to_datetime(energy_data.date_time) + pd.DateOffset(
        hours=1
    )
    # handle DST
    energy_data = energy_data.drop_duplicates(subset=["date_time"])
    energy_data = energy_data.set_index("date_time")

    energy_data = energy_data.resample("h").bfill()

    return energy_data
