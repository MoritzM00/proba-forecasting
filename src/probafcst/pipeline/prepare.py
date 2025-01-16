"""Data Preparation Stage.

This script prepares the data for the models. It loads each target into
a pandas dataframe and saves it as a parquet file. Each target data is
loaded using a specific function from the data module, which uses GET requests
to download the data from the web. Those requests are cached by default to avoid
unnecessary duplicated requests.
"""

from datetime import timedelta
from pathlib import Path
from warnings import simplefilter

import click
import requests_cache

from probafcst.data import get_bikes_data, get_energy_data, get_no2_data
from probafcst.pipeline._base import pipeline_setup
from probafcst.utils.sktime import get_holiday_indicator
from probafcst.weather import generate_weather_features

simplefilter("ignore", category=FutureWarning)


@click.command(name="prepare")
@click.option(
    "--target",
    "-t",
    default="energy",
    type=click.Choice(["energy", "bikes", "no2"]),
    help="The target data to prepare.",
)
def prepare(target: str) -> None:
    """Prepare data for the models.

    Parameters
    ----------
    target : str
        The target data to prepare.

    Raises
    ------
    ValueError
        If the target is invalid. Must be one of "energy", "bikes" or "no2".
    """
    params = pipeline_setup()

    if params.cache.enable and not requests_cache.is_installed():
        days_expire_after = params.cache.days_expire_after
        requests_cache.install_cache(
            "probafcst_cache",
            backend="sqlite",
            expire_after=timedelta(days=days_expire_after),
        )

    data_dir = Path(params.data_dir)
    data_dir.mkdir(exist_ok=True)

    match target:
        case "energy":
            data = get_energy_data(ignore_years=params.data.energy.ignore_years)
        case "bikes":
            data = get_bikes_data(start_date=params.data.bikes.start_date)
        case "no2":
            data = get_no2_data(
                start_date=params.data.no2.start_date, end_date=params.data.no2.end_date
            )
        case _:
            raise ValueError("Invalid target.")

    data = data.astype("float32")

    # generate whether features
    weather_features = generate_weather_features(
        data.index, forecast_days=14, past_days=60
    )
    selected_features = params.data[target].weather_features
    if selected_features is None:
        selected_features = []
    weather_features = weather_features[selected_features]

    # weather features are generated for the next 7 days,
    # but interpolate will fill the missing values after timezone conversion
    # therefore we save the last date of the data
    last_idx_data = data.index[-1].tz_localize(None)

    data = data.join(weather_features, how="right", validate="1:1")

    # remove timezone information and interpolate missing dates
    date_col = data.index.name
    y = data.reset_index()
    y[date_col] = y[date_col].dt.tz_localize(None)
    y = y.drop_duplicates(subset=[date_col])
    y = y.set_index(date_col)
    data = y.asfreq(params.data[target].freq)
    data.loc[:last_idx_data] = data.loc[:last_idx_data].interpolate()

    # add holidays
    is_holiday = get_holiday_indicator(data)
    data = data.assign(is_holiday=is_holiday)

    filepath = data_dir / f"{target}.parquet"
    data.to_parquet(filepath)


if __name__ == "__main__":
    prepare()
