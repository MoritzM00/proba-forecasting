"""Weather data retrieval from the Open-Meteo API."""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

KARLSRUHE_LONGITUDE = 8.4044
KARLSRUHE_LATITUDE = 49.0094


def get_openmeteo_client(expire_after=-1):
    """Get an Open-Meteo API client with cache and retry on error.

    Parameters
    ----------
    expire_after : int, optional
        Cache expiration time in seconds, by default indefinitely.

    Returns
    -------
    openmeteo_requests.Client
        Open-Meteo API client.
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=expire_after)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def get_hourly_weather_features(
    start_date: str = "2015-01-01",
    end_date: str = "2024-10-31",
    forecast_days=7,
    past_days=61,
    expire_after=-1,
    localize=True,
):
    """Get weather features for the training and forecast horizon.

    Parameters
    ----------
    start_date : str, optional
        Start date for historical weather data, by default "2015-01-01".
    end_date : str, optional
        End date for historical weather data, by default "2024-10-31".
    forecast_days : int, optional
        Number of days to forecast from today, by default 7.
    past_days : int, optional
        Number of days of past forecasts to use (instead of historical data).
    expire_after : int, optional
        Cache expiration time in seconds, by default indefinitely.
    localize : bool, optional
        Localize the weather data to Europe/Berlin timezone, by default True.

    Returns
    -------
    pd.DataFrame
        Hourly weather features.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    client = get_openmeteo_client(expire_after=expire_after)

    # end_date must be within past_days from today
    end_date = pd.Timestamp(end_date)
    start_date = pd.Timestamp(start_date)
    if end_date < pd.Timestamp("today") - pd.Timedelta(days=past_days):
        raise ValueError(f"{end_date=} must be within {past_days=} from today.")

    hourly_historical_weather = get_hourly_historical_weather(
        start_date=start_date,
        end_date=end_date,
        expire_after=expire_after,
        client=client,
    )
    hourly_forecast_weather = get_hourly_forecast_weather(
        forecast_days=forecast_days,
        past_days=past_days,
        expire_after=expire_after,
        client=client,
    )

    first_forecast_date = hourly_forecast_weather.index[0]
    hourly_historical_weather = hourly_historical_weather.loc[
        : first_forecast_date - pd.Timedelta(hours=1)
    ]

    weather_features = pd.concat(
        [hourly_historical_weather, hourly_forecast_weather], axis=0
    )

    if localize:
        # localize to Europe/Berlin timezone
        weather_features.index = weather_features.index.tz_convert("Europe/Berlin")
    return weather_features


def get_hourly_historical_weather(
    start_date: str = "2015-01-01",
    end_date: str = "2024-10-31",
    expire_after=-1,
    client=None,
):
    """Get hourly historical weather data from the Open-Meteo API."""
    if client is None:
        # Setup the Open-Meteo API client with cache and retry on error
        client = get_openmeteo_client(expire_after=expire_after)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": KARLSRUHE_LATITUDE,
        "longitude": KARLSRUHE_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_100m",
            "is_day",
            "sunshine_duration",
        ],
        "timezone": "Europe/Berlin",
    }
    responses = client.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(4).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(5).ValuesAsNumpy()
    hourly_sunshine_duration = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["wind_speed"] = hourly_wind_speed_100m
    hourly_data["is_day"] = hourly_is_day
    hourly_data["sunshine_duration"] = hourly_sunshine_duration

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe.set_index("date")


def get_hourly_forecast_weather(
    forecast_days: 7, past_days: 31, expire_after=3600, client=None
):
    """Get hourly forecast weather data from the Open-Meteo API."""
    if client is None:
        # Setup the Open-Meteo API client with cache and retry on error
        client = get_openmeteo_client(expire_after=expire_after)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": KARLSRUHE_LATITUDE,
        "longitude": KARLSRUHE_LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_120m",
            "is_day",
            "sunshine_duration",
        ],
        "timezone": "Europe/Berlin",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    responses = client.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_120m = hourly.Variables(4).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(5).ValuesAsNumpy()
    hourly_sunshine_duration = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["wind_speed"] = hourly_wind_speed_120m
    hourly_data["is_day"] = hourly_is_day
    hourly_data["sunshine_duration"] = hourly_sunshine_duration

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe.set_index("date")
