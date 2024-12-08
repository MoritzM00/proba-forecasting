"""Weather data retrieval from the Open-Meteo API."""

import openmeteo_requests
import pandas as pd
import requests_cache
from loguru import logger
from retry_requests import retry

KARLSRUHE_LONGITUDE = 8.4044
KARLSRUHE_LATITUDE = 49.0094


def generate_weather_features(
    y_index: pd.DatetimeIndex,
    openmeteo_client=None,
    forecast_days: int = 7,
    past_days: int = 60,
):
    """Generate weather futures for timeseries y.

    If y contains hourly data, the function will return hourly weather features.
    If y contains daily data, the function will return daily weather features.
    Other frequencies are not supported.

    It will use both historical and forecast weather data to generate the features.
    Historical data is used for data that is older than 60 days of the current date.
    Forecast data is used otherwise and is limited to 16 days in the future. By Default,
    7-day forecasts are included.

    Parameters
    ----------
    y : pd.DateTimeIndex
        The date time index of the timeseries.
    openmeteo_client : OpenMeteoClient
        An instance of OpenMeteoClient. If None, a new instance will be created.
    forecast_days : int
        The number of days to forecast. Default is 7.
    """
    if openmeteo_client is None:
        openmeteo_client = get_openmeteo_client(expire_after=-1)

    freq = pd.infer_freq(y_index)
    if freq is None:
        raise ValueError("Could not infer frequency of the timeseries.")
    elif freq == "h":
        historical_weather_func = get_hourly_historical_weather
        forecast_weather_func = get_hourly_forecast_weather
        offset = pd.Timedelta(hours=1)
    elif freq == "D":
        historical_weather_func = get_daily_historical_weather
        forecast_weather_func = get_daily_forecast_weather
        offset = pd.Timedelta(days=1)

    else:
        raise ValueError(f"Frequency must be in [h, D], but got {freq}.")

    start_date = y_index[0]
    end_date = y_index[-1] - pd.Timedelta(days=past_days)
    assert (
        start_date < end_date
    ), "The timeseries is too short to generate weather features."

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    logger.debug(f"Requesting historical weather data from {start_date} to {end_date}.")
    historical_weather = historical_weather_func(
        start_date, end_date, client=openmeteo_client
    )
    forecast_weather = forecast_weather_func(
        forecast_days=forecast_days, past_days=past_days + 1, client=openmeteo_client
    )

    historical_weather = historical_weather.loc[: forecast_weather.index[0] - offset]
    logger.debug(
        f"Using historical weather data from {historical_weather.index[0]} to {historical_weather.index[-1]}."
    )
    logger.debug(
        f"Using forecast weather data from {forecast_weather.index[0]} to {forecast_weather.index[-1]}."
    )

    weather_features = pd.concat([historical_weather, forecast_weather], axis=0)
    weather_features.index = weather_features.index.tz_convert("Europe/Berlin")

    if freq == "D":
        # adjust the index to match the target timeseries
        periods = len(y_index) + forecast_days + 1
        weather_features.index = pd.date_range(
            start=y_index[0], periods=periods, freq=freq, name=y_index.name
        )

    return weather_features


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


def get_daily_forecast_weather(
    forecast_days: 7, past_days: 31, expire_after=3600, client=None
):
    """Get daily forecast weather data from the Open-Meteo API."""
    if client is None:
        # Setup the Open-Meteo API client with cache and retry on error
        client = get_openmeteo_client(expire_after=expire_after)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": KARLSRUHE_LATITUDE,
        "longitude": KARLSRUHE_LONGITUDE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "daylight_duration",
            "sunshine_duration",
            "uv_index_max",
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
        ],
        "timezone": "Europe/Berlin",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    responses = client.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(5).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["daylight_duration"] = daily_daylight_duration
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe.set_index("date")


def get_daily_historical_weather(
    start_date: str = "2015-01-01",
    end_date: str = "2024-10-31",
    expire_after=-1,
    client=None,
):
    """Get daily forecast weather data from the Open-Meteo API."""
    if client is None:
        # Setup the Open-Meteo API client with cache and retry on error
        client = get_openmeteo_client(expire_after=expire_after)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": KARLSRUHE_LATITUDE,
        "longitude": KARLSRUHE_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "daylight_duration",
            "sunshine_duration",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "Europe/Berlin",
    }
    responses = client.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(5).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["daylight_duration"] = daily_daylight_duration
    daily_data["sunshine_duration"] = daily_sunshine_duration
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe.set_index("date")
