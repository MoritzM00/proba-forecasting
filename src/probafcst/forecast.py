"""Forecast Pipeline functions for probabilistic forecasting."""

import warnings
from typing import Literal

import holidays
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.compose import FeatureUnion, Logger, YtoX
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.fourier import FourierFeatures
from sktime.transformations.series.holiday import (
    HolidayFeatures,
)

warnings.simplefilter("ignore", category=FutureWarning)


def get_fourier_kwargs(
    freq: Literal["D", "h"], fourier_terms_list: list[int] | None = None
) -> dict:
    """Get Fourier Features."""
    if freq == "D":
        sp_list = [7, 365]
        if fourier_terms_list is None:
            fourier_terms_list = [5, 3]
        assert (
            len(fourier_terms_list) == 2
        ), "Two fourier terms are required for daily data."

    elif freq == "h":
        sp_list = [24, 24 * 7, 24 * 7 * 52]
        if fourier_terms_list is None:
            fourier_terms_list = [10, 5, 3]
        assert (
            len(fourier_terms_list) == 3
        ), "Three fourier terms are required for hourly data."

    return dict(sp_list=sp_list, fourier_terms_list=fourier_terms_list)


def get_featurizer(
    fourier_kwargs: dict | None = None,
    datetime_features=None,
    include_holidays=True,
    n_jobs=1,
):
    """Get Featurizer."""
    transformers = []
    if fourier_kwargs is not None and fourier_kwargs != {}:
        fourier_features = FourierFeatures(**fourier_kwargs)
        transformers.append(("fourier", fourier_features))

    if datetime_features is None:
        datetime_features = ["is_weekend"]

    transformers = [
        ("fourier", FourierFeatures(**fourier_kwargs)),
        ("datetime", DateTimeFeatures(manual_selection=datetime_features)),
    ]

    if include_holidays:
        calender = holidays.country_holidays("DE", subdiv="BW")
        holiday_features = HolidayFeatures(
            calender, return_indicator=True, return_dummies=False
        )
        transformers.append(("holidays", holiday_features))

    return FeatureUnion(transformers, n_jobs=n_jobs)


def get_pipeline(
    forecaster: BaseForecaster, featurizer: FeatureUnion, logger_name=None
):
    """Get ForecastingPipeline."""
    steps = [
        ("y_to_x", YtoX()),
        ("featurizer", featurizer),
    ]

    if logger_name is not None:
        logger = Logger(logger=logger_name, logger_backend="datalog")
        steps.append(("logger", logger))

    steps.append(("forecaster", forecaster))

    return ForecastingPipeline(steps=steps)
