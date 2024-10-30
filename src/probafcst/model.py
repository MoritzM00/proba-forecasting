"""Implementation of the models used for forecasting."""

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster


def get_model(target: str, params: dict | None = None) -> BaseForecaster:
    """Return the model for the given target."""
    if params is None:
        params = {}
    match target:
        case "energy" | "no2":
            return get_energy_model(**params)
        case "bikes":
            return get_bikes_model(**params)
        case _:
            raise ValueError(f"Unknown target: {target}")


def get_energy_model(**params) -> BaseForecaster:
    """Return the energy model."""
    return NaiveForecaster(strategy="mean", window_length=24 * 365, sp=24)


def get_bikes_model(**params) -> BaseForecaster:
    """Return the bikes model."""
    return NaiveForecaster(strategy="mean", window_length=90, sp=7)
