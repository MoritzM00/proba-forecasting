"""Training Stage."""

import pickle
from pathlib import Path

import click
import dvc.api
import pandas as pd
from omegaconf import OmegaConf
from sktime.forecasting.naive import NaiveForecaster

from probafcst.utils.paths import get_data_path, get_model_path


@click.command()
@click.option(
    "--target",
    "-t",
    default="energy",
    type=click.Choice(["energy", "bikes", "no2"]),
    help="The target data to prepare.",
)
def train(target):
    """Train the model."""
    params = dvc.api.params_show()
    params = OmegaConf.create(params)

    data_path = get_data_path(params.data_dir, target=target)

    y = pd.read_parquet(data_path)

    forecaster = NaiveForecaster(strategy="mean", sp=24)
    forecaster.fit(y)

    # Save the model
    model_dir = Path(params.model_dir)
    model_dir.mkdir(exist_ok=True)
    model_path = get_model_path(params.model_dir, target)
    with open(model_path, "wb") as f:
        pickle.dump(forecaster, f)


if __name__ == "__main__":
    train()
