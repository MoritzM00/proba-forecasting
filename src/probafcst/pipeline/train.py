"""Training Stage."""

import pickle
from pathlib import Path

import click
import dvc.api
import pandas as pd
from omegaconf import OmegaConf
from sktime.forecasting.naive import NaiveForecaster


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

    data_path = Path(params.data_dir) / params.data.energy.filename
    y = pd.read_parquet(data_path)

    forecaster = NaiveForecaster(strategy="mean", sp=24)
    forecaster.fit(y)

    # Save the model
    model_dir = Path(params.model_dir)
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{target}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(forecaster, f)


if __name__ == "__main__":
    train()
