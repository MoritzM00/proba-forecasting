"""Training Stage."""

import os
import pickle
from pathlib import Path

import click
import pandas as pd

from probafcst.models import get_model
from probafcst.pipeline._base import pipeline_setup
from probafcst.utils.paths import get_data_path, get_model_path

os.environ["PYTHONWARNINGS"] = "ignore"


@click.command()
@click.option(
    "--target",
    "-t",
    default="energy",
    type=click.Choice(["energy", "bikes"]),
    help="The target data to prepare.",
)
def train(target):
    """Train the model."""
    params = pipeline_setup()

    data_path = get_data_path(params.data_dir, target=target)

    freq = params.data[target].freq

    data = pd.read_parquet(data_path).asfreq(freq).dropna()
    data_subset = data.loc[params.train[target].cutoff :]

    target_col = params.data[target].target_col
    y = data_subset[target_col]
    X = data_subset.drop(columns=target_col)

    forecaster = get_model(
        params=params.train[target],
        quantiles=params.quantiles,
        freq=freq,
    )

    forecaster.fit(y, X=X)

    # Save the model
    model_dir = Path(params.model_dir)
    model_dir.mkdir(exist_ok=True)
    model_path = get_model_path(params.model_dir, target)
    with open(model_path, "wb") as f:
        pickle.dump(forecaster, f)


if __name__ == "__main__":
    train()
