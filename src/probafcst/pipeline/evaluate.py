"""Evaluation Stage."""

import click
import dvc.api
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
from sktime.split import SlidingWindowSplitter

from probafcst.model import get_model
from probafcst.utils.paths import get_data_path


@click.command()
@click.option(
    "--target",
    "-t",
    default="energy",
    type=click.Choice(["energy", "bikes", "no2"]),
    help="The target data to prepare.",
)
def eval(target: str):
    """Evaluate the model."""
    params = dvc.api.params_show()
    params = OmegaConf.create(params)

    data_path = get_data_path(params.data_dir, target=target)

    y = pd.read_parquet(data_path)

    forecaster = get_model(target)

    eval_params = params.eval[target]
    fh = np.arange(1, eval_params.fh)
    cv = SlidingWindowSplitter(
        fh=fh,
        window_length=eval_params.window_length,
        step_length=eval_params.step_length,
    )
    loss = PinballLoss()

    results = evaluate(forecaster, y, cv=cv, strategy="refit", scoring=loss)
    results.to_csv(f"output/eval_results_{target}.csv")


if __name__ == "__main__":
    eval()
