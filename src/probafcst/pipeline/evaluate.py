"""Evaluation Stage."""

import json
from pathlib import Path

import click
import dvc.api
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
from sktime.split import ExpandingWindowSplitter

from probafcst.model import get_model
from probafcst.plotting import plot_quantiles
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
    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=eval_params.initial_window,
        step_length=eval_params.step_length,
    )
    loss = PinballLoss(alpha=params.quantiles)

    results = evaluate(
        forecaster, y=y, cv=cv, strategy="refit", scoring=loss, return_data=True
    )
    results.iloc[:, :-3].to_csv(f"output/eval_results_{target}.csv", index=False)

    mean_pinball_loss = results["test_PinballLoss"].mean()

    # save to metrics.json
    metrics = {
        "mean_pinball_loss": mean_pinball_loss,
    }
    with open(f"output/{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plots_dir = Path(params.output_dir, "eval_plots", target)
    plots_dir.mkdir(exist_ok=True)

    nrows = min(3, len(results))
    for i, (_, row) in enumerate(results.iloc[-nrows:].iterrows()):
        fig, _ = plot_quantiles(row.y_test, row.y_pred_quantiles)
        fig.savefig(plots_dir / f"forecast_{i + 1}.png")


if __name__ == "__main__":
    eval()
