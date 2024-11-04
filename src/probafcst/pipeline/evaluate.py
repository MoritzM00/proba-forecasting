"""Evaluation Stage."""

import json
import pickle
from pathlib import Path

import click
import dvc.api
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting.probabilistic import (
    PinballLoss,
)
from sktime.split import ExpandingWindowSplitter

from probafcst.plotting import plot_quantiles
from probafcst.utils.paths import get_data_path, get_model_path


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
    sns.set_theme(style="ticks")
    params = dvc.api.params_show()
    params = OmegaConf.create(params)

    data_path = get_data_path(params.data_dir, target=target)

    y = pd.read_parquet(data_path)

    model_path = get_model_path(params.model_dir, target)
    with open(model_path, "rb") as f:
        forecaster = pickle.load(f)

    eval_params = params.eval[target]
    fh = np.arange(1, eval_params.fh)
    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=eval_params.initial_window,
        step_length=eval_params.step_length,
    )
    scoring = PinballLoss(alpha=params.quantiles)

    results = evaluate(
        forecaster,
        y=y,
        cv=cv,
        strategy="refit",
        scoring=scoring,
        return_data=True,
        error_score="raise",
    )
    results.iloc[:, :-3].to_csv(f"output/{target}_eval_results.csv", index=False)

    mean_pinball_loss = results["test_PinballLoss"].mean()
    std_pinball_loss = results["test_PinballLoss"].std()

    metrics = {
        "pinball_loss": {"mean": mean_pinball_loss, "std": std_pinball_loss},
        "avg_fit_time": results["fit_time"].mean(),
        "avg_pred_time": results["pred_quantiles_time"].mean(),
    }
    with open(f"output/{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plots_dir = Path(params.output_dir, "eval_plots", target)
    plots_dir.mkdir(exist_ok=True)

    # visualize some forecasts
    idx = [0, len(results) // 2, -1]
    for i, (_, row) in enumerate(results.iloc[idx].iterrows()):
        fig, _ = plot_quantiles(row.y_test, row.y_pred_quantiles)
        fig.savefig(plots_dir / f"forecast_{i + 1}.png", bbox_inches="tight")


if __name__ == "__main__":
    eval()
