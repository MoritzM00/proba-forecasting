"""Evaluation Stage."""

import json
import pickle
from pathlib import Path

import click
import dvc.api
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from probafcst.backtest import backtest
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
def evaluate_forecaster(target: str):
    """Evaluate the model."""
    sns.set_theme(style="ticks")
    params = dvc.api.params_show()
    params = OmegaConf.create(params)

    output_dir = Path(params.output_dir)
    data_path = get_data_path(params.data_dir, target=target)

    y = pd.read_parquet(data_path)

    model_path = get_model_path(params.model_dir, target)
    with open(model_path, "rb") as f:
        forecaster = pickle.load(f)

    eval_params = params.eval[target]
    results, metrics, predictions, _ = backtest(
        forecaster,
        y,
        forecast_steps=eval_params.fh,
        initial_window=eval_params.initial_window,
        step_length=eval_params.step_length,
        quantiles=params.quantiles,
    )
    results.to_csv(output_dir / f"{target}_eval_results.csv", index=False)

    with open(output_dir / f"{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plots_dir = output_dir / "eval_plots" / target
    plots_dir.mkdir(exist_ok=True)

    # visualize some forecasts
    idx = [0, len(results) // 2, -1]
    for i, (_, row) in enumerate(predictions.iloc[idx].iterrows()):
        fig, _ = plot_quantiles(row.y_test, row.y_pred_quantiles)
        fig.savefig(plots_dir / f"forecast_{i + 1}.png", bbox_inches="tight")

    # create box plots for each quantile loss using results frame
    # use melt for this
    melted = results[params.quantiles].melt(var_name="quantile", value_name="loss")
    melted["quantile"] = melted["quantile"].apply(lambda x: f"q{x}")

    fig, ax = plt.subplots()
    sns.boxplot(data=melted, x="quantile", y="loss", hue="quantile", ax=ax)
    fig.savefig(output_dir / f"{target}_pinball_losses.png", bbox_inches="tight")


if __name__ == "__main__":
    evaluate_forecaster()
