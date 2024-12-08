"""Evaluation Stage."""

import json
import os
from pathlib import Path

import click
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from probafcst.backtest import backtest, get_window_params
from probafcst.models import get_model
from probafcst.pipeline._base import pipeline_setup
from probafcst.plotting import plot_quantiles
from probafcst.utils.paths import get_data_path

os.environ["PYTHONWARNINGS"] = "ignore"


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
    params = pipeline_setup()

    output_dir = Path(params.output_dir)
    data_path = get_data_path(params.data_dir, target=target)

    freq = params.data[target].freq
    target_col = params.data[target].target_col
    data = pd.read_parquet(data_path).asfreq(freq).dropna()

    y = data[target_col]
    X = data.drop(columns=target_col)

    forecaster = get_model(
        params=params.train[target], n_jobs=1, quantiles=params.quantiles
    )

    eval_params = params.eval[target]

    window_params = get_window_params(
        n_years_initial_window=eval_params.n_years_initial_window,
        step_length_days=eval_params.step_length_days,
        forecast_steps_days=eval_params.forecast_steps_days,
        freq=params.data[target].freq,
    )

    results, metrics, predictions, _ = backtest(
        forecaster,
        y,
        **window_params,
        X=X,
        quantiles=params.quantiles,
        backend=params.eval.backend,
        splitter_type=params.eval.splitter_type,
    )
    results.to_csv(output_dir / f"{target}_eval_results.csv", index=False)

    with open(output_dir / f"{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plots_dir = output_dir / "eval_plots" / target
    plots_dir.mkdir(exist_ok=True)

    sns.set_theme(style="ticks")
    # visualize some forecasts
    idx = [0, len(results) // 2, -1]
    for i, (_, row) in enumerate(predictions.iloc[idx].iterrows()):
        fig, _ = plot_quantiles(row.y_test, row.y_pred_quantiles)
        fig.savefig(plots_dir / f"forecast_{i + 1}.svg", bbox_inches="tight")

    # create box plots for each quantile loss using results frame
    melted = results[params.quantiles].melt(var_name="quantile", value_name="loss")
    melted["quantile"] = melted["quantile"].apply(lambda x: f"q{x}")
    fig, ax = plt.subplots()
    sns.boxplot(data=melted, x="quantile", y="loss", hue="quantile", ax=ax)
    fig.savefig(output_dir / f"{target}_pinball_losses.svg", bbox_inches="tight")


if __name__ == "__main__":
    evaluate_forecaster()
