"""Implement a backtesting strategy for probabilistic forecasts."""

import numpy as np
import pandas as pd
from loguru import logger
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting.probabilistic import (
    PinballLoss,
)
from sktime.split import ExpandingWindowSplitter


def backtest(
    forecaster: BaseForecaster,
    y: pd.DataFrame,
    forecast_steps: int,
    initial_window: int,
    step_length: int,
    quantiles: list[float],
    X=None,
    backend: str | None = None,
):
    """Backtest a probabilistic forecaster using Pinball Loss scoring.

    Parameters
    ----------
    forecaster : sktime.forecasting.base.BaseForecaster
        A forecaster object.
    y : pd.DataFrame
        The time series data.
    forecast_steps : int
        Number of steps to forecast.
    initial_window : int
        Initial window size.
    step_length : int
        Step length.
    quantiles : list[float]
        List of quantile levels.

    Returns
    -------
    eval_results : pd.DataFrame
        Evaluation results with loss informations and window sizes.
    metrics : dict
        Dictionary with average fit time, average prediction time, and pinball losses.
    predictions : pd.DataFrame
        Predictions for each test window, including train and test series.
    """
    fh = np.arange(1, forecast_steps)
    cv = ExpandingWindowSplitter(
        fh=fh,
        initial_window=initial_window,
        step_length=step_length,
    )
    scoring = PinballLoss(alpha=quantiles, score_average=False)

    start = pd.Timestamp.now()
    results = evaluate(
        forecaster,
        y=y,
        X=X,
        cv=cv,
        strategy="refit",
        scoring=scoring,
        return_data=True,
        backend=backend,
        error_score="raise",
    )
    end = pd.Timestamp.now()

    # each entry in test_PinballLoss is a Series with the quantile levels as index
    # expand them into columns
    results = pd.concat(
        [
            results.drop(columns="test_PinballLoss"),
            results["test_PinballLoss"].apply(pd.Series),
        ],
        axis=1,
    )
    predictions = results[["y_train", "y_test", "y_pred_quantiles"]]
    eval_results = results[
        ["fit_time", "pred_quantiles_time", "len_train_window", "cutoff", *quantiles]
    ]
    eval_results = eval_results.assign(
        # TODO: might change the mean to sum instead
        # because in the submission evaluation it is done like this
        # might make sense because these scores are quite volatile over quantile levels
        test_PinballLoss=eval_results[quantiles].sum(axis=1)
    )

    metrics = {
        "avg_fit_time": results["fit_time"].mean(),
        "avg_pred_time": results["pred_quantiles_time"].mean(),
        "pinball_loss": {
            "mean": eval_results["test_PinballLoss"].mean(),
            "std": eval_results["test_PinballLoss"].std(),
        },
    }
    additional_metrics = {
        f"pinball_loss_q{q}": {
            "mean": eval_results[q].mean(),
            "std": eval_results[q].std(),
        }
        for q in quantiles
    }

    logger.info(f"Backtesting finished in {(end - start).total_seconds():.2f} seconds.")
    logger.info(
        f"Pinball Loss: {metrics['pinball_loss']['mean']:.3f} ± {metrics['pinball_loss']['std']:.3f}"
    )

    return eval_results, metrics, predictions, additional_metrics
