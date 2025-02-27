"""Implement a backtesting strategy for probabilistic forecasts."""

import math
from typing import Literal, NamedTuple, TypedDict

import numpy as np
import pandas as pd
from loguru import logger
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting.probabilistic import (
    PinballLoss,
)
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

from probafcst.metrics.coverage import compute_coverage
from probafcst.metrics.interval_score import interval_score_from_pinball_losses


class WindowParams(TypedDict):
    """NamedTuple to store window parameters for backtesting."""

    initial_window: int
    step_length: int
    forecast_steps: int


class BacktestResults(NamedTuple):
    """NamedTuple to store backtest results."""

    eval_results: pd.DataFrame
    metrics: dict
    predictions: pd.DataFrame
    additional_metrics: dict


def get_window_params(
    n_years_initial_window: float,
    step_length_days: int,
    forecast_steps_days: int,
    freq: Literal["D", "h"],
) -> WindowParams:
    """Calculate window parameters for backtesting."""
    match freq:
        case "D":
            DAY_DURATION = 1
        case "h":
            DAY_DURATION = 24
        case _:
            raise ValueError(
                f"Invalid frequency: {freq}. Only 'D' and 'h' are supported."
            )
    initial_window = math.floor(DAY_DURATION * 365 * n_years_initial_window)
    step_length = DAY_DURATION * step_length_days
    forecast_steps = DAY_DURATION * forecast_steps_days

    return WindowParams(
        initial_window=initial_window,
        step_length=step_length,
        forecast_steps=forecast_steps,
    )


def backtest(
    forecaster: BaseForecaster,
    y: pd.DataFrame,
    forecast_steps: int,
    initial_window: int,
    step_length: int,
    quantiles: list[float],
    X=None,
    backend: str | None = None,
    splitter_type: Literal["expanding", "sliding"] = "sliding",
) -> BacktestResults:
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
    backend: str, default=None
        The backend to use. If None, the default backend is used. Use 'loky' for
        parallel backtest.
    splitter_type : { 'expanding', 'sliding' }, default='sliding'
        Type of splitter to use. Either 'expanding' or 'sliding'.

    Returns
    -------
    eval_results : pd.DataFrame
        Evaluation results with loss informations and window sizes.
    metrics : dict
        Dictionary with average fit time, average prediction time, and pinball losses.
    predictions : pd.DataFrame
        Predictions for each test window, including train and test series.
    """
    fh = np.arange(1, forecast_steps + 1)
    match splitter_type.lower():
        case "expanding":
            cv = ExpandingWindowSplitter(
                fh=fh, initial_window=initial_window, step_length=step_length
            )
        case "sliding":
            cv = SlidingWindowSplitter(
                fh=fh, window_length=initial_window, step_length=step_length
            )
        case _:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
    cutoffs = cv.get_cutoffs(y)
    logger.debug(f"Using cutoffs: {y.iloc[cutoffs].index}")

    n_splits = cv.get_n_splits(y)
    scoring = PinballLoss(alpha=quantiles, score_average=False)

    logger.info(f"Starting Backtest with {n_splits} splits.")
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
        test_PinballLoss=eval_results[quantiles].sum(axis=1)
    )

    # compute interval scores from pinball losses
    alpha_50 = 0.5
    alpha_95 = 0.05
    interval_scores_50 = interval_score_from_pinball_losses(
        alpha=alpha_50,
        pl_lower=eval_results[alpha_50 / 2],
        pl_upper=eval_results[1 - alpha_50 / 2],
    )
    interval_scores_95 = interval_score_from_pinball_losses(
        alpha=alpha_95,
        pl_lower=eval_results[alpha_95 / 2],
        pl_upper=eval_results[1 - alpha_95 / 2],
    )
    coverage_mean, coverage_std = compute_coverage(predictions)

    metrics = {
        "avg_fit_time": results["fit_time"].mean(),
        "avg_pred_time": results["pred_quantiles_time"].mean(),
        "pinball_loss": {
            "mean": eval_results["test_PinballLoss"].mean(),
            "std": eval_results["test_PinballLoss"].std(),
        },
        "interval_score_50": {
            "mean": interval_scores_50.mean(),
            "std": interval_scores_50.std(),
        },
        "interval_score_95": {
            "mean": interval_scores_95.mean(),
            "std": interval_scores_95.std(),
        },
        "coverage_50": {
            "mean": coverage_mean[0.5],
            "std": coverage_std[0.5],
        },
        "coverage_95": {
            "mean": coverage_mean[0.95],
            "std": coverage_std[0.95],
        },
    }
    additional_metrics = {
        f"pinball_loss_q{q}": {
            "mean": eval_results[q].mean(),
            "std": eval_results[q].std(),
        }
        for q in quantiles
    }
    backtest_time = (end - start).total_seconds()
    additional_metrics["backtest_time"] = backtest_time

    logger.info(f"Backtest finished in {backtest_time:.2f} seconds.")
    logger.info(
        f"Pinball Loss: {metrics['pinball_loss']['mean']:.3f} ± {metrics['pinball_loss']['std']:.3f}"
    )

    return BacktestResults(eval_results, metrics, predictions, additional_metrics)
