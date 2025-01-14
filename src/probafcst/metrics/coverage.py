"""Provide empirical coverage computation for interval predictions."""

import numpy as np
from sktime.performance_metrics.forecasting.probabilistic import EmpiricalCoverage

from probafcst.utils.sktime import quantiles_to_interval_df


def compute_coverage(predictions):
    """Compute empirical coverage for 50% and 95% central prediction intervals.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing the predictions made during folds. This is the output
        of the backtest function in `probafcst.backtest`.
        The DataFrame should have the following columns:
        - 'y_train': Training values for the time series (ignored)
        - 'y_test': Actual values for the time series
        - 'y_pred_quantiles': Predicted quantiles for the time series

    Returns
    -------
    emp_coverage_mean : dict
        Dictionary containing the mean empirical coverage for each coverage level.
    emp_coverage_std : dict
        Dictionary containing the standard deviation of empirical coverage for each coverage level.
    """
    coverage = [0.5, 0.95]

    emp_coverages = {level: np.zeros(shape=len(predictions)) for level in coverage}

    coverage_scorer = EmpiricalCoverage(score_average=False, coverage=coverage)

    for i, (_, y_test, y_pred_quantiles) in predictions.iterrows():
        interval_df, _ = quantiles_to_interval_df(y_pred_quantiles)
        fold_i_coverage = coverage_scorer(y_test, interval_df).to_list()
        for level, cov in zip(coverage, fold_i_coverage):
            emp_coverages[level][i] = cov

    emp_coverage_mean = {level: np.mean(emp_coverages[level]) for level in coverage}
    emp_coverage_std = {level: np.std(emp_coverages[level]) for level in coverage}
    return emp_coverage_mean, emp_coverage_std
