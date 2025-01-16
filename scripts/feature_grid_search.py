"""Grid Search over Feature Engineering Parameters."""

import itertools
import subprocess

import click


@click.command()
@click.argument(
    "model", type=click.Choice(["lgbm", "xgb-custom", "quantreg", "catboost"])
)
def run_grid_search(model):
    """Run Grid Search over Feature Engineering Parameters."""
    # Define the parameter grid
    param_grid = {
        "include_seasonal_dummies": [True, False],
        "cyclical_encodings": [True, False],  # Note: Must match conditional logic
        "X_lag_cols": [None, []],
        "include_rolling_stats": [True, False],
    }

    # Generate all combinations of parameters using itertools.product
    param_combinations = list(
        itertools.product(
            param_grid["include_seasonal_dummies"],
            param_grid["cyclical_encodings"],
            param_grid["X_lag_cols"],
            param_grid["include_rolling_stats"],
        )
    )

    # Filter combinations to enforce conditional logic:
    # cyclical_encodings must be False if include_seasonal_dummies is False
    filtered_combinations = [
        params
        for params in param_combinations
        if not (params[0] is False and params[1] is True)
    ]

    # Iterate through valid combinations and queue experiments
    for params in filtered_combinations:
        (
            include_seasonal_dummies,
            cyclical_encodings,
            X_lag_cols,
            include_rolling_stats,
        ) = params
        # Convert None to YAML-compatible null
        X_lag_cols_value = "null" if X_lag_cols is None else []

        message = f"Grid Search for model: {model}"

        # Construct the DVC command
        command = [
            "dvc",
            "exp",
            "run",
            "--queue",
            f"--set-param=train.bikes.selected={model}",
            f"--set-param=train.energy.selected={model}",
            f"--set-param=train.bikes.{model}.include_seasonal_dummies={str(include_seasonal_dummies).lower()}",
            f"--set-param=train.bikes.{model}.cyclical_encodings={str(cyclical_encodings).lower()}",
            f"--set-param=train.bikes.{model}.X_lag_cols={X_lag_cols_value}",
            f"--set-param=train.bikes.{model}.include_rolling_stats={str(include_rolling_stats).lower()}",
            f"--set-param=train.energy.{model}.include_seasonal_dummies={str(include_seasonal_dummies).lower()}",
            f"--set-param=train.energy.{model}.cyclical_encodings={str(cyclical_encodings).lower()}",
            f"--set-param=train.energy.{model}.X_lag_cols={X_lag_cols_value}",
            f"--set-param=train.energy.{model}.include_rolling_stats={str(include_rolling_stats).lower()}",
            "--message",
            message,
        ]

        print(f"Queueing experiment with params: {message}")

        # Execute the command
        subprocess.run(command, check=True)

    print("All experiments queued. Start the queue with `dvc queue start`.")
    print(f"Total number of experiments: {len(filtered_combinations)}")
    print("You can delete them with `dvc queue remove --queued`, if needed.")


if __name__ == "__main__":
    run_grid_search()
