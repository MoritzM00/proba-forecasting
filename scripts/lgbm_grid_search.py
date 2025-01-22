"""LGBM Grid Search Script."""

import itertools
import subprocess

import click


@click.command()
def run_lgbm_grid_search():
    """Run a small Grid Search on LightGBM."""
    # Define the parameter grid
    param_grid = {
        # 31 is the default, should be smaller than 2^max_depth
        "num_leaves": [15, 31, 150],
        "max_depth": [-1, 4, 8],
        "n_estimators": [100, 250, 500, 750],
        "learning_rate": [0.1],  # Fixed learning rate
        "subsample_freq": [1.0],  # subsample every time
        "subsample": [0.8, 1.0],
        "reg_lambda": [1e-3],
    }

    # Generate all combinations of parameters using itertools.product
    param_combinations = list(itertools.product(*param_grid.values()))

    click.confirm(
        f"Queue {len(param_combinations)} experiments?",
        abort=True,
    )

    # Iterate through all combinations and queue experiments
    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))

        # set num_leaves to min(2^max_depth, num_leaves)
        if param_dict["max_depth"] != -1:
            param_dict["num_leaves"] = min(
                2 ** param_dict["max_depth"], param_dict["num_leaves"]
            )

        model = "lgbm"
        # Construct the DVC command dynamically
        command = [
            "dvc",
            "exp",
            "run",
            "--queue",
            f"--set-param=train.bikes.selected={model}",
            f"--set-param=train.energy.selected={model}",
        ]

        for dataset in ["bikes", "energy"]:
            for key, value in param_dict.items():
                command.append(
                    f"--set-param=train.{dataset}.{model}.kwargs.{key}={value}"
                )

        # Add a descriptive message
        message = ", ".join([f"{key}={value}" for key, value in param_dict.items()])
        command.extend(["--message", f"LGBM Grid Search: {message}"])

        # Execute the command
        subprocess.run(command, check=True)

    print("All experiments queued. Start the queue with `dvc queue start`.")
    print(f"Total number of experiments: {len(param_combinations)}")
    print("You can delete them with `dvc queue remove --queued`, if needed.")


if __name__ == "__main__":
    run_lgbm_grid_search()
