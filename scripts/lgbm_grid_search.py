"""LGBM Grid Search Script."""

import itertools
import subprocess

import click


@click.command()
@click.option(
    "--auto-start",
    is_flag=True,
    help="Automatically start the queue after queuing all experiments.",
)
def run_lgbm_grid_search(auto_start: bool):
    """Run a small Grid Search on LightGBM."""
    # Define the parameter grid
    param_grid = {
        # 31 is the default, should be smaller than 2^max_depth
        "num_leaves": [6, 15, 31, 62, 90],
        "min_child_samples": [5, 25, 50],
        "max_depth": [-1],
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1],  # Fixed learning rate
        # "subsample_freq": [0],  # subsample every time
        # "subsample": [1.0],
        "colsample_bytree": [0.8],
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

    if auto_start:
        print("Starting the queue...")
        subprocess.run(["dvc", "queue", "start"], check=True)
    else:
        print("All experiments queued. Start the queue with `dvc queue start`.")
        print(f"Total number of experiments: {len(param_combinations)}")
        print("You can delete them with `dvc queue remove --queued`, if needed.")


if __name__ == "__main__":
    run_lgbm_grid_search()
