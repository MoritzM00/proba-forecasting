#!/bin/bash

# Array of parameter values
n_weeks_values=(10 25 50 75 100 125 150 200 250 300)

# Nested loops to iterate over all combinations
for n_weeks in "${n_weeks_values[@]}"; do
        echo "Queueing experiment with n_weeks $n_weeks"
        dvc exp run --queue \
            --set-param "train.energy.benchmark.n_weeks=$n_weeks" \
            --set-param "train.bikes.benchmark.n_weeks=$n_weeks"
done

source .venv/bin/activate
dvc queue start -v
