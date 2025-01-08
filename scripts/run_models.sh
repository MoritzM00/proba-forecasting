#!/bin/bash

# Array of parameter values
models=("benchmark" "quantreg" "xgb-custom" "lgbm" "catboost")

# Nested loops to iterate over all combinations
for model in "${models[@]}"; do
  echo "Queueing experiment using model $model"
  dvc exp run --queue \
      --set-param "train.energy.selected=$model" \
      --set-param "train.bikes.selected=$model" \
      --message "$model
done

echo "All experiments queued. Run `dvc queue start` to start execution."
