"""Pipeline for probafcst package.

This module contains the pipeline for the probafcst package. The pipeline
is a series of steps that are executed in order to generate forecasts.

The pipeline consists of the following steps:

1. Prepare: Data is downloaded and preprocessed.
2. Train: Models are trained on the data.
3. Evaluate: Models are evaluated on the data.
4. Submit: Forecasts are generated and saved to disk.

The pipeline is managed by DVC, which ensures that each step is executed
in the correct order and only when necessary. The pipeline is defined in
the `dvc.yaml` file in the root of the project.
"""
