# Probabilistic Timeseries Forecasting Challenge

![Tests](https://img.shields.io/github/actions/workflow/status/MoritzM00/proba-forecasting/test_deploy.yaml?style=for-the-badge&label=Test%20and%20Deploy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge)][pre-commit]
![License](https://img.shields.io/github/license/MoritzM00/proba-forecasting?style=for-the-badge)

[pre-commit]: https://github.com/pre-commit/pre-commit

Repository for the Probabilistic Timeseries Forecasting Challenge. This challenge focuses on quantile forecasting for timeseries data. Results can be found [here](https://gitlab.kit.edu/nils.koster/ptsfc24_results).

## Development Guide

This guide shows how to reproduce the results of the challenge.

### Set up the environment

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Set up the environment:

```bash
make setup
source .venv/bin/activate
```

### Reproduce the results

Run `dvc repro` to reproduce the results. This requires the data to be in the correct format.

#### Data Requirements

TODO

### Documentation

The Documentation is automatically deployed to GitHub Pages.

To view the documentation locally, run:

```bash
make docs_view
```

## Credits

This project was generated with the [Light-weight Python Template](https://github.com/MoritzM00/python-template) by Moritz Mistol.
