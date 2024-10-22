# Probabilistic Timeseries Forecasting Challenge

![Tests](https://img.shields.io/github/actions/workflow/status/MoritzM00/proba-forecasting/test_deploy.yaml?style=for-the-badge&label=Test%20and%20Deploy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge)][pre-commit]
![License](https://img.shields.io/github/license/MoritzM00/Probabilistic Timeseries Forecasting Challenge?style=for-the-badge)

[pre-commit]: https://github.com/pre-commit/pre-commit

---

## Quick Start

Below you can find the quick start guide for development.

### Set up the environment

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Set up the environment:

```bash
make setup
source .venv/bin/activate
```

### Additional first-time setup

1. After setting up the environment, commit the `uv.lock` file to your repository, so that the workflow on github can use it.
2. Enable [Pre-Commit CI](https://pre-commit.ci/) for your repository.
3. Enable **Github Pages** for your documentation.
   To do that, go to the _Settings_ tab of your repository and scroll down to the _GitHub Pages_ section.
   For the _Source_ option, select _GitHub Action_. Done!

### Install new packages

To install new PyPI packages, run:

```bash
uv add <package-name>
```

To add dev-dependencies, run:

```bash
uv add --dev <package-name>
```

### Documentation

The Documentation is automatically deployed to GitHub Pages.

To view the documentation locally, run:

```bash
make docs_view
```

## Credits

This project was generated with the [Light-weight Python Template](https://github.com/MoritzM00/python-template) by Moritz Mistol.
