"""Utility functions for handling paths."""

from pathlib import Path


def get_data_path(data_dir: str, target: str = "energy") -> Path:
    """Get path to data file.

    Parameters
    ----------
    target : str
        The target data to get the path for.

    Returns
    -------
    str
        Path to the data file.
    """
    return Path(data_dir) / f"{target}.parquet"


def get_model_path(model_dir: str, target: str = "energy") -> Path:
    """Get path to model file.

    Parameters
    ----------
    target : str
        The target data to get the path for.

    Returns
    -------
    str
        Path to the model file.
    """
    return Path(model_dir) / f"{target}_model.pkl"
