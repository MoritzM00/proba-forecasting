"""Some base functions for pipeline stages."""

import os
import sys

import dvc.api
from loguru import logger
from omegaconf import OmegaConf


def pipeline_setup(log_level: str = "INFO") -> None:
    """Run setup for the pipeline."""
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                "level": log_level,
                "colorize": True,
            }
        ],
    )
    # because of joblib parallellism (loky), need to set this environment variable
    # instead of using warnings.filterwarnings because subprocesses don't inherit from them
    os.environ["PYTHONWARNINGS"] = "ignore"

    params = dvc.api.params_show()
    params = OmegaConf.create(params)
    return params
