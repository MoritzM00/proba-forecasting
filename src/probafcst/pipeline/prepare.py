"""Data Preparation Stage."""

from datetime import timedelta
from pathlib import Path
from warnings import simplefilter

import click
import dvc.api
import requests_cache
from omegaconf import OmegaConf

from probafcst.data import get_bikes_data, get_energy_data, get_no2_data

simplefilter("ignore", category=FutureWarning)


@click.command(name="prepare")
@click.option(
    "--target",
    "-t",
    default="energy",
    type=click.Choice(["energy", "bikes", "no2"]),
    help="The target data to prepare.",
)
def prepare(target: str) -> None:
    """Prepare data for the models."""
    params = dvc.api.params_show()
    params = OmegaConf.create(params)

    if params.cache.enable and not requests_cache.is_installed():
        days_expire_after = params.cache.days_expire_after
        requests_cache.install_cache(
            "probafcst_cache",
            expire_after=timedelta(days=days_expire_after),
        )

    data_dir = Path(params.data_dir)
    data_dir.mkdir(exist_ok=True)

    match target:
        case "energy":
            energy_data = get_energy_data(ignore_years=params.data.energy.ignore_years)
            energy_data.to_parquet(data_dir / "energy.parquet")
        case "bikes":
            bikes_data = get_bikes_data(start_date=params.data.bikes.start_date)
            bikes_data.to_parquet(data_dir / "bikes.parquet")
        case "no2":
            no2_data = get_no2_data(
                start_date=params.data.no2.start_date, end_date=params.data.no2.end_date
            )
            no2_data.to_parquet(data_dir / "no2.parquet")
        case _:
            raise ValueError("Invalid target.")


if __name__ == "__main__":
    prepare()
