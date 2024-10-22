"""Energy Analysis Notebook."""

import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    from darts import TimeSeries

    return TimeSeries, pd


@app.cell
def __(pd):
    # read excel file
    data = pd.read_excel(
        "data/raw/energy_load.xlsx",
        header=9,
        sheet_name="Realisierter Stromverbrauch",
        thousands=",",
        decimal=".",
        na_values="-",
    )
    return (data,)


@app.cell
def __(data):
    data.head()
    return


@app.cell
def __(data, pd):
    load = data.loc[:, ["Datum von", "Gesamt (Netzlast) [MWh]"]].dropna()
    load = load.rename(columns={"Datum von": "date", "Gesamt (Netzlast) [MWh]": "load"})

    # parse the date column with german format: 01.01.2020 00:00
    load["date"] = pd.to_datetime(load["date"], format="%d.%m.%Y %H:%M")

    # parse the DST information
    # https://www.smard.de/en/how-changing-the-clock-is-processed-in-smard-data-211236
    load["date"] = load["date"].dt.tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"
    )
    load["date"] = load["date"].dt.tz_convert("UTC")
    load = load.set_index("date")
    load.head()
    return (load,)


@app.cell
def __(load):
    load.index.duplicated().sum()
    return


@app.cell
def __(TimeSeries, load):
    ts = TimeSeries.from_dataframe(
        load, value_cols="load", fill_missing_dates=True, freq="h"
    )
    return (ts,)


@app.cell
def __(load):
    load.to_parquet("data/processed/energy_load.parquet", index=True)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
