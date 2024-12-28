# ruff: noqa
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests
from tqdm import tqdm

from datetime import datetime, date, timedelta


def get_energy_data():
    # get all available time stamps
    stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
    response = requests.get(stampsurl)
    # ignore first 7 years (don't need those in the baseline and speeds the code up a bit)
    timestamps = list(response.json()["timestamps"])[7 * 52 :]

    col_names = ["date_time", "Netzlast_Gesamt"]
    energydata = pd.DataFrame(columns=col_names)

    # loop over all available timestamps
    for stamp in tqdm(timestamps):
        dataurl = (
            "https://www.smard.de/app/chart_data/410/DE/410_DE_hour_"
            + str(stamp)
            + ".json"
        )
        response = requests.get(dataurl)
        rawdata = response.json()["series"]
        for i in range(len(rawdata)):
            rawdata[i][0] = datetime.fromtimestamp(
                int(str(rawdata[i][0])[:10])
            ).strftime("%Y-%m-%d %H:%M:%S")

        energydata = pd.concat([energydata, pd.DataFrame(rawdata, columns=col_names)])

    energydata = energydata.dropna()
    energydata["date_time"] = pd.to_datetime(energydata.date_time)
    # set date_time as index
    energydata.set_index("date_time", inplace=True)

    return energydata


# %%
df = get_energy_data()
df = df.rename(columns={"Netzlast_Gesamt": "gesamt"})
df["gesamt"] = df["gesamt"] / 1000
df.tail()
# %%
# not really necessary for latest data. useful for predictions of earlier weeks
weeks_back = 0
end_date = (date.today() - timedelta(days=weeks_back * 7 + 1)).strftime("%Y-%m-%d")
end_date
# %%
df = df[df.index < end_date]
df.tail()
# %%
# Hours from 0-23 (not 1-24!)
CONSIDERED_HOURS = [12, 16, 20]
CONSIDERED_DAYS = [4, 5]  # Friday and Saturday

df["weekday"] = df.index.weekday  # Monday=0, Sunday=6
df["hour"] = df.index.hour

df = df[df.hour.isin(CONSIDERED_HOURS)]
df = df[df.weekday.isin(CONSIDERED_DAYS)]
# %%
# Lead times are
# horizons_def = [36, 40, 44, 60, 64, 68]
# horizons = [h + 1 for h in horizons_def]
# %%
# quantile levels
tau = [0.025, 0.25, 0.5, 0.75, 0.975]

# rows correspond to horizon, columns to quantile level
pred_baseline = np.zeros((6, 5))

last_t = 100

i = 0
for day in CONSIDERED_DAYS:
    for hour in CONSIDERED_HOURS:
        cond = (df.weekday == day) & (df.hour == hour)

        pred_baseline[i, :] = np.quantile(df[cond].iloc[-last_t:]["gesamt"], q=tau)

        i += 1

pred_baseline
# %%
horizons = [36, 40, 44, 60, 64, 68]
_ = plt.plot(horizons, pred_baseline, ls="", marker="o", c="black")
_ = plt.xticks(horizons, horizons)
_ = plt.plot(
    (horizons, horizons), (pred_baseline[:, 0], pred_baseline[:, -1]), c="black"
)

# %%
df_sub = pd.DataFrame(
    {
        "forecast_date": end_date,
        "target": "energy",
        "horizon": [str(h) + " hour" for h in horizons],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4],
    }
)
df_sub

# %%
# need to change this
PATH = "/save/to/path"

df_sub.to_csv(PATH + end_date.replace("-", "") + "_power_baseline.csv", index=False)
# %%
