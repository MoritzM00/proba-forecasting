# %%
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# %%
start_date = "01/01/2019"

dataurl = (
    "https://www.eco-visio.net/api/aladdin/1.0.0/pbl/publicwebpageplus/data/"
    f"100126474?idOrganisme=4586&idPdc=100126474&interval=4&flowIds=100126474&debut={start_date}"
)
response = requests.get(dataurl)
rawdata = response.json()

# %%

df = pd.DataFrame(rawdata, columns=["date", "bike_count"])
df["bike_count"] = df["bike_count"].astype(float)
df = df.set_index(pd.to_datetime(df["date"]))
df.drop(columns=["date"], inplace=True)

df.head()
# %%
# Plot bikes time series
plt.plot(df["bike_count"])

# %%
df = df.dropna(axis=0)
df.isna().any()

# %%
# Define weekday column
df["weekday"] = df.index.weekday  # Monday=0, Sunday=6

# %%
# Lead times are
horizons = [1, 2, 3, 4, 5, 6]
horizons


# %%
def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(days=horizon)


# %%
LAST_IDX = -1
LAST_DATE = df.iloc[LAST_IDX].name

# %%
LAST_DATE

# %%
# Get time and date that correspond to the lead times (starting at the last observation in our data which should be the respective thursday 0:00)
# *Attention*: if the last timestamp in the data is not thursday 0:00, you have to adjust your lead times accordingly

horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]
horizon_date

# %%
# quantile levels

tau = [0.025, 0.25, 0.5, 0.75, 0.975]

# %%
# rows correspond to horizon, columns to quantile level
pred_baseline = np.zeros((6, 5))

last_t = 100

for i, d in enumerate(horizon_date):
    weekday = d.weekday()

    df_tmp = df.iloc[:LAST_IDX]

    cond = df_tmp.weekday == weekday

    pred_baseline[i, :] = np.quantile(df_tmp[cond].iloc[-last_t:]["bike_count"], q=tau)

# %%
pred_baseline

# %%
# Visually check if quantiles make sense

x = horizons
_ = plt.plot(x, pred_baseline, ls="", marker="o", c="black")
_ = plt.xticks(x, x)
_ = plt.plot((x, x), (pred_baseline[:, 0], pred_baseline[:, -1]), c="black")

# %%
from datetime import datetime, timedelta

date_str = datetime.today().strftime("%Y%m%d")

date_str = date.today() - timedelta(days=1)
date_str = date_str.strftime("%Y-%m-%d")
date_str

# %%
df_sub = pd.DataFrame(
    {
        "forecast_date": date_str,
        "target": "bikes",
        "horizon": [str(d) + " day" for d in horizons],
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

df_sub.to_csv(PATH + date_str + "_bikes_baseline.csv", index=False)
