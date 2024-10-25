# %%
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
start_date = "2020-01-01"
end_date = date.today()

url = (
    f"https://www.umweltbundesamt.de/api/air_data/v3/measures/csv?date_from={start_date}"
    f"&time_from=24&date_to={end_date}"
    f"&time_to=23&data%5B0%5D%5Bco%5D=5&data%5B0%5D%5Bsc%5D=2&data%5B0%5D%5Bda%5D={end_date}"
    "&data%5B0%5D%5Bti%5D=12&data%5B0%5D%5Bst%5D=282&data%5B0%5D%5Bva%5D=27&lang=en"
)

df = pd.read_csv(url, sep=";")
df = df.drop(df.index[-1])
df["Measure value"] = df["Measure value"].replace("-", np.nan)
df["Measure value"] = df["Measure value"].astype(float)

df.head()

# %%
# Extract hour from (string) Time column and get it in correct format

df["hour"] = df["Time"].str[1:3].astype(int)
df["hour"] = df["hour"].replace(24, 0)

# %%
# Create datetime column and set it as index
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["hour"].astype(str) + ":00")
df = df.set_index("datetime")

# %%
# Plot NO2 time series
plt.plot(df["Measure value"])

# %%
df.isna().any()

# %%
df[df["Measure value"].isna()].shape[0]

# %%
df = df.dropna(axis=0)

# %%
df[df["Measure value"].isna()].shape[0]

# %%
# Define weekday column
df["weekday"] = df.index.weekday  # Monday=0, Sunday=6

# %%
# Lead times are
horizons_def = [36, 40, 44, 60, 64, 68]
horizons_def

# %%
# Adapt horzions so they actually fit
horizons = [h + 1 for h in horizons_def]
horizons


# %%
def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=24 + horizon)


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
    hour = d.hour

    df_tmp = df.iloc[:LAST_IDX]

    cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())

    pred_baseline[i, :] = np.quantile(
        df_tmp[cond].iloc[-last_t:]["Measure value"], q=tau
    )

# %%
pred_baseline

# %%
# Visually check if quantiles make sense

x = horizons
_ = plt.plot(x, pred_baseline, ls="", marker="o", c="black")
_ = plt.xticks(x, x)
_ = plt.plot((x, x), (pred_baseline[:, 0], pred_baseline[:, -1]), c="black")

# %%
from datetime import date, datetime, timedelta

date_str = datetime.today().strftime("%Y%m%d")

date_str = date.today() - timedelta(days=1)
date_str = date_str.strftime("%Y-%m-%d")
date_str

# %%
df_sub = pd.DataFrame(
    {
        "forecast_date": date_str,
        "target": "no2",
        "horizon": [str(h) + " hour" for h in horizons_def],
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

df_sub.to_csv(PATH + date_str + "_no2_baseline.csv", index=False)
