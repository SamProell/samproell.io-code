import pathlib
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns

# READING TEMPERATURE DATA
temps = pd.read_csv("data/temperatures.csv")
temps.timestamp = pd.to_datetime(temps.timestamp)
temps.set_index("timestamp", inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.4))
temps.plot(y=["temp1", "temp2"], ax=ax, x_compat=True)
plt.xlabel("Time")
plt.ylabel("Body temp. (°C)")
plt.tight_layout()

# import scipy.stats
lr = scipy.stats.linregress(temps.temp1, temps.temp2)
print(f"R² = {lr.rvalue**2}")
print("R**2 =", temps.corr().loc["temp1", "temp2"]**2)
temps.eval("temp = (temp1 + temp2) / 2", inplace=True)

# READING HEART RATE DATA
# import sqlite3, pathlib
monitoring_db = pathlib.Path("~/HealthData/DBs/garmin_monitoring.db")
con = sqlite3.connect(monitoring_db.expanduser())
hr_df = (
    pd.read_sql("SELECT * FROM monitoring_hr", con, parse_dates=["timestamp"])
    .set_index("timestamp")
)
con.close()
print(hr_df.head())

# DATA WRANGLING
start, end = temps.index.min(), temps.index.max()
hrs = hr_df.query("@start < index < @end").sort_index()
hrs.rename(columns={"heart_rate": "hr"}, inplace=True)
hrs["hr_smooth"] = hrs.rolling("30min", min_periods=15, center=True).mean()

fig, ax = plt.subplots(1, 1)
twinx = ax.twinx()
hrs.plot(y="hr_smooth", ax=ax, legend=False)
temps.plot(y="temp", ax=twinx, color="C1", legend=False)
plt.figlegend()
ax.set_xlabel("Time")
ax.set_ylabel("Heart rate (BPM)")
twinx.set_ylabel("Temperature (°C)")

aux_df = pd.merge(temps, hrs, left_index=True, right_index=True, how="outer")
res_df = aux_df.resample("30min").agg(["mean", "std"]).dropna(how="all")
res_df.columns = [f"{col}_{feature}" for col, feature in res_df.columns]

# CORRELATION ANALYSIS
plt.figure()
sns.scatterplot(
    data=res_df, x="temp_mean", y="hr_mean", hue="hr_std", palette="mako_r"
)
# sns.regplot(data=res_df, x="temp_mean", y="hr_mean", scatter=False)
plt.tight_layout()

std_thresh = res_df.hr_std.quantile(0.85)
print("hr_std threshold:", std_thresh)
for df in [res_df, res_df.query("hr_std < @std_thresh")]:
    lr = scipy.stats.linregress(df[["temp_mean", "hr_mean"]].dropna(how="any"))
    print(f"R² = {lr.rvalue**2:.3f} | p = {lr.pvalue:.4g}"
          f" hr = {lr.slope:.1f} temp + {lr.intercept:.1f}")

import statsmodels.formula.api as smf
model = smf.ols(
    "hr_mean ~ temp_mean", data=res_df.query("hr_std < @std_thresh")
)
res = model.fit()
print(res.summary())

# FINAL PLOT
std_thresh = res_df.hr_std.quantile(0.85)
df = res_df.query("hr_std < @std_thresh")
lr = scipy.stats.linregress(df[["temp_mean", "hr_mean"]].dropna())

plt.figure(figsize=(6.4, 3.2))
sns.regplot(
    data=df, x="temp_mean", y="hr_mean", line_kws={"color": "C0"}, color=".6"
)
plt.xlabel("Body temperature (°C)")
plt.ylabel("Heart rate (bpm)")
plt.text(0, 1.02, f"HR = {lr.slope:.1f} \u00B7 Temp {lr.intercept:=+7.1f}",
         va="bottom", clip_on=False, transform=plt.gca().transAxes, color=".5")
plt.text(1, 1.02, f"R² = {lr.rvalue**2:.3f} | p = {lr.pvalue:.4g}",
         va="bottom", ha="right", clip_on=False, transform=plt.gca().transAxes,
         color=".5")
plt.tight_layout()

plt.show() # show all figures at once.
