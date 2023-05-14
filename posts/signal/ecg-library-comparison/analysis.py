# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
results = pd.read_csv("engzee.lenovo.csv")
for w in ["0.05", "0.1", "0.15"]:
    results[f"fm{w}"] = np.sqrt(results.eval(f"`precision{w}` * `recall{w}`"))
# %%
order = results.sort_values("exec_dur").method.unique()
# %%
fig = plt.figure()
sns.boxplot(data=results, y="method", x="exec_dur", order=order)
# plt.xticks(rotation=45, ha="right")
plt.xscale("log")
plt.xlabel("Speed (processed hours per second)")
plt.ylabel("")
fig.savefig("processing-time.png", dpi=120, bbox_inches="tight")
times = results.groupby("method")["exec_dur"].mean()
pd.merge(times.max() / times.sort_values(), times, left_index=True, right_index=True)
# %%
fig = plt.figure()
sns.boxplot(data=results, y="method", x="recall0.1", order=order)
plt.ylabel("")
plt.xlabel("Recall | Sensitivity @ 0.1 seconds")
# plt.xticks(rotation=45, ha="right")
plt.xlim(0.982, 1.001)
fig.savefig("recall0.1.png", dpi=120, bbox_inches="tight")
results.groupby("method")["recall0.1"].mean().sort_values()
# %%
fig = plt.figure()
sns.boxplot(data=results, y="method", x="precision0.1", order=order)
plt.ylabel("")
plt.xlabel("Precision | Positive predictivity @ 0.1 seconds")
# plt.xticks(rotation=45, ha="right")
plt.xlim(0.93, 1.001)
fig.savefig("precision0.1.png", dpi=120, bbox_inches="tight")
results.groupby("method")["precision0.1"].mean().sort_values()
# %%
