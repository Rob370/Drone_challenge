import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load and preprocess
file_path = "battery_data_1.csv"
df = pd.read_csv(file_path)
df.columns = [c.strip().lower() for c in df.columns]

time_col = "time [ms]"
i_col = "current [10^-2 a]"
cycle_col = "cycles number [-]"
cell_cols = [c for c in df.columns if c.startswith("cell voltage") and "[mv]" in c]

#Convert units
df["time_s"] = df[time_col] / 1000.0
df["current_a"] = df[i_col] * 1e-2
for c in cell_cols:
    df[c + "_v"] = df[c] / 1000.0
df["pack_voltage_v"] = df[[c + "_v" for c in cell_cols]].sum(axis=1)

#Sort by cycle and time
df = df.sort_values([cycle_col, "time_s"]).reset_index(drop=True)

#Compute per-sample time delta in hours
df["dt_h"] = df.groupby(cycle_col)["time_s"].diff().fillna(0) / 3600.0

#Energy integration per cycle
df["power_w"] = df["pack_voltage_v"] * df["current_a"]
df["dE_wh"] = df["power_w"] * df["dt_h"]

cycles = df.groupby(cycle_col).agg(
    energy_wh=("dE_wh", "sum"),
    duration_h=("dt_h", "sum"),
    i_mean=("current_a", "mean"),
    v_mean=("pack_voltage_v", "mean"),
    n_samples=("dE_wh", "size")
).reset_index().rename(columns={cycle_col: "cycle"})

#Detect abnormalities
def mad_based_flags(values, thresh=3.5):
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    else:
        z = 0.6745 * (x - med) / (mad + 1e-12)
    return np.abs(z) > thresh

cycles["abnormal"] = mad_based_flags(cycles["energy_wh"], thresh=3.5)

#Trend analysis
mask = cycles["energy_wh"].notna()
slope_wh_per_cycle = np.polyfit(
    cycles.loc[mask, "cycle"], cycles.loc[mask, "energy_wh"], 1
)[0]

#Summary statistics
abn = cycles["abnormal"].fillna(False)
normal = cycles.loc[~abn]
summary = {
    "n_cycles_total": int(len(cycles)),
    "n_cycles_abnormal": int(abn.sum()),
    "abnormal_cycles": cycles.loc[abn, "cycle"].tolist(),
    "energy_wh_mean_all": float(cycles["energy_wh"].mean()),
    "energy_wh_median_all": float(cycles["energy_wh"].median()),
    "energy_wh_min_all": float(cycles["energy_wh"].min()),
    "energy_wh_max_all": float(cycles["energy_wh"].max()),
    "energy_wh_mean_normal": float(normal["energy_wh"].mean()),
    "energy_wh_median_normal": float(normal["energy_wh"].median()),
    "energy_wh_min_normal": float(normal["energy_wh"].min()),
    "energy_wh_max_normal": float(normal["energy_wh"].max()),
    "slope_wh_per_cycle": float(slope_wh_per_cycle)
}

print("\n--- Cycle Energy Analysis ---")
for k, v in summary.items():
    print(f"{k}: {v}")

#Plots
plt.figure()
plt.plot(cycles["cycle"], cycles["energy_wh"], marker="o", label="Energy per cycle")
plt.scatter(
    cycles.loc[abn, "cycle"], cycles.loc[abn, "energy_wh"],
    color="red", s=80, label="Abnormal"
)
plt.xlabel("Cycle")
plt.ylabel("Energy (Wh)")
plt.title("Battery Energy per Cycle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("energy_per_cycle.png", dpi=150)
plt.close()

plt.figure()
plt.plot(cycles["cycle"], cycles["i_mean"], marker="o")
plt.xlabel("Cycle")
plt.ylabel("Mean Current (A)")
plt.title("Mean Current per Cycle")
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_current_per_cycle.png", dpi=150)
plt.close()

plt.figure()
plt.plot(cycles["cycle"], cycles["v_mean"], marker="o")
plt.xlabel("Cycle")
plt.ylabel("Mean Voltage (V)")
plt.title("Mean Voltage per Cycle")
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_voltage_per_cycle.png", dpi=150)
plt.close()

print("\nPlots saved as:")
print("  - energy_per_cycle.png")
print("  - mean_current_per_cycle.png")
print("  - mean_voltage_per_cycle.png")
