import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load and preprocess
file_path = "battery_data_1.csv"
df = pd.read_csv(file_path)
df.columns = [c.strip().lower() for c in df.columns]

time_col = "time [ms]"
i_col = "current [10^-2 a]"
cell_cols = [c for c in df.columns if c.startswith("cell voltage") and "[mv]" in c]

#Convert units
df["time_s"] = df[time_col] / 1000.0                           
df["dt_h"] = df[time_col].diff().fillna(0) / (1000 * 3600)     
df["current_a"] = df[i_col] * 1e-2                            
for c in cell_cols:
    df[c + "_v"] = df[c] / 1000.0                             
df["pack_voltage_v"] = df[[c + "_v" for c in cell_cols]].sum(axis=1)

#Capacity integration
i = df["current_a"].values
dt_h = df["dt_h"].values
i_mid = (i + np.r_[i[0], i[:-1]]) / 2
sign = np.sign(i_mid)

cap_charge_ah = np.sum(np.where(sign > 0, i_mid * dt_h, 0.0))
cap_discharge_ah = -np.sum(np.where(sign < 0, i_mid * dt_h, 0.0))
cap_charge_mah = cap_charge_ah * 1000
cap_discharge_mah = cap_discharge_ah * 1000

#Energy integration
v = df["pack_voltage_v"].values
p_mid = i_mid * v
energy_in_wh = np.sum(np.where(sign > 0, p_mid * dt_h, 0.0))
energy_out_wh = -np.sum(np.where(sign < 0, p_mid * dt_h, 0.0))

#Efficiencies
coulombic_eff = (cap_discharge_mah / cap_charge_mah) * 100 if cap_charge_mah > 0 else np.nan
energy_eff = (energy_out_wh / energy_in_wh) * 100 if energy_in_wh > 0 else np.nan

#Current limits
imax_a = df["current_a"].max()
imin_a = df["current_a"].min()

#Voltage stats
pack_v_min = df["pack_voltage_v"].min()
pack_v_max = df["pack_voltage_v"].max()

#Print summary
print("\n--- Battery Specifications ---")
print(f"Series cells: {len(cell_cols)}")
print(f"Charge capacity: {cap_charge_mah:.2f} mAh")
print(f"Discharge capacity: {cap_discharge_mah:.2f} mAh")
print(f"Coulombic efficiency: {coulombic_eff:.2f} %")
print(f"Energy in: {energy_in_wh:.3f} Wh")
print(f"Energy out: {energy_out_wh:.3f} Wh")
print(f"Energy efficiency: {energy_eff:.2f} %")
print(f"Max charge current: {imax_a*1000:.1f} mA")
print(f"Max discharge current: {-imin_a*1000:.1f} mA")
print(f"Pack voltage min: {pack_v_min:.3f} V")
print(f"Pack voltage max: {pack_v_max:.3f} V")

#Plot: Current vs Time
plt.figure()
plt.plot(df["time_s"], df["current_a"] * 1000.0)
plt.xlabel("Time (s)")
plt.ylabel("Current (mA)")
plt.title("Current vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("current_vs_time.png", dpi=150)
plt.close()

#Plot: Pack Voltage vs Time
plt.figure()
plt.plot(df["time_s"], df["pack_voltage_v"])
plt.xlabel("Time (s)")
plt.ylabel("Pack Voltage (V)")
plt.title("Pack Voltage vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("pack_voltage_vs_time.png", dpi=150)
plt.close()

print("\nPlots saved as:")
print("  - current_vs_time.png")
print("  - pack_voltage_vs_time.png")
