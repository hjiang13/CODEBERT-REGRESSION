import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "BenchMark": ["IS", "LU", "Nn", "Myocyte", "Backprop", "CG", "MG", "BT", "SP", "DC", "Lud", "Kmeans", "STREAM", "Blackscholes", "PuReMD", "Lulesh", "Hotspot", "Bfs-rodinia", "Nw", "Pathfinder"],
    "benign MSE": [0.009604, 0.001444, 0.032761, 0.081225, 0.031329, 0.073984, 0.113569, 0.0064, 0.2209, 0.0004, 0.105625, 0.251001, 0.048841, 0.013225, 0.281961, 0.003844, 0.042436, 0.008281, 0.015129, 0.0016],
    "crash MSE": [0.0081, 0.013924, 0.000361, 0.022201, 0.000625, 0.159201, 0.042025, 0.075076, 0.000036, 0.016129, 0.200704, 0.173056, 0.042436, 0.039204, 0.299209, 0.0169, 0.093636, 0.034969, 0.055696, 0.118336],
    "SDC MSE": [0.000361, 0.095481, 0.103041, 0.0361, 0.056644, 0.0441, 0.047089, 0.092416, 0.609961, 0.007056, 0.044944, 0.013924, 0.000729, 0.023409, 0.002601, 0.021609, 0.024649, 0.027225, 0.0441, 0.265225]
}

# Create DataFrame and sort by SDC MSE
df = pd.DataFrame(data)
df_sorted = df.sort_values("SDC MSE")

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_sorted["BenchMark"], df_sorted["benign MSE"], label="Benign MSE", marker='v',color='blue')
ax.plot(df_sorted["BenchMark"], df_sorted["crash MSE"], label="Crash MSE", marker='o',color='orange')
ax.plot(df_sorted["BenchMark"], df_sorted["SDC MSE"], label="SDC MSE", marker='s',color='green')

# Calculate and plot averages for each MSE category
avg_benign_mse = df["benign MSE"].mean()
avg_crash_mse = df["crash MSE"].mean()
avg_sdc_mse = df["SDC MSE"].mean()

ax.axhline(y=avg_benign_mse, color='blue', linestyle='--', label=f"Average Benign MSE: {avg_benign_mse:.4f}")
ax.axhline(y=avg_crash_mse, color='orange', linestyle='--', label=f"Average Crash MSE: {avg_crash_mse:.4f}")
ax.axhline(y=avg_sdc_mse, color='green', linestyle='--', label=f"Average SDC MSE: {avg_sdc_mse:.4f}")

# Table data
table_data=[
    ["Benign MSE", "1.000", "0.605", "0.258"],
    ["Crash MSE", "0.605", "1.000", "-0.194"],
    ["SDC MSE", "0.258", "-0.194", "1.000"]
]

# Add table to the plot
table = ax.table(cellText=table_data, colLabels=["CC", "Benign MSE", "Crash MSE", "SDC MSE"], loc='upper center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(0.5, 1.5)  # Adjust scale to match the plot aesthetics

# Additional plot settings
ax.set_xlabel("Benchmark")
ax.set_ylabel("Mean Squared Error (MSE)")
#ax.set_title("MSE Comparison Across Benchmarks")
ax.legend(loc='upper left')
plt.xticks(rotation=45)
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, wspace=0, hspace=0)
plt.show()
# Save the plot as PDF
plt.savefig("LSTM_all_line.png")