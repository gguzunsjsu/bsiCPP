import pandas as pd
import matplotlib.pyplot as plt

# Load the results
df = pd.read_csv("results_1M.csv")

# Prepare plot data
skews = df["skew"].tolist()
bsi_times = df["bsi_avg_us"].tolist()
vec_times = df["vec_avg_us"].tolist()
x = list(range(len(skews)))
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# BSI bars with dot hatch
bars_bsi = ax.bar(
        [i - width/2 for i in x],
        bsi_times,
        width,
        label="Time for BSI operation",
        edgecolor="black",
        hatch=".",
        linewidth=0.5
)

# Vector bars
bars_vec = ax.bar(
        [i + width/2 for i in x],
        vec_times,
        width,
        label="Time for vector operation",
        edgecolor="black",
        linewidth=0.5
)

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels([f"{s:.1f}" for s in skews])
ax.set_xlabel("Zipf Parameter", fontsize=12)
ax.set_ylabel("Time taken (microseconds)", fontsize=12)
ax.set_title("Comparison of BSI Dot Product and Vector Dot Product", fontsize=14, pad=15)

# Grid lines
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)

# Legend
ax.legend(frameon=False, fontsize=11, loc="upper right")

# Tidy layout
plt.tight_layout()
plt.show()
