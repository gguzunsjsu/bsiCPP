import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results_100_1M.csv")
sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
plt.figure(figsize=(8, 6))

plt.plot(
    sizes,
    df["bsi_avg_us"],
    linestyle="--",
    marker="o",
    label="Time for BSI operation",
    linewidth=1.5,
)

plt.plot(
    sizes,
    df["vec_avg_us"],
    linestyle="-.",
    marker="s",
    label="Time for vector operation",
    linewidth=1.5,
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of elements", fontsize=12)
plt.ylabel("Time taken (µs)", fontsize=12)
plt.title("Comparison of BSI Dot Product and Vector Dot Product", fontsize=14, pad=10)
plt.legend(frameon=False, fontsize=11)

plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

plt.show()
