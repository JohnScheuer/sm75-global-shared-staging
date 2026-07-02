import pandas as pd
import matplotlib.pyplot as plt

# Carrega CSV
df = pd.read_csv("results.csv")

# Cria coluna combinada
df["CONFIG"] = df["MODE"] + "_" + df["BARRIER"]

plt.figure(figsize=(10,6))

for config in df["CONFIG"].unique():
    subset = df[df["CONFIG"] == config]
    plt.plot(subset["BLOCK_SIZE"],
             subset["GB/s"],
             marker='o',
             label=config)

plt.xlabel("Block Size")
plt.ylabel("Bandwidth (GB/s)")
plt.title("SM75 Global→Shared Staging Sweep")
plt.axhline(448, linestyle='--', label='Theoretical Peak')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("bandwidth_plot.png")
plt.show()

plt.annotate("vec4 best\n(350 GB/s)",
             (kernel_ai, kernel_perf),
             textcoords="offset points",
             xytext=(10,10))