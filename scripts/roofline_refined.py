import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Hardware Specs
# ==========================================================

# SM75 (RTX 2070)
bw_sm75 = 448      # GB/s
fp32_sm75 = 7500   # GFLOPs

# SM80 (A100 approx)
bw_sm80 = 1555     # GB/s
fp32_sm80 = 19500  # GFLOPs

# ==========================================================
# Load Benchmark Data
# ==========================================================

df = pd.read_csv("results.csv")

# Arithmetic Intensity
df["AI"] = df["MODE"].apply(
    lambda m: 0.5 if m == "scalar" else 0.125
)

# Convert bandwidth to GFLOPs
df["GFLOPs"] = df["GB/s"] * df["AI"]

# ==========================================================
# Roofline Curves
# ==========================================================

ai_range = np.logspace(-3, 2, 400)

mem_roof_sm75 = bw_sm75 * ai_range
comp_roof_sm75 = np.full_like(ai_range, fp32_sm75)

mem_roof_sm80 = bw_sm80 * ai_range
comp_roof_sm80 = np.full_like(ai_range, fp32_sm80)

# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(9,6))

# SM75 Roofs
plt.loglog(ai_range, mem_roof_sm75,
           linewidth=2,
           label="SM75 Memory Roof")

plt.loglog(ai_range, comp_roof_sm75,
           linewidth=1.5,
           alpha=0.6,
           label="SM75 Compute Roof")

# SM80 Roofs (dashed)
plt.loglog(ai_range, mem_roof_sm80,
           linestyle='--',
           linewidth=2,
           label="SM80 Memory Roof")

plt.loglog(ai_range, comp_roof_sm80,
           linestyle='--',
           linewidth=1.5,
           alpha=0.6,
           label="SM80 Compute Roof")

# ==========================================================
# Plot Measured Points
# ==========================================================

markers = {
    128: "o",
    256: "s",
    512: "^"
}

colors = {
    "scalar": "blue",
    "vec4": "red"
}

for _, row in df.iterrows():
    plt.scatter(
        row["AI"],
        row["GFLOPs"],
        color=colors[row["MODE"]],
        marker=markers[row["BLOCK_SIZE"]],
        s=80,
        alpha=0.85
    )

# ==========================================================
# Highlight Best SM75 Configuration
# ==========================================================

best = df.loc[df["GB/s"].idxmax()]

plt.scatter(best["AI"],
            best["GFLOPs"],
            color="black",
            s=160,
            edgecolors="yellow",
            linewidth=1.8,
            label="Best SM75 Config")

# ==========================================================
# Predict SM80 Performance for Best Kernel
# ==========================================================

scale_factor = bw_sm80 / bw_sm75
predicted_sm80_gflops = best["GFLOPs"] * scale_factor

plt.scatter(best["AI"],
            predicted_sm80_gflops,
            color="green",
            s=160,
            edgecolors="black",
            linewidth=1.8,
            label="Predicted SM80 (Same Kernel)")

# Optional annotation
plt.annotate("Best SM75",
             (best["AI"], best["GFLOPs"]),
             textcoords="offset points",
             xytext=(8,6))

plt.annotate("Predicted SM80",
             (best["AI"], predicted_sm80_gflops),
             textcoords="offset points",
             xytext=(8,6))

# ==========================================================
# Formatting
# ==========================================================

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOPs)")
plt.title("Roofline Comparison: SM75 vs SM80 (Simulated)")

# Focus on relevant region
plt.xlim(0.08, 0.6)
plt.ylim(20, 500)

plt.legend()
plt.grid(True, which="major", linestyle="--", alpha=0.2)
plt.grid(False, which="minor")

plt.tight_layout()

# Save outputs
plt.savefig("roofline_sm75_vs_sm80_refined.png", dpi=300)
plt.savefig("roofline_sm75_vs_sm80_refined.svg")

plt.show()