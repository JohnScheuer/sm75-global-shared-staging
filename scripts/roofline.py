import numpy as np
import matplotlib.pyplot as plt

# RTX 2070 specs
peak_bandwidth = 448  # GB/s
peak_compute = 7500   # GFLOPs (7.5 TFLOPs)

# Arithmetic intensity range
ai = np.logspace(-3, 2, 100)

# Roofline model
memory_roof = peak_bandwidth * ai
compute_roof = np.full_like(ai, peak_compute)

# Kernel point
kernel_ai = 0.125
measured_bandwidth = 350
kernel_perf = measured_bandwidth * kernel_ai

plt.figure(figsize=(8,6))

plt.loglog(ai, memory_roof, label="Memory Roof")
plt.loglog(ai, compute_roof, label="Compute Roof")

plt.scatter(kernel_ai, kernel_perf, color="red", s=100, label="SM75 vec4 best")

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOPs)")
plt.title("Roofline Model - RTX 2070 (SM75)")
plt.legend()
plt.grid(True, which="both")

plt.tight_layout()
plt.savefig("roofline_sm75.png")
plt.show()
