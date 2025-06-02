import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
# Format: threads  time(s)  speedup
data = np.loadtxt("speedup_periodic.txt")
threads = data[:, 0]
times = data[:, 1]
speedup = data[:, 2]

# Plotting the speedup
plt.figure(figsize=(8, 5))
plt.plot(threads, speedup, 'o-', color='royalblue', label='Measured speedup')
plt.plot(threads, threads, 'k--', label='Ideal speedup (linear)')

# Annotations and aesthetics
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Speedup", fontsize=12)
plt.title("Parallel Speedup using OpenMP", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xticks(threads.astype(int))
plt.tight_layout()

# Save and/or show the figure
plt.savefig("speedup_plot.png", dpi=300)
plt.show()
