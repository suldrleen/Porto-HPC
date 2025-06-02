import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("landau8.txt")  # ou landau4.txt, etc.
phi = data[:, 0]
E = data[:, 1]

plt.figure(figsize=(8,6))
plt.scatter(phi, E, s=0.1, color='black')
plt.xlabel(r"$\phi/\phi_0$")
plt.ylabel("Énergie")
plt.title("Hofstadter Butterfly")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
