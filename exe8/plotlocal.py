import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('LDosAnderson_1.txt', header=None)

N_points = int(data.iloc[0, 0])
energy_max = int(data.iloc[1, 0])

lines = data.iloc[2:].values.flatten()

num_energies = (len(lines)) // (N_points + 1)  # car une ligne énergie + N_points valeurs

ldos_values = []

for i in range(num_energies):
    start = i * (N_points + 1) + 1  # +1 pour sauter la ligne énergie
    end = start + N_points
    ldos_values.extend(lines[start:end])

ldos_values = np.array(ldos_values, dtype=float)

energy = np.linspace(-energy_max, energy_max, num_energies)
states = np.arange(N_points)

X, Y = np.meshgrid(energy, states)
Z = ldos_values.reshape(N_points, num_energies)

# le reste du plot...

print(energy_max)

# Tracé du graphe
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.8)
ax.set_xlabel('Energy')
ax.set_ylabel('State')
ax.set_zlabel('DOS local graphen')
ax.view_init(30, 60)
plt.show()
