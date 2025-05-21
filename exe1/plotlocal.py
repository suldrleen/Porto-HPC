import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('DosOpen2Dlocal_1.txt', header=None)

# Extraire les valeurs
N_points = int(data.iloc[0, 0])
energy_max = int(data.iloc[1, 0])
dos_values = data.iloc[2:].values.flatten()

# Vérifier que les tailles correspondent
if len(dos_values) % N_points != 0:
    print("Erreur : le nombre total de données DOS n'est pas divisible par N_points.")
    print(f"Taille dos_values = {len(dos_values)}, N_points = {N_points}")
    exit(1)

num_energies = len(dos_values) // N_points

# Grilles pour l'affichage
energy = np.linspace(-energy_max, energy_max, num_energies)
states = np.arange(0, N_points)

X, Y = np.meshgrid(energy, states)

Z = dos_values.reshape(N_points, num_energies)

# Tracé du graphe
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.8)
ax.set_xlabel('Energy')
ax.set_ylabel('State')
ax.set_zlabel('DOS')
ax.view_init(30, 60)
plt.show()
