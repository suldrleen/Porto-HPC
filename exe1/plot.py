import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

data = pd.read_csv('DosOpen1D.txt',header=None)

# Extraire les valeurs d'énergie et de points
energy_values = int(data.iloc[1, 0])  # 2nd valeur comme énergie
N_points = int(data.iloc[0, 0])  # 1ere valeur comme nombre de points
print(energy_values)

# Extraire les valeurs de DOS
dos_values = data.iloc[2:].values.flatten()  # Toutes les valeurs restantes

# Calculer le nombre d'énergies
num_energies = int(len(dos_values) / N_points)


# Créer une grille pour les axes X et Y
energy = np.linspace(-energy_values, energy_values, num_energies)
N = np.arange(0, N_points)

X, Y = np.meshgrid(energy, N)

# Résoudre les valeurs de DOS en une matrice 2D
Z = dos_values.reshape(N_points,num_energies)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel('Energy')
ax.set_ylabel('State')
ax.set_zlabel('DOS')
ax.view_init(30, 60)
plt.show()

# Lire les données depuis le fichier
iterations = []
execution_times = []

with open("TimeExecution1op.txt", "r") as file:
    for line in file:
        parts = line.split()
        iterations.append(float(parts[0]))
        execution_times.append(float(parts[1]))

plt.plot(iterations, execution_times, marker='o')
plt.title('Parallélisation for Open 1D')
plt.xlabel('Itération')
plt.ylabel('Temps d\'exécution (secondes)')
plt.grid(True)
plt.show()