import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

data = pd.read_csv('DosPeriodic1D.txt',header=None, delim_whitespace=True)

# Extraire les valeurs d'énergie et de points
energy_values = int(data.iloc[1, 0])  # 2nd valeur comme énergie
N_points = int(data.iloc[0, 0])  # 1ere valeur comme nombre de points


# Extraire les valeurs de DOS
dos_values = data.iloc[2:].values.flatten()  # Toutes les valeurs restantes


# Calculer le nombre d'énergies
num_energies = len(dos_values)


# Créer une grille pour les axes X et Y
energy = np.linspace(-energy_values, energy_values, num_energies)


fig = plt.figure()
plt.plot(energy,dos_values)
plt.xlabel('Energy')
plt.ylabel('Dos')
plt.title('Dos periodic Boundaries 1D')
plt.show()

# Lire les données depuis le fichier
iterations = []
execution_times = []

with open("TimeExecution1D.txt", "r") as file:
    for line in file:
        parts = line.split()
        iterations.append(float(parts[0]))
        execution_times.append(float(parts[1]))

plt.plot(iterations, execution_times, marker='o')
plt.title('Parallélisation for periodic 1D')
plt.xlabel('Itération')
plt.ylabel('Temps d\'exécution (secondes)')
plt.grid(True)
plt.show()