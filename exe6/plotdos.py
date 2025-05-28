import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Lecture de ton fichier DOS ===
data = pd.read_csv('DosPeriodic3D_1.txt', header=None)

N_points = int(data.iloc[0, 0])    # nombre de points (pas utilisé ici)
energy_max = int(data.iloc[1, 0])   # énergie maximale
dos_values = data.iloc[2:].values.flatten()

num_energies = len(dos_values)
energy_values = np.linspace(-energy_max, energy_max, num_energies)

# === 2. Calcul de la DOS théorique pour 2D périodique ===
# Seulement pour |E| <= 4
E_theo = np.linspace(-4, 4, 1000)
dos_theo = np.zeros_like(E_theo)

# Éviter la division par zéro en ajoutant une petite valeur epsilon
epsilon = 1e-6
for i, E in enumerate(E_theo):
    if abs(E) < 4:
        dos_theo[i] = 1.0 / np.sqrt(4**2 - E**2 + epsilon)
    else:
        dos_theo[i] = 0

# Normalisation pour matcher les échelles (optionnel, pour que ce soit plus joli)
dos_theo *= np.max(dos_values) / np.max(dos_theo)

# === 3. Tracer ===
plt.figure(figsize=(10, 6))
plt.plot(energy_values, dos_values, label="DOS Numérique (C++)", lw=2)


plt.xlabel('Énergie')
plt.ylabel('DOS')
plt.title('DOS Numérique (3D cube)')
plt.grid(True)
plt.legend()
plt.show()

