import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Lecture du fichier DOS ===
data = pd.read_csv('DosAnderson_1.txt', header=None)

N_points = int(data.iloc[0, 0])      # nombre de points N_E
Emin = float(data.iloc[1, 0])        # énergie minimale
Emax = float(data.iloc[2, 0])        # énergie maximale

dos_values = data.iloc[3:].values.flatten().astype(float)

energy_values = np.linspace(Emin, Emax, len(dos_values))



# === 3. Tracé ===
plt.figure(figsize=(10, 6))
plt.plot(energy_values, dos_values, label="DOS Numérique (C++)", lw=2)
plt.xlabel("Énergie")
plt.ylabel("DOS")
plt.title("Densité d'États")
plt.grid(True)
plt.legend()
plt.show()

