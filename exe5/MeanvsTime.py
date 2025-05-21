import numpy as np
import matplotlib.pyplot as plt

# Fonction pour lire les fichiers et extraire les données de mean et variance
def read_data(file_name):
    # Chargement des données depuis le fichier
    data = np.loadtxt(file_name)
    
    # Extraction des colonnes
    times = data[:, 0]
    means = data[:, 1]
    variances = data[:, 2]
    
    return times, means, variances

# Spécifier les noms des fichiers (par exemple, pour 1 thread)
file_name = "mean_1.txt"

# Lire les données
times, means, variances = read_data(file_name)

# Tracer le graphique
plt.figure(figsize=(10, 5))

# Tracer la moyenne
plt.subplot(1, 2, 1)
plt.plot(times, means, label='Mean')
plt.xlabel('Time')
plt.ylabel('Mean')
plt.title('Mean vs Time')
plt.grid(True)

# Tracer la variance
plt.subplot(1, 2, 2)
plt.plot(times, variances, label='Variance', color='red')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.title('Variance vs Time')
plt.grid(True)

# Affichage
plt.tight_layout()
plt.show()
