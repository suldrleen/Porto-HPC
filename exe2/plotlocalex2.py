import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fonction pour afficher le DOS à partir du fichier généré
def plot_dos_from_file(filename):
    # Lecture du fichier DosOpen2D.txt
    data = pd.read_csv(filename, header=None)

    # Extraire les valeurs d'énergie et de points
    N_points = int(data.iloc[0, 0])  # 1ère valeur comme nombre de points
    energy_values = int(data.iloc[1, 0])  # 2ème valeur comme énergie

    # Extraire les valeurs de DOS
    dos_values = data.iloc[2:].values.flatten()  # Toutes les valeurs restantes

    # Vérification de la taille des dos_values
    print(f"Taille des dos_values : {len(dos_values)}")
    
    # Calculer le nombre d'énergies de manière fiable en fonction du nombre total de données
    # On s'attend à ce qu'il y ait un total de N_points * num_energies valeurs
    num_energies = len(dos_values) // N_points

    if len(dos_values) % N_points != 0:
        print(f"Avertissement : Le nombre total de DOS n'est pas divisible par N_points.")
    
    print(f"Nombre d'énergies : {num_energies}")

    # Créer une grille pour les axes X et Y
    energy = np.linspace(-energy_values, energy_values, num_energies)
    N = np.arange(0, N_points)

    # Créer une grille 2D
    X, Y = np.meshgrid(energy, N)

    # Résoudre les valeurs de DOS en une matrice 2D
    Z = dos_values.reshape(N_points, num_energies)

    # Tracer le graphique 3D du DOS
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
    ax.set_xlabel('Energy')
    ax.set_ylabel('State')
    ax.set_zlabel('DOS')
    ax.view_init(30, 60)
    plt.show()

# Exemple d'appel avec un fichier généré, tu peux adapter selon le nom du fichier
filename = 'DosOpen2D_1.txt'  # Remplace par le nom du fichier désiré
plot_dos_from_file(filename)
