#include <iostream>
#include <omp.h>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <complex>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

Eigen::MatrixXd hamiltonian(double t, unsigned L_A, unsigned L_B, double sigma)
{
    unsigned size = 2 * L_A * L_B;  // Taille de la matrice H (2 états par site)
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(size, size);  // Initialisation de la matrice H à zéro

    double espA = -sigma / 2;
    double espB = sigma / 2;

    for (int a = 0; a < size; a += 2) {
        int ia = (a / 2) % L_A;  // Indice dans la direction A
        int ib = (a / 2) / L_A;  // Indice dans la direction B

        int b = a + 1;  // L'état B associé à l'état A

        // Calcul des indices voisins avec conditions périodiques
        int xa_a = (a + 2) % size;  // Voisin à droite (en x)
        int ya_a = (a + 2 * L_A) % size;  // Voisin en bas (en y)

        int xa_b = (b + 2) % size;  // Voisin à droite (pour B)
        int ya_b = (b + 2 * L_A) % size;  // Voisin en bas (pour B)

        // Remplissage des termes diagonaux (potentiels)
        H(a, a) = espA;
        H(b, b) = espB;

        // Interaction entre voisins
        // Voisins à droite (A et B)
        H(a, xa_a) = -t;
        H(xa_a, a) = -t;
        H(b, xa_b) = -t;
        H(xa_b, b) = -t;

        // Voisins en bas (A et B)
        H(a, ya_a) = -t;
        H(ya_a, a) = -t;
        H(b, ya_b) = -t;
        H(ya_b, b) = -t;
    }

    return H;
}


Eigen::VectorXd LDOS(double energy, double gamma, double t,unsigned L_A, unsigned L_B, double sigma) 
{
    double N = 2 * L_A * L_B; // Taille de la matrice H (2 états par site)
    Eigen::MatrixXd H = hamiltonian(t, L_A, L_B, sigma);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H); // diagonalisation de H
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::VectorXd dos = Eigen::VectorXd::Zero(N);


    #pragma omp parallel
    {
        Eigen::VectorXd dos_private = Eigen::VectorXd::Zero(N);

        #pragma omp for schedule(static)

        for (int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                double k_x = (2 * i * M_PI) / N;
                double k_y = (2 * j * M_PI) / N; 
                for (int y = 1; y <= N; y++) //to respect orthoganility, from 1 to N (size N)
                {
                    double phi_x = sqrt(2.0 / (N + 1)) * sin(k_x * y);
                    double phi_y = sqrt(2.0 / (N+1)) * sin(k_y * y); 
                    double phi = phi_x * phi_y;
                    double energy_val = eigenvalues(i); 
                    dos_private(y-1) += (gamma / (((energy - energy_val) * (energy - energy_val)) + gamma * gamma)) * phi * phi;
                }
            }

        }

    }
    dos = (1 / (M_PI * N * N)) * dos;
    return dos;
    
}


int main()
{
    unsigned L_A = 50; // Nombre de points dans la direction A
    unsigned L_B = 25; // Nombre de points dans la direction B
    double t = 1; // hopping parameter
    double gamma = 0.1;
    double sigma = 0.2;
    int energy = 4;
    unsigned N_point = L_A * L_B; // Nombre total de points

    // Vecteurs pour stocker les temps et les accélérations
    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);


    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        std::ofstream DosFile("LDosGraphen_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir le fichier de sortie" << std::endl;
            return 1;
        }

        DosFile << N_point << std::endl;
        DosFile << energy << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        // Calcul de la densité d'états (DOS) pour différentes énergies
        for (int i = 0; i <= 2 * energy * 10; ++i) 
        {
            double e = -energy + 0.1 * i;
            Eigen::VectorXd dos = LDOS(e, gamma, t, L_A, L_B, sigma);
            for (int j = 0; j < dos.size(); ++j)
            {
                DosFile << dos(j) << '\n';
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << "s\n";

        DosFile.close();
    }

    // Calcul des accélérations (speedups)
    for (int i = 0; i < times.size(); ++i) {
        speedups(i) = times(0) / times(i);
    }

    // Sauvegarde des résultats dans un fichier speedup
    std::ofstream SpeedupFile("speedup_periodic.txt");
    if (!SpeedupFile) {
        std::cerr << "Impossible d'ouvrir le fichier speedup" << std::endl;
        return 1;
    }
    for (int i = 0; i < thread_counts.size(); ++i) {
        SpeedupFile << thread_counts(i) << " " << times(i) << " " << speedups(i) << std::endl;
    }
    SpeedupFile.close();

    return 0;
}


