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

enum AtomType { A = 0, B = 1 }; //subspace A & B

struct Hopping {
    AtomType from;
    AtomType to;
    std::array<int, 2> displacement; // déplacement dans le réseau (x, y)
    std::complex<double> amplitude;
};

inline int getIndex(int x, int y, AtomType type, int Lx, int Ly){
    x = (x +Lx) % Lx;
    y = (y + Ly) % Ly;
    return 2 * (x + Lx * y) + static_cast<int>(type);
}


Eigen::MatrixXcd buildHamiltonian(int Lx, int Ly, double t, double delta) {
    int size = 2 * Lx * Ly; //2 atoms/sites
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(size, size);

    // Potentiels sur chaque sous-réseau
    std::complex<double> E_A(delta / 2.0, 0);
    std::complex<double> E_B(-delta / 2.0, 0);

    // definition hoppings 
    std::vector<Hopping> hoppings = {
        {A, A, {0, 0}, E_A},
        {B, B, {0, 0}, E_B},
        {A, B, {0, 0}, -t},
        {A, B, {-1, 0}, -t},
        {A, B, {0, -1}, -t}
    };

    // On remplit la matrice Hamiltonien
    #pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < Lx; ++x) {
        for (int y = 0; y < Ly; ++y) {
            for (const auto& hop : hoppings) {
                int from_idx = getIndex(x, y, hop.from, Lx, Ly); //on récupère l'indice de Atome A
                int to_x = x + hop.displacement[0];
                int to_y = y + hop.displacement[1];
                int to_idx = getIndex(to_x, to_y, hop.to, Lx, Ly); //indice atom B
                // On remplit la matrice Hamiltonien en utilisant définition hopping
                H(from_idx, to_idx) += hop.amplitude;
                // Le Hamiltonien est hermitien : on met aussi la transposée conjuguée si hop.from != hop.to
                if (from_idx != to_idx) {
                    H(to_idx, from_idx) += std::conj(hop.amplitude);
                }
            }
        }
    }
    return H;
}




// === Calcule la densité d'états (DOS) pour une énergie donnée ===
double DOS(double energy, double gamma, const Eigen::VectorXd& eigenvalues) {
    double dos = 0.0;
    int size = eigenvalues.size();

    for (int i = 0; i < size; i++) {
        double diff = energy - eigenvalues(i);
        dos += gamma / (diff * diff + gamma * gamma);
    }
    dos /= (M_PI * size);
    return dos;
}

int main()
{
    int Lx = 24;
    int Ly = 24;
    double t = 1.0;
    double delta = 1.0;
    double gamma = 0.1;
    double sigma = 0.2;
    double energy = 4;
    unsigned N_point = Lx * Ly; // Nombre total de points

    // Vecteurs pour stocker les temps et les accélérations
    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);

    //Diagonalisation de H une seule fois
    Eigen::MatrixXcd H = buildHamiltonian(Lx, Ly, t, delta);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    Eigen::VectorXd eigenvalues = es.eigenvalues();


    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);
        std::vector<std::stringstream> buffers(threads);

        std::ofstream DosFile("DosGraphen_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir le fichier de sortie" << std::endl;
            return 1;
        }

        DosFile << N_point << std::endl;
        DosFile << energy << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        // Calcul de la densité d'états (DOS) pour différentes énergies
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i <= static_cast<int>(2 * energy * 5); ++i) 
        {
            int tid = omp_get_thread_num();
            double e = -energy + 0.2 * i;
            double dos = DOS(e, gamma, eigenvalues);

            buffers[tid] << dos << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << "s\n";

        for (int t = 0; t < threads; ++t) {
            DosFile << buffers[t].str();
        }

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

// Compilation : g++ -fopenmp -I /usr/include/eigen3 graphene.cpp -o graphene

