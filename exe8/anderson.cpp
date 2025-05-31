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

Eigen::MatrixXcd hamiltonian_1D(int N, double t, const std::vector<double>& epsilon, double k) {
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(N, N);

    // Diagonale avec le désordre εn
    for (int n = 0; n < N; ++n) {
        H(n, n) = epsilon[n];
    }

    // Hoppings entre voisins
    for (int n = 0; n < N - 1; ++n) {
        H(n, n + 1) = -t;
        H(n + 1, n) = -t;
    }

    // Condition aux bords tordue avec phase e^{ik}
    std::complex<double> phase = std::exp(std::complex<double>(0, 1) * k);
    H(N - 1, 0) = -t * phase;
    H(0, N - 1) = -t * std::conj(phase);

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



int main() {
    int N = 24 * 24; // taille du système
    double t = 1.0;
    double gamma = 0.1;
    double energy_max = 4.0;
    int num_points = static_cast<int>(2 * energy_max / 0.2) + 1;


    // Création d'un vecteur epsilon (désordre) avec des valeurs arbitraires (exemple zéro)
    std::vector<double> epsilon(N, 0.0);

    // Phase k (exemple à zéro)
    double k = 0.0;

    // Construction de l'Hamiltonien 1D
    Eigen::MatrixXcd H = hamiltonian_1D(N, t, epsilon, k);

    // Diagonalisation
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXcd eigenvectors = es.eigenvectors();

    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);

    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        
        std::ofstream DosFile("DosAnderson_" + std::to_string(threads) + ".txt");


        DosFile << N << std::endl;
        DosFile << energy_max << std::endl;



        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::stringstream> buffers_dos(threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::stringstream& buffer = buffers_dos[tid];
        
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < num_points; ++i) {
                double e = -energy_max + 0.2 * i;
                double dos = DOS(e, gamma, eigenvalues);
        
                buffer << dos << "\n";
            }
        }
               
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << " s\n";

        // Écriture dans les fichiers
        for (int t = 0; t < threads; ++t) {
            DosFile << buffers_dos[t].str();
        }




        DosFile.close();

    }

    for (int i = 0; i < times.size(); ++i) {
        speedups(i) = times(0) / times(i);
    }



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