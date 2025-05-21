#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <complex>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include <fstream>

// === Calcule les énergies du système périodique ===
Eigen::VectorXd H(double t, unsigned N) {
    Eigen::VectorXd en(N);

    #pragma omp parallel for schedule(static, 2) shared(en)
    for (int i = 0; i < N; i++) {
        double k_x = (2 * i * M_PI) / N;
        en(i) = -2 * t * cos(k_x);
    }
    return en;
}

// === Calcule la densité d'états (DOS) pour une énergie donnée ===
double DOS(unsigned N, double energy, double gamma, double t) {
    Eigen::VectorXd en = H(t, N);
    double dos = 0.0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:dos)
    for (int i = 0; i < N; i++) {
        dos += gamma / ((energy - en(i)) * (energy - en(i)) + gamma * gamma);
    }

    dos *= 1.0 / (M_PI * N);
    return dos;
}

int main() {
    int energy = 3;
    double gamma = 0.1, t = 1.0;
    unsigned N_point = 100;

    // Vecteurs pour stocker les temps et les accélérations
    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);

    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        std::ofstream DosFile("DosPeriodic1D_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir le fichier de sortie" << std::endl;
            return 1;
        }

        DosFile << N_point << std::endl;
        DosFile << energy << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i <= 2 * energy * 10; ++i) {
            double e = -energy + 0.1 * i;
            double dos = DOS(N_point, e, gamma, t);
            DosFile << dos << std::endl;
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
