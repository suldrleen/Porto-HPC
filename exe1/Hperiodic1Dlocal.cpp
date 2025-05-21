#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <complex>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include <fstream>

Eigen::VectorXd H(double t, unsigned N)
{
    Eigen::VectorXd en(N);

    #pragma omp parallel for schedule(dynamic,1)
    for(int i = 1; i < N + 1; i++)
    {
        double k_x = (i * M_PI) / (N + 1);
        en(i-1) = -2 * t * cos(k_x);
    }
    return en;
}

Eigen::VectorXd LDOS(int N, double energy, double gamma, double t) {
    Eigen::VectorXd en = H(t, N);
    Eigen::VectorXd dos(N);
    dos.setZero();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; i++) {
        double k_x = (2 * i * M_PI) / N;
        for (int y = 0; y < N; y++) {
            double phi = sqrt(2.0 / (N + 1)) * sin(k_x * y);
            dos(y) += (gamma / (((energy - en(i)) * (energy - en(i))) + gamma * gamma)) * phi * phi;
        }
    }
    dos = 1 / (M_PI * N) * dos;
    return dos;
}

int main() {
    int energy = 2;
    double gamma = 0.1, t = 1.0;
    unsigned N_point = 100;

    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);

    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        std::string base_name = "Periodic1Dlocal_" + std::to_string(threads);
        std::ofstream DosFile("Dos" + base_name + ".txt");

        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir Dos" + base_name + ".txt" << std::endl;
            continue;
        }

        DosFile << N_point << '\n';
        DosFile << energy << '\n';

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i <= 2 * energy * 10; ++i) {
            double e = -energy + 0.1 * i;
            Eigen::VectorXd dos = LDOS(N_point, e, gamma, t);
            for (unsigned j = 0; j < N_point; j++) {
                DosFile << dos(j) << '\n';
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << "s\n";

        DosFile.close();
    }

    for (int i = 0; i < times.size(); ++i) {
        speedups(i) = times(0) / times(i);
    }

    std::ofstream SpeedupFile("speedup_periodic_local.txt");
    for (int i = 0; i < thread_counts.size(); ++i) {
        SpeedupFile << thread_counts(i) << " " << times(i) << " " << speedups(i) << '\n';
    }
    SpeedupFile.close();

    return 0;
}
