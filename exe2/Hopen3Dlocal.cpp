#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <complex>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include <fstream>

Eigen::VectorXd H(double t, unsigned N)
{
    Eigen::VectorXd en(N * N * N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            for (int k = 0; k < N; k++) 
            {
                double k_x = (i + 1) * M_PI / (N + 1);
                double k_y = (j + 1) * M_PI / (N + 1);
                double k_z = (k + 1) * M_PI / (N + 1);

                en(i * N * N + j * N + k) = -2 * t * (cos(k_x) + cos(k_y) + cos(k_z));
            }
        }
    }
    return en;
}

Eigen::VectorXd LDOS(int N, double energy, double gamma, double t)
{
    Eigen::VectorXd en = H(t, N);
    Eigen::VectorXd dos = Eigen::VectorXd::Zero(N);

    #pragma omp parallel
    {
        Eigen::VectorXd dos_private = Eigen::VectorXd::Zero(N);

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) 
        {
            for (int j = 0; j < N; j++) 
            {
                for (int k = 0; k < N; k++) 
                {
                    double k_x = (i + 1) * M_PI / (N + 1);
                    double k_y = (j + 1) * M_PI / (N + 1);
                    double k_z = (k + 1) * M_PI / (N + 1);

                    for (int y = 1; y <= N; y++) 
                    {
                        double phi_x = sqrt(2.0 / (N + 1)) * sin(k_x * y);
                        double phi_y = sqrt(2.0 / (N + 1)) * sin(k_y * y);
                        double phi_z = sqrt(2.0 / (N + 1)) * sin(k_z * y);
                        double phi = phi_x * phi_y * phi_z;

                        dos_private(y - 1) += (gamma / ((energy - en(i * N * N + j * N + k)) * (energy - en(i * N * N + j * N + k)) + gamma * gamma)) * phi * phi;
                    }
                }
            }
        }

        #pragma omp critical
        {
            dos += dos_private;
        }
    }

    dos *= (1.0 / (M_PI * N * N * N));
    return dos;
}

int main()
{
    int energy_max = 4;
    double gamma = 0.1, t = 1.0;
    unsigned N_point = 10;

    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);

    for (int idx = 0; idx < thread_counts.size(); ++idx)
    {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        std::ofstream DosFile("DosOpen3Dlocal_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible to open output file" << std::endl;
            return 1;
        }

        DosFile << N_point << std::endl;
        DosFile << energy_max << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i <= 2 * energy_max * 10; ++i) 
        {
            double e = -energy_max + 0.1 * i;
            Eigen::VectorXd dos = LDOS(N_point, e, gamma, t);

            for (unsigned j = 0; j < N_point; j++) 
            {
                DosFile << dos(j) << std::endl;
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

    std::ofstream SpeedupFile("speedup_open3Dlocal.txt");
    for (int i = 0; i < thread_counts.size(); ++i) {
        SpeedupFile << thread_counts(i) << " " << times(i) << " " << speedups(i) << std::endl;
    }
    SpeedupFile.close();

    return 0;
}
