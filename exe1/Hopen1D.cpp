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

    #pragma omp parallel for schedule(dynamic, 16)
    for(int i = 1; i<N+1 ; i++)
    {
        double k_x = (i*M_PI)/(N+1);
        en(i-1) = -2 * t * cos(k_x);
    }
    return en;
}

double DOS(int N, double energy, double gamma, double t, int &l, std::ofstream &TimeFile, std::chrono::high_resolution_clock::time_point &start)
{
    Eigen::VectorXd en = H(t, N);
    double dos = 0.0;

    #pragma omp parallel for reduction(+:dos) schedule(dynamic, 16)
    for(int i = 0; i < N; i++)
    {
        dos += gamma / (((energy - en(i)) * (energy - en(i))) + gamma * gamma);
    }

    dos = 1 / (M_PI * N) * dos;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;


   TimeFile << l << "  " << elapsed.count() << std::endl;
    l += 1;
    
    return dos;
}

    

int main()
{
    int energy = 3;
    double gamma = 0.1, t = 1.0;
    unsigned N_point = 10000;

    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 6;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);

    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        int l = 1;
        std::string base_name = "Open1D_" + std::to_string(threads);
        std::ofstream DosFile("Dos" + base_name + ".txt");
        std::ofstream TimeFile("TimeExecution" + base_name + ".txt");

        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir Dos" + base_name + ".txt" << std::endl;
            continue;
        }

        DosFile << N_point << std::endl;
        DosFile << energy << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        for(double i = -energy; i <= energy; i += 0.1) {
            double dos = DOS(N_point, i, gamma, t, l, TimeFile, start);
            DosFile << dos << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << "s\n";

        DosFile.close();
        TimeFile.close();
    }

    for (int i = 0; i < times.size(); ++i) {
        speedups(i) = times(0) / times(i);
    }

    std::ofstream SpeedupFile("speedup_open.txt");
    for (int i = 0; i < thread_counts.size(); ++i) {
        SpeedupFile << thread_counts(i) << " " << times(i) << " " << speedups(i) << std::endl;
    }
    SpeedupFile.close();

    return 0;
}
