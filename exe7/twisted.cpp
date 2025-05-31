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
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum AtomType { A = 0, B = 1 };

struct Hopping {
    AtomType from;
    AtomType to;
    std::array<int, 2> displacement;
    std::complex<double> amplitude;
};

Eigen::Matrix2cd hamiltonian_k(double kx, double ky, double t, double delta) {
    Eigen::Matrix2cd H = Eigen::Matrix2cd::Zero();

    std::complex<double> E_A = delta / 2.0;
    std::complex<double> E_B = -delta / 2.0;

    std::vector<Hopping> hoppings = {
        {A, A, {0, 0}, E_A},
        {B, B, {0, 0}, E_B},
        {A, B, {0, 0}, -t},
        {A, B, {-1, 0}, -t},
        {A, B, {0, -1}, -t}
    };

    for (const auto& hop : hoppings) {
        int from = static_cast<int>(hop.from);
        int to = static_cast<int>(hop.to);
        double phase = kx * hop.displacement[0] + ky * hop.displacement[1];
        std::complex<double> factor = std::exp(std::complex<double>(0, 1) * phase);
        H(from, to) += hop.amplitude * factor;
        if (from != to) {
            H(to, from) += std::conj(hop.amplitude * factor);
        }
    }
    return H;
}

double DOS_MC(double energy, double gamma, double t, double delta, int Nk)
{
    double dos = 0.0;

    #pragma omp parallel reduction(+:dos)
    {
        std::random_device r;
        std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 gen(seed2);
        std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);

        #pragma omp for schedule(static, 50)
        for (int i = 0; i < Nk; ++i)
        {
            double kx = dist(gen);
            double ky = dist(gen);

            Eigen::Matrix2cd Hk = hamiltonian_k(kx, ky, t, delta);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es(Hk);
            Eigen::Vector2d eigs = es.eigenvalues();

            for (int j = 0; j < 2; ++j)
            {
                double diff = energy - eigs(j);
                dos += gamma / (diff * diff + gamma * gamma);
            }
        }
    }

    dos /= (Nk * M_PI);
    return dos;
}
// ... mêmes includes, enums, structs, fonctions hamiltonian_k et DOS_MC ...
int main()
{
    double t = 1.0;
    double delta = 0.5;
    double gamma = 0.1;
    int Nk = 100000;
    double Emax = 3.0;
    double Emin = -3.0;
    int N_E = 300;

    Eigen::VectorXi thread_counts(3);
    thread_counts << 1, 2, 4;
    Eigen::VectorXd times(3);
    Eigen::VectorXd speedups(3);

    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        // Création d'un buffer par thread pour éviter les conflits d'écriture
        std::vector<std::stringstream> buffers(threads);

        std::ofstream DosFile("DosGraphen_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir le fichier de sortie" << std::endl;
            return 1;
        }

        // Écriture des paramètres en tête du fichier (optionnel)
        DosFile << N_E << std::endl;
        DosFile << Emin << std::endl;
        DosFile << Emax << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(static)
            for (int i = 0; i <= N_E; ++i)
            {
                double energy = Emin + i * (Emax - Emin) / N_E;
                double dos = DOS_MC(energy, gamma, t, delta, Nk);
                buffers[tid] << dos << std::endl;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << " s\n";

        // Concaténer les buffers dans l'ordre des threads (pas forcément ordonné en énergie)
        for (int t = 0; t < threads; ++t) {
            DosFile << buffers[t].str();
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
