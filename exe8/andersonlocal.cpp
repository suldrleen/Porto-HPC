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




Eigen::VectorXd LDOS(double energy, double gamma, const Eigen::VectorXd& eigenvalues, const Eigen::MatrixXcd& eigenvectors) {
    int N = eigenvalues.size();
    Eigen::VectorXd ldos = Eigen::VectorXd::Zero(N);

    int nthreads = omp_get_max_threads();
    std::vector<Eigen::VectorXd> ldos_private(nthreads, Eigen::VectorXd::Zero(N));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Eigen::VectorXd& local_ldos = ldos_private[tid];

        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            double diff = energy - eigenvalues(n);
            double lorentz = gamma / (diff * diff + gamma * gamma);
            const auto& eigvec_n = eigenvectors.col(n);

            for (int i = 0; i < N; ++i) {
                double weight = std::norm(eigvec_n(i));
                local_ldos(i) += weight * lorentz;
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        ldos += ldos_private[t];
    }

    ldos /= M_PI;
    return ldos;
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

        std::vector<std::stringstream> buffers_ldos(num_points);
        
        std::ofstream LDosFile("LDosAnderson_" + std::to_string(threads) + ".txt");
        



        LDosFile << N << std::endl;
        LDosFile << energy_max << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_points; ++i) {
            double e = -energy_max + 0.2 * i;


            Eigen::VectorXd ldos = LDOS(e, gamma, eigenvalues, eigenvectors);



            buffers_ldos[i] << e << "\n";
            for (int j = 0; j < ldos.size(); ++j) {
                buffers_ldos[i] << ldos(j) << "\n";
            }
        }        
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << " s\n";




        for (int i = 0; i < num_points; ++i) {
            LDosFile << buffers_ldos[i].str();
        }




        LDosFile.close();
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
