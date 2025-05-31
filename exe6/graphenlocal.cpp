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


Eigen::VectorXd LDOS(double energy, double gamma, const Eigen::VectorXd& eigenvalues, const Eigen::MatrixXcd& eigenvectors) {
    int N = eigenvalues.size();
    Eigen::VectorXd ldos = Eigen::VectorXd::Zero(N);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        double diff = energy - eigenvalues(n);
        double lorentz = gamma / (diff * diff + gamma * gamma);
        const auto& eigvec_n = eigenvectors.col(n);  // accès direct à la colonne n

        for (int i = 0; i < N; ++i) {
            double weight = std::norm(eigvec_n(i));  // |ψ_n(i)|²

            #pragma omp atomic
            ldos(i) += weight * lorentz;
        }
    }

    ldos /= M_PI;
    return ldos;
}




int main()
{
    unsigned L_A = 25;
    unsigned L_B = 25;
    double t = 1;
    double gamma = 0.1;
    double sigma = 0.2;
    int energy = 4;
    unsigned N_point = L_A * L_B;

    Eigen::MatrixXcd H = buildHamiltonian(L_A, L_B, t, sigma);
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

        std::vector<std::stringstream> buffers(threads); // 🟩 buffers placés au bon endroit

        std::ofstream DosFile("LDosGraphen_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir le fichier de sortie" << std::endl;
            return 1;
        }

        DosFile << N_point << std::endl;
        DosFile << energy << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
        
            #pragma omp for schedule(dynamic)
            for (int i = 0; i <= 2 * energy * 10; ++i) 
            {
                double e = -energy + 0.1 * i;
                Eigen::VectorXd ldos = LDOS(e, gamma, eigenvalues, eigenvectors);
        
                buffers[tid] << e << "\n";
                for (int j = 0; j < ldos.size(); ++j) {
                    buffers[tid] << ldos(j) << "\n";
                }
            }
        }
        

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << "s\n";

        // Écrire tous les buffers
        for (int t = 0; t < threads; ++t) {
            DosFile << buffers[t].str();
        }

        DosFile.close();
    }

    // Calcul des accélérations
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

        