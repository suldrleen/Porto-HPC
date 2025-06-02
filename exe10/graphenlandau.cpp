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

// graphen lattice


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


Eigen::MatrixXcd buildHamiltonian(int Lx, int Ly, double t, double delta, double flux_ratio) {
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
        for (int y = 0; y < Ly; ++y) 
        {
            // On parcourt les hoppings pour chaque site (x, y)
            for (const auto& hop : hoppings) {
                int from_idx = getIndex(x, y, hop.from, Lx, Ly); //on récupère l'indice de Atome A
                int to_x = x + hop.displacement[0];
                int to_y = y + hop.displacement[1];
                int to_idx = getIndex(to_x, to_y, hop.to, Lx, Ly); //indice atom B

                std::complex<double> amp = hop.amplitude;

                    // Phase de Peierls si saut en y
                if (hop.displacement[1] != 0) {
                    double phase = 2.0 * M_PI * flux_ratio * x;
                    amp *= std::exp(std::complex<double>(0, phase));
                }

                H(from_idx, to_idx) += amp;
                if (from_idx != to_idx)
                    H(to_idx, from_idx) += std::conj(amp);
                }
            }
        }

        return H;
    }






int main()
{
    const int Lx = 20;
    const int Ly = 20;
    const double t = 1.0;
    const int N_flux = 50; // number of phi/phi0 values (horizontal resolution)
    double delta = 0.5; // Potentiel de désordre

    // Vecteurs pour stocker les temps et les accélérations
    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);


    
    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);
        std::vector<std::stringstream> buffers(threads);

        std::ofstream LandauFile("landau" + std::to_string(threads) + ".txt");

        auto start = std::chrono::high_resolution_clock::now();

        // Calcul de la densité d'états (DOS) pour différentes énergies
        #pragma omp parallel
        { 
            int tid = omp_get_thread_num();


            #pragma omp for schedule(dynamic)
            for (int k = 0; k <= N_flux; ++k) {
                double flux_ratio = static_cast<double>(k) / N_flux;
        
                Eigen::MatrixXcd H = buildHamiltonian(Lx, Ly, t, delta, flux_ratio);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
                const Eigen::VectorXd& eigs = es.eigenvalues();
        
                for (int i = 0; i < eigs.size(); ++i) {
                    buffers[tid] << flux_ratio << " " << eigs(i) << "\n";
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total = end - start;
        times(idx) = total.count();

        std::cout << "Threads: " << threads << ", Time: " << total.count() << "s\n";

        for (int t = 0; t < threads; ++t) {
            LandauFile << buffers[t].str();
        }

        LandauFile.close();

    }

    // Calcul des accélérations (speedups)
    for (int i = 0; i < times.size(); ++i) {
        speedups(i) = times(0) / times(i);
    }

    // Sauvegarde des résultats dans un fichier speedup
    std::ofstream SpeedupFile("speedup_periodic.txt");

    for (int i = 0; i < thread_counts.size(); ++i) {
        SpeedupFile << thread_counts(i) << " " << times(i) << " " << speedups(i) << std::endl;
    }
    SpeedupFile.close();


    return 0;

}