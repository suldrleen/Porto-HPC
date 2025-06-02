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

// square lattice


inline int getIndex(int x, int y, int Lx, int Ly){
        x = (x + Lx) % Lx;
        y = (y + Lx) % Lx;
        return x + Lx * y;
    };

Eigen::MatrixXcd buildHamiltonian(int Lx, int Ly, double t, double flux_ratio) {
    int size =  Lx * Ly; //1 atoms/sites
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(size, size);


    // On remplit la matrice Hamiltonien
    #pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < Lx; ++x) {
        for (int y = 0; y < Ly; ++y) 
        { 
            int i = getIndex(x, y, Lx, Ly);

            //saut en x

            int jx = getIndex(x+1, y, Lx, Ly);
            H(i, jx) = -t;
            H(jx, i) = -t;

            //saut en y 

            int jy = getIndex(x, y+1, Lx, Ly);
            double phase = 2.0 * M_PI * flux_ratio * x; // Landau
            std::complex<double> peierls = std::exp(std::complex<double>(0, phase));

            H(i, jy) = -t * peierls;
            H(jy, i) = -t * std::conj(peierls);

            
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
        
                Eigen::MatrixXcd H = buildHamiltonian(Lx, Ly, t, flux_ratio);
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