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


constexpr double PI = 3.14159265358979323846;
using namespace std::complex_literals;

//on veut mettre l'hamiltonian en fonction de k, en ajoutant le changement de phase associé 
//puis on va faire la diagonalisation de H(k) pour obtenir les valeurs propres
//on utilisera MonteCarlo pour l'intégration sur k

enum AtomType { A = 0, B = 1 }; //subspace A & B

struct Hopping {
    AtomType from;
    AtomType to;
    std::array<int, 2> displacement; // déplacement dans le réseau (x, y)
    std::complex<double> amplitude;
};

// === Hamiltonien H(kx, ky) 2x2 pour un vecteur d'onde (kx, ky) ===
Eigen::Matrix2cd hamiltonian_k(double kx, double ky, double t, double delta) {
    Eigen::Matrix2cd H = Eigen::Matrix2cd::Zero();

    // Potentiel sur les sous-réseaux A et B (masse delta)
    std::complex<double> E_A = delta / 2.0;
    std::complex<double> E_B = -delta / 2.0;

    // Liste des termes de hopping (ici modèle de graphene avec 3 liaisons)
    std::vector<Hopping> hoppings = {
        {A, A, {0, 0}, E_A},
        {B, B, {0, 0}, E_B},
        {A, B, {0, 0}, -t},
        {A, B, {-1, 0}, -t},
        {A, B, {0, -1}, -t}
    };

    // Remplir la matrice H(k) avec les facteurs de phase e^{i k.r}
    for (const auto& hop : hoppings) {
        int from = static_cast<int>(hop.from);
        int to = static_cast<int>(hop.to);
        double phase = kx * hop.displacement[0] + ky * hop.displacement[1];
        std::complex<double> factor = std::exp(1i * phase);
        H(from, to) += hop.amplitude * factor;
        if (from != to) {
            H(to, from) += std::conj(hop.amplitude * factor); // Hermitien
        }
    }

    return H;
}

// === Densité d'états (DOS) pour une énergie donnée par intégration Monte Carlo ===
double DOS_MC(double energy, double gamma, double t, double delta, int Nk)
{
    double dos = 0.0;

    #pragma omp parallel reduction(+:dos)
    {
        std::random_device r;
        std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 gen(seed2);  // Générateur aléatoire propre à chaque thread
        std::uniform_real_distribution<double> dist(0.0, 2 * M_PI); // échantillonner kx, ky dans [0, 2π]

        #pragma omp for schedule(static, 50)
        for (int i = 0; i < Nk; ++i)
        {
            double kx = dist(gen); // échantillonner kx
            double ky = dist(gen); //   échantillonner ky

            // Construction de H(k) pour ce point k
            Eigen::Matrix2cd Hk = hamiltonian_k(kx, ky, t, delta); // Hamiltonien 2x2
            // Diagonalisation de H(k) pour obtenir les valeurs propres 
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es(Hk);
            Eigen::Vector2d eigs = es.eigenvalues(); // deux valeurs propres pour le Hamiltonien 2x2

            // Ajout des contributions Lorentziennes
            for (int j = 0; j < 2; ++j)
            {
                double diff = energy - eigs(j);
                dos += gamma / (diff * diff + gamma * gamma);
            }
        }
    }

    // Normalisation par la densité d’échantillonnage
    dos /= (Nk * M_PI);
    return dos;
}


int main()
{
    // Paramètres physiques
    double t = 1.0;
    double delta = 0.5;
    double gamma = 0.1;
    int Nk = 100000; // nombre de points Monte Carlo
    double Emax = 3.0; // énergie max
    double Emin = -3.0;
    int N_E = 300; // nombre de points énergie

    // Vecteurs pour stocker les temps et les accélérations
    Eigen::VectorXi thread_counts(4);
    thread_counts << 1, 2, 4, 8;
    Eigen::VectorXd times(4);
    Eigen::VectorXd speedups(4);


    for (int idx = 0; idx < thread_counts.size(); ++idx) {
        int threads = thread_counts(idx);
        omp_set_num_threads(threads);

        std::ofstream DosFile("DosGraphen_" + std::to_string(threads) + ".txt");
        if (!DosFile) {
            std::cerr << "Impossible d'ouvrir le fichier de sortie" << std::endl;
            return 1;
        }

        DosFile << N_point << std::endl;
        DosFile << energy << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        // Calcul de la densité d'états (DOS) pour différentes énergies
        for (int i = 0; i <= 2 * energy * 10; ++i) 
        {
            double energy = Emin + i * (Emax - Emin) / N_E;
            double dos = DOS_MC(energy, gamma, Nk, t, delta);
            DosFile << energy << " " << dos << "\n";
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


