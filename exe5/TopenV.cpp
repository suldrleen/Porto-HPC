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
Eigen::MatrixXd H(double t_value, unsigned N, double E) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N); //création de la matrice H de taille N*N
    #pragma omp parallel for schedule(dynamic) // parallélisation de la boucle
    for(int i = 0; i < N; i++) 
    { 
        H(i,i) = -E * i; // potentiel diagonal proportionnel à n V = -E sum (n*|n><n|)
        //connexion pour conditions open
        if (i < N-1)
        {
            H(i, i + 1) = -t_value; // élément supérieur
            H(i+1, i) = -t_value; // élément inférieur
        }

    }
    return H;
}

// Initialisation de phi(n,0)
Eigen::VectorXd phi0(unsigned N, double sigma, double k0) {
    Eigen::VectorXd phi(N);
    double norm = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:norm) // parallélisation de la boucle
    for (int i = 0; i < N; i++) 
    {
        double x = i - N / 2; //centré sur N/2
        phi(i) = exp(-0.5*(x * x )/ (sigma * sigma)) * cos(-k0 * i);
        norm += phi(i) * phi(i); // somme des psi²
    }
    phi /= std::sqrt(norm); // normalisation
    return phi;
}


// Calcul de la fonction d'onde à un instant t
Eigen::VectorXcd time_evolution(const Eigen::VectorXd& phi0, const Eigen::VectorXd& eigenvalues, const Eigen::MatrixXd& eigenvectors, double time, unsigned N) {
    Eigen::VectorXcd phi_t = Eigen::VectorXcd::Zero(N);

    #pragma omp parallel
    {
        Eigen::VectorXcd local_phi_t = Eigen::VectorXcd::Zero(N);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            double projection = eigenvectors.col(i).dot(phi0);
            std::complex<double> phase = std::exp(std::complex<double>(0, -eigenvalues(i) * time));
            local_phi_t += projection * phase * eigenvectors.col(i);
        }

        #pragma omp critical
        phi_t += local_phi_t;
    }

    return phi_t;
}

// fonction pour calculer la moyenne du module en fonction du temps
double calculate_mean(const Eigen::VectorXcd& phi_t) {
    double mean = 0.0;
    unsigned N = phi_t.size();

    for (unsigned n = 0; n < N; ++n) {
        double prob = std::norm(phi_t(n)); // |phi(n,t)|^2
        mean += n * prob;
    }

    return mean;
}


// fonction pour calculer la variance du module de phi_t sur tous les points n
double calculate_variance(const Eigen::VectorXcd& phi_t) {
    double mean = calculate_mean(phi_t); // On réutilise la fonction précédente
    double mean_squared = 0.0;
    unsigned N = phi_t.size();

    for (unsigned n = 0; n < N; ++n) {
        double prob = std::norm(phi_t(n)); // |phi(n,t)|^2
        mean_squared += n * n * prob;
    }

    double variance = mean_squared - mean * mean;
    return variance;
}



int main() 
{
    // Paramètres
    unsigned N = 100; // Nombre de points
    double t = 1; // Hopping value
    double dt = 0.01; // Pas de temps
    double sigma = N/32; // Largeur de la gaussienne
    double k0 = 1.0; // Vecteur d'onde initial
    double E = 1.0; // Champs électrique

    Eigen::MatrixXd H_matrix = H(t, N, E); // Calculer la matrice H

    //Diagonalisation de H
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_matrix);
    Eigen::VectorXd eigenvalues = solver.eigenvalues(); // Les énergies propres
    Eigen::MatrixXd eigenvectors = solver.eigenvectors(); // Les vecteurs propres

    // Initialiser phi(0)
    Eigen::VectorXd phi_0 = phi0(N, sigma, k0);

    //listes de temps d'observation
    std::vector<double> observation_times = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};


    // Temps de simulation
    double T_max = 10.0; // Temps maximum de la simulation

    // Vecteurs pour calculer le speed up selon le nombre de threads
    Eigen::VectorXi thread_count(4);
    thread_count << 1, 2, 4, 8; // Nombre de threads à tester
    Eigen::VectorXd time_taken(4); // Temps pris pour chaque nombre de threads
    Eigen::VectorXd speedup(4); // Vitesse de calcul

    //création d'un file speedup.txt
    std::ofstream speedup_file("speedup.txt");
    if (!speedup_file)
     {
        std::cerr << "Erreur à l'ouverture du fichier de sortie" << std::endl;
        return 1;
    }

    for(int idx = 0; idx < thread_count.size(); idx++) {
        int threads = thread_count(idx);
        // Définir le nombre de threads
        omp_set_num_threads(thread_count(idx));

        // Démarrer le chronomètre
        auto start = std::chrono::high_resolution_clock::now();

        // Ouvrir le fichier une fois avant la boucle de temps
        std::ofstream output_file("evolution_phi_" + std::to_string(thread_count(idx)) + ".txt");
        if (!output_file) 
        {
            std::cerr << "Erreur à l'ouverture du fichier de sortie" << std::endl;
            return 1;
        }

        // Ouvrir le fichier une fois avant la boucle de temps
        std::ofstream mean_file("mean_" + std::to_string(thread_count(idx)) + ".txt");
        if (!mean_file) 
        {
            std::cerr << "Erreur à l'ouverture du fichier de sortie" << std::endl;
            return 1;
        }

        // Calcul de la fonction d'onde à différents instants
        for (double t_curr = 0.0; t_curr <= T_max; t_curr += dt) 
        {
            Eigen::VectorXcd phi_t = time_evolution(phi_0, eigenvalues, eigenvectors, t_curr, N);
            
            // Écriture des résultats dans le fichier
            output_file << t_curr;
            for (int i = 0; i < N; i++) {
                output_file << " " << phi_t(i).real(); // Écrire la partie réelle de phi_t(i)
            }
            output_file << std::endl;
            double mean = calculate_mean(phi_t);
            double variance = calculate_variance(phi_t);

            mean_file << t_curr << " " << mean << " " << variance << std::endl;
        }
        output_file.close(); // Fermer le fichier après la boucle
        mean_file.close();
        // Arrêter le chronomètre
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end - start;

        // Afficher le temps pris
        std::cout << "Temps pris avec " << threads << " threads : " << elapsed_time.count() << " secondes" << std::endl;

        // Enregistrer le temps pris pour ce nombre de threads
        time_taken(idx) = elapsed_time.count();
        speedup(idx) = time_taken(0) / time_taken(idx);
        speedup_file << thread_count(idx) << " " << time_taken(idx) << " " << speedup(idx) << std::endl;
    }
    speedup_file.close(); // Fermer le fichier de vitesse

    std::ofstream modulus_file("modulus_vs_n.txt");
    if (!modulus_file) {
        std::cerr << "Erreur à l'ouverture du fichier modulus_vs_n.txt" << std::endl;
        return 1;
    }
    
    // En-tête : temps
    modulus_file << "# n";
    for (double t_obs : observation_times) {
        modulus_file << " " << t_obs;
    }
    modulus_file << std::endl;
    
    // Calculer une fois pour chaque temps d'observation
    std::vector<Eigen::VectorXcd> all_phi_t;
    for (double t_obs : observation_times) {
        all_phi_t.push_back(time_evolution(phi_0, eigenvalues, eigenvectors, t_obs, N));
    }
    
    // Pour chaque n, écrire tous les modules
    for (unsigned n = 0; n < N; ++n) {
        modulus_file << n;
        for (const auto& phi_t : all_phi_t) {
            modulus_file << " " << std::abs(phi_t(n));
        }
        modulus_file << std::endl;
    }
    
    modulus_file.close();
    


    return 0;
}
