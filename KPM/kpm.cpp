
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include <random>
#include <numeric>
#include <unistd.h>

struct kpm 
{
  const unsigned L;//taille du systèmeme
  const double W;//amplitude de la perturbation/désordre
  const double EnergyScale; //normaliser energie entre -1 et 1
  const double t; //taux de couplage entre les sites
  unsigned index; //indice courant les vecteurs de chebyshev
  double left_send, right_send;
  double left_recv, right_recv;
  int left_neighbour, right_neighbour; // voisins gauche et droite pour parallélisation
  

  Eigen::Array<double, -1, -1> v; //vecteurs de chebyshev
  Eigen::Array<double, -1, -1> phi0; //vecteur initial
  Eigen::Array<double, -1, 1> U; //potentiel aléatoire
  std::mt19937 rnd; //gnénérateur de nombres aléatoires
  std::uniform_real_distribution<> dist; //distribution uniforme entre -1 et 1

  kpm (unsigned ll, double jj, double w) : L(ll),  W(w), EnergyScale( (2 * jj + W/2)*1.0001 ), t(jj/EnergyScale), v(L, 2), phi0(1, L), U(L) 
  { //initialise taille, energie, potentiel aléatoire etc, crée vecteur de taille, calcule une échelle d'énergie pour normaliser et initiale un gen aléatoire entropique
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    rnd.seed(seed);	
  };
  
  void initialize() 
  {
    index = 0;
    for(unsigned i = 0; i < L; i++)
      {
	v(i,index) = (dist(rnd) - 0.5) * 2 * sqrt(3); //initiake v avec valeur aléatoire
	phi0(0,i) = v(i,index);  //vecteur initial
	U(i) = W * (dist(rnd) - 0.5)/EnergyScale; //potentiel aléatoire normalisé
      }  
  };
  
  void hamiltonian()
  {
    index = 1;
    for(unsigned i = 0; i < L; i++)
    {
      // left recv et right recv sont les valeurs reçues des voisins gauche et droit, définis dans main
      // si i == 0 (1er élément), on utilise left_recv, sinon on prend l'élément précédent qui est connu de ce thread
      double left = (i == 0) ? left_recv : v(i-1, 0);
      // si i == L - 1 (dernier élément), on utilise right_recv, sinon on prend l'élément suivant qui est connu de ce thread
      double right = (i == L - 1) ? right_recv : v(i+1, 0);

      v(i, 1) = -t * (left + right) + U(i) * v(i, 0); //calcul de Hphi0

    }


    };
 

 

  void kpm_iteration() //v(n+1) = 2Hv(n)-v(n)
  {
    index++;
    unsigned i0 = (index)%2;
    unsigned i1 = (index - 1)%2;
    for(unsigned i = 0; i < L; i++)
    {
      double left = (i == 0) ? left_recv : v(i-1, i1);
      double right = (i == L - 1) ? right_recv : v(i+1, i1);

      v(i, i0) = -2 * t * (left + right) + 2 * U(i) * v(i, i1) - v(i, (index - 2) % 2); //calcul de Tn+1 = 2Hv(n)-v(n-1)

    }
  }
};

int main ()
{
  unsigned NAverages = 1; //vecteur initial nbr
  unsigned NMoments = 1024; //nombre de moments
  //unsigned L = 16384*64; //taille du système
  unsigned L_global = 16384*64; //taille du système
  unsigned nthreads = omp_get_max_threads(); //nombre de threads

  //définition des buffers en dehors de omp

  std::vector<double> left_boundary(omp_get_max_threads());
  std::vector<double> right_boundary(omp_get_max_threads());

  std::vector<Eigen::ArrayXd> mu_thread(nthreads, Eigen::ArrayXd::Zero(NMoments));

  #pragma omp parallel 
  {
    int thread_id = omp_get_thread_num();  // récupère l'identifiant du thread
    unsigned L_local = L_global / nthreads; //taille locale pour chaque thread
    int num_threads = omp_get_num_threads();


    kpm v(L_local, 1.0, 1.0); //initialisation pour chaque thread

    v.left_neighbour = (thread_id == 0) ? -1 : thread_id - 1; // voisin gauche
    v.right_neighbour = (thread_id == num_threads - 1) ? -1 : thread_id + 1; // voisin droit

    Eigen::ArrayXd& mu_local = mu_thread[thread_id];

    for (unsigned av = 0; av < NAverages; av++)
    { 
      mu_local.setZero();
      v.initialize();

      left_boundary[thread_id] = v.v(0, 0); // envoie le premier élément de v à gauche
      right_boundary[thread_id] = v.v(L_local - 1, 0); // envoie le dernier élément de v à droite

      #pragma omp barrier // synchronisation pour s'assurer que tous les threads ont mis à jour leurs frontières
      
      // Récupération des frontières des voisins
      // si le thread n'est pas le premier, il reçoit la frontière gauche de son voisin gauche
      if (v.left_neighbour != -1)
        v.left_recv = right_boundary[v.left_neighbour];
      else
        v.left_recv = 0.0;

      //si le thread n'est pas le dernier, il reçoit la frontière droite de son voisin droit
      if (v.right_neighbour != -1)
        v.right_recv = left_boundary[v.right_neighbour];
      else
      v.right_recv = 0.0;


      v.hamiltonian();

      for(unsigned m = 2; m < NMoments ; m += 2)
        {
          for(unsigned j = 0; j < 2; j++) 
          {
            v.left_send = v.v(0,j);
            v.right_send = v.v(L_local - 1, j); //envoie les bords gauche et droit pour la prochaine itération

            #pragma omp barrier // synchronisation pour s'assurer que tous les threads ont mis à jour leurs frontières

            if (v.left_neighbour != -1) 
            {
              left_boundary[v.left_neighbour] = v.left_send; // met à jour la frontière gauche du voisin
            }

            if (v.right_neighbour != -1) 
            {
              right_boundary[v.right_neighbour] = v.right_send; // met à jour la frontière droite du voisin

            }     
            
            if (v.left_neighbour != -1)
            v.left_recv = right_boundary[v.left_neighbour];
            else
            v.left_recv = 0.0;
    
            //si le thread n'est pas le dernier, il reçoit la frontière droite de son voisin droit
            if (v.right_neighbour != -1)
            v.right_recv = left_boundary[v.right_neighbour];
            else
            v.right_recv = 0.0;




            v.kpm_iteration();
            mu_local.segment(m, 2) += ( (v.phi0.matrix() * v.v.matrix()).array().transpose() );
            

          }

      
    }
    

  }

  mu_local /= NAverages;

  }

  Eigen::ArrayXd mu = Eigen::ArrayXd::Zero(NMoments);
  for (int i = 0; i < nthreads; ++i)
    mu += mu_thread[i];

  mu /= nthreads;



  std::ofstream DosFile(std::string("DosKPM") + ".txt");

  DosFile << mu/double(L_global) << std::endl;

  DosFile.close();


  return 0;

}