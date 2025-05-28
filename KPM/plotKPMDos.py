import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# === 1. Lecture du fichier ===


KPMcoef = np.loadtxt('DosKPM.txt')



#        double diff = energy - eigenvalues(i);
#        dos += gamma / (diff * diff + gamma * gamma);

def approx(coef, x):
    N = len(coef)
    Tn = np.zeros((len(x), 2))
    Tn[:, 0] = 1
    if N > 1:
        Tn[:, 1] = x

    result = coef[0]*Tn[:, 0] 
    if N>1 :
        result += coef[1]*Tn[:, 1]

    for i in range(2, N):
        Tn[:, i%2 ] = 2*x*Tn[:, (i-1)%2] - Tn[:, (i-2)%2]
        result += coef[i] *Tn[:, i%2 ]
    return result

def triangle_filter(N):
    n = np.arange(N)
    g = (2/np.pi) * (np.cos(n * (np.pi/2))* (1 - n/N))
    g[0] /= 2
    return g


def Dos(size, KPMcoef, gamma):
    N = len(KPMcoef)
    g = triangle_filter(N)
    coef_filtered = KPMcoef * g

    return approx(coef_filtered, E) / (np.pi * gamma)


gamma = 0.1
E = np.linspace(-1, 1, 1024)
dos_values = abs(Dos(KPMcoef, E, gamma))



plt.plot(E, dos_values)
plt.xlabel('Energy (E)')
plt.ylabel('DOS')
plt.title('DOS KMP')
plt.xlim(-0.2, 0.2)
plt.show()