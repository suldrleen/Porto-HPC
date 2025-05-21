import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("modulus_vs_n.txt")
n = data[:,0]
for i in range(1, data.shape[1]):
    plt.plot(n, data[:,i], label=f't={i-1} steps')

plt.xlabel('n')
plt.ylabel('|phi(n,t)|')
plt.legend()
plt.show()
