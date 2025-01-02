'''
Three different Poisson point process Monte Carlo simulations.
1: all samples in 1 step
2: a number of steps with a number of samples in each
3: many steps, most of them with 0 samples, some with 1
'''

import numpy as np
import matplotlib.pyplot as plt

lam = 10 # Poisson point process rate
T = 50 # duration
m = 8 # iterations per method

# 1st method
n1 = 1000
N1 = np.random.poisson(lam*T, m)
t1 = [np.sort(np.random.uniform(size=N1_)*T) for N1_ in N1]
x1 = [np.searchsorted(t1_, np.linspace(0, T, n1)) for t1_ in t1]

# 2nd method
n2 = 1000
x2 = np.cumsum(np.random.poisson(lam*T/n2, (m, n2)), axis=-1)

# 3rd method
n3 = 100000
x3 = np.cumsum(np.random.uniform(size=(m, n3)) < lam*T/n3, axis=-1)

plt.figure(1)
t1 = np.linspace(0, T, n1)
t2 = np.linspace(0, T, n2)
t3 = np.linspace(0, T, n3)
for i in range(m): plt.plot(t1, x1[i], c='#FF0000')
plt.plot(t2, x2.T, c='#00FF00')
plt.plot(t3, x3.T, c='#0000FF')
plt.xlim(0, T)
plt.tight_layout()
plt.show()
