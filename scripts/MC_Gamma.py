'''
A Gamma process Monte Carlo simulation.
'''

import numpy as np
import matplotlib.pyplot as plt

# numerical params
N = 8 # number of runs
n = 1000 # number of steps per run
# process params
T = 50 # duration
lam = 10 # Gamma process rate
gam = 1 # Gamma process shape

x = np.cumsum(np.random.gamma(T/n*gam**2/lam, lam/gam, (N, n)), axis=-1)

plt.figure(1)
t = np.linspace(0, T, n)
plt.plot(t, x.T)
plt.xlim(0, T)
plt.tight_layout()
plt.show()
