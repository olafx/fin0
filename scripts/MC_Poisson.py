'''
Three different Poisson point process Monte Carlo simulations.
1: all samples in 1 step, placed uniformly
2: a number of steps with a number of samples in each
3: many steps, most of them with 0 samples, some with 1
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs

makedirs(Path.cwd().parent/'out', exist_ok=True)
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['text.usetex'] = True

# plot params
plot_save = False
# numerical params
N = 8 # runs per method
n1 = 1000 # number of steps per run for 1st method
n2 = 1000 # number of steps per run for 2nd method
n3 = 100000 # number of steps per run for 3rd method
# process params
T = 10 # duration
lam = 10 # Poisson point process rate

N1 = np.random.poisson(lam*T, N)
t1 = [np.sort(np.random.uniform(size=N1_)*T) for N1_ in N1]
x1 = [np.searchsorted(t1_, np.linspace(0, T, n1)) for t1_ in t1]

x2 = np.cumsum(np.random.poisson(lam*T/n2, (N, n2)), axis=-1)

x3 = np.cumsum(np.random.uniform(size=(N, n3)) < lam*T/n3, axis=-1)

t1 = np.linspace(0, T, n1)
t2 = np.linspace(0, T, n2)
t3 = np.linspace(0, T, n3)
for x1_ in x1: plt.plot(t1, x1_, c='red', alpha=.5)
plt.plot(t2, x2.T, c='green', alpha=.5)
plt.plot(t3, x3.T, c='blue', alpha=.5)
plt.plot([], c='red', label='method 1', alpha=.5)
plt.plot([], c='green', label='method 2', alpha=.5)
plt.plot([], c='blue', label='method 3', alpha=.5)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$N_t$')
plt.legend()
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'MC_Poisson.png', bbox_inches='tight', dpi=400)
else: plt.show()
