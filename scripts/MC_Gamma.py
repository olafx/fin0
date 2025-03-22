'''
Gamma process Monte Carlo simulation.
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs

makedirs(Path.cwd().parent/'out', exist_ok=True)
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['text.usetex'] = True

# numerical params
N = 32 # number of runs
n = 1000 # number of steps per run
# process params
T = 10 # duration
alp = 2 # shape
th = 3 # scale

dt = T/n
x = np.cumsum(np.random.gamma(shape=alp*dt, scale=th, size=(N, n)), axis=-1)

t = np.linspace(0, T, n)
plt.plot(t, x.T)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$\Gamma_t$')
plt.tight_layout()
plt.savefig(Path.cwd().parent/'out'/'MC_Gamma.png', bbox_inches='tight', dpi=400)
