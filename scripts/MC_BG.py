'''
Bilateral gamma Monte Carlo simulation.
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
N = 100 # number of runs
n = 1000 # number of steps per run
# model params
T = 1 # duration
r = 0.06 # risk free interest rate
q = 0.02 # dividend yield
al1 = 1.18 # alpha^+
lam1 = 10.57 # lambda^+
al2 = 1.44 # alpha^-
lam2 = 5.57 # lambda^-
S0 = 100 # initial spot price

dt = T/n
xi = -al1*np.log(lam1/(lam1-1))-al2*np.log(lam2/(lam2+1))
gam1 = np.cumsum(np.random.gamma(dt*al1, 1/lam1, (N, n)), axis=-1)
gam2 = np.cumsum(np.random.gamma(dt*al2, 1/lam2, (N, n)), axis=-1)
x = gam1-gam2
t = np.linspace(0, T, n)
S = S0*np.exp((r-q+xi)*t+x)

plt.figure(1)
plt.plot(t, x.T)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$X_t$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd().parent/'out'/'MC_BG_1.png', bbox_inches='tight', dpi=400)
plt.figure(2)
plt.xlim(0, T)
plt.plot(t, S.T)
plt.xlabel(R'$t$')
plt.ylabel(R'$S_t$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd().parent/'out'/'MC_BG_2.png', bbox_inches='tight', dpi=400)
else: plt.show()
