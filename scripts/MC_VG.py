'''
Variance gamma model Monte Carlo simulation.
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
N = 32 # number of runs
n = 1000 # number of steps per run
# model params
T = 3 # duration
r = .05 # risk free interest rate
q = .02 # dividend yield
nu = .2 # jump
th = .05 # drift
sig = .4 # volatility
S0 = 100 # initial spot price

mu_p = .5*(th**2+2*sig**2/nu)**.5+.5*th
mu_q = .5*(th**2+2*sig**2/nu)**.5-.5*th
nu_p = mu_p**2*nu
nu_q = mu_q**2*nu
om = 1/nu*np.log(1-.5*sig**2*nu-th*nu)

gam1 = np.cumsum(np.random.gamma(T/n*mu_p**2/nu_p, nu_p/mu_p, (N, n)), axis=-1)
gam2 = np.cumsum(np.random.gamma(T/n*mu_q**2/nu_q, nu_q/mu_q, (N, n)), axis=-1)
x = gam1-gam2
t = np.linspace(0, T, n)
S = S0*np.exp((r-q+om)*t+x)

plt.figure(1)
plt.xlim(0, T)
plt.plot(t, x.T)
plt.xlabel(R'$t$')
plt.ylabel(R'$X_t$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'MC_VG_1.png', bbox_inches='tight', dpi=400)
plt.figure(2)
plt.plot(t, S.T)
plt.xlabel(R'$t$')
plt.ylabel(R'$S_t$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'MC_VG_2.png', bbox_inches='tight', dpi=400)
else: plt.show()
