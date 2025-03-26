'''
Black-Scholes-Merton Monte Carlo simulation, Euler-Maruyama/Milstein.
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
N = 50 # number of runs
n = 500 # number of steps per run
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = .02 # dividend rate
sig = .4 # volatility
S0 = 100 # initial spot price

S = np.zeros((N, n))
S[:,0] = S0
for j in range(N):
  for i in range(1, n):
    x = np.random.normal()
    S[j,i] = S[j,i-1]*(1+(r-q)*T/n+sig*(T/n)**.5*x)

ts = np.linspace(0, T, n)
plt.plot(ts, S.T)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$S_t$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'MC_BSM.png', bbox_inches='tight', dpi=400)
else: plt.show()
