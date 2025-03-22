'''
Cox-Ingersoll-Ross Monte Carlo simulation, Euler-Maruyama with reflection.
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
n = 500 # number of steps per run
# model params
T = 3 # duration
eta = .4 # level of mean reversion
kap = 1 # spread of mean reversion
th = .2 # vol-of-vol (vol-of-var)
v0 = .2 # initial volatility

dt = T/n
v = np.zeros((N, n))
v[:,0] = v0
n_reflection = 0
for j in range(N):
  for i in range(1, n):
    x = np.random.normal()
    v[j,i] = v[j,i-1]+kap*(eta-v[j,i-1])*dt+th*(v[j,i-1]*dt)**.5*x
    if v[j,i] < 0:
      v[j,i] *= -1
      n_reflection += 1

Feller_cond = 2*kap*eta >= th**2
print(f'Feller condition{'' if Feller_cond else ' not'} satisfied')
print(f'vol < 0 rate {n_reflection/(n*N):.2e}')
ts = np.linspace(0, T, n)
plt.plot(ts, v.T)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$\nu_t$')
plt.tight_layout()
plt.savefig(Path.cwd().parent/'out'/'MC_CIR.png', bbox_inches='tight', dpi=400)
