'''
Heston Monte Carlo simulation, Euler-Maruyama.
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
r = .05 # risk free interest rate
q = .02 # dividend rate
eta = .04 # level of mean reversion
kap = 1 # spread of mean reversion
th = .15 # vol-of-vol (vol-of-var)
rho = -.6 # vol-stock correlation
S0 = 120 # initial spot price
sig0 = .3 # initial volatility

S, v = np.zeros((2, N, n))
S[:,0] = S0
v[:,0] = sig0**2
n_reflection = 0
for j in range(N):
  for i in range(1, n):
    xi, zeta = np.random.randn(2)
    omega = rho*xi+(1-rho**2)**.5*zeta
    S[j,i] = S[j,i-1]*(1+(r-q)*T/n+(v[j,i-1]*T/n)**.5*xi)
    v[j,i] = v[j,i-1]+kap*(eta-v[j,i-1])*T/n+th*(v[j,i-1]*T/n)**.5*omega
    if v[j,i] < 0:
      v[j,i] *= -1
      n_reflection += 1

Feller_cond = 2*kap*eta >= th**2
print(f'Feller condition:{'' if Feller_cond else ' not'} satisfied')
print(f'vol < 0 rate {n_reflection/(n*N):.2e}')

ts = np.linspace(0, T, n)
plt.figure(1)
plt.plot(ts, S.T)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$S_t$')
plt.tight_layout()
plt.savefig(Path.cwd().parent/'out'/'MC_Heston_1.png', bbox_inches='tight', dpi=400)
plt.figure(2)
plt.plot(ts, v.T**.5)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$\sqrt{\nu_t}$')
plt.tight_layout()
plt.savefig(Path.cwd().parent/'out'/'MC_Heston_2.png', bbox_inches='tight', dpi=400)
