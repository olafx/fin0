'''
Heston Monte Carlo simulation, Milstein.
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

dt = T/n
S, v = np.zeros((2, N, n))
S[:,0] = S0
v[:,0] = sig0**2
n_reflection = 0
for j in range(N):
  for i in range(1, n):
    x1, x2 = np.random.normal(size=2)
    x3 = rho*x1+(1-rho**2)**.5*x2
    S[j,i] = S[j,i-1]*(1+(r-q)*dt+(v[j,i-1]*dt)**.5*x1)
    v[j,i] = v[j,i-1]+kap*(eta-v[j,i-1])*dt+th*(v[j,i-1]*dt)**.5*x3+.25*th**2*(x3**2-1)*dt
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
if plot_save: plt.savefig(Path.cwd()/'out'/'MC_Heston_1.png', bbox_inches='tight', dpi=400)
plt.figure(2)
plt.plot(ts, v.T**.5)
plt.xlim(0, T)
plt.xlabel(R'$t$')
plt.ylabel(R'$\sqrt{\nu_t}$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'MC_Heston_2.png', bbox_inches='tight', dpi=400)
else: plt.show()
