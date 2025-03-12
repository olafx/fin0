'''
A Heston Monte Carlo simulation.
'''

import numpy as np
import matplotlib.pyplot as plt

# numerical params
N = 32 # number of runs
n = 500 # number of steps per run
# model params
T = 3 # duration
r = 0.06 # risk free interest rate
q = 0.02 # dividend rate
eta = .4 # level of mean reversion
kappa = 1 # spread of mean reversion
theta = .2 # vol-of-vol
rho = -.3 # vol-stock correlation
# model initial condition
S0 = 100 # initial spot price
sig0 = .2 # initial volatility

Feller_cond = 2*kappa*eta >= theta**2
print(f'Feller condition:{'' if Feller_cond else ' not'} satisfied')

S, v = np.zeros((2, N, n))
S[:,0] = S0
v[:,0] = sig0**2

k = 0
for j in range(N):
  for i in range(1, n):
    xi, zeta = np.random.randn(2)
    omega = rho*xi+(1-rho**2)**.5*zeta
    S[j,i] = S[j,i-1]*(1+(r-q)*T/n+(v[j,i-1]*T/n)**.5*xi)
    v[j,i] = v[j,i-1]+kappa*(eta-v[j,i-1])*T/n+theta*(v[j,i-1]*T/n)**.5*omega
    if v[j,i] < 0: # reflection
      v[j,i] *= -1
      k += 1

print(f'negative volatility rate: {k/(n*N):.4f}')

ts = np.linspace(0, T, n)
plt.figure('Heston')
for i in range(N): plt.plot(ts, S[i])
plt.xlim(0, T)
plt.xlabel('$t$')
plt.ylabel('$S_t$')
plt.tight_layout()
plt.show()
