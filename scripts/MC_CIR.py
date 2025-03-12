'''
A Cox-Ingersoll-Ross Monte Carlo simulation.
'''

import numpy as np
import matplotlib.pyplot as plt

# numerical params
N = 32 # number of runs
n = 500 # number of steps per run
# model params
T = 3 # duration
eta = .4 # level of mean reversion
kap = 1 # spread of mean reversion
th = .2 # vol-of-vol
# model initial condition
v0 = .2 # initial volatility

print(f'Feller condition:{"" if 2*kap*eta >= th**2 else " not"} satisfied')

v = np.zeros((N, n))
v[:,0] = v0

k = 0
for j in range(N):
  for i in range(1, n):
    x = np.random.randn()
    v[j,i] = v[j,i-1]+kap*(eta-v[j,i-1])*T/n+th*(v[j,i-1]*T/n)**.5*x
    if v[j,i] < 0:
      v[j,i] *= -1
      k += 1

print(f'negative volatility rate: {k/(n*N):.2f}')

ts = np.linspace(0, T, n)
plt.figure('Cox-Ingersoll-Ross')
for i in range(N): plt.plot(ts, v[i])
plt.xlim(0, T)
plt.xlabel('$t$')
plt.ylabel('$\\nu_t$')
plt.tight_layout()
plt.show()
