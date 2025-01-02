'''
Price European calls and options via a Monte Carlo simulation of geometric
Brownian motion. This is a full Monte Carlo simulation, using nothing but the
geometric Brownian motion SDE.
'''

import numpy as np

S0 = 100
X = 120
sigma = .2
mu = .1
T = 1
n = int(1e2)
N = int(1e5)
call = True

V0 = 0
for i in range(N):
  ST = S0
  for j in range(n):
    ST += ST*mu*T/n+ST*sigma*np.random.randn()*(T/n)**.5
  if call: V0 += max(0, (ST-X)*np.exp(-mu*T))/N
  else: V0 += max(0, (X-ST)*np.exp(-mu*T))/N

print(f'{V0=:.3e}')
