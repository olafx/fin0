'''
Price European calls and options via a Monte Carlo simulation of geometric
Brownian motion. This is sort of halfway Monte Carlo, where I'm just sampling
the distribution that follows from Itô calculus.
'''

import numpy as np

S0 = 100
X = 120
sigma = .2
mu = .1
T = 1
N = int(1e6)
call = True

V0 = 0
for i in range(N):
  delta_ln_ST = np.random.randn()*sigma+(mu-sigma**2/2)*T
  ST = S0*np.exp(delta_ln_ST)
  if call: V0 += max(0, (ST-X)*np.exp(-mu*T))/N
  else: V0 += max(0, (X-ST)*np.exp(-mu*T))/N

print(f'{V0=:.3e}')
