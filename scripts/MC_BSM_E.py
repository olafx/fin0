'''
A Black-Scholes-Merton Monte Carlo simulation to price European options.
'''

import numpy as np

# numerical params
N = 40000 # number of runs
n = 500 # number of steps per run
# model params
T = 3 # duration
r = 0.06 # risk free interest rate
q = 0.02 # dividend rate
sigma = .2 # volatility
# model initial condition
S0 = 100 # initial spot price
# option params
K = 70 # strike price # 40
style = 'call'

V0 = 0
for j in range(N):
  S = S0
  for i in range(1, n):
    S *= 1+(r-q)*T/n+sigma*(T/n)**.5*np.random.randn()
  if style == 'call': V0 += max(0, (S-K))*np.exp(-r*T)/N
  elif style == 'put': V0 += max(0, (K-S))*np.exp(-r*T)/N

print(f'V0 {V0:.4e}')
