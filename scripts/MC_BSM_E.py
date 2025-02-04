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
sig = .2 # volatility
# model initial condition
S0 = 30 # initial spot price
# TODO: high efficiency: S0=100, low efficiency: S0=30
# option params
K = 70 # strike price # 40
style = 'call'

dt = T/n
V0 = 0
for j in range(N):
# Calculating the entire path in case it is needed, more general this way.
  dWt = np.random.normal(size=n)*dt**.5
  WT = np.sum(dWt)
  ST = S0*np.exp((r-q-sig**2/2)*T+sig*WT) # mine
  V0 += max(0, ST-K if style == 'call' else K-ST)
V0 *= np.exp(-r*T)/N

print(f'V0 {V0:.4e}')
