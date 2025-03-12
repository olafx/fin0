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
# option params
K = 70 # strike price # 40
style = 'call'

dt = T/n
V0s = []
for i in range(N):
  S = S0
  for j in range(n):
    x = np.random.randn()
    S *= 1+(r-q)*dt+sig*dt**.5*x
  V0s += [max(0, S-K if style == 'call' else K-S)]
V0 = np.mean(V0s)*np.exp(-r*T)
var_V0 = np.var(V0s)*np.exp(-2*r*T)

print(f'V0 {V0:.4e}')
print(f'var V0 {var_V0:.4e}')
