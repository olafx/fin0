'''
Black-Scholes-Merton Monte Carlo simulation to price European options.
'''

import numpy as np

# numerical params
N = int(4e5) # number of runs
n = 100 # number of steps per run
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = .02 # dividend rate
sig = .2 # volatility
S0 = 40 # initial spot price
# option params
K = 70 # strike price
style = 'call' # call or put

assert style in ('call', 'put')

dt = T/n
V0s = []
for i in range(N):
  S = S0
  for j in range(n):
    x = np.random.normal()
    S *= 1+(r-q)*dt+sig*dt**.5*x
  V0s += [max(0, S-K if style == 'call' else K-S)]
V0 = np.mean(V0s)*np.exp(-r*T)
var_V0 = np.var(V0s)*np.exp(-2*r*T)
se_V0 = (var_V0/N)**.5

print(f'V0 {V0:.4e} (s.e. {se_V0:.4e})')
