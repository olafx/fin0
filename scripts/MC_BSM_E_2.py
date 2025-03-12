'''
A Black-Scholes-Merton model Monte Carlo simulation to price European options
with importance sampling and antithetic variates.
'''

import numpy as np

# numerical params
N = int(1e5) # number of runs
n = 500 # number of steps per run
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = .02 # dividend rate
sig = .2 # volatility
# model initial condition
S0 = 40 # initial spot price
# option params
K = 70 # strike price
style = 'call'

lam = np.log(S0/K)/(sig*T)+(r-q)/sig-sig/2

dt = T/n
V0s = []
for i in range(N//2):
  S1, S2, W1, W2 = S0, S0, 0, 0
  for j in range(n):
    x1, x2 = np.random.randn(2)
    W1 += x1; W2 += x2
    S1 *= 1+(r-q-sig*lam)*dt+sig*dt**.5*x1
    S2 *= 1+(r-q-sig*lam)*dt+sig*dt**.5*x2
  W1 *= dt**.5; W2 *= dt**.5
  RN1 = np.exp(lam*(W1-.5*lam*T))
  RN2 = np.exp(lam*(W2-.5*lam*T))
  V0s += [RN1*max(0, S1-K if style == 'call' else K-S1),
          RN2*max(0, S2-K if style == 'call' else K-S2)]
V0 = np.mean(V0s)*np.exp(-r*T)
var_V0 = np.var(V0s)*np.exp(-2*r*T)

print(f'lam {lam:.4e}')
print(f'V0 {V0:.4e}')
print(f'var V0 {var_V0:.4e}')
