'''
An efficient Black-Scholes-Merton Monte Carlo simulation to price European
options. This introduces a fictitious interest rate lambda, which may be
positive or negative, so that the option on average expires at the money. This
improves efficiency since more samples should be taken where the derivative in
the price according to all parameters involved (model and option) is high, which
for European options is in the neighborhood of the strike price.
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

lam = np.log(K/S0)/(sig*T)-r/sig+sig/2

dt = T/n
V0 = 0
for j in range(N):
# Calculating the entire path in case it is needed, more general this way.
  dWt = np.random.normal(size=n)*dt**.5
  WT = np.sum(dWt)
  ST = S0*np.exp((r-q-sig**2/2+sig*lam)*T+sig*WT)
  weight = np.exp(-lam*WT-lam**2*T/2)
  V0 += max(0, ST-K if style == 'call' else K-ST)*weight
V0 *= np.exp(-r*T)/N

print(f'lam {lam:.4e}')
print(f'V0 {V0:.4e}')
