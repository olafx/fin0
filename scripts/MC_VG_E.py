'''
A variance-gamma model Monte Carlo simulation to price European options.
'''

import numpy as np

# numerical params
N = 10000 # number of runs
n = 1000 # number of steps per run
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = 0 # dividend yield
nu = .2 # jump
th = .05 # drift
sig = .4 # volatility
# model initial condition
S0 = 100 # initial spot price
# option params
K = 40
style = 'call'

mu_p = .5*(th**2+2*sig**2/nu)**.5+.5*th
mu_q = .5*(th**2+2*sig**2/nu)**.5-.5*th
nu_p = mu_p**2*nu
nu_q = mu_q**2*nu
om = 1/nu*np.log(1-.5*sig**2*nu-th*nu)

V0 = 0
S_mean = 0
for i in range(N):
  a1 = np.random.gamma(T/n*mu_p**2/nu_p, nu_p/mu_p, n)
  a2 = np.random.gamma(T/n*mu_q**2/nu_q, nu_q/mu_q, n)
  x = np.cumsum(a1-a2)
  S = S0*np.exp((r-q+om)*T+x[-1])
  S_mean += S/N
  if style == 'call': V0 += max(0, S-K)*np.exp(-r*T)/N
  elif style == 'put': V0 += max(0, K-S)*np.exp(-r*T)/N

S_mean_pred = S0*np.exp(r*T)
print(f'expected mean {S_mean_pred:.2e}')
print(f'actual mean {S_mean:.2e}')
print(f'option price {V0:.4e}')
