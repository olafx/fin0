'''
A variance-gamma model Monte Carlo simulation to price European options.
'''

import numpy as np

# numerical params
N = int(4e4) # number of runs
n = 1000 # number of steps per run
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = 0 # dividend yield
nu = .2 # jump
th = .05 # drift
sig = .4 # volatility
S0 = 100 # initial spot price
# option params
K = 40 # strike price
style = 'call' # call or put

assert style in ('call', 'put')

mu_p = .5*(th**2+2*sig**2/nu)**.5+.5*th
mu_q = .5*(th**2+2*sig**2/nu)**.5-.5*th
nu_p = mu_p**2*nu
nu_q = mu_q**2*nu
om = 1/nu*np.log(1-.5*sig**2*nu-th*nu)

V0s = []
for i in range(N):
  gam1 = np.random.gamma(T/n*mu_p**2/nu_p, nu_p/mu_p, n)
  gam2 = np.random.gamma(T/n*mu_q**2/nu_q, nu_q/mu_q, n)
  x = np.cumsum(gam1-gam2)
  S = S0*np.exp((r-q+om)*T+x[-1])
  V0s += [max(0, S-K if style == 'call' else K-S)]
V0 = np.mean(V0s)*np.exp(-r*T)
var_V0 = np.var(V0s)*np.exp(-2*r*T)
se_V0 = (var_V0/N)**.5

print(f'V0 {V0:.4e} (s.e. {se_V0:.4e})')
