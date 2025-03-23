'''
Heston model Monte Carlo simulation to price European options, with importance
sampling.
'''

import numpy as np

# numerical params
N = int(4e4) # number of runs
n = 200 # number of steps per run
# model params
T = 1 # duration
r = .05 # risk free interest rate
q = .02 # dividend rate
eta = .04 # level of mean reversion
kap = 1 # spread of mean reversion
th = .1 # vol-of-vol (vol-of-var)
rho = -.3 # vol-stock correlation
S0 = 120 # initial spot price
sig0 = .2 # initial volatility
# option params
K = 100 # strike price
style = 'call' # call or put

assert style in ('call', 'put')

lam = -.7

dt = T/n
V0s = []
n_reflection = 0
for i in range(N):
  S, v, W1, W2 = S0, sig0**2, 0, 0
  for j in range(n):
    x1, x2 = np.random.normal(size=2)
    W1 += x1; W2 += x2
    x3 = rho*x1+(1-rho**2)**.5*x2
    S *= 1+(r-q-lam*v**.5)*dt+(v*dt)**.5*x1
    v += kap*(eta-v)*dt+th*(v*dt)**.5*x3+.25*th**2*(x3**2-1)*dt
    if v < 0:
      v *= -1
      n_reflection += 1
  W1 *= dt**.5; W2 *= dt**.5
  RN = np.exp(lam*W1-lam*rho/(1-rho**2)**.5*W2-lam**2*T/2/(1-rho**2))
  V0s += [RN*max(0, S-K if style == 'call' else K-S)]
V0 = np.mean(V0s)*np.exp(-r*T)
var_V0 = np.var(V0s)*np.exp(-2*r*T)
se_V0 = (var_V0/N)**.5

Feller_cond = 2*kap*eta >= th**2
print(f'Feller condition{'' if 2*kap*eta >= th**2 else ' not'} satisfied')
print(f'vol < 0 rate {n_reflection/(n*N):.2e}')
print(f'lam {lam:+.4e}')
print(f'V0 {V0:.4e} (s.e. {se_V0:.4e})')
