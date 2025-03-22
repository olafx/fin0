'''
Bilateral gamma Monte Carlo simulation to price European options.
'''

import numpy as np

# numerical params
N = int(4e5) # number of runs
n = 1000 # number of steps per run
# model params
T = 1 # duration
r = 0.06 # risk free interest rate
q = 0.03 # dividend yield
al1 = 1.18 # alpha^+
lam1 = 11 # lambda^+
al2 = 1.44 # alpha^-
lam2 = 6 # lambda^-
S0 = 100 # initial spot price
# model params adjusted under new measure
lam1_ = 12 # lambda^+
lam2_ = 4 # lambda^-
# option params
K = 80 # strike price
style = 'put' # call or put

assert style in ('call', 'put')

dt = T/n
xi = -al1*np.log(lam1/(lam1-1))-al2*np.log(lam2/(lam2+1))
V0s = []
for i in range(N):
  gam1 = np.random.gamma(dt*al1, 1/lam1_, n)
  gam2 = np.random.gamma(dt*al2, 1/lam2_, n)
  gam1T = np.sum(gam1)
  gam2T = np.sum(gam2)
  xT = gam1T-gam2T
  S = S0*np.exp((r-q+xi)*T+xT)
  RN = (lam1/lam1_)**(al1*T)*(lam2/lam2_)**(al2*T)\
    *np.exp(-(lam1-lam1_)*gam1T-(lam2-lam2_)*gam2T)
  V0s += [RN*max(0, S-K if style == 'call' else K-S)]
V0 = np.mean(V0s)*np.exp(-r*T)
var_V0 = np.var(V0s)*np.exp(-2*r*T)
se_V0 = (var_V0/N)**.5

print(f'V0 {V0:.4e}')
print(f'se V0 {se_V0:.4e}')
