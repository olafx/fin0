'''
Binomial tree model for pricing European options under the Black-Scholes-Merton
model.
'''

import numpy as np

# model params
T = 3 # duration
r = .05 # risk-free interest rate
q = .0 # dividend rate
sig = .2 # volatility
# model initial conditions
S0 = 100
# options params
K = 130 # strike price
style = 'put'
# numerical params
N = 1000 # binomial tree depth

assert style in ('call', 'put')

dt = T/N
u = np.exp(sig*dt**.5)
d = np.exp(-sig*dt**.5)
p = (np.exp((r-q)*dt)-d)/(u-d)
q = 1-p

def gen_tree(n):
  S = [S0*u**(n-i)*d**i for i in range(n+1)]
  if n == N:
    V = [max(S[i]-K, 0) if style == 'call' else max(K-S[i], 0) for i in range(n+1)]
    return S, V
  return S

S_old, V_old = gen_tree(N)
for i in range(N-1, -1, -1):
  S_new = gen_tree(i)
  V_new = [(p*V_old[j]+q*V_old[j+1])*np.exp(-r*dt) for j in range(i+1)]
  S_old, V_old = S_new, V_new
V0 = V_new[0]

print(style)
print(f'V0 {V0:.4e}')
