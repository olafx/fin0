'''
Binomial tree model for pricing American options under the Black-Scholes-Merton
model.
'''

import numpy as np

# model params
T = 1 # duration
r = .05 # risk-free interest rate
q = .02 # dividend rate
sig = .2 # volatility
S0 = 90 # initial spot price
# options params
K = 100 # strike price
style = 'put'
# numerical params
N_bt = 4000 # binomial tree depth

assert style in ('call', 'put')

# Tree parameters.
dt = T/N_bt
u = np.exp(sig*dt**.5)
d = np.exp(-sig*dt**.5)
p = (np.exp((r-q)*dt)-d)/(u-d)
# Old and new tree layers.
V0, V1 = np.zeros((2, N_bt+1))
# Set up expiry bottom layer.
n_up = np.arange(N_bt+1)
S = S0*u**n_up*d**(N_bt-n_up)
V1 = np.maximum(0, S-K if style == 'call' else K-S)
# Construct higher layers iteratively.
for i_layer in range(N_bt-1, -1, -1):
# i_layer is both the index and length of the i-th layer, bottom is i=N_bt.
  V0, V1 = V1, V0
  n_up = np.arange(i_layer+1)
  S = S0*u**n_up*d**(i_layer-n_up)
# Attempt early exercise, otherwise hold.
  V_hold = np.exp(-r*dt)*(p*V0[1:]+(1-p)*V0[:-1])
  V_exer = np.maximum(0, S-K if style == 'call' else K-S)
  V1 = np.maximum(V_hold, V_exer)
V0 = V1[0]

print(f'V0 {V0:.4e}')
