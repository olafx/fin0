'''
The Black-Scholes PDE, to price European calls and puts.

FTCS PDE integration scheme is used, but backward Euler naturally.
A Neumann boundary condition is used, fixing the values to those of the
initial condition at expiry.
'''

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# numerical params
n_S = 1001 # number of spot prices (should be odd ideally, so that it contains S0)
n_t = 300000 # number of time steps
a = 3 # scale factor for S range
# model params
T = 3 # duration
r = .05 # risk free interest rate
sigma = .4 # volatility
# model initial condition
S0 = 100 # initial spot price
# option params
K = 70 # strike price
style = 'call' # call or put

t = np.linspace(0, T, n_t)
dt = T/(n_t-1)

S = S0+np.linspace(-1, 1, n_S)*S0*sigma*T*a
dS = 2/(n_S-1)*S0*sigma*T*a
V = np.maximum(0, (S-K) if style == 'call' else (K-S))

plt.figure('Black-Scholes PDE')
plt.plot(S, V, c='black')

# Neumann boundary condition: fix Gamma to 0, and Delta to 0, -1, or 1.
for i in range(n_t):
  Delta = (V[2:]-V[:-2])/(2*dS)
  Gamma = (V[2:]-2*V[1:-1]+V[:-2])/dS
  if style == 'call':
    V[0] -= dt*(r*V[0]) # Delta 0
    V[-1] -= dt*(-r*S[-1]+r*V[-1]) # Delta 1
  else:
    V[0] -= dt*(r*S[0]+r*V[0]) # Delta -1
    V[-1] -= dt*(r*V[-1]) # Delta 0
  V[1:-1] -= dt*(-sigma**2*S[1:-1]**2/2*Gamma-r*S[1:-1]*Delta+r*V[1:-1])

print(f'V0 {interpolate.CubicSpline(S, V)(S0):.4e}')
plt.xlim(S[0], S[-1])
# plt.xlim(0, 150)
# plt.ylim(0, 100)
plt.plot(S, V, c='red')
plt.tight_layout()
plt.show()
