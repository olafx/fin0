'''
The Black-Scholes PDE, to price European calls and puts.

FTCS PDE integration scheme is used, but backward Euler naturally.
A Neumann boundary condition is used, fixing the values to those of the
initial condition at expiry. Von Neumann stability analysis is performed.
'''

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs

makedirs(Path.cwd().parent/'out', exist_ok=True)
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['text.usetex'] = True

# plot params
plot_save = False
# numerical params
n_S = 301 # number of spot prices (should be odd ideally, so that it contains S0)
n_t = 100000 # number of time steps
a = 3 # scale factor for S range
# model params
T = 3 # duration
r = .05 # risk free interest rate
sig = .4 # volatility
S0 = 100 # initial spot price
# option params
K = 70 # strike price
style = 'call' # call or put

assert style in ('call', 'put')

dt = T/(n_t-1)
t = np.linspace(0, T, n_t)
S0s = S0+np.linspace(-1, 1, n_S)*S0*sig*T*a
dS0 = 2/(n_S-1)*S0*sig*T*a
stable = dt <= dS0**2/(S0s[-1]*sig)**2
assert stable
V0 = np.maximum(0, (S0s-K) if style == 'call' else (K-S0s))
plt.plot(S0s, V0, c='black')
for i in range(n_t):
# Centered difference in space.
  Delta = (V0[2:]-V0[:-2])/(2*dS0)
  Gamma = (V0[2:]-2*V0[1:-1]+V0[:-2])/dS0
  match style: # Neumann boundary condition
    case 'call': # L: Delta=0, R: Delta=1
      V0[0] -= dt*(r*V0[0])
      V0[-1] -= dt*(-r*S0s[-1]+r*V0[-1])
    case 'put': # L: Delta=-1, R: Delta=0
      V0[0] -= dt*(r*S0s[0]+r*V0[0])
      V0[-1] -= dt*(r*V0[-1])
# Forward Euler in time.
  V0[1:-1] -= dt*(-sig**2*S0s[1:-1]**2/2*Gamma-r*S0s[1:-1]*Delta+r*V0[1:-1])

print(f'V0 {CubicSpline(S0s, V0)(S0):.4e}')
plt.plot(S0s, V0, c='red')
plt.xlim(S0s[0], S0s[-1])
plt.xlabel('$S_0$')
plt.ylabel('$V_0$')
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'PDE_BSM_E.png', bbox_inches='tight', dpi=400)
else: plt.show()
