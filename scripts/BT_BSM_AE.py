'''
Plot American vs European prices under the Black-Scholes-Merton method, the
American option prices evaluated via a binomial tree model.
'''

import numpy as np
from scipy.stats import norm as normal
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
N_S0 = 100 # number of initial spot prices
N_bt = 2000 # binomial tree depth
# options params
K = 100 # strike price
style = 'put'
# model params
T = 1 # duration
r = .05 # risk-free interest rate
q = .02 # dividend rate
sig = .2 # volatility
range_S0 = (.5*K, 2*K) # initial spot price range

assert style in ('call', 'put')
Phi = normal.cdf

def V0_A(style, S0, K, r, sig, T):
  if isinstance(S0, np.ndarray) or isinstance(S0, list):
    return [V0_A(style, S0_, K, r, sig, T) for S0_ in S0]
  dt = T/N_bt
  u = np.exp(sig*dt**.5)
  d = np.exp(-sig*dt**.5)
  p = (np.exp((r-q)*dt)-d)/(u-d)
  V0, V1 = np.zeros((2, N_bt+1))
  n_up = np.arange(N_bt+1)
  S = S0*u**n_up*d**(N_bt-n_up)
  V1 = np.maximum(0, S-K if style == 'call' else K-S)
  for i_layer in range(N_bt-1, -1, -1):
    V0, V1 = V1, V0
    n_up = np.arange(i_layer+1)
    S = S0*u**n_up*d**(i_layer-n_up)
    V_hold = np.exp(-r*dt)*(p*V0[1:]+(1-p)*V0[:-1])
    V_exer = np.maximum(0, S-K if style == 'call' else K-S)
    V1 = np.maximum(V_hold, V_exer)
  return V1[0]

def V0_E(style, S0, K, r, sig, T):
  d1 = (np.log(S0/K)+(r-q+sig**2/2)*T)/(sig*T**.5)
  d2 = d1-sig*T**.5
  match style:
    case 'call': return S0*np.exp(-q*T)*Phi(d1)-K*np.exp(-r*T)*Phi(d2)
    case 'put': return -S0*np.exp(-q*T)*Phi(-d1)+K*np.exp(-r*T)*Phi(-d2)

S0s = np.linspace(*range_S0, N_S0)
V0_As = V0_A(style, S0s, K, r, sig, T)
V0_Es = V0_E(style, S0s, K, r, sig, T)
V0_Is = S0s-K if style == 'call' else K-S0s
V0_Is *= V0_Is >= 0

plt.plot(S0s, V0_Is, '--', c='black', label='intrinsic')
plt.plot(S0s, V0_As, c='red', label='American', alpha=.5)
plt.plot(S0s, V0_Es, c='blue', label='European', alpha=.5)
plt.xlim(*range_S0)
plt.xlabel(R'$S_0$')
plt.ylabel(R'$V_0$')
plt.legend()
plt.tight_layout()
if plot_save: plt.savefig(Path.cwd()/'out'/'BT_BSM_AE.png', bbox_inches='tight', dpi=400)
else: plt.show()
