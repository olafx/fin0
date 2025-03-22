'''
Calculate implied volatility from a European option.
'''

import numpy as np
from scipy.stats import norm as normal
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs

makedirs(Path.cwd().parent/'out', exist_ok=True)
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['text.usetex'] = True

# model params
T = 3 # duration
r = 0.06 # risk free interest rate
q = 0.02 # dividend rate
sig0 = .2 # initial volatility estimate
range_sig = (1e-2, 1.5) # range of volatilies to consider
S0 = 100 # spot price
# option params
K = 70 # strike price
V0 = 3.6691e+01 # option price
style = 'call' # call or put

assert style in ('call', 'put')
phi = normal.cdf

def price(S0, K, T, r, sig):
  d1 = (np.log(S0/K)+(r-q+sig**2/2)*T)/(sig*T**.5)
  d2 = d1-sig*T**.5
  match style:
    case 'call': return S0*np.exp(-q*T)*phi(d1)-K*np.exp(-r*T)*phi(d2)
    case 'put': return -S0*np.exp(-q*T)*phi(-d1)+K*np.exp(-r*T)*phi(-d2)

def price_diff(sig, S0, K, T, r, V0):
  return price(S0, K, T, r, sig)-V0

sig = root_scalar(price_diff, args=(S0, K, T, r, V0), bracket=range_sig, x0=sig0, method='bisect').root
print(f'vol {sig:.4e}')

sigs = np.linspace(*range_sig, 128)
V0s = [price(S0, K, T, r, sig) for sig in sigs]
plt.plot(sigs, V0s, c='black')
plt.plot([sigs[0], sigs[-1]], [V0, V0], c='red')
plt.plot([sig, sig], [V0s[0], V0s[-1]], c='red')
plt.xlim(*range_sig)
plt.ylim(V0s[0], V0s[-1])
plt.xlabel(R'$\sigma$')
plt.ylabel(R'$V_0$')
plt.tight_layout()
plt.savefig(Path.cwd().parent/'out'/'BSM_E_vol.png', bbox_inches='tight', dpi=400)
