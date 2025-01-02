'''
Calculate implied volatility from a European option.
'''

import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# model params
T = 3 # duration
r = 0.06 # risk free interest rate
q = 0.02 # dividend rate
sigma0 = .2 # initial volatility estimate
range_sigma = [1e-2, 1.5] # range of volatilies to consider
# model initial condition
S0 = 100 # initial spot price
# option params
K = 70 # strike price
V0 = 3.6691e+01
style = 'call' # call or put

assert style in ('call', 'put')
phi = norm.cdf

def price(S0, K, T, r, sigma):
  d1 = (np.log(S0/K)+(r-q+sigma**2/2)*T)/(sigma*T**.5)
  d2 = d1-sigma*T**.5
  if style == 'call': return S0*np.exp(-q*T)*phi(d1)-K*np.exp(-r*T)*phi(d2)
  elif style == 'put': return -S0*np.exp(-q*T)*phi(-d1)+K*np.exp(-r*T)*phi(-d2)

def price_diff(sigma, S0, K, T, r, V0):
  return price(S0, K, T, r, sigma)-V0

sigma = root_scalar(price_diff, args=(S0, K, T, r, V0), bracket=range_sigma, x0=sigma0, method='bisect').root
print(f'vol {sigma:.4e}')

sigmas = np.linspace(*range_sigma, 128)
V0s = [price(S0, K, T, r, sigma) for sigma in sigmas]
plt.plot(sigmas, V0s, c='#000000')
plt.plot([sigmas[0], sigmas[-1]], [V0, V0], c='#FF0000')
plt.plot([sigma, sigma], [V0s[0], V0s[-1]], c='#FF0000')
plt.xlim(*range_sigma)
plt.ylim(V0s[0], V0s[-1])
plt.xlabel('$\\sigma$')
plt.ylabel('$V_0$')
plt.tight_layout()
plt.show()
