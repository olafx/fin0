'''
Pricing European options via an integral solution of the probability
distribution of the price at expiry, obtained from Itô calculus.
'''

import numpy as np
from scipy.integrate import quad

def V0(S0, X, mu, sigma, T, call=True):
  a = 1 if call else -1
  def integrand(u):
    value = max(0, a*(np.exp(u)-X))
    return value*np.exp(-mu*T)*(2*np.pi*sigma**2*T)**(-1/2)*np.exp(-(u-(np.log(S0)+(mu-sigma**2/2)*T))**2/(2*sigma**2*T))
  res, err = quad(integrand, -10, 10)
  if err/res > 1e-3: print('WARNING: less than 99.9% accurate')
  return res

# Perform the integration from -inf to inf

print(f'deep ITM call, low sigma {V0(S0=100, X=40, mu=0, sigma=.2, T=1, call=True)=:.4e}')
print(f'deep ITM call, high sigma {V0(S0=100, X=40, mu=0, sigma=.3, T=1, call=True)=:.4e}')

print(f'deep OTM call, low sigma {V0(S0=40, X=100, mu=0, sigma=.2, T=1, call=True)=:.4e}')
print(f'deep OTM call, high sigma {V0(S0=40, X=100, mu=0, sigma=.3, T=1, call=True)=:.4e}')

print(f'deep ITM put, low sigma {V0(S0=40, X=100, mu=0, sigma=.2, T=1, call=False)=:.4e}')
print(f'deep ITM put, high sigma {V0(S0=40, X=100, mu=0, sigma=.3, T=1, call=False)=:.4e}')

print(f'deep OTM put, low sigma {V0(S0=100, X=40, mu=0, sigma=.2, T=1, call=False)=:.4e}')
print(f'deep OTM put, high sigma {V0(S0=100, X=40, mu=0, sigma=.3, T=1, call=False)=:.4e}')
