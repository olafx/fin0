'''
Evaluate the analytical solution of the Black-Scholes equation for European
options.
'''

from scipy.stats import norm
import numpy as np

def V0(S0, X, mu, sigma, T, call:bool):
  d1 = (np.log(S0/X)+(mu+sigma**2/2)*T)/(sigma*T**.5)
  d2 = d1-sigma*T**.5
  if call: return S0*norm.cdf(d1)-X*np.exp(-mu*T)*norm.cdf(d2)
  else: return X*np.exp(-mu*T)*norm.cdf(-d2)-S0*norm.cdf(-d1)

print(f'deep ITM call, low sigma {V0(S0=100, X=40, mu=0, sigma=.2, T=1, call=True)=:.4e}')
print(f'deep ITM call, high sigma {V0(S0=100, X=40, mu=0, sigma=.3, T=1, call=True)=:.4e}')
print(f'deep OTM call, low sigma {V0(S0=40, X=100, mu=0, sigma=.2, T=1, call=True)=:.4e}')
print(f'deep OTM call, high sigma {V0(S0=40, X=100, mu=0, sigma=.3, T=1, call=True)=:.4e}')
print(f'deep ITM put, low sigma {V0(S0=40, X=100, mu=0, sigma=.2, T=1, call=False)=:.4e}')
print(f'deep ITM put, high sigma {V0(S0=40, X=100, mu=0, sigma=.3, T=1, call=False)=:.4e}')
print(f'deep OTM put, low sigma {V0(S0=100, X=40, mu=0, sigma=.2, T=1, call=False)=:.4e}')
print(f'deep OTM put, high sigma {V0(S0=100, X=40, mu=0, sigma=.3, T=1, call=False)=:.4e}')
