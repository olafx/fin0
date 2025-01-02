'''
Analytically pricing and hedging (Delta) European options under the
Black-Scholes-Merton model.
'''

import numpy as np
from scipy.stats import norm

# model params
T = 3 # duration
r = .05 # risk free interest rate
q = .0 # dividend rate
sigma = .2 # volatility
# model initial condition
S0 = 100 # initial spot price
# option params
K = 130 # strike price
style = 'call' # call or put

assert style in ('call', 'put')
Phi = norm.cdf
phi = norm.pdf

# V0, Delta, Gamma, Vega, Theta
def price(S0, sigma, T, style):
  d1 = (np.log(S0/K)+(r-q+sigma**2/2)*T)/(sigma*T**.5)
  d2 = d1-sigma*T**.5
  Gamma = np.exp(-q*T)*phi(d1)/(S0*sigma*T**.5)
  Vega = S0*np.exp(-q*T)*T**.5*phi(d1)
  if style == 'call':
    V0 = S0*np.exp(-q*T)*Phi(d1)-K*np.exp(-r*T)*Phi(d2)
    Delta = np.exp(-q*T)*Phi(d1)
    Theta = -(S0*sigma*np.exp(-q*T))/(2*T**.5)*phi(d1)-r*K*np.exp(-r*T)*Phi(d2)+q*S0*np.exp(-q*T)*Phi(d1)
  else:
    V0 = -S0*np.exp(-q*T)*Phi(-d1)+K*np.exp(-r*T)*Phi(-d2)
    Delta = -np.exp(-q*T)*Phi(-d1)
    Theta = -(S0*sigma*np.exp(-q*T))/(2*T**.5)*phi(d1)+r*K*np.exp(-r*T)*Phi(-d2)-q*S0*np.exp(-q*T)*Phi(-d1)
  return V0, Delta, Gamma, Vega, Theta

V0, Delta, Gamma, Vega, Theta = price(S0, sigma, T, style)

h = 1e-3
x = lambda h: price(S0+h, sigma, T, style)[0]
Delta2 = (x(h)-x(-h))/(2*h)
Gamma2 = (x(h)-2*x(0)+x(-h))/h**2
x = lambda h: price(S0, sigma+h, T, style)[0]
Vega2 = (x(h)-x(-h))/(2*h)
x = lambda h: price(S0, sigma, T+h, style)[0]
Theta2 = -(x(h)-x(-h))/(2*h)

print(style)
print(f'V0     {V0:.4e}')
print(f'Delta {Delta:+.4e} {Delta2:+.4e}')
print(f'Gamma {Gamma:+.4e} {Gamma2:+.4e}')
print(f'Vega  {Vega:+.4e} {Vega:+.4e}')
print(f'Theta {Theta:+.4e} {Theta2:+.4e}')
