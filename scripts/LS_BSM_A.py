'''
Longstaff-Schwartz algorithm for pricing American options under the
Black-Scholes-Merton model.

(Not completed.)
'''

import numpy as np

# model params
T = 1 # duration
r = .05 # risk free interest rate
q = .02 # dividend rate
sig = .2 # volatility
S0 = 90 # initial spot price
# option params
K = 100 # strike price
style = 'put'

assert style in ('call', 'put')
