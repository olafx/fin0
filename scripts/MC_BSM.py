'''
A Black-Scholes-Merton Monte Carlo simulation.
'''

import numpy as np
import matplotlib.pyplot as plt

# numerical params
N = 100 # number of runs
n = 500 # number of steps per run
# model params
T = 3 # duration
r = 0.06 # risk free interest rate
q = 0.02 # dividend rate
sigma = .4 # volatility
# model initial condition
S0 = 100 # initial spot price

S = np.zeros((N, n))
S[:,0] = S0

# Updating the stock price directly, which is not a terribly good idea.
for j in range(N):
  for i in range(1, n):
    S[j,i] = S[j,i-1]*(1+(r-q)*T/n+sigma*(T/n)**.5*np.random.randn())

ts = np.linspace(0, T, n)
S_mean = np.mean(S, axis=0)
S_mean_pred = S0*np.exp((r-q)*ts)

plt.figure('Black-Scholes-Merton')
for i in range(N): plt.plot(ts, S[i])
plt.xlim(0, T)
plt.xlabel('$t$')
plt.ylabel('$S_t$')
plt.tight_layout()
plt.figure('mean S_t vs expected risk free')
plt.xlim(0, T)
plt.plot(ts, S_mean, c='#000000')
plt.plot(ts, S_mean_pred, c='#FF0000')
plt.xlabel('$t$')
plt.ylabel('$\\langle S_t\\rangle$')
plt.tight_layout()
plt.show()
