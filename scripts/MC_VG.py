'''
A variance-gamma model Monte Carlo simulation.
'''

import numpy as np
import matplotlib.pyplot as plt

# numerical params
N = 256 # number of runs
n = 1000 # number of steps per run
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = 0 # dividend yield
nu = .2 # jump
th = .05 # drift
sig = .4 # volatility
# model initial condition
S0 = 100

mu_p = .5*(th**2+2*sig**2/nu)**.5+.5*th
mu_q = .5*(th**2+2*sig**2/nu)**.5-.5*th
nu_p = mu_p**2*nu
nu_q = mu_q**2*nu
om = 1/nu*np.log(1-.5*sig**2*nu-th*nu)

a1 = np.random.gamma(T/n*mu_p**2/nu_p, nu_p/mu_p, (N, n))
a2 = np.random.gamma(T/n*mu_q**2/nu_q, nu_q/mu_q, (N, n))
Gam_1 = np.cumsum(a1, axis=-1)
Gam_2 = np.cumsum(a2, axis=-1)
x = np.cumsum(a1-a2, axis=-1)

t = np.linspace(0, T, n)
S = S0*np.exp((r-q+om)*t+x)
S_mean = np.mean(S, axis=0)
S_mean_pred = S0*np.exp((r-q)*t)

plt.figure(1)
plt.xlim(0, T)
plt.plot(t, x.T)
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.tight_layout()
plt.figure(2)
plt.xlim(0, T)
plt.plot(t, Gam_1.T, c='#FF0000')
plt.plot(t, Gam_2.T, c='#00FF00')
plt.xlabel('$t$')
plt.tight_layout()
plt.figure(3)
plt.plot(t, S.T)
plt.xlabel('$t$')
plt.ylabel('$S_t$')
plt.tight_layout()
plt.figure(4)
plt.plot(t, S_mean, c='#000000')
plt.plot(t, S_mean_pred, c='#FF0000')
plt.xlim(0, T)
plt.xlabel('$t$')
plt.ylabel('$\\langle S_t\\rangle$')
plt.tight_layout()
plt.show()
