'''
Variance-gamma pricing of European options via the Carr-Madan formula,
numerically using the FFT.
'''

import numpy as np
from scipy import interpolate as intp

# numerical params
N = 4096 # resolution
alp = 1.5 # alpha, see Carr-Madan paper
eta = .25 # eta, see Carr-Madan paper
# model params
T = 3 # duration
r = .06 # risk free interest rate
q = 0 # dividend yield
nu = .2 # jump
th = .05 # drift
sig = .4 # volatility
S0 = 100 # initial spot price
# option params
K = 40 # strike price
style = 'call' # call or put

assert style in ('call', 'put')

lam = 2*np.pi/(N*eta)
b = lam*N/2
k = np.arange(-b, b, lam)
v = np.arange(0, N*eta, eta)
sw = (3+(-1)**np.arange(1, N+1))/3
sw[0] = 1/3
u = v-(alp+1)*1j
om = 1/nu*np.log(1-.5*sig**2*nu-th*nu)
phi0 = np.exp(1j*u*(np.log(S0)+(r-q+om)*T))
phiVG = (1-1j*u*th*nu+.5*sig**2*nu*u**2)**(-T/nu)
phi = phi0*phiVG
rho = np.exp(-r*T)*phi/(alp**2+alp-v**2+1j*(2*alp+1)*v)
A = rho*np.exp(1j*v*b)*eta*sw
Z = np.fft.fft(A).real
Cs = np.exp(-alp*k)*Z/np.pi
V0 = intp.CubicSpline(np.exp(k), Cs)(K)
if style == 'put': V0 += -S0*np.exp(-q*T)+np.exp(-r*T)*K

print(f'V0 {V0:.4e}')
