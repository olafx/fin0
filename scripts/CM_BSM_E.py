'''
Black-Scholes-Merton pricing of European options via the Carr-Madan formula,
numerically via the FFT.
'''

import numpy as np
from scipy import interpolate

# numerical params
N = 4096
alp = 1.5
eta = .25
# model params
T = 3 # duration
r = 0.05 # risk free interest rate
q = 0 # dividend yield
sig = .4 # volatility
# model initial condition
S0 = 100 # initial spot price
# option params
K = 70 # strike price
style = 'call' # call or put

lam = 2*np.pi/(N*eta)
b = lam*N/2
k = np.arange(-b, b, lam)
v = np.arange(0, N*eta, eta)
sw = (3+(-1)**np.arange(1, N+1))/3
sw[0] = 1/3
u = v-(alp+1)*1j
# phi2BS = np.exp(1j*u*(-.5*sig**2*T)-.5*u**2*sig**2*T)
# phi2 = np.exp(1j*u*(np.log(S0)+r*T))*phi2BS
# phi = phi2
phi0 = np.exp(1j*u*(np.log(S0)+(r-q-.5*sig**2)*T))
phiBS = np.exp(-.5*sig**2*T*u**2)
phi = phi0*phiBS
rho = np.exp(-r*T)*phi/(alp**2+alp-v**2+1j*(2*alp+1)*v)
A = rho*np.exp(1j*v*b)*eta*sw
Z = np.fft.fft(A).real
Cs = np.exp(-alp*k)*Z/np.pi
V = interpolate.CubicSpline(np.exp(k), Cs)(K)
if style == 'put': V += -S0+np.exp(-r*T)*K

print(f'{V:.4e}')
