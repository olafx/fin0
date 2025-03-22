'''
Bilateral gamma pricing of European options via the Carr-Madan formula,
numerically using the FFT.
'''

import numpy as np
from scipy.interpolate import CubicSpline

# numerical params
N = 4096 # resolution
alp = 1.5 # alpha, see Carr-Madan paper
eta = .25 # eta, see Carr-Madan paper
# model params
T = 1 # duration
r = 0.06 # risk free interest rate
q = 0.03 # dividend yield
al1 = 1.18 # alpha^+
lam1 = 10.57 # lambda^+
al2 = 1.44 # alpha^-
lam2 = 5.57 # lambda^-
S0 = 100 # initial spot price
# option params
K = 80 # strike price
style = 'put' # call or put

assert style in ('call', 'put')

lam = 2*np.pi/(N*eta)
b = lam*N/2
k = np.arange(-b, b, lam)
v = np.arange(0, N*eta, eta)
sw = (3+(-1)**np.arange(1, N+1))/3
sw[0] = 1/3
u = v-(alp+1)*1j
xi = -al1*np.log(lam1/(lam1-1))-al2*np.log(lam2/(lam2+1))
phi0 = np.exp(1j*u*(np.log(S0)+(r-q+xi)*T))
phiBG = (lam1/(lam1-1j*u))**(T*al1)*(lam2/(lam2+1j*u))**(T*al2)
phi = phi0*phiBG
rho = np.exp(-r*T)*phi/(alp**2+alp-v**2+1j*(2*alp+1)*v)
A = rho*np.exp(1j*v*b)*eta*sw
Z = np.fft.fft(A).real
Cs = np.exp(-alp*k)*Z/np.pi
V0 = CubicSpline(np.exp(k), Cs)(K)
if style == 'put': V0 += -S0*np.exp(-q*T)+np.exp(-r*T)*K

print(f'V0 {V0:.4e}')
