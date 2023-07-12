import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from emir_func import *

N = 10000
omega_0 = k_0/a_0
k = a_0*np.sqrt(omega_0**2-M_GW**2)
k = 1e1
eta = np.logspace(17, 18, N)
H_square = np.vectorize(Hubble)(eta) ** 2
ang_freq_square = np.vectorize(ang_freq)(eta) ** 2
a = np.vectorize(scale_fac)(eta)

k_idx = 0
print(a_0)
for i in range(0, len(eta)):
    if a[i] >= a_0:
        print(a[i])
        k_idx = i
        break

print("Current eta_0 = "+f'{eta[k_idx]}')

time = scipy.integrate.cumtrapz(a, eta, initial=0)



fig, (ax1) = plt.subplots(1)

#ax1.plot(eta, a, label='a')
#ax1.axhline(y=(k/M_GW)**2, linestyle="dashed",
#            linewidth=1, label=r'$k_0/M_{GW}$')
#ax1.plot(eta, time, label='Physical time')
#res, err = scipy.integrate.quad(scale_fac, 0, 2.105)

# print('Time: ' + f'{res}')

#roots = scipy.optimize.fsolve(
#    lambda x: scipy.integrate.quad(scale_fac, 0, x)[0] - 1, 1)
# print('Roots: ', roots[0])


eta = np.logspace(-7, 1, N)
a_k_1 = omega_M*H_0**2+H_0*np.sqrt(4*omega_R*k**2+(H_0*omega_M)**2)/(2*k**2)
#print('Using inverse with Wolfalph: ', a_k_1)
idx_1 = 0
H_square = np.vectorize(Hubble)(eta)**2
ang_freq_square = np.vectorize(ang_freq)(eta) ** 2

for i in range(0, len(eta)):
    if H_square[i] <= ang_freq_square[i]:
        eta_k = eta[i]
        eq_idx = i
        break
print('a_k = ', scale_fac(eta_k))
print('eta_k = ', eta_k)
a = np.vectorize(scale_fac)(eta)
ax1.plot(a, H_square,label='H', color='orange')
ax1.plot(a, ang_freq_square,label='omega', color='blue')
ax1.axhline(y=k, label='k')

ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Scale Factor')

'''

'''

fig, ax2 = plt.subplots(1, figsize=(10, 10))
ax2.plot(eta,a)
ax2.set_xscale('log')

'''
ax2.plot([0, 10], [0, 10], '--', color='gray')

a = H_0**2*omega_M
b = H_0**2*omega_R
c = H_0**2*omega_L
x = np.linspace(.1,10,10000)
y = reg_N(a,b,c,x)
y_min = y.min()
ax2.plot(x,y, label="regular")

inverse1 = inv_of_H(a,b,c,x, c1=1, c2=1)
inverse1 = np.where(x >= y_min, inverse1, np.nan)
ax2.plot(x,inverse1, label="inverse1")

inverse2 = inv_of_H(a,b,c,x, c1=1, c2=-1)
inverse2 = np.where(x >= y_min, inverse2, np.nan)
ax2.plot(x,inverse2, label="inverse2")

inverse3 = inv_approx(a,b,x, c1=1)
inverse3 = np.where(x >= y_min, inverse3, np.nan)
ax2.plot(x,inverse3, label="inverse3")

inverse4 = inv_approx(a,b,x, c1=-1)
inverse4 = np.where(x >= y_min, inverse4, np.nan)
ax2.plot(x,inverse4, label="inverse4")

ax2.legend(loc="best")

'''
#plt.savefig("emir/emir_calc_figs/inverse_of_hubble.pdf")
plt.show()
