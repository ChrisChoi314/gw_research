from math import log10, floor
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import mpmath
from scipy.integrate import odeint
from blue_func import *

N = 1000

# plt.style.use('dark_background')


H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14

tau = np.logspace(10, 20, N)
k_arr = [1e-18, 1e-3, 1e9]
colors = ['red', 'green', 'blue']
k_arr = [1e-18]
idx = 0

def abs_sq_h1(n, t):
    return float(mpmath.fabs(mpmath.hankel1(n, t))**2)


def jv(n, t):
    return float(mpmath.jv(n, t))


def v_k_3(f, m, tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m / H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
    C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
    lamb = m*tau_m**2 / (2*H_inf*tau_r**2)
    D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) -
                            C_1*np.sin(lamb-np.pi/8))
    D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    return 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * \
        (D_1*np.cos(k*tau) + D_2*np.sin(k*tau))

def P_massless(f, m, tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    print(scipy.special.gamma(nu), scipy.special.jv(-3/4,m/(2*H_inf)),scipy.special.jv(1/4, m/(2*H_inf)) )
    C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m / H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
    C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
    lamb = m*tau_m**2 / (2*H_inf*tau_r**2)  
    D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) -
                            C_1*np.sin(lamb-np.pi/8))
    D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
    v_k_reg3 = 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * \
        (D_1*np.cos(k*tau) + D_2*np.sin(k*tau))

    return 4*k**3*np.abs(v_k_reg3)**2/(np.pi**2*M_pl**2*(tau/(tau_r**2*H_inf))**2)


def omega_GW_massless(f, m, tau):
    k = f*2*np.pi
    nu = (9/4 - m**2 / H_inf**2)**.5
    return np.pi**2/(3*H_0**2)*f**3*P_massless(f, m, tau)


def P_T(f, tau):
    return (f*2*np.pi)**2/(2*np.pi*scale_fac(tau)**2*(M_pl)**2)*(f*2*np.pi*tau)*abs_sq_h1(3/2,tau)


m_arr = [.5*H_inf, .8*H_inf]
f = np.logspace(-18, 5, N)

a_r = 1/(tau_r*H_inf)
k = 1
tau_end = 5*tau_m
m = .8*H_inf
nu = np.sqrt(9/4 - m**2 / H_inf**2)
tau_init = -200*tau_r
tau = np.logspace(1,19, N)

fig, (ax1) = plt.subplots(1)  # , figsize=(22, 14))

tau_arr = [tau_m*5*(eta_0/eta_rm), eta_0, eta_0/1e5]
for t in tau_arr:
    '''ax1.plot(
        f, P_massless(f,m,t), "--",
        label=f'blue: tau = {t}',alpha=.7
    )
    ax1.plot(
        f, P_T(f,t) , label=f'nick: tau = {t}'
    )

    ax1.plot(
        f, P_massless(f, m ,t) , label=f'blue: tau = {t}', linestyle = 'dashed' 
    )

    ax1.plot(
        f, omega_GW_massless(f, m ,t) , label=f'omega: tau = {t}', linestyle = '-.' 
    )
    '''
ax1.plot(f, omega_GW_massless(f, m, tau_m*5*(eta_0/eta_rm)), label=f'omega', alpha = .6)

def average(arr):
    outpt = []
    i= 2
    for i in range(len(arr)-1):
        outpt += [(arr[i-2]+arr[i-1]+ arr[i])/3]
    outpt += [arr[len(arr) -1]]
    return outpt


ax1.plot(f, average(average(omega_GW_massless(f, m, tau_m*5*(eta_0/eta_rm)))), label='averaged', alpha = 1)


ax1.set_xscale("log")
ax1.set_yscale("log")
'''
fig, (ax2) = plt.subplots(1)  # , figsize=(22, 14))
N_neg_r = 0
N_pos_r = 0
N_m = 0
for val in tau:
    if val <= -tau_r:
        N_neg_r += 1
    if val <= tau_r:
        N_pos_r += 1
    if val <= tau_m:
        N_m += 1

isReal = False


def scale_fac(conf_time):
    if conf_time < tau_r:
        return -1/(H_inf*conf_time)
    else:
        return a_r * conf_time / tau_r


def d_scale_fac_dz(conf_time):
    if conf_time < tau_r:
        return -2/(H_inf*conf_time**3)
    else:
        return 0


def mu(conf_time):
    if conf_time < tau_m:
        return m
    else:
        return 0


# The input for the homogeneous version of the equation, just chi''(u) + 2/u chi'(u) + chi(u) = 0
def M_derivs_homo(M, u):
    return [M[1], -(k**2 + (scale_fac(u)*mu(u))**2 - d_scale_fac_dz(u) / scale_fac(u)) * M[0]]


# horrors beyond my comprehension for the central difference method
xxx = np.array([tau[0] - 0.0000001, tau[0] + 0.0000001])
xx = (np.sqrt(-np.pi*xxx) / 2 * scipy.special.hankel1(nu, -k*xxx))[:2]
xx = xx[1] - xx[0]
xx /= xxx[1] - xxx[0]


# Get the homogeneous solution using scipy.integrate.odeint
if isReal:
    v_0 = np.sqrt(-np.pi*tau[0]) / 2 * \
        scipy.special.hankel1(nu, -k*tau[0])
    v_prime_0 = xx
else:
    v_0 = np.sqrt(-np.pi*tau[0]) / 2 * \
        scipy.special.hankel1(nu, -k*tau[0]).imag
    v_prime_0 = xx.imag


v, v_prime = odeint(M_derivs_homo, [v_0, v_prime_0], tau[0:N_neg_r]).T


ax2.plot(
    tau[0:N_neg_r], v, label=r"Numerical solution", color="black"
)

if isReal:
    ax2.plot(
        tau[0:N_neg_r], np.sqrt(-np.pi*tau[0:N_neg_r]) / 2 *
        scipy.special.hankel1(nu, -k*tau[0:N_neg_r]), "--", color="orange",
        label=r"Analyt Soln: $\frac{\sqrt{-\pi \tau}}{2} H_\nu^{(1)} (-k\tau)$"
    )
else:
    ax2.plot(
        tau[0:N_neg_r], -1j*np.sqrt(-np.pi*tau[0:N_neg_r]) / 2 *
        scipy.special.hankel1(nu, -k*tau[0:N_neg_r]), "--", color="orange",
        label=r"Analyt Soln: $\frac{\sqrt{-\pi \tau}}{2} H_\nu^{(1)} (-k\tau)$"
    )


v_0_rest = v[N_neg_r-1]
xxxxx = np.array([tau[N_neg_r-1] - 0.0000001, tau[N_neg_r-1] + 0.0000001])
if isReal:
    xxxx = (np.sqrt(-np.pi*xxxxx) / 2 *
            scipy.special.hankel1(nu, -k*xxxxx))[:2]
else:
    xxxx = -1j*(np.sqrt(-np.pi*xxxxx) / 2 *
                scipy.special.hankel1(nu, -k*xxxxx))[:2]
xxxx = xxxx[1] - xxxx[0]
xxxx /= xxxxx[1] - xxxxx[0]
v_prime_0_rest = xxxx
v_rest, v_prime_rest = odeint(
    M_derivs_homo, [v_0_rest, v_prime_0_rest], tau[N_pos_r:]).T


ax2.plot(
    tau[N_pos_r:], v_rest, color="black"
)

ax2.set_xlabel(r"$\tau$ (conformal time)")

C_1 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m /H_inf * scipy.special.jv(-3/4, m/(2*H_inf)) + (1-2*nu)*scipy.special.jv(1/4, m/(2*H_inf)))
C_2 = -1j*np.sqrt(np.pi)*2**(-7/2 + nu)*(k*tau_r)**(-nu)*scipy.special.gamma(nu)*(2*m / H_inf * scipy.special.jv(3/4, m/(2*H_inf)) - (1-2*nu)*scipy.special.jv(-1/4, m/(2*H_inf)))
v_k_reg2 = 2/np.sqrt(np.pi*a_r * tau[N_pos_r:N_m] / tau_r*m)*(C_1*np.cos(a_r * tau[N_pos_r:N_m] / tau_r *m*tau[N_pos_r:N_m]/2 - np.pi/8) + C_2*np.sin(a_r * tau[N_pos_r:N_m] / tau_r*m*tau[N_pos_r:N_m]/2 + np.pi/8))


ax2.plot(
    tau[N_pos_r:N_m], -1j*v_k_reg2, "--", color="cyan",
    label=r"Analyt Solution: $\frac{2}{\sqrt{\pi am}}[C_1 \cos(\frac{am\tau}{2} - \frac{\pi}{8}) + C_2 \sin(\frac{am\tau}{2} + \frac{\pi}{8})]$"
)

lamb = m*tau_m**2 / (2*H_inf*tau_r**2)
D_1 = -np.sin(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
D_2 = np.cos(k*tau_m)*(C_2*np.cos(lamb+np.pi/8) - C_1*np.sin(lamb-np.pi/8))
v_k_reg3 = 2/k*np.sqrt(m*tau_m / (np.pi*H_inf*tau_r**2)) * \
    (D_1*np.cos(k*tau[N_m:] + k*(22.903 - 22.8915)) +
     D_2*np.sin(k*tau[N_m:]+k*(22.903 - 22.8915))) * .998
ax2.plot(
    tau[N_m:], -1j*v_k_reg3, "--",
    color="magenta",
    label=r"Analyt Solution: $\frac{2}{k}\sqrt{\frac{m\tau_m}{\pi H_{inf} \tau_r^2}}[D_1 \cos(k\tau) + D_2 \sin(k\tau)]$"
)

# ax2.set_xlim(-tau_end, tau_end)
ax2.set_ylabel(r"$v_k(\tau)$")


ax2.axvline(x=-tau_r, color='red', linestyle='dashed',
            linewidth=1)
ax2.axvline(x=tau_r, color='red', linestyle='dashed',
            linewidth=1, label=r'$\pm \tau_r$')
ax2.axvline(x=tau_m, color='blue', linestyle='dashed',
            linewidth=1, label=r'$\tau_m$')


'''

plt.clf() 
f = np.logspace(-19, -11, N)
M_GW = 1e-8
plt.plot(f, ak(f))
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$a_k$')
plt.title(r"$a_k(f)$ from Emir's paper")
plt.xscale("log")
plt.yscale("log") 

plt.savefig("blue/test_figs/fig1.pdf")
plt.legend()
plt.show()
