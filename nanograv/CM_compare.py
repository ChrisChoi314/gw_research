import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

fs = 12
plt.rcParams.update({'font.size': fs})
# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.
f_yr = 1/(365*24*3600)
gamma_12p5 = 13/3
gamma_15 = 3.2  # from page 4 of https://iopscience.iop.org/article/10.3847/2041-8213/acdac6/pdf
A_12p5 = -16
A_15 = -14.19382002601611

gamma_cp = gamma_15
A_cp = A_15

# # Common Process Spectral Model Comparison Plot (Figure 1) # #

# Definition for powerlaw and broken powerlaw for left side of Figure 1


def powerlaw_vec(f, f_0, log10_A=A_cp, gamma=gamma_cp):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma-3) * f**(-gamma) * f_0)


def powerlaw(f, log10_A=A_cp, gamma=gamma_cp):
    return np.sqrt((10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma-3) * f**(-gamma) * f[0])


hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]
    log_likelihood = f['log_likelihood'][:, burnin::extra_thin]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]
burnin = 0
thin = 1

# fig, axs = plt.subplots(2, 1, figsize = (10,7), sharex=True)
plt.figure(figsize=(10,4.5))

N = 1000
f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f


def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)

num_freqs = 30
freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)
# Finally realized the log10_A and gamma I needed were in https://zenodo.org/records/8067506 in the
# NANOGrav15yr_CW-Analysis_v1.0.0/15yr_cw_analysis-main/data/15yr_quickCW_detection.h5 file.
A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

# plt.scatter(gamma_arr, A_arr, color='black')
# gamma_arr = np.zeros((PL_30freq_num,30))
# A_arr = np.zeros((PL_30freq_num,30))
PL = np.zeros((67, num_freqs))
for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)
plt.plot(freqs, h**2*omega_GW(freqs, -15.6, 4.7),
         linestyle='dashed', color='black')


BBN_f = np.logspace(np.log10(f_BBN), 9)
plt.fill_between(BBN_f, BBN_f*0+h**2*1e-5,
                 BBN_f * 0 + 1e1, alpha=0.5, color='orchid')
plt.text(10**(-7.5), 1e-6, r"BBN Bound", fontsize=15)

plt.xlabel(r'$f$ Hz')
plt.ylabel(r'$h_0^2\Omega_{GW}$')


P_prim_k = 2.43e-10
beta = H_eq**2 * a_eq**4 / (2)


def A(k):
    return np.where(k >= 0., np.sqrt(P_prim_k*np.pi**2/(2*k**3)), -1.)


def enhance_approx(x):
    if x < M_GW:
        return 0.
    val = a_0 * np.sqrt(x**2 - M_GW**2)
    if k_0 < val:
        return 1.
    elif val <= k_0 and val >= k_c:
        if val >= k_eq:
            output = (x**2 / M_GW**2 - 1)**(-3/4)
            return output
        if val < k_eq and k_eq < k_0:
            output = k_eq/(np.sqrt(2)*k_0)(x**2 / M_GW**2 - 1)**(-5/4)
            return output
        if k_eq > k_0:
            output = (x**2 / M_GW**2 - 1)**(-5/4)
            return output
    elif val <= k_c:
        beta = H_eq**2 * a_eq**4 / (2)
        a_k_0_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_0**2 + beta)) / (
            2. * a_eq * k_0**2
        )
        if abs(x**2 / M_GW**2 - 1) < 1e-25:
            return 0.
        output = a_c/a_k_0_GR*np.sqrt(k_c/k_0)*(x**2 / M_GW**2 - 1)**(-1/2)
        return output
   

M_arr = [8.6e-24*1e-9, 8.2e-24*1e-9]
M_arr = [k / hbar for k in M_arr] + [2e-9*2*np.pi]
color_arr = ['red', 'blue', 'green']
text = ['2023 Wang et al.', '2023 Wu et al.', 'NG15 Freq Bound']
idx = 0

for M_GW in M_arr:
    omega_0 = np.logspace(math.log(M_GW, 10), np.log10(6e-8*2*np.pi), N)
    k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
    a_k = ak(k)  # uses multithreading to run faster
    omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
    k_prime = (
        a_0 * omega_0
    )
    f = omega_0/(2*np.pi) 
    
    Omega = omega_0*a_k**3*omega_k/a_0/k**2*P_prim_k

    plt.plot(f, h**2*2*np.pi**2*Omega/(3*H_0**2)*(f)**2, color=color_arr[idx],
                  label=r'$M_{GW} = $' + f'{round_it(M_GW*hbar, 2)}'+r' GeV/$c^2$' + ' ('+text[idx] + ')')   
    idx+=1
plt.xlim(1e-9,1e-7)
plt.ylim(1e-15,1e-5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$f$ [Hz])')
plt.ylabel(r'$h_0^2\Omega_{GW}$')
plt.legend(loc='lower right')
plt.grid(alpha=.2)
plt.savefig('nanograv/CM_compare_figs/fig0.pdf')
plt.show()
