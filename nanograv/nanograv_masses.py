import numpy as np
import matplotlib.pyplot as plt
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
import seaborn as sns   
import pandas as pd

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


# determine placement of frequency components
Tspan = 12.893438736619137 * (365 * 86400)  # psr.toas.max() - psr.toas.min() #
freqs_30 = 1.0 * np.arange(1, 31) / Tspan


chain_DE438_30f_vary = np.loadtxt(
    './blue/data/12p5yr_DE438_model2a_cRN30freq_gammaVary_chain.gz', usecols=[90, 91, 92], skiprows=25000)
chain_DE438_30f_vary = chain_DE438_30f_vary[::4]

# Pull MLV params
DE438_vary_30cRN_idx = np.argmax(chain_DE438_30f_vary[:, -1])

# Make MLV Curves
PL_30freq = powerlaw(
    freqs_30, log10_A=chain_DE438_30f_vary[:, 1][DE438_vary_30cRN_idx], gamma=chain_DE438_30f_vary[:, 0][DE438_vary_30cRN_idx])
PL_30freq_num = int(chain_DE438_30f_vary[:, 0].shape[0] / 5.)
PL_30freq_array = np.zeros((PL_30freq_num, 30))

gamma_arr = np.zeros((PL_30freq_num, 30))
A_arr = np.zeros((PL_30freq_num, 30))
for ii in range(PL_30freq_num):
    PL_30freq_array[ii] = np.log10(powerlaw(
        freqs_30, log10_A=chain_DE438_30f_vary[ii*5, 1], gamma=chain_DE438_30f_vary[ii*5, 0]))
    A_arr[ii] = chain_DE438_30f_vary[ii*5, 1]
    gamma_arr[ii] = chain_DE438_30f_vary[ii*5, 0]


hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1

with h5py.File(hdf_file, 'r') as f:
    Ts = f['T-ladder'][...]
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]
    # samples = f['samples_cold'][...]
    log_likelihood = f['log_likelihood'][:, burnin::extra_thin]
    # log_likelihood = f['log_likelihood'][...]
    par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
    acc_fraction = f['acc_fraction'][...]
    fisher_diag = f['fisher_diag'][...]
burnin = 0
thin = 1


# Make Figure
# plt.style.use('dark_background')
plt.figure(figsize=(12, 5))

# Left Hand Side Of Plot

# plt.semilogx(freqs_30, (PL_30freq_array.mean(axis=0)), color='C2', label='PL (30 freq.)', ls='dashdot')
# plt.fill_between(freqs_30, (PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0)), (PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0)), color='C2', alpha=0.15)


# This commented out portion was my attempt to get the violin plots for the 12.5 and 15 year data set for the HD_correlated free spectral process from
# Fig 1 of https://iopscience.iop.org/article/10.3847/2041-8213/abd401/pdf and Fig 3 of https://iopscience.iop.org/article/10.3847/2041-8213/acdc91/pdf#bm_apjlacdc91eqn73
# respectively. I was able to successfully get the 12.5 one, as seen in blue/nanograv_masses_figs/fig2.pdf

'''chain_DE438_FreeSpec = np.loadtxt('blue/data/12p5yr_DE438_model2a_PSDspectrum_chain.gz', usecols=np.arange(90,120), skiprows=30000)
chain_DE438_FreeSpec = chain_DE438_FreeSpec[::5]
vpt = plt.violinplot(np.log10(chain_DE438_FreeSpec), positions=(freqs_30), widths=0.05*freqs_30, showextrema=False)
for pc in vpt['bodies']:
    pc.set_facecolor(
        'k')
    pc.set_alpha(0.3)'''

# this is with the 15 year data set

dir = '30f_fs{hd+mp+dp+cp}_ceffyl_hd-only'
dir = '30f_fs{hd+mp+dp}_ceffyl_hd-only'
dir = '30f_fs{cp}_ceffyl'
dir = '30f_fs{hd}_ceffyl'
freqs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/freqs.npy')
rho = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/log10rhogrid.npy')
density = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/density.npy')
bandwidth = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/bandwidths.npy')
density = np.transpose(density[0]) 
'''vpt = plt.violinplot(10**density,positions=freqs,widths=.05*freqs,showextrema=False)
for pc in vpt['bodies']:
    pc.set_facecolor('k')
    pc.set_alpha(0.3)
'''

N = 1000
f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f

def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)


OMG_30freq_array = np.zeros((PL_30freq_num, 30))
for ii in range(PL_30freq_num):
    OMG_30freq_array[ii] = np.log10(
        h**2*omega_GW(freqs_30, A_arr[ii], gamma_arr[ii]))

num = 67
num_freqs = 30
#freqs = np.logspace(np.log10(2e-9), np.log10(6e-8), num_freqs)

with open('blue/data/v1p1_all_dict.json', 'r') as f:
    data = json.load(f)
# This was a failed attempt to try to get the log10_A and gamma from the blue/data/v1p1_all_dict.json file
'''
A_arr,gamma_arr = []
i = 0
for key in data.keys():
    if 'log10_A' in key:
        A_arr.append(data[key])
    if 'gamma' in key:
        gamma_arr.append(data[key])
A_arr = np.array(A_arr)
gamma_arr = np.array(gamma_arr)
'''

# Finally realized the log10_A and gamma I needed were in https://zenodo.org/records/8067506 in the
# NANOGrav15yr_CW-Analysis_v1.0.0/15yr_cw_analysis-main/data/15yr_quickCW_detection.h5 file.
A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

# plt.scatter(gamma_arr, A_arr, color='black')
# gamma_arr = np.zeros((PL_30freq_num,30))
# A_arr = np.zeros((PL_30freq_num,30))
PL = np.zeros((67, num_freqs))
#plt.fill_between(np.log10(freqs), OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0), OMG_15.mean(axis=0) +
#                 2*OMG_15.std(axis=0), color='orange',label='2$\sigma$ posterior of GWB', alpha=0.5)

#plt.fill_between(np.log10(freqs), PL.mean(axis=0) - 2*PL.std(axis=0), PL.mean(axis=0) +
#                 2*PL.std(axis=0), color='orange', label='2$\sigma$ posterior of GWB', alpha=0.5)

#plt.fill_between((freqs), np.log10(PL.mean(axis=0) - 2*PL.std(axis=0)), np.log10(PL.mean(axis=0) +
#                 2*PL.std(axis=0)), color='skyblue', label='2$\sigma$ posterior of GWB', alpha=0.5)

'''
for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0)), color='orange', alpha=0.7)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0)), color='orange', alpha=0.5)
plt.fill_between(freqs, 10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0)), 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0)), color='orange', alpha=0.3)

plt.plot(freqs, h**2*omega_GW(freqs, -15.6, 4.7),
         linestyle='dashed', color='black', label='SMBHB spectrum')

'''        
# plt.fill_between(np.log10(freqs), PL.mean(axis=0) - 2*PL.std(axis=0), PL.mean(axis=0) + 2*PL.std(axis=0), color='orange', alpha=0.5)
# plt.fill_between(log10_f, omega_15.mean(axis=0) - 2*omega_15.std(axis=0), omega_15.mean(axis=0) + 2*omega_15.std(axis=0), color='orange', alpha=0.75)

# trying to reproduce Emma's fig 1 in https://arxiv.org/pdf/2102.12428.pdf
# plt.fill_between(np.log10(freqs_30), OMG_30freq_array.mean(axis=0) - 2*OMG_30freq_array.std(axis=0), OMG_30freq_array.mean(axis=0) + 2*OMG_30freq_array.std(axis=0), color='pink', alpha=0.75)
# plt.fill_between(freqs_30, PL_30freq_array.mean(axis=0) - PL_30freq_array.std(axis=0), PL_30freq_array.mean(axis=0) + PL_30freq_array.std(axis=0), color='pink', alpha=0.55)

# This part plots the energy densities of massive gravitons from the Mukohyama Blue tilted paper https://arxiv.org/pdf/1808.02381.pdf
H_inf = 1e8
tau_r = 1
tau_m = 1e10
H_14 = H_inf/1e14
a_r = 1/(tau_r*H_inf)


def omega_GW_approx(f, m):
    f_8 = f/(2e8)
    nu = (9/4 - m**2 / H_inf**2)**.5
    return 1e-15 * tau_m/tau_r * H_14**(nu+1/2)*f_8**(3-2*nu)


tau_m = 1e21*tau_r
M_GW = .3*H_inf
#plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_approx(freqs, M_GW)), color='blue',
#         label=r'MG - Blue-tilted, $m = 0.3H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')
M_GW = .6*H_inf
#plt.plot(freqs,h**2*omega_GW_approx(freqs, M_GW), linestyle='dashed',
#         color='blue', label=r'MG - Blue-tilted, $m = 0.6H_{inf}$, $\frac{\tau_m}{\tau_r} = 10^{21}$')

M_GW = 0
tau_m = 1
# print(np.log10(h**2*omega_GW_approx(freqs, M_GW)))
#plt.plot(np.log10(freqs), np.log10(h**2*omega_GW_approx(freqs, M_GW)), linestyle='dashed',
#         color='green', label=r'GR - Blue-tilted')

BBN_f = np.logspace(-10, 9)
# plt.fill_between(np.log10(BBN_f), np.log10(BBN_f*0+1e-5),
 #                np.log10(BBN_f * 0 + 1e1), alpha=0.5, color='orchid')
#plt.text(-8.5, -5.4, r"BBN", fontsize=15)


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


linestyle_arr = ['solid', 'dashed', 'solid']
M_arr = [4.3e-23*1e-9, 1.2e-22*1e-9, 0]
M_arr = [k / hbar for k in M_arr]
linestyle_arr = ['solid', 'dashed', 'solid']
color_arr = ['red', 'red', 'green']
text = ['2023 NANOGrav', '2016 LIGO', 'GR']
idx = 0
N = 2000
'''

for M_GW in M_arr:
    if M_GW == 0:
        omega_0 = np.logspace(-10, -1, N)
        omega_0 = np.logspace(-20, -1, N)
    else:
        omega_0 = np.logspace(math.log(M_GW, 10), math.log(.1*2*np.pi, 10), N)
    k = np.where(omega_0 >= M_GW, a_0 * np.sqrt(omega_0**2 - M_GW**2), -1.)
    a_k = ak(k)  # uses multithreading to run faster
    omega_k = np.sqrt((k / a_k) ** 2 + M_GW**2)
    k_prime = (
        a_0 * omega_0
    )
    a_k_prime_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_prime**2 + beta)) / (
        2 * a_eq * k_prime**2
    )

    gamma_k_t_0 = A(k)*np.sqrt(omega_k * a_k**3 / (omega_0*a_0**3))
    P = np.where(omega_0 <= M_GW, np.nan, omega_0**2 /
                 (omega_0**2-M_GW**2)*(2*k**3/np.pi**2)*gamma_k_t_0**2)
    P = omega_0**2/(omega_0**2-M_GW**2)*(2*k**3/np.pi**2)  # *y_k_0**2
    gamma_k_GR_t_0 = A(k_prime)*a_k_prime_GR/a_0
    P_GR = (2*k_prime**3/np.pi**2)*gamma_k_GR_t_0**2  # *y_k_0**2
    S = np.where(omega_0 <= M_GW, np.nan, k_prime * a_k / (k * a_k_prime_GR)
                 * np.sqrt(omega_k * a_k / (omega_0 * a_0)))
    f = omega_0/(2*np.pi)

    if idx < 2:
        plt.plot(np.log10(f), np.log10(h**2*2*np.pi**2*(P_GR*S**2 / (4*f))/(3*H_0**2)*(f)**(3)),
                 linestyle=linestyle_arr[idx], color='red', label=r'MG - Emir Gum. - $M_{GW}=$' + f'{round_it(M_GW*hbar, 2)}'+r' GeV/$c^2$' + ' ('+text[idx] + ')')
    else:
        plt.plot(np.log10(f), np.log10(h**2*2*np.pi**2*(P_GR/(4*f))/(3*H_0**2)  
                 * (f)**(3)), color='green', label=r'GR - Emir Gum.')
    N_extra = math.log(10)
    a_k_0_GR = (beta + np.sqrt(beta) * np.sqrt(4 * a_eq**2 * k_0**2 + beta)) / (
        2 * a_eq * k_0**2
    )
    S_peak_anal = a_c*np.sqrt(k_0*k_c)/(a_k_0_GR*a_0*H_0)*np.e**N_extra
    S_peak_num = 0
    var = False
    for s in S:
        # print(s)
        if s != np.nan and var == False:
            S_peak_num = s
            var = True
    # print(f'M_GW (in Hz): {M_GW}, S_peak_anal: {S_peak_anal}, S_peak_num: {S_peak_num}')
    T_obs = 15/f_yr
    if M_GW == 0:
        break
        #print(f'amplif factor: {1e-4*(T_obs/H_0)**(-4)*(M_GW/H_0)**(-3) *math.log(np.e**N_extra*M_GW/H_0)}')
    idx += 1
'''
plt.gca()
plt.clf()

from scipy.stats import rv_histogram

# load the logpdfs

logpdfs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/density.npy') 
logpdfs = logpdfs[0]
# load frequencies

freqs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/freqs.npy')

nfreqs = len(freqs)

# create ‘histogram’ bin edges

grid_points = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/log10rhogrid.npy')
binedges = 0.5 * (grid_points[1:] + grid_points[:-1])

dx = binedges[1] - binedges[0]

binedges = np.insert(binedges, 0, binedges[0]-dx)

binedges = np.insert(binedges, -1, binedges[-1]+dx)

# create and sample from continuous rv objects

samples = np.zeros((10000, nfreqs))

for ii in range(nfreqs):
    rv = rv_histogram((np.exp(logpdfs[ii]), binedges), density=True)
    samples[:,ii] = rv.rvs(size=10000)
    print(np.exp(logpdfs[ii]))

import seaborn as sns

rms = []
bin = []
data_set = []
for i in range(len(density)):
    for j in range(len(density[0])):
        rms.append(density[i,j])
        bin.append(freqs[j])
        data_set.append('15')

df = pd.DataFrame()
df["RMS"] = rms
df["Fbin"] = bin
df["Data Set"] = data_set
sns.violinplot(x=df["Fbin"], y = df["RMS"], log_scale=True)

vpt = plt.violinplot((h**2*(10**samples)**2*8*np.pi**4*freqs**5/H_0**2/freqs[0]), positions=freqs,widths=0.1*freqs, showextrema=False)
print(samples)
for pc in vpt['bodies']:

    pc.set_facecolor('k')
    pc.set_alpha(0.1)

# Plot Labels
# plt.title(r'NANOGrav 15-year data and Mu')
# plt.xlabel('$\gamma_{cp}$')
plt.xlabel(r'log$_{10}(f$ Hz)')
plt.xlabel(r'$f$ [Hz]')  
plt.ylabel(r'log$_{10}(h_0^2\Omega_{GW})$')
plt.ylim(-13,-4)
plt.ylim(1e-14, 1e-4)
plt.xlim(1e-9, 1e-7)
plt.grid(which='major', alpha=.2)
plt.grid(which='minor', alpha=.2)
plt.xscale("log")
plt.yscale("log")
plt.legend(loc='lower left').set_visible(False)
plt.grid(alpha=.2)
plt.savefig('nanograv/nanograv_masses_figs/fig8.pdf')
plt.show()