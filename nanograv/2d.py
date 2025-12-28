import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import enterprise.constants as const
import math
import h5py
import json
from nanograv_func import *
from ng_blue_func import *

fs = 12
plt.rcParams.update({'font.size': fs})

# Much of this code is taken from the NANOGrav collaboration's github page, where they have code that generates certain plots from their set of 4 (or 5?) papers.
hdf_file = "blue/data/15yr_quickCW_detection.h5"

# specify how much of the first samples to discard
burnin = 0
extra_thin = 1
with h5py.File(hdf_file, 'r') as f:
    samples_cold = f['samples_cold'][0, burnin::extra_thin, :]


N = 1000
f = np.linspace(-9, math.log(3e-7, 10), N)
f = np.logspace(-8.6, -7, 30)
freqs_30 = f

def omega_GW(f, A_cp, gamma):
    return 2*np.pi**2*(10**A_cp)**2*f_yr**2/(3*H_0**2)*(f/f_yr)**(5-gamma)


dir = '30f_fs{hd}_ceffyl'
freqs = np.load('blue/data/NANOGrav15yr_KDE-FreeSpectra_v1.0.0/' + dir + '/freqs.npy')
num_freqs = 30

A_arr = samples_cold[:, -1]
gamma_arr = samples_cold[:, -2]
OMG_15 = np.zeros((67, num_freqs))

freqs_boundary = [freqs[0], freqs[-1]]

Omega_sig_f_low = []
Omega_sig_f_high = []

for ii in range(67):
    OMG_15[ii] = np.log10(h**2*omega_GW(freqs, A_arr[ii], gamma_arr[ii]))

Omega_sig_f_low += [10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0))[0], 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0))[0]]
Omega_sig_f_low += [10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0))[0], 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0))[0]]
Omega_sig_f_low += [10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0))[0], 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0))[0]]

Omega_sig_f_high += [10**(OMG_15.mean(axis=0) - 1*OMG_15.std(axis=0))[-1], 10**(OMG_15.mean(axis=0) + 1*OMG_15.std(axis=0))[-1]]
Omega_sig_f_high += [10**(OMG_15.mean(axis=0) - 2*OMG_15.std(axis=0))[-1], 10**(OMG_15.mean(axis=0) + 2*OMG_15.std(axis=0))[-1]]
Omega_sig_f_high += [10**(OMG_15.mean(axis=0) - 3*OMG_15.std(axis=0))[-1], 10**(OMG_15.mean(axis=0) + 3*OMG_15.std(axis=0))[-1]]

#fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(6,10),constrained_layout = True)
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3,
    figsize=(18, 5),
)

fig.subplots_adjust(top=0.85, bottom=0.15)

 
N = 2000
num_freqs = N
H_inf = 1  # in GeV
tau_r = 5.494456683825391e-7  # calculated from equation (19)
a_r = 1/(tau_r*H_inf)

f_UV = 1/tau_r / (2*np.pi)
tau_m = 1e27*tau_r

freqs = np.logspace(-19,np.log10(f_UV),num_freqs)

M_arr = np.linspace(0.00000001, 1.499999, N)
colors = ['black', 'black']
lw = 2
thicker = 1.5

omega_arr = [omega_GW_full(freqs_boundary[0], M_arr*H_inf, H_inf, tau_r, tau_m), omega_GW_full(freqs_boundary[1], M_arr*H_inf, H_inf, tau_r, tau_m)]
last_idx = len(omega_arr[0])-1
partition_idx_low = [last_idx,last_idx,last_idx,last_idx,last_idx,last_idx]
partition_idx_high = [last_idx,last_idx,last_idx,last_idx,last_idx,last_idx]
i = last_idx
while i >= 0:
    for j in range(6):
        if omega_arr[0][i] <= Omega_sig_f_low[j]:
            partition_idx_low[j] -= 1
        if omega_arr[1][i] <= Omega_sig_f_high[j]:
            partition_idx_high[j] -= 1
    i-=1

ax1.plot(
    M_arr, omega_arr[0],
    color='black', linewidth=lw,
    label=r'$f_{\mathrm{low}}$'
)

ax1.plot(
    M_arr, omega_arr[1],
    color='black', linewidth=lw,
    linestyle='dashed',
    label=r'$f_{\mathrm{high}}$'
)

for i in range(3):
    ax1.plot(M_arr[partition_idx_low[2*i+1]:partition_idx_low[2*i]], omega_arr[0][partition_idx_low[2*i+1]:partition_idx_low[2*i]] , color = 'orange', alpha = 0.7-i*0.2, linewidth = lw+thicker)
    ax1.plot(M_arr[partition_idx_low[2*i+1]:partition_idx_low[2*i]], omega_arr[1][partition_idx_low[2*i+1]:partition_idx_low[2*i]] , color = 'orange', alpha = 0.7-i*0.2, linewidth = lw+thicker)

ax1.set_yscale('log')
ax1.set_xlabel(r'$m/H_{\mathrm{inf}}$')
ax1.set_xlim(0,1.5)

tau_m_arr = np.logspace(5,31, N)
M_GW = H_inf

omega_arr = [omega_GW_full(freqs_boundary[0],  M_GW, H_inf, tau_r, tau_m_arr*tau_r), omega_GW_full(freqs_boundary[1], M_GW, H_inf, tau_r, tau_m_arr*tau_r)]
partition_idx_low = [0,0,0,0,0,0]
partition_idx_high = [0,0,0,0,0,0]
for i in range(len(omega_arr[0])):
    for j in range(6):
        if omega_arr[0][i] <= Omega_sig_f_low[j]:
            partition_idx_low[j] += 1
        if omega_arr[1][i] <= Omega_sig_f_high[j]:
            partition_idx_high[j] += 1
ax2.plot(tau_m_arr, omega_GW_full(freqs_boundary[0], M_GW, H_inf, tau_r, tau_m_arr*tau_r), color = colors[0], linewidth = lw)
ax2.plot(tau_m_arr, omega_GW_full(freqs_boundary[1], M_GW, H_inf, tau_r, tau_m_arr*tau_r), color = colors[1], linewidth = lw, linestyle='dashed')

for i in range(3):
    ax2.plot(tau_m_arr[partition_idx_low[2*i]:partition_idx_low[2*i+1]], omega_arr[0][partition_idx_low[2*i]:partition_idx_low[2*i+1]] , color = 'orange', alpha = 0.7-i*0.2, linewidth = lw+thicker)
    ax2.plot(tau_m_arr[partition_idx_low[2*i]:partition_idx_low[2*i+1]], omega_arr[1][partition_idx_low[2*i]:partition_idx_low[2*i+1]] , color = 'orange', alpha = 0.7-i*0.2, linewidth = lw+thicker)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'$\tau_m / \tau_r$')
ax2.set_xlim(1e10,1e30)
ax2.set_ylim(1e-29,1e-5)


H_inf = np.logspace(-24, 14, N) # in GeV

tau_r = 1/(a_r*H_inf)

tau_m = 1e27*tau_r 
M_GW = H_inf

omega_arr = [omega_GW_full(freqs_boundary[0],  M_GW, H_inf, tau_r, tau_m), omega_GW_full(freqs_boundary[1], M_GW, H_inf, tau_r, tau_m)]
partition_idx_low = [last_idx,last_idx,last_idx,last_idx,last_idx,last_idx]
partition_idx_high = [last_idx,last_idx,last_idx,last_idx,last_idx,last_idx]
i = last_idx
while i >= 0:
    for j in range(6):
        if omega_arr[0][i] <= Omega_sig_f_low[j]:
            partition_idx_low[j] -= 1
        if omega_arr[1][i] <= Omega_sig_f_high[j]:
            partition_idx_high[j] -= 1
    i-=1

ax3.plot(H_inf, omega_GW_full(freqs_boundary[0], M_GW, H_inf, tau_r, tau_m), color = colors[0], linewidth = lw)
ax3.plot(H_inf, omega_GW_full(freqs_boundary[1], M_GW, H_inf, tau_r, tau_m), color = colors[1], linewidth = lw, linestyle='dashed')

for i in range(3):
    ax3.plot(H_inf[partition_idx_low[2*i+1]:partition_idx_low[2*i]], omega_arr[0][partition_idx_low[2*i+1]:partition_idx_low[2*i]] , color = 'orange', alpha = 0.7-i*0.2, linewidth = lw+thicker)   
    ax3.plot(H_inf  [partition_idx_low[2*i+1]:partition_idx_low[2*i]], omega_arr[1][partition_idx_low[2*i+1]:partition_idx_low[2*i]] , color = 'orange', alpha = 0.7-i*0.2, linewidth = lw+thicker)



ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel(r'$H_{\mathrm{inf}}$ [GeV]')
ax3.set_xlim(1e-20,1e10)
ax3.set_ylim(1e-18,1e7)

axs = [ax1,ax2,ax3]

for i in range(3):
    axs[i].set_ylabel(r'$h_0^2\Omega_{\mathrm{GW}}$')
    axs[i].grid(which='major', alpha=.2)
    axs[i].grid(which='minor', alpha=.2)
fig.legend(
    loc='upper center',
    ncol=2,
    frameon=False,
    fontsize=20,
    bbox_to_anchor=(0.5, 1.02)
)
plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig('nanograv/2d_figs/fig1_horizontal_legend.pdf')
plt.show()