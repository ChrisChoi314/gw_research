import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from emir_func import *  # provides: H_0, omega_M, omega_R, omega_L, M_GW, k_0, k_c, k_eq,
                         # scale_fac, d_scale_fac_dz, d_scale_fac_dz2, diffeqMG, solve_one, give_eta, etc.

# --- match your normalization ---
P_prim_k = 2.43e-10

fs = 15
plt.rcParams.update({'font.size': fs})

def A_of_k(k):
    k = np.asarray(k, dtype=float)
    return np.sqrt(P_prim_k * np.pi**2 / (2.0 * np.maximum(k, 1e-300)**3))

def w0_of_k(k, a0=1.0):
    return np.sqrt((k/a0)**2 + M_GW**2)

def gamma_env_wkb(k, a0=1.0):
    ak = solve_one(float(k))
    wk = np.sqrt((k/ak)**2 + M_GW**2)
    w0 = w0_of_k(k, a0=a0)
    return A_of_k(k) * np.sqrt(wk * ak**3 / (w0 * a0**3))

def gamma_env_num_window(k, Nosc=120, n_eta=1400, tail_frac=0.25, rtol=1e-6, atol=1e-9):
    k = float(k)

    a_k = solve_one(k)
    eta_k = give_eta(a_k)

    eta_start = eta_k / 10.0
    eta_end   = eta_k + (2.0*np.pi*Nosc)/k
    etas = np.linspace(eta_start, eta_end, n_eta)

    a_init  = scale_fac(etas[0])
    ap_init = d_scale_fac_dz(etas[0])
    u0  = a_init * A_of_k(k)
    up0 = ap_init * A_of_k(k)

    def rhs(eta, Y):
        u, up = Y
        du, dup = diffeqMG([u, up], eta, k)
        return [du, dup]

    sol = solve_ivp(
        rhs,
        (etas[0], etas[-1]),
        [float(u0), float(up0)],
        t_eval=etas,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=(etas[-1]-etas[0]) / 400.0,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for k={k}: {sol.message}")

    u = sol.y[0]
    a = np.array([scale_fac(x) for x in etas])
    gamma = u / a

    n_tail = max(80, int(tail_frac * len(etas)))
    g_tail = gamma[-n_tail:]
    gamma_end = np.sqrt(2.0*np.mean(g_tail**2))  # peak ≈ sqrt(2)*RMS
    a_end = scale_fac(eta_end)
    w_end = w0_of_k(k, a0=a_end)
    return gamma_end, a_end, w_end

def sqrtP_from_gamma(k, gamma_env):
    w0 = w0_of_k(k, a0=1.0)
    pref = w0**2 / (w0**2 - M_GW**2)
    P = pref * (2.0*k**3/np.pi**2) * (gamma_env**2)
    return np.sqrt(P)

# --- choose k range + sampling ---
kmin = 1e-4 * k_0
kmax = 1e+2 * k_0
Nk   = 17
k_list = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

# --- compute curves ---
y_num = []
y_wkb = []

for i, k in enumerate(k_list):
    print(f"[{i+1}/{len(k_list)}] k/k0={k/k_0:.2e}", flush=True)
    g_end, a_end, w_end = gamma_env_num_window(k)
    gW = gamma_env_wkb(k)

    a0 = 1.0
    w0 = w0_of_k(k, a0=a0)

    # propagate numerical envelope from window end -> today using WKB scaling
    g0_num = g_end * np.sqrt((w_end * a_end**3) / (w0 * a0**3))

    y_num.append(sqrtP_from_gamma(k, g0_num))
    y_wkb.append(sqrtP_from_gamma(k, gW))


y_num = np.array(y_num)
y_wkb = np.array(y_wkb)

# --------- ONLY CHANGE: x-axis is omega_0 instead of k/k0 ----------
omega0_list = w0_of_k(k_list, a0=1.0)

# --- plot (no markers; WKB solid, numerical dotted) ---
fig, ax = plt.subplots(figsize=(8.0, 5.4))
ax.plot(omega0_list, y_wkb, lw=3.0, ls='-', label='WKB Approximation')
ax.plot(omega0_list, y_num, lw=3.0, ls=':', label='Numerical ODE ')


# --- label the vertical markers ---
ymin, ymax = ax.get_ylim()
y_text = ymin * 3.0  # a little above the bottom on log-y



ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\omega_0$ [Hz]')
ax.set_ylabel(r'$[P(\omega_0)]^{1/2}$ ')
#ax.grid(True, which='both', alpha=0.3)
ax.legend()

plt.savefig("wkb_num_figs/wkb.pdf", bbox_inches="tight")
plt.show()
