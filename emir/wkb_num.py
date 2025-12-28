import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from emir_func import *  # expects: M_GW, k_0, k_c, k_eq, K (or K=0),
                         # scale_fac, d_scale_fac_dz, d_scale_fac_dz2,
                         # diffeqMG, solve_one, give_eta

os.makedirs("wkb_num_figs", exist_ok=True)

P_prim_k = 2.43e-10

def A_of_k(k):
    k = np.asarray(k, dtype=float)
    return np.sqrt(P_prim_k * np.pi**2 / (2.0 * np.maximum(k, 1e-300)**3))

def omega_phys(k, a):
    return np.sqrt((k/a)**2 + M_GW**2)

def Omega_conformal(k, eta):
    # frequency for u-equation: u'' + Omega^2 u = 0
    a = scale_fac(eta)
    term = k**2 + (a*M_GW)**2 - d_scale_fac_dz2(eta)/a + 2.0*K
    return np.sqrt(max(term, 0.0))

def gamma_env_wkb_today(k, a0=1.0):
    ak = solve_one(float(k))
    wk = omega_phys(k, ak)
    w0 = omega_phys(k, a0)
    return A_of_k(k) * np.sqrt(wk * ak**3 / (w0 * a0**3))

def gamma_env_num_window(k, Nosc=120, n_eta=1600, tail_frac=0.25, rtol=1e-6, atol=1e-9):
    """
    Numerical NO-WKB solve near entry, but with a window sized by conformal frequency:
      eta_end = eta_k + (2π Nosc)/Omega_entry
    Returns: gamma_env_end, eta_end, a_end, w_end
    """
    k = float(k)

    a_k = solve_one(k)
    eta_k = give_eta(a_k)

    eta_start = eta_k / 10.0

    Omega_entry = Omega_conformal(k, eta_k)
    if Omega_entry <= 0:
        return np.nan, np.nan, np.nan, np.nan

    eta_end = eta_k + (2.0*np.pi*Nosc)/Omega_entry
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
        max_step=(etas[-1]-etas[0]) / 600.0,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for k={k}: {sol.message}")

    u = sol.y[0]
    a = np.array([scale_fac(x) for x in etas])
    gamma = u / a

    n_tail = max(80, int(tail_frac * len(etas)))
    g_tail = gamma[-n_tail:]
    gamma_env_end = np.sqrt(2.0*np.mean(g_tail**2))  # peak ≈ sqrt(2)*RMS

    a_end = scale_fac(eta_end)
    w_end = omega_phys(k, a_end)
    return gamma_env_end, eta_end, a_end, w_end

def sqrtP_today_from_gamma0(k, gamma0, a0=1.0):
    w0 = omega_phys(k, a0)
    pref = w0**2 / (w0**2 - M_GW**2)
    P = pref * (2.0*k**3/np.pi**2) * (gamma0**2)
    return np.sqrt(P)

# --- go lower in k/k0 ---
kmin = 1e-10 * k_0    # <<<<< push down as far as you like
kmax = 1e+2  * k_0
Nk   = 30             # smoother curve across more decades
k_list = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

a0 = 1.0  # set to a_0 if your convention uses non-1 today

y_num = []
y_wkb = []
k_ok  = []

for i, k in enumerate(k_list):
    print(f"[{i+1}/{len(k_list)}] k/k0={k/k_0:.2e}", flush=True)

    try:
        g_end, eta_end, a_end, w_end = gamma_env_num_window(k, Nosc=120, n_eta=1600)
        if not np.isfinite(g_end):
            continue

        # propagate numerical envelope to today using WKB scaling
        w0 = omega_phys(k, a0)
        g0_from_num = g_end * np.sqrt((w_end * a_end**3) / (w0 * a0**3))

        g0_wkb = gamma_env_wkb_today(k, a0=a0)

        y_num.append(sqrtP_today_from_gamma0(k, g0_from_num, a0=a0))
        y_wkb.append(sqrtP_today_from_gamma0(k, g0_wkb,      a0=a0))
        k_ok.append(k)

    except Exception as e:
        print(f"  skipping k/k0={k/k_0:.2e} due to: {e}", flush=True)
        continue

k_ok  = np.array(k_ok)
y_num = np.array(y_num)
y_wkb = np.array(y_wkb)

fig, ax = plt.subplots(figsize=(8.0, 5.4))
ax.plot(k_ok/k_0, y_wkb, lw=3.0, ls='-', label='WKB (today)')
ax.plot(k_ok/k_0, y_num, lw=3.0, ls=':', label='Numerical (to η_end) + WKB propagation')

ax.axvline(k_c/k_0,  ls='--', lw=1.8)
ax.axvline(k_eq/k_0, ls='--', lw=1.8)
ax.axvline(1.0,      ls='--', lw=1.8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k/k_0$')
ax.set_ylabel(r'$[P(\omega_0)]^{1/2}$')
ax.set_title(r'WKB vs numerical with consistent late-time propagation')
ax.grid(True, which='both', alpha=0.3)
ax.legend()

plt.savefig("wkb_num_figs/wkb_quality_sqrtP_vs_k_lowk_extended.pdf", bbox_inches="tight")
plt.show()
