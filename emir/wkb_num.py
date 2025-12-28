import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------- import your background + ODE definitions --------------------
# This assumes your "other file" is emir_func.py (or rename accordingly).
# It must provide: H_0, omega_M, omega_R, omega_L, M_GW, k_0, k_c, k_eq, a_eq, H_eq, K
# and functions: scale_fac(eta), d_scale_fac_dz(eta), d_scale_fac_dz2(eta),
# diffeqMG(M, eta, k), solve_one(k), give_eta(a)
from emir_func import *  # noqa

# -------------------- primordial normalization (match your script) --------------------
P_prim_k = 2.43e-10

def A_of_k(k):
    # Your A(k) definition for scale-invariant P_prim. For ratios it cancels,
    # but keep consistent with your pipeline.
    k = np.asarray(k, dtype=float)
    return np.sqrt(P_prim_k * np.pi**2 / (2.0 * np.maximum(k, 1e-300)**3))

def w0_of_k(k, a0=1.0):
    return np.sqrt((k/a0)**2 + M_GW**2)

def gamma_env_wkb(k, a0=1.0):
    """
    WKB/thin-horizon envelope for gamma today:
        gamma_env ≈ A(k) * sqrt( ω_k a_k^3 / (ω0 a0^3) )
    where a_k is determined by your solve_one(k).
    """
    ak = solve_one(float(k))
    wk = np.sqrt((k/ak)**2 + M_GW**2)
    w0 = w0_of_k(k, a0=a0)
    return A_of_k(k) * np.sqrt(wk * ak**3 / (w0 * a0**3))

def gamma_env_num_window(k, a0=1.0, Nosc=200, n_eta=2000, tail_frac=0.2,
                         rtol=1e-6, atol=1e-9):
    """
    Numerical NO-WKB solve of u'' + [k^2 + a^2 M^2 - a''/a + 2K] u = 0
    in a finite window around/after entry so it runs fast.

    Steps:
      1) compute a_k from solve_one(k)
      2) eta_k from give_eta(a_k)
      3) integrate from eta_k/10 to eta_k + Nosc*(2π/k)
      4) convert u -> gamma = u/a and estimate envelope from RMS tail
    """
    k = float(k)

    # Entry scale factor/time
    a_k = solve_one(k)
    eta_k = give_eta(a_k)

    # Window: start a bit before entry, end after Nosc oscillations
    eta_start = eta_k / 10.0
    eta_end   = eta_k + (2.0*np.pi*Nosc)/k

    etas = np.linspace(eta_start, eta_end, n_eta)

    # ICs: superhorizon-ish gamma≈A(k), gamma'≈0  => u=a*gamma, u'≈a'*A(k)
    a_init  = scale_fac(etas[0])
    ap_init = d_scale_fac_dz(etas[0])
    u0  = a_init * A_of_k(k)
    up0 = ap_init * A_of_k(k)

    def rhs(eta, Y):
        # diffeqMG returns [u', u''] given M=[u,u']
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
        max_step=(etas[-1]-etas[0]) / 400.0,  # prevents pathological micro-steps
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for k={k}: {sol.message}")

    u = sol.y[0]
    a = np.array([scale_fac(x) for x in etas])
    gamma = u / a

    # Envelope estimate near end: peak ≈ sqrt(2)*RMS over last tail window
    n_tail = max(80, int(tail_frac * len(etas)))
    g_tail = gamma[-n_tail:]
    gamma_env = np.sqrt(2.0*np.mean(g_tail**2))
    return gamma_env

def sqrtP_from_gamma(k, gamma_env, a0=1.0):
    """
    Same mapping you use:
      P(ω0) = ω0^2/(ω0^2 - M^2) * (2k^3/π^2) * gamma^2
    Plot sqrt(P).
    """
    w0 = w0_of_k(k, a0=a0)
    pref = w0**2 / (w0**2 - M_GW**2)
    P = pref * (2.0*k**3/np.pi**2) * (gamma_env**2)
    return np.sqrt(P)

# Choose k range relative to k0 (tweak for your slide)
kmin = 1e-4 * k_0
kmax = 1e+2 * k_0
Nk   = 14
k_list = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

# Window controls: smaller Nosc => faster
Nosc = 120        # 80–200 is a good range
n_eta = 1400      # 800–2000 is fine for a slide-quality comparison

y_num = []
y_wkb = []

for i, k in enumerate(k_list):
    print(f"[{i+1}/{len(k_list)}] k/k0={k/k_0:.2e}", flush=True)

    gN = gamma_env_num_window(k, Nosc=Nosc, n_eta=n_eta, tail_frac=0.25,
                                rtol=1e-6, atol=1e-9)
    gW = gamma_env_wkb(k)

    y_num.append(sqrtP_from_gamma(k, gN))
    y_wkb.append(sqrtP_from_gamma(k, gW))

y_num = np.array(y_num)
y_wkb = np.array(y_wkb)

fig, ax = plt.subplots(figsize=(8.0, 5.4))
ax.plot(k_list/k_0, y_num, marker='o', lw=2.5, label='Numerical ODE (window after entry)')
ax.plot(k_list/k_0, y_wkb, marker='s', lw=2.5, ls='--', label='WKB envelope')

# Optional regime markers (comment out if you don’t want them)
ax.axvline(k_c/k_0,  ls=':', lw=2)
ax.axvline(k_eq/k_0, ls=':', lw=2)
ax.axvline(1.0,      ls=':', lw=2)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k/k_0$')
ax.set_ylabel(r'$[P]^{1/2}$ (proxy from envelope)')
ax.set_title(r'WKB vs numerical (no-WKB) mode amplitude entering $P$')
ax.grid(True, which='both', alpha=0.3)
ax.legend()

plt.savefig("wkb_num_figs/wkb_quality_sqrtP_vs_k.pdf", bbox_inches="tight")
plt.show()
