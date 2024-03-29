import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

c_g = 1  # speed of graviton
hbar = 6.582119569e-25  # GeV / Hz
c = 3e8 / 3.086e25  # m * Hz * (1Gpc / 3.086e25m) = Gpc * Hz
M_GW = 2e-7*hbar  # Hz * GeV / Hz = GeV
H_0 = M_GW/1e10  # my val, from page 13 in Emir paper
M_pl = 1.22089e19  # GeV
H_0 = 1/2997.9*.7*1000 * c*hbar  # GeV, Jacob's val
k_0 = 1e10*H_0
k_c = 1e4*H_0  # both k_c and k_0 defined in same place in Emir paper
omega_M = .3
omega_R = 8.5e-5
omega_L = .7
eta_rm = .1

eta_ml = 12.5
K = 0
M_pl /= hbar

T = 6.58e-25
L = 1.97e-16
m2Gpc = 3.1e25

MGW = 2e-7


def hz2gpc(hz): return hz*(T/L)*m2Gpc
def gpc2hz(gpc): return gpc*(1/m2Gpc)*L/T


M_pl = hz2gpc(M_pl)
M_GW = MGW  # hz2gpc(MGW) # in Gpc
H_0 = 1/2997.9*.7*1000  # Gpc^-1
H_0 = M_GW/1e10
k_0 = 1e10*H_0  # in Gpc
k_c = 1e4*H_0
eta_rm = .1
a_c = k_c / M_GW
a_0 = k_0 / M_GW
eta_0 = 1.763699706

eta_0 = 1.826e17
eta_rm = 4*np.sqrt(omega_R)/omega_M/H_0


def a_rd(eta):
    return H_0*np.sqrt(omega_R)*eta


def a_md(eta):
    return H_0**2*.25*omega_M*eta**2


def scale_fac(conf_time):
    if conf_time < eta_rm:
        return H_0*np.sqrt(omega_R)*conf_time
    else:
        return H_0**2*.25*omega_M*conf_time**2


def d_scale_fac_dz(conf_time):
    if conf_time < eta_rm:
        return 0.
    else:
        return 2*H_0**2*.25*omega_M


def diffeqMG(M, u):
    r = [M[1], -((c_g*k)**2 + (scale_fac(u)*M_GW)**2 -
                 d_scale_fac_dz(u) / scale_fac(u) + 2*K*c_g**2) * M[0]]
    return r


def diffeqGR(M, u):
    r = [M[1], -((c_g*k)**2 -
                 d_scale_fac_dz(u) / scale_fac(u) + 2*K*c_g**2) * M[0]]
    return r


def Hubble(conf_t):
    return H_0*np.sqrt(omega_M / scale_fac(conf_t)**3 + omega_R / scale_fac(conf_t)**4 + omega_L)


def ang_freq(conf_t,k):
    return np.sqrt(k**2/scale_fac(conf_t)**2 + M_GW**2)

def Hubble_a(x):
    return H_0*np.sqrt(omega_M / x**3 + omega_R / x**4 + omega_L)


def ang_freq_a(x,k):
    return np.sqrt(k**2/x**2 + M_GW**2)


def ang_freq_GR(conf_t,k):
    return np.sqrt(k**2/scale_fac(conf_t)**2)


def normalize(array):
    max = array.max()
    array /= max
    return array


def normalize_0(array):
    max = array[0]
    array /= max
    return array

# Calculates x ** (1 / r) where r is an odd positive integer


def odd_root(x, r):
    '''
    return np.sign(x) * np.abs(x) ** (1 / r)
    '''
    return x ** (1 / r)


def inv_of_H(a, b, c, x, c1, c2):
    '''
    plus_minus = 1
    plus_minus_2 = -1
    print('inv_of_H')
    print(-4 * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 -
                                         2 * x**6)**2)
    print(
        (2 * x**2)/(3 * c) + (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 -
                                                                                                                                                              2 * x**6)**2), 3)) + odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c))
    print(
        (4 * x**2)/(3 * c) - (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)) - odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c) + plus_minus_2 * (2 * a) /
        (c * np.sqrt((1 + 0j) * (2 * x**2)/(3 * c) + (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4 + 0j) * (12 * b * c * + x**4)**3 + (27 * a**2 * c + 72 * b * c * x **
                                                                                                                                                                             2 - 2 * x**6)**2), 3)) + odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4 + 0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c))))
    '''

    result = (
        c1 * np.sqrt(
            (2 * x**2)/(3 * c) + (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 -2 * x**6)**2), 3)) + odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c)
        )
        +
        c2 * np.sqrt(
            (4 * x**2)/(3 * c) - (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)) - odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c) - c1 * (2 * a) /
            (c * np.sqrt((1 + 0j) * (2 * x**2)/(3 * c) + (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4 + 0j) * (12 * b * c * + x**4)**3 + (27 * a**2 * c + 72 * b * c * x **2 - 2 * x**6)**2), 3)) + odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4 + 0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c)))
        )
    )/2
    print(result)
    result = result.real

    return result


def inv_approx(a, b, x, c1):
    return (a + c1*np.sqrt(a**2 + 4*b*x**2)) / (2*x**2)


def reg_N(a, b, c, x):
    return np.sqrt(a/x + b/x**2 + c*x**2)

def my_sqrt(x):
    return np.sqrt(x + 0.j)

def inverse_H_omega(x):
    pm1 = -1 # first is -1
    pm2 = 1 # first is 1

    #print((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3)
    
    result = pm2*0.5 * my_sqrt(-(0.666667 * x**2)/(M_GW**2 - H_0**2 * omega_L) + (0.264567*(2 * x**6 + 72 * H_0**2 * (M_GW**2 - H_0**2 * omega_L) * omega_R*x**2 + 27 * H_0**4 * (M_GW**2 - H_0**2 * omega_L) * omega_M**2 + my_sqrt((2 * x**6 + 72 * H_0**2 * (M_GW**2 - H_0**2 * omega_L) * omega_R * x**2 + 27 * H_0**4 * (M_GW**2 - H_0**2 * omega_L) * omega_M**2)**2 - 4 * (x**4 - 12 * H_0**2 * (M_GW**2 - H_0**2 * omega_L) * omega_R)**3))**(1/3))/(M_GW**2 - H_0**2 * omega_L) + (0.419974 * (12 * omega_L * omega_R * H_0**4 - 12 * M_GW**2 * omega_R * H_0**2 + x**4))/((M_GW**2 - H_0**2*omega_L)*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3)))+ pm1*0.5*my_sqrt(-pm2*(2*omega_M*H_0**2)/((H_0**2*omega_L - M_GW**2)*my_sqrt(-(0.666667*x**2)/(M_GW**2 - H_0**2*omega_L) + (0.264567*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*( 
        M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3))/(M_GW**2 - H_0**2*omega_L) + (0.419974*(12*omega_L*omega_R*H_0**4 - 12*M_GW**2*omega_R*H_0**2 + x**4))/((M_GW**2 - H_0**2*omega_L)*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3)))) - (0.264567*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3))/(M_GW**2 - H_0**2*omega_L) - (0.419974*(12*omega_L*omega_R*H_0**4 - 12*M_GW**2*omega_R*H_0**2 + x**4))/((M_GW**2 - H_0**2*omega_L)*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + 
            my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3)) - (1.33333*x**2)/(M_GW**2 - H_0**2*omega_L))
    result = result.imag
    return result

def inverse_H_omega2(x):
    pm1 = -1 # first is -1
    pm2 = 1 # first is 1
    
    output = pm2*0.5 * my_sqrt((2*omega_M*H_0**2)/((H_0**2*omega_L - M_GW**2)*my_sqrt(-(0.666667 * x**2)/(M_GW**2 - H_0**2 * omega_L) + (0.264567*(2 * x**6 + 72 * H_0**2 * (M_GW**2 - H_0**2 * omega_L) * omega_R*x**2 + 27 * H_0**4 * (M_GW**2 - H_0**2 * omega_L) * omega_M**2 + my_sqrt((2 * x**6 + 72 * H_0**2 * (M_GW**2 - H_0**2 * omega_L) * omega_R * x**2 + 27 * H_0**4 * (M_GW**2 - H_0**2 * omega_L) * omega_M**2)**2 - 4 * (x**4 - 12 * H_0**2 * (M_GW**2 - H_0**2 * omega_L) * omega_R)**3))**(1/3))/(M_GW**2 - H_0**2 * omega_L) + (0.419974 * (12 * omega_L * omega_R * H_0**4 - 12 * M_GW**2 * omega_R * H_0**2 + x**4))/((M_GW**2 - H_0**2*omega_L)*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3))))-(0.264567*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(
        M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3))/(M_GW**2 - H_0**2*omega_L) - (0.419974*(12*omega_L*omega_R*H_0**4 - 12*M_GW**2*omega_R*H_0**2 + x**4))/((M_GW**2 - H_0**2*omega_L)*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3)) - (1.33333 *x**2)/(M_GW**2 - H_0**2*omega_L)) - 0.5*my_sqrt(-(0.666667*x**2)/M_GW**2 - H_0**2*omega_L+ (0.264567*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3))/(M_GW**2 - H_0**2*omega_L) + (0.419974*(12*omega_L*omega_R*H_0**4 - 12*M_GW**2*omega_R*H_0**2 + x**4))/((M_GW**2 - H_0**2*omega_L)*(2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2 + my_sqrt((2*x**6 + 72*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R*x**2 + 27*H_0**4*(M_GW**2 - H_0**2*omega_L)*omega_M**2)**2 - 4*(x**4 - 12*H_0**2*(M_GW**2 - H_0**2*omega_L)*omega_R)**3))**(1/3)))
    return output.real