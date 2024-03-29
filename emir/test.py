import numpy as  np
from sympy import *

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

k_arr = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]

x, y, z, t = symbols('x y z t')

'''
def sqrt(x):
    print(x)
    return np.sqrt(x + 0.j)
'''

def scale_fac(conf_time):
    if conf_time < eta_rm:
        return H_0*np.sqrt(omega_R)*conf_time
    else:
        # return H_0**2*.25*omega_M*conf_time**2
        return (H_0**2*.25*omega_M) * (((a_eq/(H_0**2*.25*omega_M))**(1/2) - a_eq/(H_0*np.sqrt(omega_R))) + conf_time)**2

def odd_root(x, r):
    # return np.sign(x) * np.abs(x) ** (1 / r)
    return x**(1/r)

# sols = list(solveset(y**2*x**2 + x**4*M_GW**2 - H_0**2* (omega_M * x + omega_R + x**4*omega_L),x))

for k in k_arr:
    pass
    # print(k, N(sols[0].subs({y: k}), 5))

    # sols = list(solveset((y**2*x**2 + x**4*M_GW**2 - H_0**2* (omega_M * x + omega_R + x**4*omega_L)).subs({y: k}),x))

def solve(k):
    from sympy import nroots, abs, symbols, re, im
    x, y = symbols('x y z t')
    sols = list(nroots((y**2*x**2 + x**4*M_GW**2 - H_0**2* (omega_M * x + omega_R + x**4*omega_L)).subs({y: k}), n=100, maxsteps=10000))
    for sol in sols:
        if abs(im(sol)) < 1e-20 and re(sol) >= 0:
            return float(re(sol))
    raise RuntimeError("no solution")

    # print((-y**6 - 4.896e-50*y**2 - 7.776e-81))
    # print((-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3))
    '''
    solveset(y**2*x**2 + x**4*M_GW**2 - H_0**2* (omega_M * x + omega_R + x**4*omega_L),x)
    res = [
        ((-2041241.45231932*sqrt(-y**2 - 0.629960524947436*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3)) - 2886751.34594813*sqrt(-y**2 + 0.314980262473718*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3) - 4.40908153700972e-41/sqrt(-y**2 - 0.629960524947436*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3)))) if (5.20833333333334e+25*y**4 - 8.5e-25 == 0) else (-2041241.45231932*sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 0.499999999999999*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) - 2886751.34594813*sqrt(-y**2 + 4.80000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 4.40908153700972e-41/sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 0.499999999999999*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3))))), 
        ((-2041241.45231932*sqrt(-y**2 - 0.629960524947436*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3)) + 2886751.34594813*sqrt(-y**2 + 0.314980262473718*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3) - 4.40908153700972e-41/sqrt(-y**2 - 0.629960524947436*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3)))) if (5.20833333333334e+25*y**4 - 8.5e-25 == 0) else (-2041241.45231932*sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 0.499999999999999*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) + 2886751.34594813*sqrt(-y**2 + 4.80000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 4.40908153700972e-41/sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 0.499999999999999*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3))))), 

        ((2041241.45231932*sqrt(-y**2 - 0.629960524947436*odd_root(-y**6 - 4.896e-50*y**2 - 7.776e-81, 3)) - 2886751.34594813*sqrt(-y**2 + 0.314980262473718*odd_root(-y**6 - 4.896e-50*y**2 - 7.776e-81,3) + 4.40908153700972e-41/sqrt(-y**2 - 0.629960524947436*odd_root(-y**6 - 4.896e-50*y**2 - 7.776e-81,3)))) if (5.20833333333334e+25*y**4 - 8.5e-25 == 0) else (2041241.45231932*sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/odd_root(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81,3) + 0.499999999999999*odd_root(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81, 3)) - 2886751.34594813*sqrt(-y**2 + 4.80000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/odd_root(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81,3) - 0.25*odd_root(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81,3) + 4.40908153700972e-41/sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/odd_root(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81,3) + 0.499999999999999*odd_root(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81,3))))), 
        
        ((2041241.45231932*sqrt(-y**2 - 0.629960524947436*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3)) + 2886751.34594813*sqrt(-y**2 + 0.314980262473718*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3) + 4.40908153700972e-41/sqrt(-y**2 - 0.629960524947436*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**(1/3)))) if (5.20833333333334e+25*y**4 - 8.5e-25 == 0) else (2041241.45231932*sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 0.499999999999999*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) + 2886751.34594813*sqrt(-y**2 + 4.80000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 4.40908153700972e-41/sqrt(-y**2 - 9.60000000000001e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 0.499999999999999*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)))))
    ]
    print(k, N(res[2].subs({y: k}), 5))
    '''

# Unused functions that I moved into here

def a_rd(eta):
    return H_0*np.sqrt(omega_R)*eta


def a_md(eta):
    return H_0**2*.25*omega_M*eta**2


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
    '''

    result = (
        c1 * np.sqrt(
            (2 * x**2)/(3 * c) + (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)) + odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c)
        )
        +
        c2 * np.sqrt(
            (4 * x**2)/(3 * c) - (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)) - odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4+0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c) - c1 * (2 * a) /
            (c * np.sqrt((1 + 0j) * (2 * x**2)/(3 * c) + (2**(1/3) * (12 * b * c + x**4))/(3 * c * odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4 + 0j) * (12 * b * c * + x**4)**3 + (27 * a**2 * c + 72 * b * c * x ** 2 - 2 * x**6)**2), 3)) + odd_root(27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6 + np.sqrt(-(4 + 0j) * (12 * b * c + x**4)**3 + (27 * a**2 * c + 72 * b * c * x**2 - 2 * x**6)**2), 3)/(3 * 2**(1/3) * c)))
        )
    )/2
    result = result.real
    return result


def inv_approx(a, b, x, c1):
    return (a + c1*np.sqrt(a**2 + 4*b*x**2)) / (2*x**2)


def reg_N(a, b, c, x):
    return np.sqrt(a/x + b/x**2 + c*x**2)

def my_sqrt(x):
    return np.sqrt(x + 0.j)

def H_omega(x):
    A, B, C, D, E = H_0, omega_R, omega_M, omega_L, M_GW

    return x*my_sqrt(A**2 * (C/x**3 + B*x**4 + D) - E**2)
def inverse_H_omega(x):
    pm1 = -1 # first is -1
    pm2 = 1 # first is 1

    A, B, C, D, E = H_0, omega_R, omega_M, omega_L, M_GW

    result = pm2*0.5 * my_sqrt(-(0.666667 * x**2)/(E**2 - A**2 * D) + (0.264567*(2 * x**6 + 72 * A**2 * (E**2 - A**2 * D) * B*x**2 + 27 * A**4 * (E**2 - A**2 * D) * C**2 + my_sqrt((2 * x**6 + 72 * A**2 * (E**2 - A**2 * D) * B * x**2 + 27 * A**4 * (E**2 - A**2 * D) * C**2)**2 - 4 * (x**4 - 12 * A**2 * (E**2 - A**2 * D) * B)**3))**(1/3))/(E**2 - A**2 * D) + (0.419974 * (12 * D * B * A**4 - 12 * E**2 * B * A**2 + x**4))/((E**2 - A**2*D)*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3)))+ pm1*0.5*my_sqrt(-pm2*(2*C*A**2)/((A**2*D - E**2)*my_sqrt(-(0.666667*x**2)/(E**2 - A**2*D) + (0.264567*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*( 
            E**2 - A**2*D)*B)**3))**(1/3))/(E**2 - A**2*D) + (0.419974*(12*D*B*A**4 - 12*E**2*B*A**2 + x**4))/((E**2 - A**2*D)*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3)))) - (0.264567*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3))/(E**2 - A**2*D) - (0.419974*(12*D*B*A**4 - 12*E**2*B*A**2 + x**4))/((E**2 - A**2*D)*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + 
                my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3)) - (1.33333*x**2)/(E**2 - A**2*D))
    result = result.real
    return result

def inverse_H_omega2(x):
    pm2 = 1 # first is 1

    A, B, C, D, E = H_0, omega_R, omega_M, omega_L, M_GW

    output = pm2*0.5 * my_sqrt((2*C*A**2)/((A**2*D - E**2)*my_sqrt(-(0.666667 * x**2)/(E**2 - A**2 * D) + (0.264567*(2 * x**6 + 72 * A**2 * (E**2 - A**2 * D) * B*x**2 + 27 * A**4 * (E**2 - A**2 * D) * C**2 + my_sqrt((2 * x**6 + 72 * A**2 * (E**2 - A**2 * D) * B * x**2 + 27 * A**4 * (E**2 - A**2 * D) * C**2)**2 - 4 * (x**4 - 12 * A**2 * (E**2 - A**2 * D) * B)**3))**(1/3))/(E**2 - A**2 * D) + (0.419974 * (12 * D * B * A**4 - 12 * E**2 * B * A**2 + x**4))/((E**2 - A**2*D)*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3))))-(0.264567*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(
        E**2 - A**2*D)*B)**3))**(1/3))/(E**2 - A**2*D) - (0.419974*(12*D*B*A**4 - 12*E**2*B*A**2 + x**4))/((E**2 - A**2*D)*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3)) - (1.33333 *x**2)/(E**2 - A**2*D)) - 0.5*my_sqrt(-(0.666667*x**2)/E**2 - A**2*D+ (0.264567*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3))/(E**2 - A**2*D) + (0.419974*(12*D*B*A**4 - 12*E**2*B*A**2 + x**4))/((E**2 - A**2*D)*(2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2 + my_sqrt((2*x**6 + 72*A**2*(E**2 - A**2*D)*B*x**2 + 27*A**4*(E**2 - A**2*D)*C**2)**2 - 4*(x**4 - 12*A**2*(E**2 - A**2*D)*B)**3))**(1/3)))
    return output.real

def sympy1(y):
    sqrt = my_sqrt
    output = (-2041241.45231932*sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) - 2886751.34594813*sqrt(-y**2 + 4.8e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 4.40908153700972e-41/sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3))))
    return output.real

def sympy2(y):
    sqrt = my_sqrt
    output = (-2041241.45231932*sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) + 2886751.34594813*sqrt(-y**2 + 4.8e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 4.40908153700972e-41/sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3))))
    return output.real

def sympy3(y):
    sqrt = my_sqrt
    output = (2041241.45231932*sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) - 2886751.34594813*sqrt(-y**2 + 4.8e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 4.40908153700972e-41/sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3))))
    return output.real

def sympy4(y):
    sqrt = my_sqrt
    output = (2041241.45231932*sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3)) + 2886751.34594813*sqrt(-y**2 + 4.8e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) - 0.25*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + 4.40908153700972e-41/sqrt(-y**2 - 9.6e-27*(8.5e-25 - 5.20833333333334e+25*y**4)/(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3) + .5*(1.0*y**6 + 4.896e-50*y**2 + sqrt((1.632e-50 - y**4)**3 + 1.0*(-y**6 - 4.896e-50*y**2 - 7.776e-81)**2) + 7.776e-81)**(1/3))))
    return output.real


# from emir_P.py 
eta = np.logspace(1, 18, N)
# eta = np.logspace(-3, -1, N)
a = np.vectorize(scale_fac)(eta)
# a = normalize_0(a)
v_0 = 1
v_prime_0 = 0
eta_0_idx = 0
for i in range(len(eta)):
    if eta[i] >= eta_0:
        eta_0_idx = i
        break
for k in k_arr:
    v, v_prime = odeint(diffeqGR, [v_0, v_prime_0], eta, args=(k,)).T
    # print(v[eta_0_idx]/a[eta_0_idx])
    # ax1.plot(eta, v/a, label=f"{k}" + r" $Hz$")
    v, v_prime = odeint(diffeqMG, [v_0, v_prime_0], eta, args=(k,)).T
    # print(v[eta_0_idx]/a[eta_0_idx])
    # ax2.plot(eta, v/a, label=f"{k}" + r" $Hz$")
eta = np.logspace(-7, 1, N)
# k = np.logspace(-5,5, N)