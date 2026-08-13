# src/simulation/tmm_engine.py
"""
Transfer Matrix Method (TMM) engine.
Standard characteristic matrix formulation with oblique incidence support.
"""
import numpy as np


def tmm_calc(stack, lam_nm, theta_deg=0.0, pol="TE"):
    """
    Standard TMM calculation.
    
    stack: list of {"d": thickness_nm, "eps": complex permittivity}
           First and last entries are semi-infinite input/output media (d ignored).
    lam_nm: wavelength in nm
    theta_deg: incidence angle in degrees
    pol: "TE" or "TM"
    
    Returns: (R_power, T_power, A)  where A = 1 - R - T
    """
    lam = lam_nm * 1e-9
    k0 = 2 * np.pi / lam

    n_list = [np.sqrt(layer["eps"] + 0j) for layer in stack]
    d_list = [layer["d"] * 1e-9 for layer in stack]

    # Snell's law: n0*sin(theta0) = ni*sin(thetai)
    theta0 = np.deg2rad(theta_deg)
    cos_t = []
    for n in n_list:
        sin_ti = n_list[0] * np.sin(theta0) / n
        cos_ti = np.sqrt(1 - sin_ti**2 + 0j)
        if cos_ti.imag < 0:
            cos_ti = -cos_ti
        cos_t.append(cos_ti)

    # Admittances
    if pol == "TE":
        eta = [n * ct for n, ct in zip(n_list, cos_t)]
    else:
        eta = [n / ct for n, ct in zip(n_list, cos_t)]

    # Characteristic matrix product (layers 1..N-2, excluding input/output)
    M = np.eye(2, dtype=complex)
    for i in range(1, len(stack) - 1):
        delta = k0 * n_list[i] * cos_t[i] * d_list[i]
        layer_M = np.array([
            [np.cos(delta), -1j * np.sin(delta) / eta[i]],
            [-1j * eta[i] * np.sin(delta), np.cos(delta)]
        ])
        M = M @ layer_M

    # Fresnel coefficients from total matrix
    num_r = (M[0, 0] + M[0, 1] * eta[-1]) * eta[0] - (M[1, 0] + M[1, 1] * eta[-1])
    den = (M[0, 0] + M[0, 1] * eta[-1]) * eta[0] + (M[1, 0] + M[1, 1] * eta[-1])
    r = num_r / den
    t = 2 * eta[0] / den

    R = float(np.abs(r) ** 2)
    T = float(np.abs(t) ** 2 * np.real(eta[-1]) / np.real(eta[0]))
    A = max(0.0, 1.0 - R - T)
    return R, T, A


# Backward-compatible aliases
def _tmm_rt_normal(stack, lam_nm):
    R, T, _ = tmm_calc(stack, lam_nm, theta_deg=0.0, pol="TE")
    return R, T

def _tmm_rt_oblique(stack, lam_nm, theta_rad, polarization="TE"):
    R, T, _ = tmm_calc(stack, lam_nm, theta_deg=np.rad2deg(theta_rad), pol=polarization)
    return R, T
