# src/simulation/tmm_struct_c_aniso.py
"""
TMM for Structure C with ANISOTROPIC EMA.

Key change: instead of isotropic f = (Wx*Wy)/P^2,
use directional fill fractions:
  - TE (Ey): f_y = Wy/P  (E-field sees y-dimension)
  - TM (Ex): f_x = Wx/P  (E-field sees x-dimension)

This gives A_tmm_te != A_tmm_tm when Wx != Wy.
"""
import numpy as np
from .materials import get_metal_permittivity, get_sio2_permittivity
from .tmm_engine import _tmm_rt_oblique, _tmm_rt_normal


def compute_tmm_batch(params, wavelengths, metal="Cr"):
    """
    Batch TMM with anisotropic EMA for Structure C.
    params: [N, 7] - [P, Wx, Wy, t_Cr, d_SiO2, theta, phi]
    wavelengths: [Nw] nm
    """
    N = params.shape[0]
    Nw = len(wavelengths)
    
    A_te = np.zeros((N, Nw))
    R_te = np.zeros((N, Nw))
    T_te = np.zeros((N, Nw))
    A_tm = np.zeros((N, Nw))
    R_tm = np.zeros((N, Nw))
    T_tm = np.zeros((N, Nw))

    eps_metal_all = get_metal_permittivity(wavelengths, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths)

    for i in range(N):
        P, Wx, Wy, t_Cr, d_SiO2, theta_deg, phi_deg = params[i]
        
        # Anisotropic EMA: directional fill fractions
        f_x = Wx / P   # TM (Ex) sees x-dimension
        f_y = Wy / P   # TE (Ey) sees y-dimension
        
        theta_rad = np.deg2rad(theta_deg)

        for j in range(Nw):
            eps_m = eps_metal_all[j]
            eps_sio2 = eps_sio2_all[j]

            # Different effective permittivity for each polarization
            eps_eff_te = f_y * eps_m + (1 - f_y) * 1.0
            eps_eff_tm = f_x * eps_m + (1 - f_x) * 1.0

            # TE stack
            stack_te = [
                {"d": 0,      "eps": 1.0 + 0j},
                {"d": t_Cr,   "eps": eps_eff_te},
                {"d": d_SiO2, "eps": eps_sio2},
                {"d": 100.0,  "eps": eps_m},
                {"d": 0,      "eps": 2.25 + 0j},
            ]
            # TM stack
            stack_tm = [
                {"d": 0,      "eps": 1.0 + 0j},
                {"d": t_Cr,   "eps": eps_eff_tm},
                {"d": d_SiO2, "eps": eps_sio2},
                {"d": 100.0,  "eps": eps_m},
                {"d": 0,      "eps": 2.25 + 0j},
            ]

            if theta_deg < 0.1:
                R_t, T_t = _tmm_rt_normal(stack_te, wavelengths[j])
                R_te[i, j] = R_t; T_te[i, j] = T_t
                A_te[i, j] = max(0.0, 1.0 - R_t - T_t)
                
                R_m, T_m = _tmm_rt_normal(stack_tm, wavelengths[j])
                R_tm[i, j] = R_m; T_tm[i, j] = T_m
                A_tm[i, j] = max(0.0, 1.0 - R_m - T_m)
            else:
                R_t, T_t = _tmm_rt_oblique(stack_te, wavelengths[j], theta_rad, "TE")
                R_te[i, j] = R_t; T_te[i, j] = T_t
                A_te[i, j] = max(0.0, 1.0 - R_t - T_t)
                
                R_m, T_m = _tmm_rt_oblique(stack_tm, wavelengths[j], theta_rad, "TM")
                R_tm[i, j] = R_m; T_tm[i, j] = T_m
                A_tm[i, j] = max(0.0, 1.0 - R_m - T_m)

    return {
        "A_tmm_te": A_te, "R_tmm_te": R_te, "T_tmm_te": T_te,
        "A_tmm_tm": A_tm, "R_tmm_tm": R_tm, "T_tmm_tm": T_tm,
    }
