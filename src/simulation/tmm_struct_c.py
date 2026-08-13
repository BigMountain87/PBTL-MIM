# src/simulation/tmm_struct_c.py
"""
TMM for Structure C: Dual-Polarization Rectangular MIM.

Layer stack (3 layers):
  Air | EMA(rect Wx×Wy) | SiO₂(d) | Cr_mirror(100nm) | Glass

EMA is polarization-independent: f = (Wx * Wy) / P²
TMM outputs identical TE/TM → NN learns the polarization correction.

Output: 6 channels (A_TE, R_TE, T_TE, A_TM, R_TM, T_TM) — but TE=TM from TMM.
"""

import numpy as np
from .materials import get_metal_permittivity, get_sio2_permittivity
from .tmm_engine import _tmm_rt_oblique, _tmm_rt_normal


def compute_tmm_batch(params: np.ndarray, wavelengths: np.ndarray,
                      metal: str = "Cr") -> dict:
    """
    Batch TMM+EMA for Structure C (dual-pol).

    params: [N, 7] — [P, Wx, Wy, t_Cr, d_SiO2, theta, phi]
    wavelengths: [Nλ] nm
    metal: "Cr" | "Ti" | "Au"

    Returns: {"A_tmm_te": [N,Nλ], "R_tmm_te": [N,Nλ], "T_tmm_te": [N,Nλ],
              "A_tmm_tm": [N,Nλ], "R_tmm_tm": [N,Nλ], "T_tmm_tm": [N,Nλ]}
    """
    N = params.shape[0]
    Nw = len(wavelengths)
    
    # TMM outputs (TE and TM computed separately at oblique incidence)
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
        f = (Wx * Wy) / P**2
        theta_rad = np.deg2rad(theta_deg)

        for j in range(Nw):
            eps_m = eps_metal_all[j]
            eps_sio2 = eps_sio2_all[j]

            eps_eff = f * eps_m + (1 - f) * 1.0

            stack = [
                {"d": 0,      "eps": 1.0 + 0j},
                {"d": t_Cr,   "eps": eps_eff},
                {"d": d_SiO2, "eps": eps_sio2},
                {"d": 100.0,  "eps": eps_m},
                {"d": 0,      "eps": 2.25 + 0j},
            ]

            if theta_deg < 0.1:
                R, T = _tmm_rt_normal(stack, wavelengths[j])
                # At normal incidence, TE = TM
                R_te[i, j] = R_tm[i, j] = R
                T_te[i, j] = T_tm[i, j] = T
                A_te[i, j] = A_tm[i, j] = max(0.0, 1.0 - R - T)
            else:
                # TE polarization
                R_t, T_t = _tmm_rt_oblique(stack, wavelengths[j], theta_rad, "TE")
                R_te[i, j] = R_t
                T_te[i, j] = T_t
                A_te[i, j] = max(0.0, 1.0 - R_t - T_t)
                
                # TM polarization
                R_m, T_m = _tmm_rt_oblique(stack, wavelengths[j], theta_rad, "TM")
                R_tm[i, j] = R_m
                T_tm[i, j] = T_m
                A_tm[i, j] = max(0.0, 1.0 - R_m - T_m)

    return {
        "A_tmm_te": A_te, "R_tmm_te": R_te, "T_tmm_te": T_te,
        "A_tmm_tm": A_tm, "R_tmm_tm": R_tm, "T_tmm_tm": T_tm,
    }
