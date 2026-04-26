# src/simulation/tmm_struct_a.py
"""
TMM for Structure A: Asymmetric Dual-Dielectric Dual-Cavity MIM.

Layer stack (7 layers):
  Air | EMA(rect Wx×Wy) | SiO₂(d₁) | Cr_mid(t_mid) | TiO₂(d₂) | EMA(sq W₂) | Cr_mirror(100nm) | Glass

EMA: Maxwell-Garnett linear mixing
  f_top = (Wx * Wy) / P²
  f_bot = (W₂ / P)²
"""

import numpy as np
from .materials import get_metal_permittivity, get_sio2_permittivity, get_tio2_permittivity
from .tmm_engine import _tmm_rt_oblique, _tmm_rt_normal


def compute_tmm_batch(params: np.ndarray, wavelengths: np.ndarray,
                      metal: str = "Cr") -> dict:
    """
    Batch TMM+EMA for Structure A.

    params: [N, 10] — [P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta]
    wavelengths: [Nλ] nm
    metal: "Cr" | "Ti" | "Au"

    Returns: {"A_tmm": [N,Nλ], "R_tmm": [N,Nλ], "T_tmm": [N,Nλ]}
    """
    N = params.shape[0]
    Nw = len(wavelengths)
    A_out = np.zeros((N, Nw))
    R_out = np.zeros((N, Nw))
    T_out = np.zeros((N, Nw))

    eps_metal_all = get_metal_permittivity(wavelengths, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths)
    eps_tio2_all = get_tio2_permittivity(wavelengths)

    for i in range(N):
        P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta_deg = params[i]
        f_top = (Wx * Wy) / P**2
        f_bot = (W2 / P)**2
        theta_rad = np.deg2rad(theta_deg)

        for j in range(Nw):
            eps_m = eps_metal_all[j]
            eps_sio2 = eps_sio2_all[j]
            eps_tio2 = eps_tio2_all[j]

            # EMA for patterned layers
            eps_eff_top = f_top * eps_m + (1 - f_top) * 1.0
            eps_eff_bot = f_bot * eps_m + (1 - f_bot) * 1.0

            stack = [
                {"d": 0,     "eps": 1.0 + 0j},        # Air (input)
                {"d": t1,    "eps": eps_eff_top},       # EMA top Cr rect
                {"d": d1,    "eps": eps_sio2},          # SiO₂ spacer
                {"d": t_mid, "eps": eps_m},             # Cr mid
                {"d": d2,    "eps": eps_tio2},          # TiO₂ spacer
                {"d": t2,    "eps": eps_eff_bot},       # EMA bot Cr square
                {"d": 100.0, "eps": eps_m},             # Cr mirror
                {"d": 0,     "eps": 2.25 + 0j},        # Glass (output)
            ]

            if theta_deg < 0.1:  # normal incidence
                R, T = _tmm_rt_normal(stack, wavelengths[j])
            else:
                R, T = _tmm_rt_oblique(stack, wavelengths[j], theta_rad, "TE")

            R_out[i, j] = R
            T_out[i, j] = T
            A_out[i, j] = max(0.0, 1.0 - R - T)

    return {"A_tmm": A_out, "R_tmm": R_out, "T_tmm": T_out}
