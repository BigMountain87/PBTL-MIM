# src/simulation/tmm_struct_b.py
"""
TMM for Structure B: Ring-Disk Fano MIM.

Layer stack (3 layers):
  Air | EMA(ring+disk) | SiO₂(d) | Cr_mirror(100nm) | Glass

EMA fill fraction: f = (π(R_out²-R_in²) + πR_disk²) / P²
"""

import numpy as np
from .materials import get_metal_permittivity, get_sio2_permittivity
from .tmm_engine import _tmm_rt_oblique, _tmm_rt_normal


def compute_tmm_batch(params: np.ndarray, wavelengths: np.ndarray,
                      metal: str = "Cr") -> dict:
    """
    Batch TMM+EMA for Structure B.

    params: [N, 8] — [P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta, phi]
    wavelengths: [Nλ] nm
    metal: "Cr" | "Ti" | "Au"

    Returns: {"A_tmm": [N,Nλ], "R_tmm": [N,Nλ], "T_tmm": [N,Nλ]}
    """
    N = params.shape[0]
    Nw = len(wavelengths)
    A_out = np.zeros((N, Nw))
    R_out_arr = np.zeros((N, Nw))
    T_out = np.zeros((N, Nw))

    eps_metal_all = get_metal_permittivity(wavelengths, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths)

    for i in range(N):
        P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta_deg, phi_deg = params[i]
        
        # Fill fraction: ring area + disk area over unit cell
        area_ring = np.pi * (R_out**2 - R_in**2)
        area_disk = np.pi * R_disk**2
        f = (area_ring + area_disk) / P**2
        f = np.clip(f, 0, 1)
        
        theta_rad = np.deg2rad(theta_deg)

        for j in range(Nw):
            eps_m = eps_metal_all[j]
            eps_sio2 = eps_sio2_all[j]

            # EMA for ring+disk pattern
            eps_eff = f * eps_m + (1 - f) * 1.0

            stack = [
                {"d": 0,      "eps": 1.0 + 0j},    # Air
                {"d": t_Cr,   "eps": eps_eff},       # EMA ring+disk
                {"d": d_SiO2, "eps": eps_sio2},      # SiO₂ spacer
                {"d": 100.0,  "eps": eps_m},          # Cr mirror
                {"d": 0,      "eps": 2.25 + 0j},    # Glass
            ]

            if theta_deg < 0.1:
                R, T = _tmm_rt_normal(stack, wavelengths[j])
            else:
                R, T = _tmm_rt_oblique(stack, wavelengths[j], theta_rad, "TE")

            R_out_arr[i, j] = R
            T_out[i, j] = T
            A_out[i, j] = max(0.0, 1.0 - R - T)

    return {"A_tmm": A_out, "R_tmm": R_out_arr, "T_tmm": T_out}
