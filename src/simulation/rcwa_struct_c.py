# src/simulation/rcwa_struct_c.py
"""
Structure C: Dual-Polarization Rectangular MIM Absorber (TORCWA GPU).

Layer stack:
  Air | Patterned Cr (rect Wx*Wy, t_Cr) | SiO2 (d) | Cr mirror (100nm) | Glass

7 Parameters: P, Wx, Wy, t_Cr, d_SiO2, theta, phi
Output: 6 channels (A_TE, R_TE, T_TE, A_TM, R_TM, T_TM)
"""

import numpy as np
import torch
import torcwa
from tqdm import tqdm
from .materials import get_metal_permittivity, get_sio2_permittivity
from .rcwa_utils import compute_RT_batch, compute_RT_batch_TM

DESIGN_SPACE = {
    "P":      {"min": 300, "max": 800, "unit": "nm"},
    "Wx":     {"min": 50,  "max": 720, "unit": "nm"},  # < 0.9P
    "Wy":     {"min": 50,  "max": 720, "unit": "nm"},  # < 0.9P
    "t_Cr":   {"min": 20,  "max": 80,  "unit": "nm"},
    "d_SiO2": {"min": 50,  "max": 200, "unit": "nm"},
    "theta":  {"min": 0,   "max": 60,  "unit": "deg"},
    "phi":    {"min": 0,   "max": 45,  "unit": "deg"},
}

PARAM_NAMES = list(DESIGN_SPACE.keys())
PARAM_MIN = np.array([DESIGN_SPACE[k]["min"] for k in PARAM_NAMES], dtype=np.float64)
PARAM_MAX = np.array([DESIGN_SPACE[k]["max"] for k in PARAM_NAMES], dtype=np.float64)

WAVELENGTH = {"start": 400, "stop": 1800, "n_pts": 100}

RCWA_SETTINGS = {"grid": (64, 64), "order": [5, 5]}

_AZI_PERTURB = np.deg2rad(0.01)


def _build_sim(freq, P, Wx, Wy, t_Cr, d_SiO2, eps_m, eps_sio2,
               theta_rad, phi_rad, amplitude, order, Nx, Ny,
               sim_dtype, device):
    """Build and solve one TORCWA simulation."""
    torcwa.rcwa_geo.Lx = P
    torcwa.rcwa_geo.Ly = P
    torcwa.rcwa_geo.nx = Nx
    torcwa.rcwa_geo.ny = Ny
    torcwa.rcwa_geo.grid()

    geo = torcwa.rcwa_geo.rectangle(Wx=Wx, Wy=Wy, Cx=P/2, Cy=P/2)
    eps_pat = (geo * eps_m + (1.0 - geo) * 1.0).to(dtype=sim_dtype, device=device)

    sim = torcwa.rcwa(freq=freq, order=order, L=[P, P],
                      dtype=sim_dtype, device=device)

    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=theta_rad, azi_ang=phi_rad)
    sim.source_planewave(amplitude=amplitude, direction='forward')

    sim.add_layer(thickness=t_Cr, eps=eps_pat)
    sim.add_layer(thickness=d_SiO2, eps=eps_sio2)
    sim.add_layer(thickness=100.0, eps=eps_m)

    sim.solve_global_smatrix()
    return sim


def simulate_single(params, wavelengths_nm, metal="Cr", device=None):
    """
    Simulate Structure C (dual-pol) for a single parameter set.
    Returns 6 channels: A_TE, R_TE, T_TE, A_TM, R_TM, T_TM.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = float(params["P"])
    Wx = float(params["Wx"])
    Wy = float(params["Wy"])
    t_Cr = float(params["t_Cr"])
    d_SiO2 = float(params["d_SiO2"])
    theta_deg = float(params["theta"])
    phi_deg = float(params["phi"])
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg) if phi_deg > 0.1 else (
        _AZI_PERTURB if theta_deg > 0.1 else 0.0)

    Nx, Ny = RCWA_SETTINGS["grid"]
    order = RCWA_SETTINGS["order"]
    sim_dtype = torch.complex128

    eps_metal_all = get_metal_permittivity(wavelengths_nm, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths_nm)

    n_wl = len(wavelengths_nm)
    A_te = np.zeros(n_wl)
    R_te = np.zeros(n_wl)
    T_te = np.zeros(n_wl)
    A_tm = np.zeros(n_wl)
    R_tm = np.zeros(n_wl)
    T_tm = np.zeros(n_wl)

    for i, lam in enumerate(wavelengths_nm):
        freq = 1.0 / lam
        eps_m = complex(eps_metal_all[i])
        eps_sio2 = float(np.real(eps_sio2_all[i]))

        # TE simulation (Ey = 1)
        sim_te = _build_sim(freq, P, Wx, Wy, t_Cr, d_SiO2, eps_m, eps_sio2,
                            theta_rad, phi_rad, [0.0, 1.0], order, Nx, Ny,
                            sim_dtype, device)
        r_te, t_te = compute_RT_batch(sim_te, order)
        A_te[i] = 1.0 - r_te - t_te
        R_te[i] = r_te
        T_te[i] = t_te

        # TM simulation (Ex = 1)
        sim_tm = _build_sim(freq, P, Wx, Wy, t_Cr, d_SiO2, eps_m, eps_sio2,
                            theta_rad, phi_rad, [1.0, 0.0], order, Nx, Ny,
                            sim_dtype, device)
        r_tm, t_tm = compute_RT_batch_TM(sim_tm, order)
        A_tm[i] = 1.0 - r_tm - t_tm
        R_tm[i] = r_tm
        T_tm[i] = t_tm

    return A_te, R_te, T_te, A_tm, R_tm, T_tm


def generate_dataset(n_samples, wavelengths_nm, metal="Cr", seed=42, device=None):
    """Generate dataset with Latin Hypercube Sampling."""
    from scipy.stats import qmc

    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=rng)
    samples_unit = sampler.random(n=n_samples)

    all_params = []
    for i in range(n_samples):
        p = {}
        for j, name in enumerate(PARAM_NAMES):
            p[name] = PARAM_MIN[j] + samples_unit[i, j] * (PARAM_MAX[j] - PARAM_MIN[j])
        max_w = 0.9 * p["P"]
        p["Wx"] = min(p["Wx"], max_w)
        p["Wy"] = min(p["Wy"], max_w)
        all_params.append(p)

    n_wl = len(wavelengths_nm)
    A_te_all = np.zeros((n_samples, n_wl))
    R_te_all = np.zeros((n_samples, n_wl))
    T_te_all = np.zeros((n_samples, n_wl))
    A_tm_all = np.zeros((n_samples, n_wl))
    R_tm_all = np.zeros((n_samples, n_wl))
    T_tm_all = np.zeros((n_samples, n_wl))
    params_array = np.zeros((n_samples, len(PARAM_NAMES)))

    for i in tqdm(range(n_samples), desc=f"RCWA-C ({metal})"):
        p = all_params[i]
        a_te, r_te, t_te, a_tm, r_tm, t_tm = simulate_single(
            p, wavelengths_nm, metal=metal, device=device)
        A_te_all[i] = a_te
        R_te_all[i] = r_te
        T_te_all[i] = t_te
        A_tm_all[i] = a_tm
        R_tm_all[i] = r_tm
        T_tm_all[i] = t_tm
        for j, name in enumerate(PARAM_NAMES):
            params_array[i, j] = p[name]

    return {
        "params": params_array, "param_names": PARAM_NAMES,
        "A_TE": A_te_all, "R_TE": R_te_all, "T_TE": T_te_all,
        "A_TM": A_tm_all, "R_TM": R_tm_all, "T_TM": T_tm_all,
        "wavelengths": wavelengths_nm, "metal": metal,
        "structure": "C_dual_pol",
    }
