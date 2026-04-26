# src/simulation/rcwa_struct_a.py
"""
Structure A: Asymmetric Dual-Dielectric Dual-Cavity MIM Absorber (TORCWA GPU).

Layer stack:
  Air | Patterned Cr (rect Wx*Wy, t1) | SiO2 (d1) | Cr mid (t_mid) |
  TiO2 (d2) | Patterned Cr (square W2, t2) | Cr mirror (100nm) | Glass

10 Parameters: P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta
"""

import numpy as np
import torch
import torcwa
from tqdm import tqdm
from .rcwa_utils import compute_RT_batch
from .materials import get_metal_permittivity, get_sio2_permittivity, get_tio2_permittivity

DESIGN_SPACE = {
    "P":     {"min": 300, "max": 600, "unit": "nm"},
    "Wx":    {"min": 50,  "max": 540, "unit": "nm"},  # < 0.9P
    "Wy":    {"min": 50,  "max": 540, "unit": "nm"},  # < 0.9P
    "W2":    {"min": 50,  "max": 540, "unit": "nm"},  # < 0.9P
    "t1":    {"min": 10,  "max": 80,  "unit": "nm"},
    "t2":    {"min": 10,  "max": 80,  "unit": "nm"},
    "t_mid": {"min": 5,   "max": 30,  "unit": "nm"},
    "d1":    {"min": 30,  "max": 200, "unit": "nm"},
    "d2":    {"min": 30,  "max": 200, "unit": "nm"},
    "theta": {"min": 0,   "max": 45,  "unit": "deg"},
}

PARAM_NAMES = list(DESIGN_SPACE.keys())
PARAM_MIN = np.array([DESIGN_SPACE[k]["min"] for k in PARAM_NAMES], dtype=np.float64)
PARAM_MAX = np.array([DESIGN_SPACE[k]["max"] for k in PARAM_NAMES], dtype=np.float64)

WAVELENGTH = {"start": 380, "stop": 780, "n_pts": 100}

RCWA_SETTINGS = {"grid": (64, 64), "order": [5, 5]}

# Small azimuthal perturbation to break symmetry
_AZI_PERTURB = np.deg2rad(0.01)


def simulate_single(params, wavelengths_nm, metal="Cr", device=None):
    """Simulate Structure A for a single parameter set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = float(params["P"])
    Wx = float(params["Wx"])
    Wy = float(params["Wy"])
    W2 = float(params["W2"])
    t1 = float(params["t1"])
    t2 = float(params["t2"])
    t_mid = float(params["t_mid"])
    d1 = float(params["d1"])
    d2 = float(params["d2"])
    theta_deg = float(params["theta"])
    theta_rad = np.deg2rad(theta_deg)
    azi_rad = _AZI_PERTURB if theta_deg > 0.1 else 0.0

    Nx, Ny = RCWA_SETTINGS["grid"]
    order = RCWA_SETTINGS["order"]
    sim_dtype = torch.complex128

    eps_metal_all = get_metal_permittivity(wavelengths_nm, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths_nm)
    eps_tio2_all = get_tio2_permittivity(wavelengths_nm)

    # Setup geometry grids
    torcwa.rcwa_geo.Lx = P
    torcwa.rcwa_geo.Ly = P
    torcwa.rcwa_geo.nx = Nx
    torcwa.rcwa_geo.ny = Ny
    torcwa.rcwa_geo.grid()

    geo_top = torcwa.rcwa_geo.rectangle(Wx=Wx, Wy=Wy, Cx=P/2, Cy=P/2)
    geo_bot = torcwa.rcwa_geo.rectangle(Wx=W2, Wy=W2, Cx=P/2, Cy=P/2)

    n_wl = len(wavelengths_nm)
    A_arr = np.zeros(n_wl)
    R_arr = np.zeros(n_wl)
    T_arr = np.zeros(n_wl)

    for i, lam in enumerate(wavelengths_nm):
        freq = 1.0 / lam

        eps_m = complex(eps_metal_all[i])
        eps_sio2 = float(np.real(eps_sio2_all[i]))
        eps_tio2 = float(np.real(eps_tio2_all[i]))

        eps_top = (geo_top * eps_m + (1.0 - geo_top) * 1.0).to(dtype=sim_dtype, device=device)
        eps_bot = (geo_bot * eps_m + (1.0 - geo_bot) * 1.0).to(dtype=sim_dtype, device=device)

        sim = torcwa.rcwa(freq=freq, order=order, L=[P, P],
                          dtype=sim_dtype, device=device)

        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=2.25)
        sim.set_incident_angle(inc_ang=theta_rad, azi_ang=azi_rad)
        sim.source_planewave(amplitude=[0.0, 1.0], direction='forward')

        sim.add_layer(thickness=t1, eps=eps_top)
        sim.add_layer(thickness=d1, eps=eps_sio2)
        sim.add_layer(thickness=t_mid, eps=eps_m)
        sim.add_layer(thickness=d2, eps=eps_tio2)
        sim.add_layer(thickness=t2, eps=eps_bot)
        sim.add_layer(thickness=100.0, eps=eps_m)

        sim.solve_global_smatrix()
        R_total, T_total = compute_RT_batch(sim, order)

        A_arr[i] = 1.0 - R_total - T_total
        R_arr[i] = R_total
        T_arr[i] = T_total

    return A_arr, R_arr, T_arr


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
        p["W2"] = min(p["W2"], max_w)
        all_params.append(p)

    n_wl = len(wavelengths_nm)
    A_all = np.zeros((n_samples, n_wl))
    R_all = np.zeros((n_samples, n_wl))
    T_all = np.zeros((n_samples, n_wl))
    params_array = np.zeros((n_samples, len(PARAM_NAMES)))

    for i in tqdm(range(n_samples), desc=f"RCWA-A ({metal})"):
        p = all_params[i]
        A, R, T = simulate_single(p, wavelengths_nm, metal=metal, device=device)
        A_all[i] = A
        R_all[i] = R
        T_all[i] = T
        for j, name in enumerate(PARAM_NAMES):
            params_array[i, j] = p[name]

    return {
        "params": params_array, "param_names": PARAM_NAMES,
        "A": A_all, "R": R_all, "T": T_all,
        "wavelengths": wavelengths_nm, "metal": metal,
        "structure": "A_dual_cavity",
    }
