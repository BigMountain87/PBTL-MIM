# src/simulation/rcwa_struct_b.py
"""
Structure B: Ring-Disk Fano Resonance MIM Absorber (TORCWA GPU).

Layer stack:
  Air | Patterned Cr (ring+disk, t_Cr) | SiO2 (d) | Cr mirror (100nm) | Glass

8 Parameters: P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta, phi
"""

import numpy as np
import torch
import torcwa
from tqdm import tqdm
from .rcwa_utils import compute_RT_batch
from .materials import get_metal_permittivity, get_sio2_permittivity

DESIGN_SPACE = {
    "P":      {"min": 300, "max": 800, "unit": "nm"},
    "R_out":  {"min": 80,  "max": 350, "unit": "nm"},   # < 0.45P
    "R_in":   {"min": 30,  "max": 300, "unit": "nm"},   # < R_out
    "R_disk": {"min": 10,  "max": 100, "unit": "nm"},   # < R_in
    "t_Cr":   {"min": 20,  "max": 80,  "unit": "nm"},
    "d_SiO2": {"min": 50,  "max": 200, "unit": "nm"},
    "theta":  {"min": 0,   "max": 60,  "unit": "deg"},
    "phi":    {"min": 0,   "max": 45,  "unit": "deg"},
}

PARAM_NAMES = list(DESIGN_SPACE.keys())
PARAM_MIN = np.array([DESIGN_SPACE[k]["min"] for k in PARAM_NAMES], dtype=np.float64)
PARAM_MAX = np.array([DESIGN_SPACE[k]["max"] for k in PARAM_NAMES], dtype=np.float64)

WAVELENGTH = {"start": 400, "stop": 1800, "n_pts": 100}

# dtype: complex64 ~5x faster, half VRAM, identical absorption to <0.001 pp vs
# complex128 (see CONVERGENCE_C.md). Use complex128 for exact legacy reproduction.
# adaptive_order: legacy [5,5] is under-converged (A_TE off by up to ~6.8 pp for
# large-period ring-disks); when True, pick the order per wavelength via adaptive_order().
RCWA_SETTINGS = {"grid": (64, 64), "order": [5, 5], "dtype": torch.complex64,
                 "adaptive_order": True}

# Small azimuthal perturbation to break symmetry and avoid singular matrices
_AZI_PERTURB = np.deg2rad(0.01)


def adaptive_order(lam_nm, P):
    """P/lambda-adaptive Fourier order for Structure B (ring-disk). Convergence is
    driven by the number of diffraction channels (P/lambda) and the staircased curved
    edges, NOT by anisotropy. Calibrated from B_big(P=750) needing ~N=17 at short
    lambda vs B_sml(P=400) ~N=13 (see CONVERGENCE_C.md).

    Residual note: large-period ring-disks at short lambda are still not fully
    converged at N=17 (Richardson suggests N>=21); documented, not eliminated."""
    pl = P / lam_nm
    if pl >= 1.3:
        N = 17
    elif pl >= 0.8:
        N = 13
    else:
        N = 9
    return [N, N]


def _make_ring_disk_geometry(P, R_out, R_in, R_disk, Nx, Ny):
    """Create ring + disk geometry on grid."""
    x = torch.linspace(0, P, Nx+1)[:-1] + P/(2*Nx)
    y = torch.linspace(0, P, Ny+1)[:-1] + P/(2*Ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    cx, cy = P / 2.0, P / 2.0
    r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    ring = (r >= R_in) & (r <= R_out)
    disk = r <= R_disk
    geo = (ring | disk).float()
    return geo


def simulate_single(params, wavelengths_nm, metal="Cr", device=None):
    """Simulate Structure B for a single parameter set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = float(params["P"])
    R_out = float(params["R_out"])
    R_in = float(params["R_in"])
    R_disk = float(params["R_disk"])
    t_Cr = float(params["t_Cr"])
    d_SiO2 = float(params["d_SiO2"])
    theta_deg = float(params["theta"])
    phi_deg = float(params["phi"])
    theta_rad = np.deg2rad(theta_deg)
    # Add perturbation to avoid singular matrix at phi=0 with oblique incidence
    phi_rad = np.deg2rad(phi_deg) if phi_deg > 0.1 else (
        _AZI_PERTURB if theta_deg > 0.1 else 0.0)

    Nx, Ny = RCWA_SETTINGS["grid"]
    use_adaptive = RCWA_SETTINGS.get("adaptive_order", True)
    fixed_order = RCWA_SETTINGS["order"]
    sim_dtype = RCWA_SETTINGS.get("dtype", torch.complex64)

    eps_metal_all = get_metal_permittivity(wavelengths_nm, metal)
    eps_sio2_all = get_sio2_permittivity(wavelengths_nm)

    geo = _make_ring_disk_geometry(P, R_out, R_in, R_disk, Nx, Ny)

    n_wl = len(wavelengths_nm)
    A_arr = np.zeros(n_wl)
    R_arr = np.zeros(n_wl)
    T_arr = np.zeros(n_wl)

    for i, lam in enumerate(wavelengths_nm):
        freq = 1.0 / lam
        order = adaptive_order(lam, P) if use_adaptive else fixed_order
        eps_m = complex(eps_metal_all[i])
        eps_sio2 = float(np.real(eps_sio2_all[i]))

        eps_pat = (geo * eps_m + (1.0 - geo) * 1.0).to(dtype=sim_dtype, device=device)

        sim = torcwa.rcwa(freq=freq, order=order, L=[P, P],
                          dtype=sim_dtype, device=device, stable_eig_grad=False)

        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=2.25)
        sim.set_incident_angle(inc_ang=theta_rad, azi_ang=phi_rad)
        sim.source_planewave(amplitude=[0.0, 1.0], direction='forward')

        sim.add_layer(thickness=t_Cr, eps=eps_pat)
        sim.add_layer(thickness=d_SiO2, eps=eps_sio2)
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
        p["R_out"] = min(p["R_out"], 0.45 * p["P"])
        p["R_in"] = min(p["R_in"], p["R_out"] - 10)
        p["R_in"] = max(p["R_in"], 20)
        p["R_disk"] = min(p["R_disk"], p["R_in"] - 10)
        p["R_disk"] = max(p["R_disk"], 5)
        all_params.append(p)

    n_wl = len(wavelengths_nm)
    A_all = np.zeros((n_samples, n_wl))
    R_all = np.zeros((n_samples, n_wl))
    T_all = np.zeros((n_samples, n_wl))
    params_array = np.zeros((n_samples, len(PARAM_NAMES)))

    for i in tqdm(range(n_samples), desc=f"RCWA-B ({metal})"):
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
        "structure": "B_ring_disk",
    }
