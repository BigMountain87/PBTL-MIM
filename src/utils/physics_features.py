#!/usr/bin/env python3
"""
Physics-informed feature computation for all structures.

Systematic framework with 6 categories:
  1. Cavity resonance (Fabry-Perot phase)
  2. Fill fraction (effective medium)
  3. Sub-wavelength ratio (diffraction regime)
  4. Skin depth ratio (metal absorption)
  5. Optical path length (interference)
  6. Angle & geometry

Each structure instantiates the relevant features from its layer stack.
"""

import numpy as np
from src.simulation.materials import (
    get_sio2_permittivity, get_tio2_permittivity, get_metal_permittivity
)


def _cavity_phase_features(n_cav, d_cav, theta_rad, wavelengths_nm):
    """Fabry-Perot round-trip phase: cos and sin components.
    
    Args:
        n_cav: (Nlam,) refractive index of cavity material
        d_cav: (N,) cavity thickness per sample
        theta_rad: (N,) incidence angle in radians
        wavelengths_nm: (Nlam,)
    Returns:
        cos_phase, sin_phase: each (N, Nlam)
    """
    sin_ti = np.clip(np.sin(theta_rad[:, None]) / n_cav[None, :], -1, 1)
    cos_ti = np.sqrt(1 - sin_ti**2)
    phase = 4 * np.pi * n_cav[None, :] * d_cav[:, None] * cos_ti / wavelengths_nm[None, :]
    return np.cos(phase), np.sin(phase)


def _skin_depth(wavelengths_nm, metal="Cr"):
    """Skin depth: delta = lambda / (4*pi*k_metal). Returns (Nlam,)."""
    eps_m = get_metal_permittivity(wavelengths_nm, metal)
    k_metal = np.imag(np.sqrt(eps_m))
    return wavelengths_nm / (4 * np.pi * k_metal + 1e-30)


def _metal_alpha(wavelengths_nm, metal="Cr"):
    """Metal absorption coefficient: 4*pi*k/lambda. Returns (Nlam,)."""
    eps_m = get_metal_permittivity(wavelengths_nm, metal)
    k_metal = np.imag(np.sqrt(eps_m))
    return 4 * np.pi * k_metal / wavelengths_nm


# ============================================================
# Structure A: Asymmetric Dual-Dielectric Dual-Cavity
# Params: P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta
# Layers: Air | Cr(rect) | SiO2 | Cr_mid | TiO2 | Cr(sq) | Cr_mirror | Glass
# Features: 17
# ============================================================

def compute_physics_features_A(params, wavelengths_nm, metal="Cr"):
    """17 physics features for Structure A."""
    N, Nlam = len(params), len(wavelengths_nm)
    P, Wx, Wy, W2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    t1, t2, t_mid = params[:, 4], params[:, 5], params[:, 6]
    d1, d2, theta = params[:, 7], params[:, 8], params[:, 9]
    theta_rad = np.deg2rad(theta)

    n_sio2 = np.sqrt(np.real(get_sio2_permittivity(wavelengths_nm)))
    n_tio2 = np.sqrt(np.real(get_tio2_permittivity(wavelengths_nm)))
    sd = _skin_depth(wavelengths_nm, metal)

    feats = []

    # 1. Cavity resonance: SiO2 + TiO2 (4 features)
    cos_s, sin_s = _cavity_phase_features(n_sio2, d1, theta_rad, wavelengths_nm)
    cos_t, sin_t = _cavity_phase_features(n_tio2, d2, theta_rad, wavelengths_nm)
    feats += [cos_s, sin_s, cos_t, sin_t]

    # 2. Fill fraction (2 features)
    feats.append(np.tile((Wx * Wy / P**2)[:, None], (1, Nlam)))
    feats.append(np.tile((W2**2 / P**2)[:, None], (1, Nlam)))

    # 3. Sub-wavelength ratio (3 features)
    feats.append(P[:, None] / wavelengths_nm[None, :])
    feats.append(Wx[:, None] / wavelengths_nm[None, :])
    feats.append(W2[:, None] / wavelengths_nm[None, :])

    # 4. Skin depth ratio (3 features)
    feats.append(t1[:, None] / sd[None, :])
    feats.append(t2[:, None] / sd[None, :])
    feats.append(t_mid[:, None] / sd[None, :])

    # 5. Optical path length (2 features)
    feats.append(n_sio2[None, :] * d1[:, None] / wavelengths_nm[None, :])
    feats.append(n_tio2[None, :] * d2[:, None] / wavelengths_nm[None, :])

    # 6. Angle & geometry (3 features)
    feats.append(np.tile(np.cos(theta_rad[:, None]), (1, Nlam)))
    feats.append(np.tile((Wy / (Wx + 1e-10))[:, None], (1, Nlam)))
    feats.append(np.tile(_metal_alpha(wavelengths_nm, metal)[None, :], (N, 1)))

    return np.stack(feats, axis=-1).astype(np.float32)  # (N, Nlam, 17)


# ============================================================
# Structure B: Ring-Disk Fano Resonance
# Params: P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta, phi
# Layers: Air | Cr(ring+disk) | SiO2 | Cr_mirror | Glass
# Features: 13
# ============================================================

def compute_physics_features_B(params, wavelengths_nm, metal="Cr"):
    """13 physics features for Structure B."""
    N, Nlam = len(params), len(wavelengths_nm)
    P, R_out, R_in, R_disk = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    t_Cr, d_SiO2 = params[:, 4], params[:, 5]
    theta, phi = params[:, 6], params[:, 7]
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    n_sio2 = np.sqrt(np.real(get_sio2_permittivity(wavelengths_nm)))
    sd = _skin_depth(wavelengths_nm, metal)

    feats = []

    # 1. Cavity resonance: SiO2 only (2 features)
    cos_s, sin_s = _cavity_phase_features(n_sio2, d_SiO2, theta_rad, wavelengths_nm)
    feats += [cos_s, sin_s]

    # 2. Fill fraction: ring + disk (2 features)
    ff_ring = np.pi * (R_out**2 - R_in**2) / P**2  # ring area / unit cell
    ff_disk = np.pi * R_disk**2 / P**2               # disk area / unit cell
    feats.append(np.tile(ff_ring[:, None], (1, Nlam)))
    feats.append(np.tile(ff_disk[:, None], (1, Nlam)))

    # 3. Sub-wavelength ratio (2 features)
    feats.append(P[:, None] / wavelengths_nm[None, :])
    feats.append(R_out[:, None] / wavelengths_nm[None, :])

    # 4. Skin depth ratio (1 feature)
    feats.append(t_Cr[:, None] / sd[None, :])

    # 5. Optical path length (1 feature)
    feats.append(n_sio2[None, :] * d_SiO2[:, None] / wavelengths_nm[None, :])

    # 6. Angle & geometry (5 features)
    feats.append(np.tile(np.cos(theta_rad[:, None]), (1, Nlam)))
    feats.append(np.tile(np.cos(phi_rad[:, None]), (1, Nlam)))
    feats.append(np.tile((R_in / (R_out + 1e-10))[:, None], (1, Nlam)))    # ring ratio
    feats.append(np.tile((R_disk / (R_out + 1e-10))[:, None], (1, Nlam)))  # disk/ring ratio
    feats.append(np.tile(_metal_alpha(wavelengths_nm, metal)[None, :], (N, 1)))

    return np.stack(feats, axis=-1).astype(np.float32)  # (N, Nlam, 13)


# ============================================================
# Structure C: Dual-Polarization Rectangular
# Params: P, Wx, Wy, t_Cr, d_SiO2, theta, phi
# Layers: Air | Cr(rect) | SiO2 | Cr_mirror | Glass
# Features: 11
# ============================================================

def compute_physics_features_C(params, wavelengths_nm, metal="Cr"):
    """11 physics features for Structure C."""
    N, Nlam = len(params), len(wavelengths_nm)
    P, Wx, Wy = params[:, 0], params[:, 1], params[:, 2]
    t_Cr, d_SiO2 = params[:, 3], params[:, 4]
    theta, phi = params[:, 5], params[:, 6]
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    n_sio2 = np.sqrt(np.real(get_sio2_permittivity(wavelengths_nm)))
    sd = _skin_depth(wavelengths_nm, metal)

    feats = []

    # 1. Cavity resonance: SiO2 only (2 features)
    cos_s, sin_s = _cavity_phase_features(n_sio2, d_SiO2, theta_rad, wavelengths_nm)
    feats += [cos_s, sin_s]

    # 2. Fill fraction (1 feature)
    feats.append(np.tile((Wx * Wy / P**2)[:, None], (1, Nlam)))

    # 3. Sub-wavelength ratio (2 features)
    feats.append(P[:, None] / wavelengths_nm[None, :])
    feats.append(Wx[:, None] / wavelengths_nm[None, :])

    # 4. Skin depth ratio (1 feature)
    feats.append(t_Cr[:, None] / sd[None, :])

    # 5. Optical path length (1 feature)
    feats.append(n_sio2[None, :] * d_SiO2[:, None] / wavelengths_nm[None, :])

    # 6. Angle & geometry (4 features)
    feats.append(np.tile(np.cos(theta_rad[:, None]), (1, Nlam)))
    feats.append(np.tile(np.cos(phi_rad[:, None]), (1, Nlam)))
    feats.append(np.tile((Wy / (Wx + 1e-10))[:, None], (1, Nlam)))  # aspect ratio
    feats.append(np.tile(_metal_alpha(wavelengths_nm, metal)[None, :], (N, 1)))

    return np.stack(feats, axis=-1).astype(np.float32)  # (N, Nlam, 11)


# ============================================================
# Differentiable (PyTorch) versions for inverse design
# ============================================================

def compute_physics_features_A_torch(geo, lam, metal="Cr"):
    """Differentiable version for Structure A inverse design.
    
    Args:
        geo: (B, 10) tensor [P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta]
        lam: scalar wavelength in nm (float)
        metal: metal name
    Returns:
        (B, 17) tensor
    """
    P, Wx, Wy, W2 = geo[:, 0], geo[:, 1], geo[:, 2], geo[:, 3]
    t1, t2, t_mid = geo[:, 4], geo[:, 5], geo[:, 6]
    d1, d2, theta = geo[:, 7], geo[:, 8], geo[:, 9]
    theta_rad = theta * (np.pi / 180.0)

    lam_np = np.array([lam])
    n_sio2 = float(np.sqrt(np.real(get_sio2_permittivity(lam_np))[0]))
    n_tio2 = float(np.sqrt(np.real(get_tio2_permittivity(lam_np))[0]))
    eps_m = get_metal_permittivity(lam_np, metal)[0]
    k_m = float(np.imag(np.sqrt(eps_m)))
    sd = lam / (4 * np.pi * k_m + 1e-30)
    alpha_m = 4 * np.pi * k_m / lam

    feats = []

    # Cavity phases
    for n_c, d_c in [(n_sio2, d1), (n_tio2, d2)]:
        sin_ti = torch.clamp(torch.sin(theta_rad) / n_c, -1, 1)
        cos_ti = torch.sqrt(1 - sin_ti**2)
        phase = 4 * np.pi * n_c * d_c * cos_ti / lam
        feats.append(torch.cos(phase))
        feats.append(torch.sin(phase))

    feats.append(Wx * Wy / P**2)
    feats.append(W2**2 / P**2)
    feats.append(P / lam)
    feats.append(Wx / lam)
    feats.append(W2 / lam)
    feats.append(t1 / sd)
    feats.append(t2 / sd)
    feats.append(t_mid / sd)
    feats.append(torch.cos(theta_rad))
    feats.append(n_sio2 * d1 / lam)
    feats.append(n_tio2 * d2 / lam)
    feats.append(Wy / (Wx + 1e-10))
    feats.append(torch.full_like(P, alpha_m))

    return torch.stack(feats, dim=-1)


def compute_physics_features_B_torch(geo, lam, metal="Cr"):
    """Differentiable version for Structure B inverse design.
    
    Args:
        geo: (B, 8) tensor [P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta, phi]
        lam: scalar wavelength in nm
    Returns:
        (B, 13) tensor
    """
    import torch
    P, R_out, R_in, R_disk = geo[:, 0], geo[:, 1], geo[:, 2], geo[:, 3]
    t_Cr, d_SiO2 = geo[:, 4], geo[:, 5]
    theta, phi = geo[:, 6], geo[:, 7]
    theta_rad = theta * (np.pi / 180.0)
    phi_rad = phi * (np.pi / 180.0)

    lam_np = np.array([lam])
    n_sio2 = float(np.sqrt(np.real(get_sio2_permittivity(lam_np))[0]))
    eps_m = get_metal_permittivity(lam_np, metal)[0]
    k_m = float(np.imag(np.sqrt(eps_m)))
    sd = lam / (4 * np.pi * k_m + 1e-30)
    alpha_m = 4 * np.pi * k_m / lam

    feats = []

    # Cavity phase
    sin_ti = torch.clamp(torch.sin(theta_rad) / n_sio2, -1, 1)
    cos_ti = torch.sqrt(1 - sin_ti**2)
    phase = 4 * np.pi * n_sio2 * d_SiO2 * cos_ti / lam
    feats.append(torch.cos(phase))
    feats.append(torch.sin(phase))

    # Fill fractions
    feats.append(np.pi * (R_out**2 - R_in**2) / P**2)
    feats.append(np.pi * R_disk**2 / P**2)

    feats.append(P / lam)
    feats.append(R_out / lam)
    feats.append(t_Cr / sd)
    feats.append(n_sio2 * d_SiO2 / lam)
    feats.append(torch.cos(theta_rad))
    feats.append(torch.cos(phi_rad))
    feats.append(R_in / (R_out + 1e-10))
    feats.append(R_disk / (R_out + 1e-10))
    feats.append(torch.full_like(P, alpha_m))

    return torch.stack(feats, dim=-1)


def compute_physics_features_C_torch(geo, lam, metal="Cr"):
    """Differentiable version for Structure C inverse design.
    
    Args:
        geo: (B, 7) tensor [P, Wx, Wy, t_Cr, d_SiO2, theta, phi]
        lam: scalar wavelength in nm
    Returns:
        (B, 11) tensor
    """
    import torch
    P, Wx, Wy = geo[:, 0], geo[:, 1], geo[:, 2]
    t_Cr, d_SiO2 = geo[:, 3], geo[:, 4]
    theta, phi = geo[:, 5], geo[:, 6]
    theta_rad = theta * (np.pi / 180.0)
    phi_rad = phi * (np.pi / 180.0)

    lam_np = np.array([lam])
    n_sio2 = float(np.sqrt(np.real(get_sio2_permittivity(lam_np))[0]))
    eps_m = get_metal_permittivity(lam_np, metal)[0]
    k_m = float(np.imag(np.sqrt(eps_m)))
    sd = lam / (4 * np.pi * k_m + 1e-30)
    alpha_m = 4 * np.pi * k_m / lam

    feats = []

    sin_ti = torch.clamp(torch.sin(theta_rad) / n_sio2, -1, 1)
    cos_ti = torch.sqrt(1 - sin_ti**2)
    phase = 4 * np.pi * n_sio2 * d_SiO2 * cos_ti / lam
    feats.append(torch.cos(phase))
    feats.append(torch.sin(phase))

    feats.append(Wx * Wy / P**2)
    feats.append(P / lam)
    feats.append(Wx / lam)
    feats.append(t_Cr / sd)
    feats.append(n_sio2 * d_SiO2 / lam)
    feats.append(torch.cos(theta_rad))
    feats.append(torch.cos(phi_rad))
    feats.append(Wy / (Wx + 1e-10))
    feats.append(torch.full_like(P, alpha_m))

    return torch.stack(feats, dim=-1)


# ============================================================
# Convenience dispatcher
# ============================================================

PHYS_FEAT_FN = {
    "A": compute_physics_features_A,
    "B": compute_physics_features_B,
    "C": compute_physics_features_C,
}

PHYS_FEAT_TORCH_FN = {
    "A": compute_physics_features_A_torch,
    "B": compute_physics_features_B_torch,
    "C": compute_physics_features_C_torch,
}

N_PHYS_FEATURES = {"A": 17, "B": 13, "C": 11}
