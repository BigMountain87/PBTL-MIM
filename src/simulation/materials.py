# src/simulation/materials.py
"""
Optical material permittivity functions.
Cr, Ti, Au: Johnson & Christy tabulated data + CubicSpline interpolation.
SiO2: Malitson Sellmeier (1965).
TiO2: Devore Sellmeier (1951), ordinary ray.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def get_cr_permittivity(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Cr permittivity from Johnson & Christy 1974."""
    jc_data = np.array([
        [250, 3.07, 3.52], [275, 3.13, 3.73], [300, 3.17, 3.93],
        [325, 3.18, 4.13], [350, 3.18, 4.30], [375, 3.17, 4.46],
        [400, 3.15, 4.59], [450, 3.08, 4.83], [500, 3.00, 5.02],
        [550, 2.91, 5.19], [600, 2.83, 5.33], [650, 2.76, 5.46],
        [700, 2.70, 5.57], [750, 2.65, 5.67], [800, 2.61, 5.77],
        [900, 2.55, 5.95], [1000, 2.52, 6.10], [1200, 2.49, 6.36],
        [1400, 2.49, 6.58], [1600, 2.51, 6.78],
    ])
    wl, n_arr, k_arr = jc_data[:,0], jc_data[:,1], jc_data[:,2]
    cs_n = CubicSpline(wl, n_arr)
    cs_k = CubicSpline(wl, k_arr)
    return (cs_n(wavelengths_nm) + 1j * cs_k(wavelengths_nm)) ** 2


def get_ti_permittivity(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Ti permittivity from Johnson & Christy."""
    jc_ti = np.array([
        [300, 3.26, 3.33], [350, 3.47, 3.64], [400, 3.64, 3.92],
        [450, 3.76, 4.17], [500, 3.82, 4.39], [550, 3.84, 4.58],
        [600, 3.82, 4.74], [700, 3.71, 5.00], [800, 3.56, 5.22],
        [900, 3.41, 5.40], [1000, 3.27, 5.56], [1200, 3.04, 5.83],
        [1400, 2.87, 6.06], [1600, 2.74, 6.26],
    ])
    wl, n_arr, k_arr = jc_ti[:,0], jc_ti[:,1], jc_ti[:,2]
    cs_n = CubicSpline(wl, n_arr)
    cs_k = CubicSpline(wl, k_arr)
    return (cs_n(wavelengths_nm) + 1j * cs_k(wavelengths_nm)) ** 2


def get_au_permittivity(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Au permittivity from Johnson & Christy."""
    jc_au = np.array([
        [300, 1.54, 1.90], [350, 0.92, 1.95], [400, 0.39, 2.03],
        [450, 0.23, 2.50], [500, 0.19, 2.98], [550, 0.17, 3.47],
        [600, 0.17, 3.93], [650, 0.19, 4.37], [700, 0.23, 4.80],
        [750, 0.28, 5.20], [800, 0.34, 5.58], [900, 0.52, 6.30],
        [1000, 0.71, 6.97], [1200, 1.09, 8.21], [1400, 1.50, 9.39],
        [1600, 1.96, 10.5],
    ])
    wl, n_arr, k_arr = jc_au[:,0], jc_au[:,1], jc_au[:,2]
    cs_n = CubicSpline(wl, n_arr)
    cs_k = CubicSpline(wl, k_arr)
    return (cs_n(wavelengths_nm) + 1j * cs_k(wavelengths_nm)) ** 2


def get_sio2_permittivity(wavelengths_nm: np.ndarray) -> np.ndarray:
    """SiO2 permittivity from Malitson Sellmeier equation (1965)."""
    lam_um = wavelengths_nm / 1000.0
    l2 = lam_um ** 2
    n_sq = 1.0 + 0.6961663 * l2 / (l2 - 0.0684043**2) \
             + 0.4079426 * l2 / (l2 - 0.1162414**2) \
             + 0.8974794 * l2 / (l2 - 9.896161**2)
    return n_sq  # real, no absorption


def get_tio2_permittivity(wavelengths_nm: np.ndarray) -> np.ndarray:
    """TiO2 permittivity from Devore Sellmeier (1951), ordinary ray."""
    lam_um = wavelengths_nm / 1000.0
    l2 = lam_um ** 2
    # Devore 1951: ordinary ray
    n_sq = 5.913 + 0.2441 / (l2 - 0.0803)
    return n_sq  # real, no absorption


# Mapping for convenience
METAL_EPS_FN = {
    "Cr": get_cr_permittivity,
    "Ti": get_ti_permittivity,
    "Au": get_au_permittivity,
}


def get_metal_permittivity(wavelengths_nm, metal="Cr"):
    """Get metal permittivity by name."""
    return METAL_EPS_FN[metal](wavelengths_nm)
