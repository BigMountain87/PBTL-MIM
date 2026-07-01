# src/simulation/materials.py
"""
Optical material permittivity functions.

METALS (Cr, Ti, Au, Cu): measured Johnson & Christy tabulated n,k from
refractiveindex.info (CC0), in data/ref/{M}_JC.txt, CubicSpline-interpolated.
  - Cr, Ti: J&C 1974 (transition metals, Phys Rev B 9, 5056), 188-1937 nm
  - Au, Cu: J&C 1972 (noble metals, Phys Rev B 6, 4370), 188-1937 nm
These cover 400-1800 nm with measurements (no extrapolation) and match the cited
source. (The pre-redesign hand-entered tables did NOT match J&C; they have been
removed.)

DIELECTRICS:
  SiO2: Malitson Sellmeier (1965), 0.21-6.7 um (lossless formula).
  TiO2: measured Siefke 2016 (ALD amorphous thin film, complex n,k), data/ref/
        TiO2_Siefke.txt, 380-1840 nm. Replaces the old Devore rutile formula (which
        was lossless, rutile, and extrapolated); note amorphous n (~2.3-2.7) is lower
        than rutile (~2.5-3.0), so Structure A shifts vs the legacy data.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ref")
_NK_CACHE = {}


def _load_jc(metal):
    if metal not in _NK_CACHE:
        a = np.loadtxt(os.path.join(_REF_DIR, f"{metal}_JC.txt"))  # cols: nm n k
        _NK_CACHE[metal] = (a[:, 0], a[:, 1], a[:, 2])
    return _NK_CACHE[metal]


def _jc_eps(wavelengths_nm, metal):
    wl, n_arr, k_arr = _load_jc(metal)
    cs_n = CubicSpline(wl, n_arr)
    cs_k = CubicSpline(wl, k_arr)
    return (cs_n(wavelengths_nm) + 1j * cs_k(wavelengths_nm)) ** 2


# ---- metals: measured Johnson & Christy --------------------------------------
def get_cr_permittivity(wavelengths_nm):
    """Cr permittivity, measured J&C 1974 (transition metals), 188-1937 nm."""
    return _jc_eps(wavelengths_nm, "Cr")


def get_ti_permittivity(wavelengths_nm):
    """Ti permittivity, measured J&C 1974 (transition metals), 188-1937 nm."""
    return _jc_eps(wavelengths_nm, "Ti")


def get_au_permittivity(wavelengths_nm):
    """Au permittivity, measured J&C 1972 (noble metals), 188-1937 nm."""
    return _jc_eps(wavelengths_nm, "Au")


def get_cu_permittivity(wavelengths_nm):
    """Cu permittivity, measured J&C 1972 (noble metals), 188-1937 nm."""
    return _jc_eps(wavelengths_nm, "Cu")


# ---- dielectrics (Sellmeier formulas verified) -------------------------------
def get_sio2_permittivity(wavelengths_nm):
    """SiO2 permittivity from Malitson Sellmeier (1965)."""
    lam_um = wavelengths_nm / 1000.0
    l2 = lam_um ** 2
    n_sq = 1.0 + 0.6961663 * l2 / (l2 - 0.0684043**2) \
             + 0.4079426 * l2 / (l2 - 0.1162414**2) \
             + 0.8974794 * l2 / (l2 - 9.896161**2)
    return n_sq


_TIO2_CACHE = {}


def get_tio2_permittivity(wavelengths_nm):
    """TiO2 permittivity, measured Siefke 2016 (ALD amorphous thin film, n,k),
    380-1840 nm. Complex (band-edge absorption near 400 nm; transparent in NIR).
    n smooth-interpolated (CubicSpline), k linear-interpolated & clamped >=0."""
    if "d" not in _TIO2_CACHE:
        a = np.loadtxt(os.path.join(_REF_DIR, "TiO2_Siefke.txt"))  # nm n k
        _TIO2_CACHE["d"] = (a[:, 0], a[:, 1], a[:, 2])
    wl, n_arr, k_arr = _TIO2_CACHE["d"]
    n = CubicSpline(wl, n_arr)(wavelengths_nm)
    k = np.clip(np.interp(wavelengths_nm, wl, k_arr), 0.0, None)
    return (n + 1j * k) ** 2


# ---- Rakic 1998 Lorentz-Drude (optional analytic; noble/simple metals) -------
# NOTE: NOT used for any reported result. The paper uses measured Johnson-Christy n,k
# (MATERIAL_MODEL='jc', the default); this analytic model is retained only as an option.
# eps(w) = 1 - f0*wp^2/(w^2 + i*G0*w) + sum_{j>=1} fj*wp^2/((wj^2 - w^2) - i*Gj*w)
_HC_EV_NM = 1239.841984
_RAKIC_LD = {
    "Au": {"wp": 9.03,  "f": [0.760, 0.024, 0.010, 0.071, 0.601, 4.384],
           "G": [0.053, 0.241, 0.345, 0.870, 2.494, 2.214],
           "w": [0.000, 0.415, 0.830, 2.969, 4.304, 13.32]},
    "Cu": {"wp": 10.83, "f": [0.575, 0.061, 0.104, 0.723, 0.638],
           "G": [0.030, 0.378, 1.056, 3.213, 4.305],
           "w": [0.000, 0.291, 2.957, 5.300, 11.18]},
    "Ag": {"wp": 9.01,  "f": [0.845, 0.065, 0.124, 0.011, 0.840, 5.646],
           "G": [0.048, 3.886, 0.452, 0.065, 0.916, 2.419],
           "w": [0.000, 0.816, 4.481, 8.185, 9.083, 20.29]},
    "Al": {"wp": 14.98, "f": [0.523, 0.227, 0.050, 0.166, 0.030],
           "G": [0.047, 0.333, 0.312, 1.351, 3.382],
           "w": [0.000, 0.162, 1.544, 1.808, 3.473]},
}


def get_metal_permittivity_rakic(wavelengths_nm, metal):
    """Lorentz-Drude permittivity (Rakic 1998); noble/simple metals only."""
    p = _RAKIC_LD[metal]
    wp2 = p["wp"] ** 2
    w = _HC_EV_NM / np.asarray(wavelengths_nm, dtype=float)
    eps = np.ones_like(w, dtype=complex)
    eps = eps - p["f"][0] * wp2 / (w ** 2 + 1j * p["G"][0] * w)
    for j in range(1, len(p["f"])):
        eps = eps + p["f"][j] * wp2 / ((p["w"][j] ** 2 - w ** 2) - 1j * p["G"][j] * w)
    return eps


METAL_EPS_FN = {  # measured J&C
    "Cr": get_cr_permittivity, "Ti": get_ti_permittivity,
    "Au": get_au_permittivity, "Cu": get_cu_permittivity,
}

# "jc" = measured Johnson & Christy (default); "rakic" = Lorentz-Drude analytic.
MATERIAL_MODEL = "jc"


def get_metal_permittivity(wavelengths_nm, metal="Cr"):
    """Get metal permittivity by name, dispatching on MATERIAL_MODEL."""
    if MATERIAL_MODEL == "rakic":
        return get_metal_permittivity_rakic(wavelengths_nm, metal)
    return METAL_EPS_FN[metal](wavelengths_nm)
