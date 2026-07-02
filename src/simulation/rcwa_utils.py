# src/simulation/rcwa_utils.py
"""
Shared RCWA utility functions for TORCWA.
Optimized R/T computation using batch S_parameters.

Grazing-order fix (COS_MIN): torcwa's x/y power normalization contains a
longitudinal-field correction ~ sqrt(1 + (K/kz)^2) * sqrt(kz) that becomes
ill-conditioned for grazing propagating orders (diffraction angle -> 90 deg, i.e.
real(kz) small relative to the medium index) at oblique incidence. Such an order's
true diffraction efficiency -> 0, but the RCWA solve inflates it, producing R+T >> 1
(the source of the unphysical negative A in struct_C_500.npz at oblique incidence;
e.g. row 186 gave R=9.44). We exclude orders whose propagation cosine
real(kz_norm)/n_medium is below COS_MIN (i.e. diffraction angle > ~84 deg) from the
power sum, separately for the reflection (input) and transmission (output) media. At
normal incidence such orders carry ~0 polarization weight, so well-behaved cases are
unchanged; for opaque MIM absorbers (T~0, specular order dominant) the excluded real
power is negligible. Residual cases that still violate R+T<=1 are a torcwa oblique
limitation -- callers should treat A<0 as a low-confidence flag.
"""

import os
import torch
import numpy as np

# Grazing-order cutoff. Default 0.10 (diffraction angle > ~84 deg). Configurable via
# RCWA_COS_MIN so the residual oblique inflation (orders at ~60-84 deg whose torcwa
# power_norm is inflated but whose individual DE stays <1, summing to R+T>1 for some
# oblique geometries -- most prominent for the Au ring-disk Structure B) can be removed
# by tightening the cutoff; the excluded grazing orders carry ~0 true longitudinal power.
COS_MIN = max(float(os.environ.get("RCWA_COS_MIN", "0.10")), 1e-3)  # floor >0: COS_MIN=0 would
# keep exactly-grazing orders and divide the power normalization by a vanishing k_z (hang/NaN).


def _nongrazing_mask(sim, eps_med):
    """Boolean mask (aligned with the [ox outer, oy inner] order grid) selecting
    orders that are propagating AND not near-grazing in a medium of permittivity
    eps_med (mu=1): real(kz_norm)/sqrt(eps_med) > COS_MIN."""
    em = eps_med.item() if torch.is_tensor(eps_med) else eps_med
    n_med = float(np.sqrt(abs(np.real(complex(em)))))
    kz = torch.sqrt(eps_med * 1.0 - sim.Kx_norm_dn**2 - sim.Ky_norm_dn**2)
    return torch.real(kz) / n_med > COS_MIN


def _per_order_DE(sim, order, pols, port, mask):
    """Per-order diffraction efficiency (summed over the two output components)
    for the given input polarization components, masked to non-grazing orders."""
    all_orders = [[ox, oy] for ox in range(-order[0], order[0]+1)
                           for oy in range(-order[1], order[1]+1)]
    de = None
    for pol in pols:
        s = sim.S_parameters(orders=all_orders, direction='forward', port=port,
                             polarization=pol, power_norm=True)
        de = torch.abs(s)**2 if de is None else de + torch.abs(s)**2
    de = torch.where(mask, de, torch.zeros_like(de))
    # Physical cap: a single propagating order cannot carry >100% of incident power.
    # torcwa inflates grazing orders near Wood's anomalies (the source of R+T>1); drop them.
    de = torch.where(de > 1.0, torch.zeros_like(de), de)
    return torch.sum(de).item()


def _sum_RT(sim, order, pols):
    mask_r = _nongrazing_mask(sim, sim.eps_in)    # reflection -> input medium
    mask_t = _nongrazing_mask(sim, sim.eps_out)   # transmission -> output medium
    R_total = _per_order_DE(sim, order, pols, 'reflection', mask_r)
    T_total = _per_order_DE(sim, order, pols, 'transmission', mask_t)
    return R_total, T_total


def compute_RT_batch(sim, order):
    """Total R and T for TE input (Ey, amplitude=[0,1]) via batch S_parameters,
    with grazing-order exclusion (see module docstring).
      R = sum_nongrazing |S_refl_yy|^2 + |S_refl_xy|^2
      T = sum_nongrazing |S_trans_yy|^2 + |S_trans_xy|^2
    """
    return _sum_RT(sim, order, ['yy', 'xy'])


def compute_RT_batch_TM(sim, order):
    """Total R and T for TM input (Ex, amplitude=[1,0]); see compute_RT_batch.
      R = sum_nongrazing |S_refl_xx|^2 + |S_refl_yx|^2
      T = sum_nongrazing |S_trans_xx|^2 + |S_trans_yx|^2
    """
    return _sum_RT(sim, order, ['xx', 'yx'])
