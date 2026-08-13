#!/usr/bin/env python3
"""Clean RCWA re-validation - no NN imports, pure RCWA only."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.simulation.rcwa_struct_a import simulate_single, PARAM_NAMES
from src.utils.data_utils import get_bounds

device = torch.device("cuda")
_, PMIN, PMAX = get_bounds("A")

inv = np.load("results/inverse_design/inverse_results.npz")
m0_params = inv["m0_params"]
mp_params = inv["mp_params"]
target_A = inv["target_A"]
target_params = inv["target_params"]
wl = inv["wavelengths"]
N = len(m0_params)

print(f"Re-validating {N} targets...")

def safe_simulate(params_vec):
    p = {n: float(params_vec[i]) for i, n in enumerate(PARAM_NAMES)}
    max_w = 0.9 * p["P"]
    p["Wx"] = min(p["Wx"], max_w)
    p["Wy"] = min(p["Wy"], max_w)
    p["W2"] = min(p["W2"], max_w)
    for i, n in enumerate(PARAM_NAMES):
        p[n] = np.clip(p[n], float(PMIN[i]), float(PMAX[i]))
    A, R, T = simulate_single(p, wl, metal="Cr", device=device)
    return A

m0_mae = np.zeros(N)
mp_mae = np.zeros(N)
m0_rcwa = np.zeros((N, len(wl)))
mp_rcwa = np.zeros((N, len(wl)))

for ti in range(N):
    A_m0 = safe_simulate(m0_params[ti])
    A_mp = safe_simulate(mp_params[ti])
    m0_rcwa[ti] = A_m0
    mp_rcwa[ti] = A_mp
    m0_mae[ti] = np.mean(np.abs(A_m0 - target_A[ti])) * 100
    mp_mae[ti] = np.mean(np.abs(A_mp - target_A[ti])) * 100
    print(f"  T{ti+1:>2}: tgt={target_A[ti].mean():.3f} | M0_A={A_m0.mean():.3f} MAE={m0_mae[ti]:.1f}% | Mp_A={A_mp.mean():.3f} MAE={mp_mae[ti]:.1f}%")

mp_wins = (mp_mae < m0_mae).sum()
print(f"\n{'='*60}")
print(f"M0 MAE:  {m0_mae.mean():.2f}% +/- {m0_mae.std():.2f}%")
print(f"Mp MAE:  {mp_mae.mean():.2f}% +/- {mp_mae.std():.2f}%")
print(f"Mp wins: {mp_wins}/{N} ({mp_wins/N*100:.0f}%)")
print(f"MAE<10%: M0={np.sum(m0_mae<10)}/{N}, Mp={np.sum(mp_mae<10)}/{N}")
print(f"MAE<5%:  M0={np.sum(m0_mae<5)}/{N}, Mp={np.sum(mp_mae<5)}/{N}")

np.savez("results/inverse_design/rcwa_clean.npz",
         m0_mae=m0_mae, mp_mae=mp_mae, m0_rcwa=m0_rcwa, mp_rcwa=mp_rcwa,
         target_A=target_A, wavelengths=wl, m0_params=m0_params, mp_params=mp_params)
print("Saved.")
