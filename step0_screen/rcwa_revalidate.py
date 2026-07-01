#!/usr/bin/env python3
"""
Re-run RCWA validation on saved inverse design results.
Fixes GPU memory/autograd state issue from original run.
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gc

from src.simulation.rcwa_struct_a import simulate_single, PARAM_NAMES
from src.utils.data_utils import get_bounds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_, PMIN, PMAX = get_bounds("A")

# Load inverse design results
ckpt_dir = "results/inverse_design"
inv = np.load(f"{ckpt_dir}/inverse_results.npz", allow_pickle=True)
m0_params = inv["m0_params"]
mp_params = inv["mp_params"]
target_A = inv["target_A"]
target_params = inv["target_params"]
wavelengths = inv["wavelengths"]

N_TARGETS = len(m0_params)
print(f"Re-validating {N_TARGETS} targets with RCWA...")

def rcwa_validate(params_vec, wavelengths_nm):
    """Run RCWA on designed params with proper GPU cleanup."""
    # Clear GPU state
    torch.cuda.empty_cache()
    gc.collect()

    p = {n: float(params_vec[i]) for i, n in enumerate(PARAM_NAMES)}
    # Enforce constraints
    max_w = 0.9 * p["P"]
    p["Wx"] = min(p["Wx"], max_w)
    p["Wy"] = min(p["Wy"], max_w)
    p["W2"] = min(p["W2"], max_w)
    for i, n in enumerate(PARAM_NAMES):
        p[n] = np.clip(p[n], float(PMIN[i]), float(PMAX[i]))

    with torch.no_grad():
        A, R, T = simulate_single(p, wavelengths_nm, metal="Cr", device=device)

    return A, R, T


# First: verify with a known sample
print("\n--- Sanity check: original target sample ---")
p_orig = target_params[0]
p_dict = {n: float(p_orig[i]) for i, n in enumerate(PARAM_NAMES)}
A_check, R_check, T_check = simulate_single(p_dict, wavelengths, metal="Cr", device=device)
print(f"  Original sample avg_A={target_A[0].mean():.4f}, re-sim avg_A={A_check.mean():.4f}")
print(f"  Match: {np.allclose(A_check, target_A[0], atol=0.01)}")

# Re-validate all designs
print("\n--- RCWA Re-validation ---")
m0_rcwa_A = np.zeros_like(target_A)
mp_rcwa_A = np.zeros_like(target_A)
m0_mae = np.zeros(N_TARGETS)
mp_mae = np.zeros(N_TARGETS)

for ti in range(N_TARGETS):
    print(f"\nTarget {ti+1}/{N_TARGETS} (target avg_A={target_A[ti].mean():.3f})")

    # M0 design
    A_m0, _, _ = rcwa_validate(m0_params[ti], wavelengths)
    m0_rcwa_A[ti] = A_m0
    m0_mae[ti] = np.mean(np.abs(A_m0 - target_A[ti])) * 100

    # M_phys design
    A_mp, _, _ = rcwa_validate(mp_params[ti], wavelengths)
    mp_rcwa_A[ti] = A_mp
    mp_mae[ti] = np.mean(np.abs(A_mp - target_A[ti])) * 100

    print(f"  M0 design:     avg_A={A_m0.mean():.4f}, RCWA MAE={m0_mae[ti]:.2f}%")
    print(f"  M_phys design: avg_A={A_mp.mean():.4f}, RCWA MAE={mp_mae[ti]:.2f}%")
    print(f"  M0 params:     {[f'{v:.1f}' for v in m0_params[ti]]}")
    print(f"  M_phys params: {[f'{v:.1f}' for v in mp_params[ti]]}")

# Summary
print("\n" + "="*70)
print("CORRECTED INVERSE DESIGN RESULTS")
print("="*70)

mp_wins = np.sum(mp_mae < m0_mae)
print(f"\nRCWA MAE (mean):  M0 = {m0_mae.mean():.2f}% ± {m0_mae.std():.2f}%,  M_phys = {mp_mae.mean():.2f}% ± {mp_mae.std():.2f}%")
print(f"RCWA MAE (median): M0 = {np.median(m0_mae):.2f}%,  M_phys = {np.median(mp_mae):.2f}%")
print(f"Win rate:         M_phys wins {mp_wins}/{N_TARGETS} ({mp_wins/N_TARGETS*100:.0f}%)")
print(f"Success rate (MAE<10%): M0 = {(m0_mae<10).sum()}/{N_TARGETS},  M_phys = {(mp_mae<10).sum()}/{N_TARGETS}")
print(f"Success rate (MAE<5%):  M0 = {(m0_mae<5).sum()}/{N_TARGETS},  M_phys = {(mp_mae<5).sum()}/{N_TARGETS}")

# Per-target table
print(f"\n{'Target':>7} | {'tgt avg_A':>10} | {'M0 MAE':>8} | {'M_phys MAE':>11} | {'Winner':>7}")
print("-"*55)
for ti in range(N_TARGETS):
    winner = "M_phys" if mp_mae[ti] < m0_mae[ti] else "M0"
    print(f"  {ti+1:>5} | {target_A[ti].mean():>10.3f} | {m0_mae[ti]:>7.2f}% | {mp_mae[ti]:>10.2f}% | {winner:>7}")

# Save corrected results
np.savez(f"{ckpt_dir}/inverse_results_corrected.npz",
         target_A=target_A, target_params=target_params,
         m0_params=m0_params, mp_params=mp_params,
         m0_rcwa_A=m0_rcwa_A, mp_rcwa_A=mp_rcwa_A,
         m0_mae=m0_mae, mp_mae=mp_mae,
         wavelengths=wavelengths)
print(f"\nCorrected results saved to {ckpt_dir}/inverse_results_corrected.npz")

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Top row: 3 example spectra
for col, ti in enumerate([0, N_TARGETS//2, N_TARGETS-1]):
    ax = axes[0, col]
    ax.plot(wavelengths, target_A[ti], 'k-', lw=2, label='Target')
    ax.plot(wavelengths, m0_rcwa_A[ti], 'r--', lw=1.5, label=f'M0 (MAE={m0_mae[ti]:.1f}%)')
    ax.plot(wavelengths, mp_rcwa_A[ti], 'b-', lw=1.5, label=f'M_phys (MAE={mp_mae[ti]:.1f}%)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorption')
    ax.set_title(f'Target {ti+1} (avg A={target_A[ti].mean():.2f})')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

# Scatter
ax = axes[1, 0]
ax.scatter(m0_mae, mp_mae, c='steelblue', s=50, edgecolors='navy', alpha=0.8)
lim = max(m0_mae.max(), mp_mae.max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
ax.set_xlabel('M0 RCWA MAE (%)')
ax.set_ylabel('M_phys RCWA MAE (%)')
ax.set_title(f'M_phys wins {mp_wins}/{N_TARGETS}')

# Box
ax = axes[1, 1]
bp = ax.boxplot([m0_mae, mp_mae], tick_labels=['M0', 'M_phys'], patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#FFB3B3')
bp['boxes'][1].set_facecolor('#B3D9FF')
ax.set_ylabel('RCWA MAE (%)')
ax.set_title('Design Error Distribution')

# Bar
ax = axes[1, 2]
metrics = ['Mean\nMAE', 'Median\nMAE', 'Success\n(MAE<10%)', 'Success\n(MAE<5%)']
m0_vals = [m0_mae.mean(), np.median(m0_mae), (m0_mae<10).sum()/N_TARGETS*100, (m0_mae<5).sum()/N_TARGETS*100]
mp_vals = [mp_mae.mean(), np.median(mp_mae), (mp_mae<10).sum()/N_TARGETS*100, (mp_mae<5).sum()/N_TARGETS*100]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, m0_vals, w, label='M0', color='#FF7777')
ax.bar(x + w/2, mp_vals, w, label='M_phys', color='#77AAFF')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel('Value')
ax.set_title('Aggregate Comparison')
ax.legend()

plt.suptitle('Inverse Design: M0 vs M_phys (RCWA-validated, corrected)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{ckpt_dir}/inverse_corrected.png", dpi=150, bbox_inches='tight')
print(f"Plot saved to {ckpt_dir}/inverse_corrected.png")

print("\nDONE!")
