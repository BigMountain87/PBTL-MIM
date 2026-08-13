#!/usr/bin/env python3
"""
Full audit of all data, models, and inverse design results.
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.utils.data_utils import normalize_params, denormalize_params, get_bounds
from src.simulation.materials import (get_sio2_permittivity, get_tio2_permittivity,
                                       get_metal_permittivity)
from src.simulation.rcwa_struct_a import simulate_single, PARAM_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("PART 1: DATA QUALITY AUDIT")
print("="*80)

# Load both datasets
for fname in ["struct_A_vis_100.npz", "struct_A_vis_500.npz"]:
    fpath = f"data/raw/{fname}"
    d = np.load(fpath, allow_pickle=True)
    params = d["params"]
    A = d["A"]
    R = d["R"]
    wl = d["wavelengths"]

    print(f"\n--- {fname} ---")
    print(f"  Samples: {len(params)}, Wavelengths: {len(wl)} ({wl.min():.0f}-{wl.max():.0f} nm)")
    print(f"  Params shape: {params.shape}")

    # Check energy conservation
    T = 1 - A - R  # approximate
    energy = A + R + T
    print(f"  A range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"  R range: [{R.min():.4f}, {R.max():.4f}]")

    # Check for bad samples
    bad_A = np.any((A < -0.01) | (A > 1.01), axis=1)
    bad_R = np.any((R < -0.01) | (R > 1.01), axis=1)
    good = np.all((R>=0)&(R<=1)&(A>=0)&(A<=1), axis=1)
    print(f"  Bad A samples: {bad_A.sum()}/{len(params)}")
    print(f"  Bad R samples: {bad_R.sum()}/{len(params)}")
    print(f"  Good samples: {good.sum()}/{len(params)}")

    # Parameter statistics
    _, pmin, pmax = get_bounds("A")
    print(f"\n  Parameter statistics:")
    for i, name in enumerate(PARAM_NAMES):
        vals = params[:, i]
        print(f"    {name:>6}: [{vals.min():.1f}, {vals.max():.1f}] (bounds: [{pmin[i]:.0f}, {pmax[i]:.0f}])")

    # Spectrum statistics for good samples
    gi = np.where(good)[0]
    A_good = A[gi]
    R_good = R[gi]
    print(f"\n  Good sample spectrum stats:")
    print(f"    Avg absorption (mean): {A_good.mean():.4f}")
    print(f"    Avg absorption (std):  {A_good.mean(axis=1).std():.4f}")
    print(f"    Max absorption: {A_good.max():.4f}")
    print(f"    Min absorption: {A_good.min():.4f}")
    print(f"    Avg reflectance: {R_good.mean():.4f}")

    # Check spectral diversity
    avg_abs = A_good.mean(axis=1)
    print(f"\n  Per-sample avg absorption distribution:")
    for pct in [0, 10, 25, 50, 75, 90, 100]:
        print(f"    {pct:>3}th percentile: {np.percentile(avg_abs, pct):.4f}")


print("\n\n" + "="*80)
print("PART 2: RCWA SPOT CHECK (3 random training samples)")
print("="*80)

# Load 500-sample dataset
d500 = np.load("data/raw/struct_A_vis_500.npz", allow_pickle=True)
params_all = d500["params"].astype(np.float32)
A_all = d500["A"].astype(np.float32)
R_all = d500["R"].astype(np.float32)
wl = d500["wavelengths"].astype(np.float32)

good = np.all((R_all>=0)&(R_all<=1)&(A_all>=0)&(A_all<=1), axis=1)
gi = np.where(good)[0]

# Spot check: re-simulate 3 random samples
rng = np.random.default_rng(123)
check_idx = rng.choice(gi, size=3, replace=False)

for ci in check_idx:
    p = {n: float(params_all[ci, i]) for i, n in enumerate(PARAM_NAMES)}
    print(f"\n  Sample {ci}: params = {[f'{v:.1f}' for v in params_all[ci]]}")

    A_rcwa, R_rcwa, T_rcwa = simulate_single(p, wl, metal="Cr", device=device)

    mae_A = np.mean(np.abs(A_rcwa - A_all[ci])) * 100
    mae_R = np.mean(np.abs(R_rcwa - R_all[ci])) * 100
    print(f"  Re-simulated vs stored: A_MAE={mae_A:.4f}%, R_MAE={mae_R:.4f}%")
    print(f"  Stored A avg={A_all[ci].mean():.4f}, Re-sim A avg={A_rcwa.mean():.4f}")
    print(f"  Energy: A+R+T = {(A_rcwa+R_rcwa+T_rcwa).mean():.6f}")


print("\n\n" + "="*80)
print("PART 3: INVERSE DESIGN FAILURE ANALYSIS")
print("="*80)

# Load inverse design results
inv = np.load("results/inverse_design/inverse_results.npz", allow_pickle=True)
m0_params = inv["m0_params"]
mp_params = inv["mp_params"]
target_A = inv["target_A"]
target_params = inv["target_params"]
m0_rcwa_A = inv["m0_rcwa_A"]
mp_rcwa_A = inv["mp_rcwa_A"]

print(f"\nDesigned params analysis ({len(m0_params)} targets):")
_, pmin, pmax = get_bounds("A")

print(f"\n  {'Param':>8} | {'Target (mean)':>14} | {'M0 design (mean)':>17} | {'M_phys design (mean)':>21}")
print("-" * 70)
for i, name in enumerate(PARAM_NAMES):
    t_mean = target_params[:, i].mean()
    m0_mean = m0_params[:, i].mean()
    mp_mean = mp_params[:, i].mean()
    print(f"  {name:>8} | {t_mean:>14.1f} | {m0_mean:>17.1f} | {mp_mean:>21.1f}")

print(f"\n  Designed params bounds check:")
for i, name in enumerate(PARAM_NAMES):
    m0_oob = np.any((m0_params[:, i] < pmin[i]) | (m0_params[:, i] > pmax[i]))
    mp_oob = np.any((mp_params[:, i] < pmin[i]) | (mp_params[:, i] > pmax[i]))
    print(f"    {name:>8}: M0 out-of-bounds={m0_oob}, M_phys out-of-bounds={mp_oob}")

# Check: are M0 and M_phys designs actually different?
param_diff = np.mean(np.abs(m0_params - mp_params), axis=0)
print(f"\n  Mean |M0 - M_phys| per param:")
for i, name in enumerate(PARAM_NAMES):
    print(f"    {name:>8}: {param_diff[i]:.2f} nm/deg")

# Check RCWA results for designed structures
print(f"\n  RCWA absorption of designed structures:")
print(f"    M0 designs:    avg_A = {m0_rcwa_A.mean():.4f} (range [{m0_rcwa_A.min():.4f}, {m0_rcwa_A.max():.4f}])")
print(f"    M_phys designs: avg_A = {mp_rcwa_A.mean():.4f} (range [{mp_rcwa_A.min():.4f}, {mp_rcwa_A.max():.4f}])")
print(f"    Targets:       avg_A = {target_A.mean():.4f} (range [{target_A.min():.4f}, {target_A.max():.4f}])")

# Are ALL designs producing near-zero absorption?
print(f"\n  Per-target designed vs target absorption:")
for ti in range(min(5, len(target_A))):
    print(f"    Target {ti+1}: target_avg={target_A[ti].mean():.3f}, "
          f"M0_design_avg={m0_rcwa_A[ti].mean():.3f}, "
          f"M_phys_design_avg={mp_rcwa_A[ti].mean():.3f}")

# Re-validate one designed structure manually
print(f"\n  Manual re-validation of M0 design for target 1:")
p0 = m0_params[0]
print(f"    Designed params: {[f'{v:.1f}' for v in p0]}")
p_dict = {n: float(p0[i]) for i, n in enumerate(PARAM_NAMES)}
# Enforce constraints
max_w = 0.9 * p_dict["P"]
for wn in ["Wx", "Wy", "W2"]:
    p_dict[wn] = min(p_dict[wn], max_w)
for i, n in enumerate(PARAM_NAMES):
    p_dict[n] = np.clip(p_dict[n], float(pmin[i]), float(pmax[i]))
print(f"    After constraints: {[f'{p_dict[n]:.1f}' for n in PARAM_NAMES]}")

A_v, R_v, T_v = simulate_single(p_dict, wl, metal="Cr", device=device)
print(f"    RCWA result: avg_A={A_v.mean():.4f}, avg_R={R_v.mean():.4f}, avg_T={T_v.mean():.4f}")
print(f"    A+R+T = {(A_v+R_v+T_v).mean():.6f}")


print("\n\n" + "="*80)
print("PART 4: FORWARD MODEL RE-EVALUATION")
print("="*80)

# Check: does the forward model actually predict well on test samples?
# And does it predict correctly at the inverse-designed params?

# Rebuild models
from step0_screen.inverse_design import M0 as M0Model, MPhys, compute_physics_features

ckpt_dir = "results/inverse_design"
geo_dim = 11

params_good = params_all[gi]
phys = compute_physics_features(params_good, wl, "Cr")
n_phys = phys.shape[-1]
Nlam = len(wl)

phys_flat = phys.reshape(-1, n_phys)
phys_mean = phys_flat.mean(0)
phys_std = phys_flat.std(0) + 1e-8

m0 = M0Model(geo_dim).to(device)
m0.load_state_dict(torch.load(f"{ckpt_dir}/m0_350.pt", map_location=device))
m0.eval()

mp = MPhys(geo_dim, n_phys).to(device)
mp.load_state_dict(torch.load(f"{ckpt_dir}/mphys_350.pt", map_location=device))
mp.eval()

params_norm = normalize_params(params_good, "A")
wl_norm = (wl - wl.min()) / (wl.max() - wl.min())

# Test on a few known samples
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(len(gi))
test_idx = all_idx[-50:]

print(f"\nForward model prediction on 5 test samples:")
for ti in range(5):
    si = test_idx[ti]
    true_A = A_all[gi[si]]

    # Build input
    pn = params_norm[si]
    x_geo = np.concatenate([wl_norm[:,None], np.tile(pn, (Nlam,1))], axis=1).astype(np.float32)
    x_geo_t = torch.tensor(x_geo).to(device)

    phys_i = phys[si]  # [Nlam, 17]
    phys_n = ((phys_i - phys_mean) / phys_std).astype(np.float32)
    phys_t = torch.tensor(phys_n).to(device)

    with torch.no_grad():
        pred_m0 = m0(x_geo_t)["A"].cpu().numpy()
        pred_mp = mp(x_geo_t, p=phys_t)["A"].cpu().numpy()

    mae_m0 = np.mean(np.abs(pred_m0 - true_A)) * 100
    mae_mp = np.mean(np.abs(pred_mp - true_A)) * 100
    print(f"  Sample {si}: true_avg_A={true_A.mean():.3f}, M0_pred={pred_m0.mean():.3f} (MAE={mae_m0:.2f}%), M_phys_pred={pred_mp.mean():.3f} (MAE={mae_mp:.2f}%)")

# Now check: what does the forward model predict for the inverse-designed params?
print(f"\nForward model prediction at DESIGNED params (vs RCWA truth):")
for ti in range(5):
    p_designed = m0_params[ti]
    pn_designed = normalize_params(p_designed.reshape(1,-1), "A")[0]

    x_geo = np.concatenate([wl_norm[:,None], np.tile(pn_designed, (Nlam,1))], axis=1).astype(np.float32)
    x_geo_t = torch.tensor(x_geo).to(device)

    with torch.no_grad():
        pred_m0 = m0(x_geo_t)["A"].cpu().numpy()

    print(f"  Design {ti+1}: M0_pred_avg_A={pred_m0.mean():.3f}, target_avg_A={target_A[ti].mean():.3f}, RCWA_avg_A={m0_rcwa_A[ti].mean():.3f}")
    print(f"    Designed params: {[f'{v:.1f}' for v in p_designed]}")
    print(f"    Surrogate thinks MAE={np.mean(np.abs(pred_m0 - target_A[ti]))*100:.2f}%, RCWA truth MAE={np.mean(np.abs(m0_rcwa_A[ti] - target_A[ti]))*100:.2f}%")

print("\n\n" + "="*80)
print("PART 5: OVERALL ASSESSMENT")
print("="*80)

# Collect all metrics
print(f"""
FORWARD MODEL:
  M0 test MAE:     ~7.9%
  M_phys test MAE: ~6.1%
  Improvement:     ~23%

DATA EFFICIENCY (previous results):
  n=50:  M0=13.28%, M_phys=10.03% (24.5% improvement)
  n=100: M0=11.89%, M_phys=8.90%  (25.2% improvement)
  n=200: M0=9.67%,  M_phys=7.49%  (22.6% improvement)
  n=350: M0=8.68%,  M_phys=6.23%  (28.3% improvement)

INVERSE DESIGN:
  Success rate: 0/20 (both models)
  All designs produce near-zero absorption in RCWA
  Forward model MAE of 6-8% is insufficient for inverse design
""")

print("DONE - Full audit complete")
