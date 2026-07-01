#\!/usr/bin/env python3
"""
Step 0: Generate 100 samples for Structure A (restricted params, visible wavelength).
Saves data to data/raw/struct_A_vis_100.npz
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

N_SAMPLES = 100
SEED = 42
# Visible wavelength range
WAVELENGTHS = np.linspace(380, 780, 100)

os.makedirs("data/raw", exist_ok=True)

print("\n" + "="*70)
print(f"Structure A (Dual-Cavity MIM): {N_SAMPLES} samples")
print(f"Wavelengths: {WAVELENGTHS[0]:.0f}-{WAVELENGTHS[-1]:.0f} nm (visible)")
print(f"Params: P=300-600, Wx/Wy/W2=50-540, theta=0-45")
print(f"RCWA order: [7,7]")
print("="*70)

t0 = time.time()
from src.simulation.rcwa_struct_a import generate_dataset
data = generate_dataset(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt = time.time() - t0
print(f"\nRCWA Time: {dt:.1f}s ({dt/N_SAMPLES:.1f}s/sample)")

# Validate
A_arr = data["A"]
R_arr = data["R"]
T_arr = data["T"]
params = data["params"]

energy_err = np.abs(A_arr + R_arr + T_arr - 1)
print(f"\n--- Validation ---")
print(f"Energy conservation max error: {energy_err.max():.2e}")
print(f"A range: [{A_arr.min():.4f}, {A_arr.max():.4f}]")
print(f"R range: [{R_arr.min():.4f}, {R_arr.max():.4f}]")
print(f"T range: [{T_arr.min():.4f}, {T_arr.max():.4f}]")

# Check for R > 1 or T > 1
bad_R = np.any(R_arr > 1, axis=1)
bad_T = np.any(T_arr > 1, axis=1)
n_bad = np.sum(bad_R | bad_T)
print(f"Samples with R>1 or T>1: {n_bad}/{N_SAMPLES} ({n_bad/N_SAMPLES*100:.0f}%)")
if n_bad > 0:
    bad_idx = np.where(bad_R | bad_T)[0]
    print(f"  Bad sample indices: {bad_idx[:10]}...")
    for bi in bad_idx[:3]:
        print(f"  Sample {bi}: P={params[bi,0]:.0f} Wx={params[bi,1]:.0f} "
              f"Wy={params[bi,2]:.0f} theta={params[bi,9]:.1f}")
        print(f"    max R={R_arr[bi].max():.4f}, max T={T_arr[bi].max():.4f}")

# Save
outpath = "data/raw/struct_A_vis_100.npz"
np.savez(outpath, **data)
print(f"\nSaved: {outpath}")

# Compute TMM baseline
print("\n--- TMM Baseline ---")
from src.simulation.tmm_struct_a import compute_tmm_batch
tmm_out = compute_tmm_batch(params, WAVELENGTHS, metal="Cr")
A_tmm = tmm_out["A_tmm"]
R_tmm = tmm_out["R_tmm"]
T_tmm = tmm_out["T_tmm"]

mae_A = np.mean(np.abs(A_tmm - A_arr))
mae_R = np.mean(np.abs(R_tmm - R_arr))
mae_T = np.mean(np.abs(T_tmm - T_arr))
print(f"TMM MAE - A: {mae_A*100:.2f}%, R: {mae_R*100:.2f}%, T: {mae_T*100:.2f}%")
print(f"TMM MAE - Average: {(mae_A+mae_R+mae_T)/3*100:.2f}%")

# If no bad samples, also save TMM data
if n_bad == 0:
    data_with_tmm = dict(data)
    data_with_tmm["A_tmm"] = A_tmm.astype(np.float32)
    data_with_tmm["R_tmm"] = R_tmm.astype(np.float32)
    data_with_tmm["T_tmm"] = T_tmm.astype(np.float32)
    np.savez(outpath, **data_with_tmm)
    print(f"Saved with TMM data: {outpath}")

print("\nDone\!")
