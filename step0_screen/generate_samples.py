#!/usr/bin/env python3
"""
Step 0: Generate 100 samples per structure using TORCWA GPU + compute TMM baseline.
Saves data to data/raw/struct_{A,B,C}_100.npz
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

PYTHON = 'sys.executable'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

N_SAMPLES = 100
SEED = 42
WAVELENGTHS = np.linspace(400, 1800, 100)

os.makedirs('data/raw', exist_ok=True)

# ===== Structure A =====
print("\n" + "="*70)
print(f"Generating Structure A: {N_SAMPLES} samples")
print("="*70)
t0 = time.time()
from src.simulation.rcwa_struct_a import generate_dataset as gen_a
data_a = gen_a(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt_a = time.time() - t0
print(f"  Time: {dt_a:.1f}s ({dt_a/N_SAMPLES:.1f}s/sample)")

# Validate
A_a, R_a, T_a = data_a["A"], data_a["R"], data_a["T"]
energy_err_a = np.max(np.abs(A_a + R_a + T_a - 1))
print(f"  Energy conservation error: {energy_err_a:.2e}")
print(f"  A range: [{A_a.min():.4f}, {A_a.max():.4f}]")
print(f"  R range: [{R_a.min():.4f}, {R_a.max():.4f}]")
print(f"  T range: [{T_a.min():.4f}, {T_a.max():.4f}]")

np.savez('data/raw/struct_A_100.npz', **data_a)
print("  Saved: data/raw/struct_A_100.npz")

# Compute TMM baseline
from src.simulation.tmm_struct_a import compute_tmm_batch as tmm_a
tmm_a_out = tmm_a(data_a["params"], WAVELENGTHS, metal="Cr")
tmm_mae_a = np.mean(np.abs(tmm_a_out["A_tmm"] - A_a))
print(f"  TMM-only MAE (absorption): {tmm_mae_a:.4f} ({tmm_mae_a*100:.2f}%)")

# ===== Structure B =====
print("\n" + "="*70)
print(f"Generating Structure B: {N_SAMPLES} samples")
print("="*70)
t0 = time.time()
from src.simulation.rcwa_struct_b import generate_dataset as gen_b
data_b = gen_b(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt_b = time.time() - t0
print(f"  Time: {dt_b:.1f}s ({dt_b/N_SAMPLES:.1f}s/sample)")

A_b, R_b, T_b = data_b["A"], data_b["R"], data_b["T"]
energy_err_b = np.max(np.abs(A_b + R_b + T_b - 1))
print(f"  Energy conservation error: {energy_err_b:.2e}")
print(f"  A range: [{A_b.min():.4f}, {A_b.max():.4f}]")
print(f"  R range: [{R_b.min():.4f}, {R_b.max():.4f}]")
print(f"  T range: [{T_b.min():.4f}, {T_b.max():.4f}]")

np.savez('data/raw/struct_B_100.npz', **data_b)
print("  Saved: data/raw/struct_B_100.npz")

from src.simulation.tmm_struct_b import compute_tmm_batch as tmm_b
tmm_b_out = tmm_b(data_b["params"], WAVELENGTHS, metal="Cr")
tmm_mae_b = np.mean(np.abs(tmm_b_out["A_tmm"] - A_b))
print(f"  TMM-only MAE (absorption): {tmm_mae_b:.4f} ({tmm_mae_b*100:.2f}%)")

# ===== Structure C =====
print("\n" + "="*70)
print(f"Generating Structure C: {N_SAMPLES} samples")
print("="*70)
t0 = time.time()
from src.simulation.rcwa_struct_c import generate_dataset as gen_c
data_c = gen_c(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt_c = time.time() - t0
print(f"  Time: {dt_c:.1f}s ({dt_c/N_SAMPLES:.1f}s/sample)")

A_te_c, R_te_c, T_te_c = data_c["A_TE"], data_c["R_TE"], data_c["T_TE"]
A_tm_c, R_tm_c, T_tm_c = data_c["A_TM"], data_c["R_TM"], data_c["T_TM"]
energy_err_te = np.max(np.abs(A_te_c + R_te_c + T_te_c - 1))
energy_err_tm = np.max(np.abs(A_tm_c + R_tm_c + T_tm_c - 1))
print(f"  Energy conservation error: TE={energy_err_te:.2e}, TM={energy_err_tm:.2e}")
print(f"  A_TE range: [{A_te_c.min():.4f}, {A_te_c.max():.4f}]")
print(f"  A_TM range: [{A_tm_c.min():.4f}, {A_tm_c.max():.4f}]")
pol_diff = np.mean(np.abs(A_te_c - A_tm_c))
print(f"  Mean |A_TE - A_TM|: {pol_diff:.4f}")

np.savez('data/raw/struct_C_100.npz', **data_c)
print("  Saved: data/raw/struct_C_100.npz")

from src.simulation.tmm_struct_c import compute_tmm_batch as tmm_c
tmm_c_out = tmm_c(data_c["params"], WAVELENGTHS, metal="Cr")
tmm_mae_te = np.mean(np.abs(tmm_c_out["A_tmm_te"] - A_te_c))
tmm_mae_tm = np.mean(np.abs(tmm_c_out["A_tmm_tm"] - A_tm_c))
print(f"  TMM-only MAE: TE={tmm_mae_te:.4f} ({tmm_mae_te*100:.2f}%), "
      f"TM={tmm_mae_tm:.4f} ({tmm_mae_tm*100:.2f}%)")

# ===== Summary =====
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Structure':<15} {'RCWA time':<15} {'TMM MAE':<15} {'Energy err':<15}")
print(f"{'A (Dual-Cav)':<15} {f'{dt_a:.0f}s':<15} {f'{tmm_mae_a*100:.2f}%':<15} {f'{energy_err_a:.1e}':<15}")
print(f"{'B (Ring-Disk)':<15} {f'{dt_b:.0f}s':<15} {f'{tmm_mae_b*100:.2f}%':<15} {f'{energy_err_b:.1e}':<15}")
print(f"{'C TE (Dual-P)':<15} {f'{dt_c:.0f}s':<15} {f'{tmm_mae_te*100:.2f}%':<15} {f'{energy_err_te:.1e}':<15}")
print(f"{'C TM (Dual-P)':<15} {'---':<15} {f'{tmm_mae_tm*100:.2f}%':<15} {f'{energy_err_tm:.1e}':<15}")
print(f"\nIdeal TMM MAE range: 10-20% (too low → NN unnecessary, too high → TMM useless)")
