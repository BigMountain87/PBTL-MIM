#!/usr/bin/env python3
"""Generate 500 samples for Structure C (Dual-Polarization, 400-1800nm)."""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

N_SAMPLES = 500
SEED = 42
WAVELENGTHS = np.linspace(400, 1800, 100)

os.makedirs("data/raw", exist_ok=True)

print(f"\nStructure C: {N_SAMPLES} samples, {WAVELENGTHS[0]:.0f}-{WAVELENGTHS[-1]:.0f}nm", flush=True)

t0 = time.time()
from src.simulation.rcwa_struct_c import generate_dataset
data = generate_dataset(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt = time.time() - t0
print(f"\nRCWA Time: {dt:.1f}s ({dt/N_SAMPLES:.1f}s/sample)", flush=True)

A_TE = data["A_TE"]
A_TM = data["A_TM"]
params = data["params"]

bad_te = np.any(A_TE < -0.01, axis=1)
bad_tm = np.any(A_TM < -0.01, axis=1)
bad_any = bad_te | bad_tm
print(f"A_TE range: [{A_TE.min():.4f}, {A_TE.max():.4f}]", flush=True)
print(f"A_TM range: [{A_TM.min():.4f}, {A_TM.max():.4f}]", flush=True)
print(f"Bad samples (neg A): {bad_any.sum()}/{N_SAMPLES} ({bad_any.sum()/N_SAMPLES*100:.1f}%)", flush=True)

outpath = "data/raw/struct_C_500.npz"
np.savez(outpath, **data)
print(f"Saved: {outpath}", flush=True)
print(f"Done! Total time: {dt/60:.1f} min", flush=True)
