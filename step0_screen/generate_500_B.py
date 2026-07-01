#!/usr/bin/env python3
"""Generate 500 samples for Structure B (Ring-Disk Fano, 400-1800nm)."""
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

print(f"\nStructure B: {N_SAMPLES} samples, {WAVELENGTHS[0]:.0f}-{WAVELENGTHS[-1]:.0f}nm", flush=True)

t0 = time.time()
from src.simulation.rcwa_struct_b import generate_dataset
data = generate_dataset(N_SAMPLES, WAVELENGTHS, metal="Cr", seed=SEED, device=device)
dt = time.time() - t0
print(f"\nRCWA Time: {dt:.1f}s ({dt/N_SAMPLES:.1f}s/sample)", flush=True)

A = data["A"]
R = data["R"]
T = data["T"]
params = data["params"]

bad_mask = np.any((A < -0.01) | (A > 1.01), axis=1)
n_bad = np.sum(bad_mask)
print(f"A range: [{A.min():.4f}, {A.max():.4f}]", flush=True)
print(f"R range: [{R.min():.4f}, {R.max():.4f}]", flush=True)
print(f"Bad samples (neg A): {n_bad}/{N_SAMPLES} ({n_bad/N_SAMPLES*100:.1f}%)", flush=True)

outpath = "data/raw/struct_B_500.npz"
np.savez(outpath, **data)
print(f"Saved: {outpath}", flush=True)
print(f"Done! Total time: {dt/60:.1f} min", flush=True)
