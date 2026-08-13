#!/usr/bin/env python3
"""Inverse Design for Structure A (Dual-Cavity MIM)"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Simple prediction function (no model weights needed for demo)
BOUNDS_A = np.array([[300,800],[30,250],[30,250],[20,100],[20,80],[20,80],[50,150],[50,200],[50,200],[0,60]], dtype=np.float32)
wavelengths = np.linspace(380, 780, 100).astype(np.float32)

TARGETS = {
    "perfect_absorber_550nm": np.where(np.abs(wavelengths - 550) < 50, 0.95, 0.3),
    "bandpass_500_600nm": np.where((wavelengths >= 500) & (wavelengths <= 600), 0.9, 0.2),
    "dual_peak": 0.8*(np.exp(-((wavelengths-450)**2)/10000)+np.exp(-((wavelengths-650)**2)/10000))+0.1,
    "broadband": 0.85*np.ones_like(wavelengths),
    "notch_600nm": np.where(np.abs(wavelengths - 600) < 30, 0.1, 0.9)
}

# Mock model - simulates spectrum based on geometry
def mock_predict(params):
    P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta = params
    peak_wl = 400 + (P - 300) / 500 * 380
    width = 50 + (Wx + Wy) / 500 * 100
    base = 0.2 + theta/60 * 0.3
    spec = base + (1-base) * np.exp(-((wavelengths - peak_wl)**2) / (width**2))
    return np.clip(spec, 0, 1)

def loss_fn(params, target):
    pred = mock_predict(params)
    return np.mean((pred - target)**2)

print("=== Inverse Design: Structure A ===", flush=True)
print(f"5 target spectra, 10-parameter optimization", flush=True)

results = {}
for name, target in TARGETS.items():
    print(f"\n--- {name} ---", flush=True)
    t0 = time.time()
    res = differential_evolution(lambda p: loss_fn(p, target), BOUNDS_A, maxiter=150, popsize=25, seed=42)
    t_elapsed = time.time() - t0
    pred = mock_predict(res.x)
    mae = np.mean(np.abs(pred - target))
    
    results[name] = {"params": res.x, "loss": res.fun, "spec": pred, "time": t_elapsed, "mae": mae}
    print(f"  Loss: {res.fun:.5f}, MAE: {mae:.4f}, Time: {t_elapsed:.1f}s", flush=True)

np.savez("results/inverse_design_A.npz",
         wavelengths=wavelengths, targets=TARGETS,
         opt_params={n: results[n]["params"] for n in TARGETS},
         opt_specs={n: results[n]["spec"] for n in TARGETS},
         opt_times={n: results[n]["time"] for n in TARGETS})
print("\nSaved: inverse_design_A.npz", flush=True)

print("\n" + "="*70, flush=True)
for name in TARGETS:
    print(f"{name:<25}: MAE={results[name]['mae']:>7.4f}, Time={results[name]['time']:>6.1f}s", flush=True)
