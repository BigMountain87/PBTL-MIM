#!/usr/bin/env python3
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import differential_evolution
from src.simulation.tmm_struct_a import compute_tmm_batch

wavelengths = np.linspace(380, 780, 100).astype(np.float32)
BOUNDS_A = np.array([[300,800],[30,250],[30,250],[20,100],[20,80],[20,80],[50,150],[50,200],[50,200],[0,60]], dtype=np.float32)

TARGETS = {
    "perfect_550nm": np.where(np.abs(wavelengths - 550) < 50, 0.95, 0.3),
    "bandpass_500_600nm": np.where((wavelengths >= 500) & (wavelengths <= 600), 0.9, 0.2),
    "dual_peak": 0.8*(np.exp(-((wavelengths-450)**2)/10000)+np.exp(-((wavelengths-650)**2)/10000))+0.1,
    "broadband": 0.85*np.ones_like(wavelengths),
    "notch_600nm": np.where(np.abs(wavelengths-600)<30, 0.1, 0.9)
}

print("=== Inverse Design A (TMM) ===", flush=True)

def loss_fn(params, target):
    try:
        out = compute_tmm_batch(params[np.newaxis, :], wavelengths)
        spec = out["A_tmm"][0]
        return np.mean((spec - target)**2)
    except:
        return 999.0

results = {}
for name, target in TARGETS.items():
    print(f"\n--- {name} ---", flush=True)
    t0 = time.time()
    res = differential_evolution(lambda p: loss_fn(p, target), BOUNDS_A, maxiter=120, popsize=25, seed=42, workers=1)
    t_elapsed = time.time() - t0
    spec = compute_tmm_batch(res.x[np.newaxis, :], wavelengths)["A_tmm"][0]
    mae = np.mean(np.abs(spec - target))
    results[name] = {"mae": mae, "time": t_elapsed, "params": res.x}
    print(f"  MAE: {mae:.4f}, Time: {t_elapsed:.1f}s", flush=True)

print("\n" + "="*70, flush=True)
for name in TARGETS:
    r = results[name]
    print(f"{name:<25}: MAE={r['mae']:.4f}, Time={r['time']:.1f}s", flush=True)
