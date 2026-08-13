#!/usr/bin/env python3
"""Quick validation: 2 samples × 3 structures, 30 wavelengths, batch method."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

wavelengths = np.linspace(400, 1800, 30)

# ---- Structure A ----
print("\n" + "="*60)
print("Structure A: Dual-Cavity Asymmetric MIM")
print("="*60)
from src.simulation.rcwa_struct_a import simulate_single as sim_a

test_params_a = [
    {"P": 500, "Wx": 200, "Wy": 250, "W2": 180, "t1": 40, "t2": 30,
     "t_mid": 15, "d1": 100, "d2": 80, "theta": 0},
    {"P": 600, "Wx": 300, "Wy": 200, "W2": 250, "t1": 50, "t2": 40,
     "t_mid": 10, "d1": 150, "d2": 120, "theta": 30},
]

for j, p in enumerate(test_params_a):
    t0 = time.time()
    A, R, T = sim_a(p, wavelengths, device=device)
    dt = time.time() - t0
    energy_err = np.max(np.abs(A + R + T - 1))
    valid = (np.all(A >= -1e-6) and np.all(R >= -1e-6) and np.all(T >= -1e-6)
             and energy_err < 1e-6)
    print(f"  Sample {j}: A=[{A.min():.4f},{A.max():.4f}] "
          f"R=[{R.min():.4f},{R.max():.4f}] T=[{T.min():.4f},{T.max():.4f}] "
          f"energy_err={energy_err:.2e} time={dt:.1f}s {'PASS' if valid else 'FAIL'}")

# ---- Structure B ----
print("\n" + "="*60)
print("Structure B: Ring-Disk Fano MIM")
print("="*60)
from src.simulation.rcwa_struct_b import simulate_single as sim_b

test_params_b = [
    {"P": 500, "R_out": 200, "R_in": 120, "R_disk": 50,
     "t_Cr": 50, "d_SiO2": 100, "theta": 0, "phi": 0},
    {"P": 600, "R_out": 250, "R_in": 150, "R_disk": 60,
     "t_Cr": 40, "d_SiO2": 150, "theta": 30, "phi": 20},
]

for j, p in enumerate(test_params_b):
    t0 = time.time()
    A, R, T = sim_b(p, wavelengths, device=device)
    dt = time.time() - t0
    energy_err = np.max(np.abs(A + R + T - 1))
    valid = (np.all(A >= -1e-6) and np.all(R >= -1e-6) and np.all(T >= -1e-6)
             and energy_err < 1e-6)
    print(f"  Sample {j}: A=[{A.min():.4f},{A.max():.4f}] "
          f"R=[{R.min():.4f},{R.max():.4f}] T=[{T.min():.4f},{T.max():.4f}] "
          f"energy_err={energy_err:.2e} time={dt:.1f}s {'PASS' if valid else 'FAIL'}")

# ---- Structure C ----
print("\n" + "="*60)
print("Structure C: Dual-Polarization Rectangular MIM")
print("="*60)
from src.simulation.rcwa_struct_c import simulate_single as sim_c

test_params_c = [
    {"P": 500, "Wx": 200, "Wy": 300, "t_Cr": 50, "d_SiO2": 100,
     "theta": 0, "phi": 0},
    {"P": 600, "Wx": 350, "Wy": 200, "t_Cr": 40, "d_SiO2": 150,
     "theta": 30, "phi": 20},
]

for j, p in enumerate(test_params_c):
    t0 = time.time()
    A_te, R_te, T_te, A_tm, R_tm, T_tm = sim_c(p, wavelengths, device=device)
    dt = time.time() - t0
    energy_err_te = np.max(np.abs(A_te + R_te + T_te - 1))
    energy_err_tm = np.max(np.abs(A_tm + R_tm + T_tm - 1))
    valid = (energy_err_te < 1e-6 and energy_err_tm < 1e-6
             and np.all(A_te >= -1e-6) and np.all(A_tm >= -1e-6))
    pol_diff = np.mean(np.abs(A_te - A_tm))
    print(f"  Sample {j}: TE A=[{A_te.min():.4f},{A_te.max():.4f}] "
          f"TM A=[{A_tm.min():.4f},{A_tm.max():.4f}] "
          f"TE-TM diff={pol_diff:.4f}")
    print(f"    energy_err TE={energy_err_te:.2e} TM={energy_err_tm:.2e} "
          f"time={dt:.1f}s {'PASS' if valid else 'FAIL'}")

print("\n" + "="*60)
print("All validation tests complete!")
print("="*60)
