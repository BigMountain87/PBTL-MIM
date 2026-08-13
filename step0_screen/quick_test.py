#!/usr/bin/env python3
"""
Quick test: 5 samples per structure to verify TORCWA works correctly.
Checks energy conservation and physical validity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

wavelengths = np.linspace(400, 1800, 50)  # fewer points for quick test
n_test = 3

def test_structure_a():
    """Test Structure A: Dual-cavity MIM"""
    print("\n" + "="*60)
    print("Structure A: Asymmetric Dual-Dielectric Dual-Cavity MIM")
    print("="*60)
    from src.simulation.rcwa_struct_a import simulate_single

    test_params = [
        {"P": 500, "Wx": 200, "Wy": 200, "W2": 180, "t1": 40, "t2": 30,
         "t_mid": 15, "d1": 100, "d2": 80, "theta": 0},
        {"P": 600, "Wx": 300, "Wy": 250, "W2": 200, "t1": 50, "t2": 40,
         "t_mid": 10, "d1": 150, "d2": 120, "theta": 20},
        {"P": 400, "Wx": 180, "Wy": 160, "W2": 150, "t1": 30, "t2": 25,
         "t_mid": 20, "d1": 80, "d2": 60, "theta": 40},
    ]

    for i, p in enumerate(test_params[:n_test]):
        t0 = time.time()
        A, R, T = simulate_single(p, wavelengths, metal="Cr", device=device)
        dt = time.time() - t0

        energy = A + R + T
        energy_err = np.max(np.abs(energy - 1.0))
        a_range = (np.min(A), np.max(A))
        r_range = (np.min(R), np.max(R))

        status = "PASS" if energy_err < 1e-6 and np.all(A >= -0.01) and np.all(R >= -0.01) else "FAIL"
        print(f"  Sample {i}: A=[{a_range[0]:.4f},{a_range[1]:.4f}] "
              f"R=[{r_range[0]:.4f},{r_range[1]:.4f}] "
              f"|A+R+T-1|_max={energy_err:.2e} [{status}] ({dt:.1f}s)")

    return True


def test_structure_b():
    """Test Structure B: Ring-Disk Fano MIM"""
    print("\n" + "="*60)
    print("Structure B: Ring-Disk Fano Resonance MIM")
    print("="*60)
    from src.simulation.rcwa_struct_b import simulate_single

    test_params = [
        {"P": 500, "R_out": 200, "R_in": 120, "R_disk": 50, "t_Cr": 40,
         "d_SiO2": 100, "theta": 0, "phi": 0},
        {"P": 600, "R_out": 250, "R_in": 150, "R_disk": 60, "t_Cr": 50,
         "d_SiO2": 150, "theta": 20, "phi": 15},
        {"P": 400, "R_out": 160, "R_in": 100, "R_disk": 40, "t_Cr": 30,
         "d_SiO2": 80, "theta": 40, "phi": 30},
    ]

    for i, p in enumerate(test_params[:n_test]):
        t0 = time.time()
        A, R, T = simulate_single(p, wavelengths, metal="Cr", device=device)
        dt = time.time() - t0

        energy = A + R + T
        energy_err = np.max(np.abs(energy - 1.0))
        a_range = (np.min(A), np.max(A))
        r_range = (np.min(R), np.max(R))

        status = "PASS" if energy_err < 1e-6 and np.all(A >= -0.01) and np.all(R >= -0.01) else "FAIL"
        print(f"  Sample {i}: A=[{a_range[0]:.4f},{a_range[1]:.4f}] "
              f"R=[{r_range[0]:.4f},{r_range[1]:.4f}] "
              f"|A+R+T-1|_max={energy_err:.2e} [{status}] ({dt:.1f}s)")

    return True


def test_structure_c():
    """Test Structure C: Dual-Polarization MIM"""
    print("\n" + "="*60)
    print("Structure C: Dual-Polarization Rectangular MIM")
    print("="*60)
    from src.simulation.rcwa_struct_c import simulate_single

    test_params = [
        {"P": 500, "Wx": 200, "Wy": 300, "t_Cr": 40, "d_SiO2": 100,
         "theta": 0, "phi": 0},
        {"P": 600, "Wx": 350, "Wy": 200, "t_Cr": 50, "d_SiO2": 150,
         "theta": 20, "phi": 15},
        {"P": 400, "Wx": 180, "Wy": 250, "t_Cr": 30, "d_SiO2": 80,
         "theta": 40, "phi": 30},
    ]

    for i, p in enumerate(test_params[:n_test]):
        t0 = time.time()
        A_te, R_te, T_te, A_tm, R_tm, T_tm = simulate_single(
            p, wavelengths, metal="Cr", device=device)
        dt = time.time() - t0

        energy_te = A_te + R_te + T_te
        energy_tm = A_tm + R_tm + T_tm
        err_te = np.max(np.abs(energy_te - 1.0))
        err_tm = np.max(np.abs(energy_tm - 1.0))

        # Polarization difference (should be nonzero for Wx != Wy)
        pol_diff = np.mean(np.abs(A_te - A_tm))

        status = "PASS" if err_te < 1e-6 and err_tm < 1e-6 else "FAIL"
        print(f"  Sample {i}: A_TE=[{np.min(A_te):.4f},{np.max(A_te):.4f}] "
              f"A_TM=[{np.min(A_tm):.4f},{np.max(A_tm):.4f}] "
              f"pol_diff={pol_diff:.4f} "
              f"|err|_max=({err_te:.2e},{err_tm:.2e}) [{status}] ({dt:.1f}s)")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Quick Test: 3 Structures × 3 Samples (TORCWA GPU)")
    print("=" * 60)

    try:
        test_structure_a()
    except Exception as e:
        print(f"  Structure A FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        test_structure_b()
    except Exception as e:
        print(f"  Structure B FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        test_structure_c()
    except Exception as e:
        print(f"  Structure C FAILED: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("Quick Test Complete")
    print("=" * 60)
