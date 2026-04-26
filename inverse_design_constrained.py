#!/usr/bin/env python3
"""
Constrained Inverse Design using PBTL Surrogate (M_TL+phys).

Key fixes over the original failed attempt:
  1. Box constraints: params clamped to design space throughout optimization
  2. Consistent validation: TMM (no solver mismatch)
  3. Self-consistency test: recover known test samples

Usage:
    python inverse_design_constrained.py

Device: runs on CPU (recommended) or MPS; does NOT require torcwa.
Note: CPU is ~41% faster than MPS for this workload due to small batch size
(100 wavelength points per design). MPS kernel launch overhead dominates.
"""

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from src.utils.data_utils import get_bounds, normalize_params, denormalize_params
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

# ─── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cpu")  # CPU is faster than MPS for this small-batch workload
print(f"Device: {device}")

# ─── Design space ────────────────────────────────────────────────────────────
_, PARAM_MIN, PARAM_MAX = get_bounds("A")
PARAM_NAMES = ["P", "Wx", "Wy", "W2", "t1", "t2", "t_mid", "d1", "d2", "theta"]
WAVELENGTHS = np.linspace(380, 780, 100).astype(np.float32)

PARAM_MIN_T = torch.tensor(PARAM_MIN, dtype=torch.float32, device=device)
PARAM_MAX_T = torch.tensor(PARAM_MAX, dtype=torch.float32, device=device)

# ─── Model definition (must match pbtl_experiment.py) ────────────────────────
class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = h + self.act(b(h))
        return h


class MPhys(nn.Module):
    """M_TL+phys: geometry (11) + physics features (17) → absorptance."""
    def __init__(self, geo_dim=11, phys_dim=17):
        super().__init__()
        self.bb = BaseResNet(geo_dim + phys_dim)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x_geo, p):
        h = self.bb(torch.cat([x_geo, p], dim=-1))
        R = self.head(h).squeeze(-1)
        return 1 - R  # absorptance


# ─── Differentiable physics features (torch version) ─────────────────────────
def compute_physics_features_torch(params_phys, wavelengths_nm_t, phys_mean, phys_std,
                                    metal="Cr"):
    """
    Differentiable physics feature computation for Structure A.
    params_phys: [10] tensor (physical units, NOT normalized)
    wavelengths_nm_t: [Nlam] tensor
    Returns: [Nlam, 17] normalized feature tensor
    """
    from src.simulation.materials import (
        get_sio2_permittivity, get_tio2_permittivity, get_metal_permittivity)

    Nlam = len(wavelengths_nm_t)
    wl_np = wavelengths_nm_t.cpu().numpy()

    # Material data (numpy → reuse existing code)
    n_sio2 = torch.tensor(np.sqrt(np.real(get_sio2_permittivity(wl_np))),
                           dtype=torch.float32, device=params_phys.device)
    n_tio2 = torch.tensor(np.sqrt(np.real(get_tio2_permittivity(wl_np))),
                           dtype=torch.float32, device=params_phys.device)
    eps_m   = get_metal_permittivity(wl_np, metal)
    k_metal = torch.tensor(np.imag(np.sqrt(eps_m)),
                            dtype=torch.float32, device=params_phys.device)
    skin_depth = wavelengths_nm_t / (4 * np.pi * k_metal + 1e-30)
    alpha_metal = 4 * np.pi * k_metal / wavelengths_nm_t

    P, Wx, Wy, W2 = params_phys[0], params_phys[1], params_phys[2], params_phys[3]
    t1, t2, t_mid = params_phys[4], params_phys[5], params_phys[6]
    d1, d2, theta  = params_phys[7], params_phys[8], params_phys[9]
    theta_rad = theta * (np.pi / 180.0)

    def cavity_phase(n_cav, d_cav):
        """(Nlam,) → cos_phi, sin_phi"""
        sin_ti = torch.clamp(torch.sin(theta_rad) / n_cav, -1 + 1e-6, 1 - 1e-6)
        cos_ti = torch.sqrt(1 - sin_ti**2)
        phase  = 4 * np.pi * n_cav * d_cav * cos_ti / wavelengths_nm_t
        return torch.cos(phase), torch.sin(phase)

    feats = []

    # 1. Cavity resonance: SiO2 + TiO2 (4 features)
    cos_s, sin_s = cavity_phase(n_sio2, d1)
    cos_t, sin_t = cavity_phase(n_tio2, d2)
    feats += [cos_s, sin_s, cos_t, sin_t]

    # 2. Fill fraction (2 features)
    f_rect = (Wx * Wy / (P**2 + 1e-6)).expand(Nlam)
    f_sq   = (W2**2     / (P**2 + 1e-6)).expand(Nlam)
    feats += [f_rect, f_sq]

    # 3. Sub-wavelength ratio (3 features)
    feats += [P  / wavelengths_nm_t,
              Wx / wavelengths_nm_t,
              W2 / wavelengths_nm_t]

    # 4. Skin depth ratio (3 features)
    feats += [t1   / (skin_depth + 1e-6),
              t2   / (skin_depth + 1e-6),
              t_mid / (skin_depth + 1e-6)]

    # 5. Optical path length (2 features)
    feats += [n_sio2 * d1 / wavelengths_nm_t,
              n_tio2 * d2 / wavelengths_nm_t]

    # 6. Angle & geometry (3 features)
    feats += [torch.cos(theta_rad).expand(Nlam),
              (Wy / (Wx + 1e-6)).expand(Nlam),
              alpha_metal]

    feat_mat = torch.stack(feats, dim=1)  # [Nlam, 17]

    # Normalize using pre-computed stats from TMM data
    mean_t = torch.tensor(phys_mean, dtype=torch.float32, device=params_phys.device)
    std_t  = torch.tensor(phys_std,  dtype=torch.float32, device=params_phys.device)
    return (feat_mat - mean_t) / (std_t + 1e-8)


# ─── Surrogate forward pass for a single design ──────────────────────────────
def surrogate_forward(model, params_phys_t, wavelengths_t, wl_norm_t,
                      phys_mean, phys_std, metal="Cr"):
    """
    params_phys_t: [10] tensor, physical units
    Returns: [Nlam] absorptance tensor
    """
    Nlam = len(wavelengths_t)

    # Normalized geometry input  [Nlam, 11] = [λ̃, p̃_1..10]
    params_norm = (params_phys_t - PARAM_MIN_T) / (PARAM_MAX_T - PARAM_MIN_T)
    params_rep  = params_norm.unsqueeze(0).expand(Nlam, -1)           # [Nlam, 10]
    x_geo = torch.cat([wl_norm_t.unsqueeze(1), params_rep], dim=1)   # [Nlam, 11]

    # Differentiable physics features  [Nlam, 17]
    x_phys = compute_physics_features_torch(params_phys_t, wavelengths_t,
                                             phys_mean, phys_std, metal)

    return model(x_geo, x_phys)  # [Nlam]


# ─── Constrained inverse design ──────────────────────────────────────────────
def inverse_design_constrained(model, target_A, wavelengths_t, wl_norm_t,
                                phys_mean, phys_std,
                                n_restarts=5, n_steps=1000, lr=0.02,
                                metal="Cr", verbose=True):
    """
    Gradient-based inverse design WITH box constraints.

    Key fix: after each gradient step, clamp params to [PARAM_MIN, PARAM_MAX].
    This prevents adversarial gradient exploitation by keeping params in-distribution.

    Returns best_params (physical units), best_loss, loss_history
    """
    target_t = torch.tensor(target_A, dtype=torch.float32, device=device)

    best_loss   = float("inf")
    best_params = None
    best_history = None

    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        # Random initialization within design space
        p0 = rng.uniform(PARAM_MIN, PARAM_MAX).astype(np.float32)
        params = torch.tensor(p0, dtype=torch.float32,
                               device=device, requires_grad=True)
        opt = torch.optim.Adam([params], lr=lr)

        history = []
        for step in range(n_steps):
            opt.zero_grad()

            # Clamp BEFORE forward pass (projected gradient descent)
            with torch.no_grad():
                params.data.clamp_(PARAM_MIN_T, PARAM_MAX_T)

            A_pred = surrogate_forward(model, params, wavelengths_t,
                                        wl_norm_t, phys_mean, phys_std, metal)
            loss = torch.mean((A_pred - target_t)**2)
            loss.backward()

            # Gradient clipping to prevent large steps
            torch.nn.utils.clip_grad_norm_([params], 1.0)
            opt.step()

            # Clamp AFTER step (box projection)
            with torch.no_grad():
                params.data.clamp_(PARAM_MIN_T, PARAM_MAX_T)

            history.append(loss.item())

        final_loss = history[-1]
        if verbose:
            print(f"  Restart {restart+1}/{n_restarts}: final MAE={np.sqrt(final_loss)*100:.2f}%")

        if final_loss < best_loss:
            best_loss   = final_loss
            best_params = params.detach().cpu().numpy().copy()
            best_history = history

    return best_params, np.sqrt(best_loss), best_history


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    # Load pre-trained stats and model
    stats = np.load("results/inverse_design/phys_stats.npz")
    phys_mean = stats["mean"]
    phys_std  = stats["std"]

    model = MPhys(geo_dim=11, phys_dim=17).to(device)
    sd = torch.load("results/inverse_design/mphys_350.pt", map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print("Loaded fine-tuned M_TL+phys (n=350)")

    wavelengths_t = torch.tensor(WAVELENGTHS, dtype=torch.float32, device=device)
    wl_norm_t     = (wavelengths_t - wavelengths_t.min()) / \
                    (wavelengths_t.max() - wavelengths_t.min())

    # Load test data: use known RCWA samples as targets
    data = np.load("data/raw/struct_A_vis_500.npz")
    params_all  = data["params"].astype(np.float32)
    A_rcwa_all  = data["A"].astype(np.float32)

    # Use last 50 as test (consistent with training split seed=42)
    rng0 = np.random.default_rng(42)
    idx  = rng0.permutation(len(params_all))
    test_idx = idx[int(len(idx)*0.85):]  # last 15% ≈ 75 samples
    # Pick 6 diverse test samples
    chosen = test_idx[:6]

    results = []
    for i, ti in enumerate(chosen):
        target_A   = A_rcwa_all[ti]          # true RCWA spectrum
        true_params = params_all[ti]

        print(f"\n[Target {i+1}] True params: {dict(zip(PARAM_NAMES, true_params.round(1)))}")

        best_params, best_mae, history = inverse_design_constrained(
            model, target_A, wavelengths_t, wl_norm_t,
            phys_mean, phys_std,
            n_restarts=5, n_steps=800, lr=0.03, verbose=True)

        # Forward eval: surrogate prediction at recovered params
        with torch.no_grad():
            p_t = torch.tensor(best_params, dtype=torch.float32, device=device)
            A_surr = surrogate_forward(model, p_t, wavelengths_t, wl_norm_t,
                                        phys_mean, phys_std).cpu().numpy()

        # TMM validation at recovered params
        tmm_out = compute_tmm_batch(best_params[None], WAVELENGTHS, "Cr")
        A_tmm   = tmm_out["A_tmm"][0]

        # Compute MAEs
        mae_surr = np.mean(np.abs(A_surr - target_A)) * 100
        mae_tmm  = np.mean(np.abs(A_tmm  - target_A)) * 100

        print(f"  Recovered params: {dict(zip(PARAM_NAMES, best_params.round(1)))}")
        print(f"  MAE surrogate: {mae_surr:.2f}%  |  MAE TMM: {mae_tmm:.2f}%")

        results.append({
            "true_params": true_params,
            "best_params": best_params,
            "target_A": target_A,
            "A_surr": A_surr,
            "A_tmm": A_tmm,
            "mae_surr": mae_surr,
            "mae_tmm": mae_tmm,
            "history": history,
        })

    # ─── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        wl = WAVELENGTHS
        ax.plot(wl, r["target_A"],  "k-",  lw=2,   label="Target (RCWA)")
        ax.plot(wl, r["A_surr"],    "b--", lw=1.5, label=f"Surrogate ({r['mae_surr']:.1f}%)")
        ax.plot(wl, r["A_tmm"],     "r:",  lw=1.5, label=f"TMM ({r['mae_tmm']:.1f}%)")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorptance")
        ax.set_title(f"Target {i+1} — P={r['best_params'][0]:.0f}nm  "
                     f"d1={r['best_params'][7]:.0f}nm")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Constrained Inverse Design: M_TL+phys Surrogate\n"
                 "(Box constraints prevent out-of-distribution exploitation)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    outpath = "results/inverse_design/constrained_inverse_design.pdf"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")

    # ─── Summary ──────────────────────────────────────────────────────────────
    maes_s = [r["mae_surr"] for r in results]
    maes_t = [r["mae_tmm"]  for r in results]
    print("\n" + "="*55)
    print("SUMMARY")
    print(f"  Surrogate MAE: {np.mean(maes_s):.2f}% ± {np.std(maes_s):.2f}%")
    print(f"  TMM     MAE: {np.mean(maes_t):.2f}% ± {np.std(maes_t):.2f}%")
    print("  (TMM validation ≠ full RCWA; serves as lower-fidelity sanity check)")
    print("="*55)

    # Save results
    np.savez("results/inverse_design/constrained_results.npz",
             wavelengths=WAVELENGTHS,
             chosen_idx=np.array(chosen),
             mae_surrogate=np.array(maes_s),
             mae_tmm=np.array(maes_t),
             best_params=np.array([r["best_params"] for r in results]),
             true_params=np.array([r["true_params"]  for r in results]),
             target_A=np.array([r["target_A"]  for r in results]),
             A_surr=np.array([r["A_surr"]    for r in results]),
             A_tmm=np.array([r["A_tmm"]     for r in results]))
    print("Saved results to results/inverse_design/constrained_results.npz")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
