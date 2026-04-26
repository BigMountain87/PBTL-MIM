#!/usr/bin/env python3
"""
Tandem Network for Inverse Design of Structure A.

Architecture:
  InverseNet(spectrum → params_norm) → FrozenForwardSurrogate(params → spectrum)

Training:
  Phase 1 (supervised) : loss = MSE(params_pred, params_true)
  Phase 2 (tandem)     : loss = MSE(forward(params_pred), spectrum_target)
                         (forward surrogate frozen)

Usage:
    python tandem_inverse_design.py
"""

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.utils.data_utils import get_bounds, normalize_params, denormalize_params
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("cpu")
WAVELENGTHS = np.linspace(380, 780, 100).astype(np.float32)
PARAM_NAMES = ["P", "Wx", "Wy", "W2", "t1", "t2", "t_mid", "d1", "d2", "theta"]

_, PARAM_MIN, PARAM_MAX = get_bounds("A")
PARAM_MIN_T = torch.tensor(PARAM_MIN, dtype=torch.float32)
PARAM_MAX_T = torch.tensor(PARAM_MAX, dtype=torch.float32)

# ─── Forward surrogate (exact match to inverse_design_constrained.py) ────────
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
    def __init__(self, geo_dim=11, phys_dim=17):
        super().__init__()
        self.bb = BaseResNet(geo_dim + phys_dim)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x_geo, p):
        h = self.bb(torch.cat([x_geo, p], dim=-1))
        R = self.head(h).squeeze(-1)
        return 1 - R  # absorptance


def surrogate_forward(model, params_phys_t, wavelengths_t, wl_norm_t, phys_mean, phys_std):
    """params_phys_t: (10,) tensor in physical units → (n_wl,) absorptance"""
    from src.simulation.materials import (
        get_sio2_permittivity, get_tio2_permittivity, get_metal_permittivity)

    Nlam = len(wavelengths_t)
    params_norm = (params_phys_t - PARAM_MIN_T) / (PARAM_MAX_T - PARAM_MIN_T)
    params_rep  = params_norm.unsqueeze(0).expand(Nlam, -1)
    x_geo = torch.cat([wl_norm_t.unsqueeze(1), params_rep], dim=1)  # (Nlam, 11)

    # Physics features (numpy-based, then normalize)
    wl_np = wavelengths_t.detach().cpu().numpy()
    p_np  = params_phys_t.detach().cpu().numpy()
    ph_np = compute_physics_features_A(p_np[None], wl_np)[0].astype(np.float32)
    ph_t  = torch.tensor(ph_np, dtype=torch.float32, device=device)
    phys_mean_t = torch.tensor(phys_mean, dtype=torch.float32, device=device)
    phys_std_t  = torch.tensor(phys_std,  dtype=torch.float32, device=device)
    x_phys = (ph_t - phys_mean_t) / (phys_std_t + 1e-8)  # (Nlam, 17)

    return model(x_geo, x_phys)  # (Nlam,)


def surrogate_forward_batch(model, params_batch, wavelengths_t, wl_norm_t, phys_mean, phys_std):
    """params_batch: (B, 10) physical units → (B, n_wl) absorptance"""
    return torch.stack([
        surrogate_forward(model, params_batch[i], wavelengths_t, wl_norm_t,
                          phys_mean, phys_std)
        for i in range(params_batch.shape[0])
    ], dim=0)


# ─── Inverse Network ──────────────────────────────────────────────────────────
class InverseNet(nn.Module):
    """Spectrum (100,) → normalized params (10,) in [0, 1]"""
    def __init__(self, spec_dim=100, param_dim=10, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(spec_dim, hidden)
        self.act   = nn.SiLU()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden, param_dim)

    def forward(self, spec):
        h = self.act(self.fc_in(spec))
        for b in self.blocks:
            h = h + self.act(b(h))
        return torch.sigmoid(self.fc_out(h))  # (B, 10) in [0,1]


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    data = np.load("data/raw/struct_A_vis_500.npz")
    params_all = data["params"].astype(np.float32)
    A_all      = data["A"].astype(np.float32)

    # Normalize params to [0, 1]
    params_norm = (params_all - PARAM_MIN) / (PARAM_MAX - PARAM_MIN + 1e-8)

    # Same split as training: seed=42, 85% train
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(params_all))
    n_train = int(len(idx) * 0.85)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    X_tr = torch.tensor(A_all[train_idx],      dtype=torch.float32)
    y_tr = torch.tensor(params_norm[train_idx], dtype=torch.float32)
    X_te = torch.tensor(A_all[test_idx],        dtype=torch.float32)
    y_te = torch.tensor(params_norm[test_idx],  dtype=torch.float32)

    return X_tr, y_tr, X_te, y_te, params_all, A_all, test_idx


# ─── Training phases ──────────────────────────────────────────────────────────
def phase1_supervised(inv_net, X_tr, y_tr, epochs=500, batch=64, lr=1e-3):
    """Supervised: spectrum → params (MSE on normalized params)"""
    opt = torch.optim.Adam(inv_net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X_tr.shape[0]
    losses = []

    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            loss = F.mse_loss(inv_net(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        losses.append(ep_loss / n_batches)
        if (ep + 1) % 100 == 0:
            print(f"  [Phase1] ep {ep+1}/{epochs}  loss={losses[-1]:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    return losses


def phase2_tandem(inv_net, fwd_model, X_tr, wavelengths_t, wl_norm_t,
                  phys_mean, phys_std, epochs=300, batch=16, lr=3e-4):
    """Tandem: freeze forward, train inverse with spectrum reconstruction loss"""
    for p in fwd_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(inv_net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X_tr.shape[0]
    losses = []

    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0; n_batches = 0

        for i in range(0, n, batch):
            idx_b = perm[i:i+batch]
            spec_target = X_tr[idx_b]          # (B, 100)
            params_norm = inv_net(spec_target)  # (B, 10) in [0,1]

            # Denormalize to physical units
            params_raw = params_norm * (PARAM_MAX_T - PARAM_MIN_T) + PARAM_MIN_T
            params_raw = params_raw.clamp(PARAM_MIN_T, PARAM_MAX_T)

            # Forward through surrogate (sample-by-sample, differentiable)
            A_pred = surrogate_forward_batch(fwd_model, params_raw,
                                             wavelengths_t, wl_norm_t,
                                             phys_mean, phys_std)  # (B, 100)

            loss = F.mse_loss(A_pred, spec_target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1

        sched.step()
        losses.append(ep_loss / n_batches)
        if (ep + 1) % 50 == 0:
            print(f"  [Phase2] ep {ep+1}/{epochs}  loss={losses[-1]:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    return losses


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(inv_net, fwd_model, X_te, y_te, wavelengths_t, wl_norm_t,
             phys_mean, phys_std, params_all, A_all, test_idx):
    inv_net.eval()
    maes_surr = []
    maes_tmm  = []

    with torch.no_grad():
        params_norm_pred = inv_net(X_te)  # (N_te, 10)
        params_raw_pred  = (params_norm_pred * (PARAM_MAX_T - PARAM_MIN_T)
                            + PARAM_MIN_T).clamp(PARAM_MIN_T, PARAM_MAX_T)

    for i in range(X_te.shape[0]):
        target_A = X_te[i].numpy()
        pvec     = params_raw_pred[i].numpy()

        # Surrogate MAE
        with torch.no_grad():
            A_surr = surrogate_forward(fwd_model, params_raw_pred[i],
                                       wavelengths_t, wl_norm_t,
                                       phys_mean, phys_std).numpy()
        maes_surr.append(np.mean(np.abs(A_surr - target_A)) * 100)

        # TMM MAE
        tmm_out = compute_tmm_batch(pvec[None], WAVELENGTHS, "Cr")
        A_tmm   = tmm_out["A_tmm"][0]
        maes_tmm.append(np.mean(np.abs(A_tmm - target_A)) * 100)

    return np.array(maes_surr), np.array(maes_tmm), params_raw_pred.numpy()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results/inverse_design", exist_ok=True)
    wavelengths_t = torch.tensor(WAVELENGTHS, dtype=torch.float32)
    wl_norm_t     = (wavelengths_t - wavelengths_t.min()) / \
                    (wavelengths_t.max() - wavelengths_t.min())

    # Load forward surrogate
    stats     = np.load("results/inverse_design/phys_stats.npz")
    phys_mean = stats["mean"]
    phys_std  = stats["std"]

    fwd_model = MPhys(geo_dim=11, phys_dim=17).to(device)
    sd = torch.load("results/inverse_design/mphys_350.pt", map_location=device)
    fwd_model.load_state_dict(sd)
    fwd_model.eval()
    print("Loaded forward surrogate M_TL+phys (n=350)")

    # Load data
    X_tr, y_tr, X_te, y_te, params_all, A_all, test_idx = load_data()
    print(f"Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

    # Build inverse network
    inv_net = InverseNet(spec_dim=100, param_dim=10, hidden=256, n_blocks=4)
    n_params = sum(p.numel() for p in inv_net.parameters())
    print(f"InverseNet: {n_params:,} parameters")
    print()

    # ── Phase 1: Supervised ───────────────────────────────────────────────────
    print("=" * 55)
    print("PHASE 1: Supervised (spectrum → params)")
    print("=" * 55)
    t0 = time.time()
    loss1 = phase1_supervised(inv_net, X_tr, y_tr, epochs=500, batch=64, lr=1e-3)
    print(f"Phase 1 done: {time.time()-t0:.1f}s, final loss={loss1[-1]:.5f}")

    # ── Phase 2: Tandem ───────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("PHASE 2: Tandem fine-tuning (frozen forward surrogate)")
    print("=" * 55)
    t0 = time.time()
    loss2 = phase2_tandem(inv_net, fwd_model, X_tr, wavelengths_t, wl_norm_t,
                          phys_mean, phys_std, epochs=300, batch=16, lr=3e-4)
    print(f"Phase 2 done: {time.time()-t0:.1f}s, final loss={loss2[-1]:.5f}")

    # ── Evaluation on test set ────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("EVALUATION on test set")
    print("=" * 55)
    maes_surr, maes_tmm, recovered = evaluate(
        inv_net, fwd_model, X_te, y_te, wavelengths_t, wl_norm_t,
        phys_mean, phys_std, params_all, A_all, test_idx)

    print(f"  Surrogate MAE: {maes_surr.mean():.2f}% ± {maes_surr.std():.2f}%")
    print(f"  TMM      MAE: {maes_tmm.mean():.2f}% ± {maes_tmm.std():.2f}%")

    # ── RCWA validation on the same 6 targets ────────────────────────────────
    print()
    print("=" * 55)
    print("RCWA VALIDATION: same 6 targets as gradient-based")
    print("=" * 55)

    # Load previous constrained results for comparison
    prev = np.load("results/inverse_design/constrained_rcwa_validation.npz")
    target_A_6  = prev["target_A"]   # (6, 100)
    mae_grad    = prev["mae_rcwa"]   # (6,)

    try:
        import grcwa
        from validate.rcwa_validate_constrained_grcwa import simulate_structA_grcwa
        can_rcwa = True
    except ImportError:
        can_rcwa = False
        print("  (grcwa not available; TMM used as proxy)")

    # Get same 6 test targets
    rng0 = np.random.default_rng(42)
    idx_all = rng0.permutation(len(params_all))
    test_idx6 = idx_all[int(len(idx_all) * 0.85):][:6]

    inv_net.eval()
    mae_tandem_surr = np.zeros(6)
    mae_tandem_rcwa = np.zeros(6)
    recovered_6     = np.zeros((6, 10))

    for i, ti in enumerate(test_idx6):
        spec_t = torch.tensor(A_all[ti], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_norm = inv_net(spec_t)[0]
            p_raw  = (p_norm * (PARAM_MAX_T - PARAM_MIN_T) + PARAM_MIN_T
                      ).clamp(PARAM_MIN_T, PARAM_MAX_T)
        recovered_6[i] = p_raw.numpy()

        # Surrogate MAE
        with torch.no_grad():
            A_surr = surrogate_forward(fwd_model, p_raw, wavelengths_t, wl_norm_t,
                                       phys_mean, phys_std).numpy()
        mae_tandem_surr[i] = np.mean(np.abs(A_surr - target_A_6[i])) * 100

        # RCWA or TMM MAE
        if can_rcwa:
            A_r, _, _ = simulate_structA_grcwa(recovered_6[i], WAVELENGTHS)
            mae_tandem_rcwa[i] = np.mean(np.abs(A_r - target_A_6[i])) * 100
        else:
            tmm_out = compute_tmm_batch(recovered_6[i][None], WAVELENGTHS, "Cr")
            A_r = tmm_out["A_tmm"][0]
            mae_tandem_rcwa[i] = np.mean(np.abs(A_r - target_A_6[i])) * 100

        print(f"  Target {i+1}: surrogate={mae_tandem_surr[i]:.2f}%  "
              f"{'RCWA' if can_rcwa else 'TMM'}={mae_tandem_rcwa[i]:.2f}%  "
              f"(grad-RCWA={mae_grad[i]:.2f}%)")

    label = "RCWA" if can_rcwa else "TMM"
    print()
    print(f"  Tandem  {label} avg : {mae_tandem_rcwa.mean():.2f}% ± {mae_tandem_rcwa.std():.2f}%")
    print(f"  Gradient {label} avg: {mae_grad.mean():.2f}% ± {mae_grad.std():.2f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    wl = WAVELENGTHS

    for i, ti in enumerate(test_idx6):
        ax = axes[i]
        ax.plot(wl, target_A_6[i], "k-", lw=2.0, label="Target (RCWA gt)")

        # Tandem prediction
        spec_t = torch.tensor(A_all[ti], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_norm = inv_net(spec_t)[0]
            p_raw  = (p_norm * (PARAM_MAX_T - PARAM_MIN_T) + PARAM_MIN_T
                      ).clamp(PARAM_MIN_T, PARAM_MAX_T)
            A_surr = surrogate_forward(fwd_model, p_raw, wavelengths_t, wl_norm_t,
                                       phys_mean, phys_std).numpy()
        ax.plot(wl, A_surr, "b-", lw=1.5,
                label=f"Tandem surrogate ({mae_tandem_surr[i]:.1f}%)")

        # Previous gradient-based (RCWA)
        ax.plot(wl, prev["A_rcwa_recovered"][i], "r--", lw=1.5,
                label=f"Grad-based RCWA ({mae_grad[i]:.1f}%)")

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorptance")
        ax.set_title(f"Target {i+1}  —  P={recovered_6[i][0]:.0f} nm")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Tandem Network vs. Gradient-Based Inverse Design\n"
        "Blue: Tandem surrogate prediction  |  Red dashed: Gradient-based RCWA",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_pdf = "results/inverse_design/tandem_inverse_design.pdf"
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_pdf}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(inv_net.state_dict(), "results/inverse_design/tandem_invnet.pt")
    np.savez("results/inverse_design/tandem_results.npz",
             wavelengths=WAVELENGTHS,
             mae_surr_test=maes_surr,
             mae_tmm_test=maes_tmm,
             mae_tandem_surr_6=mae_tandem_surr,
             mae_tandem_rcwa_6=mae_tandem_rcwa,
             mae_grad_rcwa_6=mae_grad,
             recovered_6=recovered_6,
             loss_phase1=np.array(loss1),
             loss_phase2=np.array(loss2))
    print("Data saved: results/inverse_design/tandem_results.npz")
    print("Model saved: results/inverse_design/tandem_invnet.pt")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
