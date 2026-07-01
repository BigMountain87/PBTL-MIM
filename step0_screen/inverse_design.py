#!/usr/bin/env python3
"""
Inverse Design: M0 vs M_phys comparison.
1. Train both models on 350 samples
2. Select 20 target spectra from test set
3. Gradient-based inverse design (sigmoid reparameterization, 10 restarts)
4. Validate designed structures with RCWA
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, denormalize_params, get_bounds
from src.simulation.materials import (get_sio2_permittivity, get_tio2_permittivity,
                                       get_metal_permittivity)
from src.simulation.rcwa_struct_a import simulate_single, PARAM_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1. Model definitions (same as data_efficiency_500.py)
# ============================================================
def compute_physics_features(params, wavelengths_nm, metal="Cr"):
    N = len(params)
    Nlam = len(wavelengths_nm)
    P, Wx, Wy, W2 = params[:,0], params[:,1], params[:,2], params[:,3]
    t1, t2, t_mid = params[:,4], params[:,5], params[:,6]
    d1, d2, theta = params[:,7], params[:,8], params[:,9]
    theta_rad = np.deg2rad(theta)

    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_tio2 = get_tio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    n_tio2 = np.sqrt(np.real(eps_tio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4 * np.pi * k_metal)

    feats = []
    for n_cav, d_cav in [(n_sio2, d1), (n_tio2, d2)]:
        sin_ti = np.clip(np.sin(theta_rad[:,None]) / n_cav[None,:], -1, 1)
        cos_ti = np.sqrt(1 - sin_ti**2)
        phase = 4*np.pi * n_cav[None,:] * d_cav[:,None] * cos_ti / wavelengths_nm[None,:]
        feats.append(np.cos(phase))
        feats.append(np.sin(phase))

    feats.append(np.tile((Wx*Wy/P**2)[:,None], (1,Nlam)))
    feats.append(np.tile((W2**2/P**2)[:,None], (1,Nlam)))
    feats.append(P[:,None] / wavelengths_nm[None,:])
    feats.append(Wx[:,None] / wavelengths_nm[None,:])
    feats.append(W2[:,None] / wavelengths_nm[None,:])
    feats.append(t1[:,None] / skin_depth[None,:])
    feats.append(t2[:,None] / skin_depth[None,:])
    feats.append(t_mid[:,None] / skin_depth[None,:])
    feats.append(np.tile(np.cos(theta_rad[:,None]), (1,Nlam)))
    feats.append(n_sio2[None,:] * d1[:,None] / wavelengths_nm[None,:])
    feats.append(n_tio2[None,:] * d2[:,None] / wavelengths_nm[None,:])
    feats.append(np.tile((Wy/(Wx+1e-10))[:,None], (1,Nlam)))
    alpha = 4*np.pi*k_metal / wavelengths_nm
    feats.append(np.tile(alpha[None,:], (N,1)))

    return np.stack(feats, axis=-1).astype(np.float32)


# Physics features computation in PyTorch (differentiable for inverse design)
def compute_physics_features_torch(params_phys, wavelengths_nm, phys_mean, phys_std, metal="Cr"):
    """
    Compute physics features in PyTorch for backprop through inverse design.
    params_phys: [10] physical params tensor (P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta)
    wavelengths_nm: [Nlam] numpy array
    Returns: [Nlam, 17] normalized physics features
    """
    Nlam = len(wavelengths_nm)
    wl_t = torch.tensor(wavelengths_nm, dtype=torch.float32, device=params_phys.device)

    P, Wx, Wy, W2 = params_phys[0], params_phys[1], params_phys[2], params_phys[3]
    t1, t2, t_mid = params_phys[4], params_phys[5], params_phys[6]
    d1, d2, theta_deg = params_phys[7], params_phys[8], params_phys[9]
    theta_rad = theta_deg * (3.14159265 / 180.0)

    # Material properties (non-differentiable constants)
    eps_sio2 = torch.tensor(np.real(get_sio2_permittivity(wavelengths_nm)), dtype=torch.float32, device=params_phys.device)
    eps_tio2 = torch.tensor(np.real(get_tio2_permittivity(wavelengths_nm)), dtype=torch.float32, device=params_phys.device)
    eps_metal_np = get_metal_permittivity(wavelengths_nm, metal)
    k_metal = torch.tensor(np.imag(np.sqrt(eps_metal_np)), dtype=torch.float32, device=params_phys.device)

    n_sio2 = torch.sqrt(eps_sio2)
    n_tio2 = torch.sqrt(eps_tio2)
    skin_depth = wl_t / (4 * 3.14159265 * k_metal + 1e-10)

    feats = []
    for n_cav, d_cav in [(n_sio2, d1), (n_tio2, d2)]:
        sin_ti = torch.clamp(torch.sin(theta_rad) / n_cav, -1, 1)
        cos_ti = torch.sqrt(1 - sin_ti**2)
        phase = 4 * 3.14159265 * n_cav * d_cav * cos_ti / wl_t
        feats.append(torch.cos(phase))
        feats.append(torch.sin(phase))

    feats.append(torch.full((Nlam,), (Wx*Wy/(P**2)).item(), device=params_phys.device))
    feats.append(torch.full((Nlam,), (W2**2/(P**2)).item(), device=params_phys.device))
    feats.append(P / wl_t)
    feats.append(Wx / wl_t)
    feats.append(W2 / wl_t)
    feats.append(t1 / skin_depth)
    feats.append(t2 / skin_depth)
    feats.append(t_mid / skin_depth)
    feats.append(torch.full((Nlam,), torch.cos(theta_rad).item(), device=params_phys.device))
    feats.append(n_sio2 * d1 / wl_t)
    feats.append(n_tio2 * d2 / wl_t)
    feats.append(torch.full((Nlam,), (Wy/(Wx+1e-10)).item(), device=params_phys.device))
    alpha = 4 * 3.14159265 * k_metal / wl_t
    feats.append(alpha)

    X_phys = torch.stack(feats, dim=-1)  # [Nlam, 17]

    # Normalize
    pm = torch.tensor(phys_mean, dtype=torch.float32, device=params_phys.device)
    ps = torch.tensor(phys_std, dtype=torch.float32, device=params_phys.device)
    return (X_phys - pm) / ps


class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden,hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden,hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks: h = h + self.act(b(h))
        return h

class M0(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1); return {"A": 1-R, "R": R}

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1)); R = self.head(h).squeeze(-1); return {"A": 1-R, "R": R}


def train_model(model, dl_tr, dl_vl, epochs=5000, lr=1e-3, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None

    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if has_phys: x,a,r,p = batch; out = model(x, p=p)
            else: x,a,r = batch; out = model(x)
            loss = crit(out["A"],a) + crit(out["R"],r)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()

        if (ep+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                for batch in dl_vl:
                    if has_phys: x,a,r,p = batch; out = model(x, p=p)
                    else: x,a,r = batch; out = model(x)
                    vl += nn.functional.l1_loss(out["A"],a,reduction="sum").item()
                    vn += len(a)
                vm = vl/vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k:v.clone() for k,v in model.state_dict().items()}
                if (ep+1) % 1000 == 0:
                    print(f"    Epoch {ep+1}: val_MAE={vm*100:.2f}%, best={best_vl*100:.2f}%")

    if best_st: model.load_state_dict(best_st)
    return model


# ============================================================
# 2. Data loading
# ============================================================
datapath = "data/raw/struct_A_vis_500.npz"
print(f"Loading: {datapath}")
data = np.load(datapath, allow_pickle=True)
params_all = data["params"].astype(np.float32)
A_all = data["A"].astype(np.float32)
R_all = data["R"].astype(np.float32)
wavelengths = data["wavelengths"].astype(np.float32)
Nlam = len(wavelengths)

good = np.all((R_all>=0)&(R_all<=1)&(A_all>=0)&(A_all<=1), axis=1)
gi = np.where(good)[0]
N = len(gi)
params = params_all[gi]; A_arr = A_all[gi]; R_arr = R_all[gi]
print(f"Data: {N} good samples, {Nlam} wavelengths")

# Physics features
phys = compute_physics_features(params, wavelengths, "Cr")
n_phys = phys.shape[-1]

# Normalize
params_norm = normalize_params(params, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + params.shape[1]

# Flatten
params_rep = np.repeat(params_norm[:,None,:], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None,:,None], (N,1,1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_flat = phys.reshape(-1, n_phys).astype(np.float32)
phys_mean = X_phys_flat.mean(0, keepdims=True)
phys_std = X_phys_flat.std(0, keepdims=True) + 1e-8
X_phys_n = ((X_phys_flat - phys_mean) / phys_std).astype(np.float32)
A_flat = A_arr.reshape(-1).astype(np.float32)
R_flat = R_arr.reshape(-1).astype(np.float32)

def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])

# Split: test=50, val=50, train=350
N_TEST, N_VAL, N_TRAIN = 50, 50, 350
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N)
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST+N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST+N_VAL)]

rng2 = np.random.default_rng(42)
perm = rng2.permutation(len(remaining))
tr_idx = remaining[perm[:N_TRAIN]]

print(f"Train: {N_TRAIN}, Val: {N_VAL}, Test: {N_TEST}")

# Build datasets
def make_tensors(idx):
    rows = get_rows(idx)
    return (torch.tensor(X_geo[rows]).to(device),
            torch.tensor(A_flat[rows]).to(device),
            torch.tensor(R_flat[rows]).to(device),
            torch.tensor(X_phys_n[rows]).to(device))

X_tr, A_tr, R_tr, P_tr = make_tensors(tr_idx)
X_vl, A_vl, R_vl, P_vl = make_tensors(val_idx)
X_te, A_te, R_te, P_te = make_tensors(test_idx)

dl_tr_m0 = DataLoader(TensorDataset(X_tr,A_tr,R_tr), batch_size=512, shuffle=True)
dl_tr_ph = DataLoader(TensorDataset(X_tr,A_tr,R_tr,P_tr), batch_size=512, shuffle=True)
dl_vl_m0 = DataLoader(TensorDataset(X_vl,A_vl,R_vl), batch_size=2048)
dl_vl_ph = DataLoader(TensorDataset(X_vl,A_vl,R_vl,P_vl), batch_size=2048)

# ============================================================
# 3. Train models
# ============================================================
print("\n" + "="*60)
print("PHASE 1: Training M0 and M_phys (n_train=350)")
print("="*60)

SEED = 42
set_seed(SEED)
print("\nTraining M0...")
m0 = M0(geo_dim).to(device)
m0 = train_model(m0, dl_tr_m0, dl_vl_m0, epochs=5000, has_phys=False)

set_seed(SEED)
print("\nTraining M_phys...")
mp = MPhys(geo_dim, n_phys).to(device)
mp = train_model(mp, dl_tr_ph, dl_vl_ph, epochs=5000, has_phys=True)

# Test MAE
m0.eval(); mp.eval()
with torch.no_grad():
    out_m0 = m0(X_te); out_mp = mp(X_te, p=P_te)
    mae_m0 = nn.functional.l1_loss(out_m0["A"], A_te).item()
    mae_mp = nn.functional.l1_loss(out_mp["A"], A_te).item()
print(f"\nTest MAE: M0={mae_m0*100:.2f}%, M_phys={mae_mp*100:.2f}%")

# Save checkpoints
ckpt_dir = "results/inverse_design"
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(m0.state_dict(), f"{ckpt_dir}/m0_350.pt")
torch.save(mp.state_dict(), f"{ckpt_dir}/mphys_350.pt")
np.savez(f"{ckpt_dir}/phys_stats.npz", mean=phys_mean.squeeze(), std=phys_std.squeeze())
print(f"Checkpoints saved to {ckpt_dir}")

# ============================================================
# 4. Inverse design
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Inverse Design (gradient-based optimization)")
print("="*60)

_, PMIN, PMAX = get_bounds("A")
PMIN_T = torch.tensor(PMIN, dtype=torch.float32, device=device)
PMAX_T = torch.tensor(PMAX, dtype=torch.float32, device=device)
wl_norm_t = torch.tensor(wl_norm, dtype=torch.float32, device=device)

N_TARGETS = 20
N_RESTARTS = 10
N_STEPS = 500
LR = 0.05

# Select 20 targets from test set (diverse spectra)
test_params = params[test_idx]
test_A = A_arr[test_idx]  # [N_TEST, Nlam]

# Pick targets: sort by average absorption, pick evenly spaced
avg_abs = test_A.mean(axis=1)
sorted_idx = np.argsort(avg_abs)
target_indices = sorted_idx[np.linspace(0, len(sorted_idx)-1, N_TARGETS, dtype=int)]
print(f"Selected {N_TARGETS} targets (avg absorption range: {avg_abs[target_indices].min():.3f} - {avg_abs[target_indices].max():.3f})")


def surrogate_predict_m0(params_norm_t, wl_norm_t):
    """Forward pass through M0: params_norm [10] -> A [Nlam]"""
    Nlam = len(wl_norm_t)
    x_geo = torch.cat([wl_norm_t.unsqueeze(1), params_norm_t.unsqueeze(0).expand(Nlam,-1)], dim=1)
    out = m0(x_geo)
    return out["A"]


def surrogate_predict_mphys(params_phys_t, params_norm_t, wl_norm_t, phys_m, phys_s):
    """Forward pass through M_phys: params [10] -> A [Nlam]"""
    Nlam = len(wl_norm_t)
    x_geo = torch.cat([wl_norm_t.unsqueeze(1), params_norm_t.unsqueeze(0).expand(Nlam,-1)], dim=1)
    x_phys = compute_physics_features_torch(params_phys_t, wavelengths, phys_m, phys_s)
    out = mp(x_geo, p=x_phys)
    return out["A"]


def inverse_design_m0(target_A_t, n_restarts=N_RESTARTS, n_steps=N_STEPS, lr=LR):
    """Gradient-based inverse design through M0."""
    best_loss = float('inf')
    best_params = None

    for r in range(n_restarts):
        # Sigmoid reparameterization
        raw = torch.randn(10, device=device) * 0.3
        raw.requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            p_norm = torch.sigmoid(raw)

            # Constraint penalty: Wx, Wy, W2 < 0.9*P
            p_phys = p_norm * (PMAX_T - PMIN_T) + PMIN_T
            P_val = p_phys[0]
            penalty = (torch.relu(p_phys[1] - 0.9*P_val)**2 +
                       torch.relu(p_phys[2] - 0.9*P_val)**2 +
                       torch.relu(p_phys[3] - 0.9*P_val)**2) * 10.0

            A_pred = surrogate_predict_m0(p_norm, wl_norm_t)
            loss = torch.mean((A_pred - target_A_t)**2) + penalty
            loss.backward()
            opt.step()

        with torch.no_grad():
            final_loss = torch.mean((A_pred - target_A_t)**2).item()
            if final_loss < best_loss:
                best_loss = final_loss
                p_norm_final = torch.sigmoid(raw)
                best_params = (p_norm_final * (PMAX_T - PMIN_T) + PMIN_T).detach().cpu().numpy()

    return best_params, best_loss


def inverse_design_mphys(target_A_t, n_restarts=N_RESTARTS, n_steps=N_STEPS, lr=LR):
    """Gradient-based inverse design through M_phys."""
    best_loss = float('inf')
    best_params = None

    for r in range(n_restarts):
        raw = torch.randn(10, device=device) * 0.3
        raw.requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            p_norm = torch.sigmoid(raw)
            p_phys = p_norm * (PMAX_T - PMIN_T) + PMIN_T

            # Constraint penalty
            P_val = p_phys[0]
            penalty = (torch.relu(p_phys[1] - 0.9*P_val)**2 +
                       torch.relu(p_phys[2] - 0.9*P_val)**2 +
                       torch.relu(p_phys[3] - 0.9*P_val)**2) * 10.0

            A_pred = surrogate_predict_mphys(p_phys, p_norm, wl_norm_t,
                                              phys_mean.squeeze(), phys_std.squeeze())
            loss = torch.mean((A_pred - target_A_t)**2) + penalty
            loss.backward()
            opt.step()

        with torch.no_grad():
            final_loss = torch.mean((A_pred - target_A_t)**2).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = p_phys.detach().cpu().numpy()

    return best_params, best_loss


# Run inverse design for all targets
results_m0 = []
results_mp = []

for ti, tidx in enumerate(target_indices):
    target_A = test_A[tidx]
    target_A_t = torch.tensor(target_A, dtype=torch.float32, device=device)
    true_params = test_params[tidx]

    print(f"\nTarget {ti+1}/{N_TARGETS} (avg_A={target_A.mean():.3f})")

    # M0 inverse design
    t0 = time.time()
    p_m0, loss_m0 = inverse_design_m0(target_A_t)
    t_m0 = time.time() - t0

    # M_phys inverse design
    t0 = time.time()
    p_mp, loss_mp = inverse_design_mphys(target_A_t)
    t_mp = time.time() - t0

    results_m0.append({"params": p_m0, "surr_loss": loss_m0, "time": t_m0})
    results_mp.append({"params": p_mp, "surr_loss": loss_mp, "time": t_mp})

    print(f"  M0:     surr_loss={loss_m0:.6f}, time={t_m0:.1f}s")
    print(f"  M_phys: surr_loss={loss_mp:.6f}, time={t_mp:.1f}s")

# ============================================================
# 5. RCWA validation
# ============================================================
print("\n" + "="*60)
print("PHASE 3: RCWA Validation")
print("="*60)

def rcwa_validate(params_vec, wavelengths_nm):
    """Run RCWA on designed params and return A spectrum."""
    p = {n: float(params_vec[i]) for i, n in enumerate(PARAM_NAMES)}
    # Enforce constraints
    max_w = 0.9 * p["P"]
    p["Wx"] = min(p["Wx"], max_w)
    p["Wy"] = min(p["Wy"], max_w)
    p["W2"] = min(p["W2"], max_w)
    # Clamp to bounds
    for i, n in enumerate(PARAM_NAMES):
        p[n] = np.clip(p[n], float(PMIN[i]), float(PMAX[i]))

    A, R, T = simulate_single(p, wavelengths_nm, metal="Cr", device=device)
    return A


rcwa_results_m0 = []
rcwa_results_mp = []

for ti, tidx in enumerate(target_indices):
    target_A = test_A[tidx]
    print(f"\nRCWA validating target {ti+1}/{N_TARGETS}...")

    # M0 design
    A_rcwa_m0 = rcwa_validate(results_m0[ti]["params"], wavelengths)
    mae_m0 = np.mean(np.abs(A_rcwa_m0 - target_A)) * 100

    # M_phys design
    A_rcwa_mp = rcwa_validate(results_mp[ti]["params"], wavelengths)
    mae_mp = np.mean(np.abs(A_rcwa_mp - target_A)) * 100

    rcwa_results_m0.append({"A_rcwa": A_rcwa_m0, "mae": mae_m0})
    rcwa_results_mp.append({"A_rcwa": A_rcwa_mp, "mae": mae_mp})

    print(f"  M0:     RCWA MAE = {mae_m0:.2f}%")
    print(f"  M_phys: RCWA MAE = {mae_mp:.2f}%")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "="*70)
print("INVERSE DESIGN RESULTS: M0 vs M_phys")
print("="*70)

mae_m0_all = np.array([r["mae"] for r in rcwa_results_m0])
mae_mp_all = np.array([r["mae"] for r in rcwa_results_mp])
loss_m0_all = np.array([r["surr_loss"] for r in results_m0])
loss_mp_all = np.array([r["surr_loss"] for r in results_mp])
time_m0_all = np.array([r["time"] for r in results_m0])
time_mp_all = np.array([r["time"] for r in results_mp])

# Per-target comparison
print(f"\n{'Target':>7} | {'M0 RCWA MAE':>12} | {'M_phys RCWA MAE':>16} | {'M0 SurrLoss':>12} | {'M_phys SurrLoss':>16} | {'Winner':>7}")
print("-"*85)
mp_wins = 0
for ti in range(N_TARGETS):
    winner = "M_phys" if rcwa_results_mp[ti]["mae"] < rcwa_results_m0[ti]["mae"] else "M0"
    if winner == "M_phys": mp_wins += 1
    print(f"  {ti+1:>5} | {rcwa_results_m0[ti]['mae']:>10.2f}% | {rcwa_results_mp[ti]['mae']:>14.2f}% | {results_m0[ti]['surr_loss']:>12.6f} | {results_mp[ti]['surr_loss']:>16.6f} | {winner:>7}")

print(f"\n--- Aggregate ---")
print(f"RCWA MAE (mean):  M0 = {mae_m0_all.mean():.2f}% ± {mae_m0_all.std():.2f}%,  M_phys = {mae_mp_all.mean():.2f}% ± {mae_mp_all.std():.2f}%")
print(f"RCWA MAE (median): M0 = {np.median(mae_m0_all):.2f}%,  M_phys = {np.median(mae_mp_all):.2f}%")
print(f"Surr Loss (mean): M0 = {loss_m0_all.mean():.6f},  M_phys = {loss_mp_all.mean():.6f}")
print(f"Time (mean):      M0 = {time_m0_all.mean():.1f}s,  M_phys = {time_mp_all.mean():.1f}s")
print(f"Win rate:         M_phys wins {mp_wins}/{N_TARGETS} ({mp_wins/N_TARGETS*100:.0f}%)")
print(f"Success rate (MAE<10%): M0 = {(mae_m0_all<10).sum()}/{N_TARGETS},  M_phys = {(mae_mp_all<10).sum()}/{N_TARGETS}")
print(f"Success rate (MAE<5%):  M0 = {(mae_m0_all<5).sum()}/{N_TARGETS},  M_phys = {(mae_mp_all<5).sum()}/{N_TARGETS}")

# Save all results
np.savez(f"{ckpt_dir}/inverse_results.npz",
         target_indices=target_indices,
         target_A=test_A[target_indices],
         target_params=test_params[target_indices],
         m0_params=np.array([r["params"] for r in results_m0]),
         mp_params=np.array([r["params"] for r in results_mp]),
         m0_surr_loss=loss_m0_all,
         mp_surr_loss=loss_mp_all,
         m0_rcwa_A=np.array([r["A_rcwa"] for r in rcwa_results_m0]),
         mp_rcwa_A=np.array([r["A_rcwa"] for r in rcwa_results_mp]),
         m0_rcwa_mae=mae_m0_all,
         mp_rcwa_mae=mae_mp_all,
         wavelengths=wavelengths)
print(f"\nResults saved to {ckpt_dir}/inverse_results.npz")

# ============================================================
# 7. Visualization
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Top row: 3 example target spectra comparisons
for col, ti in enumerate([0, N_TARGETS//2, N_TARGETS-1]):
    ax = axes[0, col]
    tidx_real = target_indices[ti]
    target_A = test_A[tidx_real]
    ax.plot(wavelengths, target_A, 'k-', lw=2, label='Target (RCWA)')
    ax.plot(wavelengths, rcwa_results_m0[ti]["A_rcwa"], 'r--', lw=1.5, label=f'M0 design (MAE={rcwa_results_m0[ti]["mae"]:.1f}%)')
    ax.plot(wavelengths, rcwa_results_mp[ti]["A_rcwa"], 'b-', lw=1.5, label=f'M_phys design (MAE={rcwa_results_mp[ti]["mae"]:.1f}%)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorption')
    ax.set_title(f'Target {ti+1} (avg A={target_A.mean():.2f})')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

# Bottom-left: scatter plot
ax = axes[1, 0]
ax.scatter(mae_m0_all, mae_mp_all, c='steelblue', s=50, edgecolors='navy', alpha=0.8)
lim = max(mae_m0_all.max(), mae_mp_all.max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
ax.set_xlabel('M0 RCWA MAE (%)')
ax.set_ylabel('M_phys RCWA MAE (%)')
ax.set_title(f'M_phys wins {mp_wins}/{N_TARGETS}')
ax.set_aspect('equal')

# Bottom-center: box plot
ax = axes[1, 1]
bp = ax.boxplot([mae_m0_all, mae_mp_all], labels=['M0', 'M_phys'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#FFB3B3')
bp['boxes'][1].set_facecolor('#B3D9FF')
ax.set_ylabel('RCWA MAE (%)')
ax.set_title('Distribution of Design Error')

# Bottom-right: bar chart summary
ax = axes[1, 2]
metrics = ['Mean MAE', 'Median MAE', 'Success\n(MAE<10%)', 'Success\n(MAE<5%)']
m0_vals = [mae_m0_all.mean(), np.median(mae_m0_all),
           (mae_m0_all<10).sum()/N_TARGETS*100, (mae_m0_all<5).sum()/N_TARGETS*100]
mp_vals = [mae_mp_all.mean(), np.median(mae_mp_all),
           (mae_mp_all<10).sum()/N_TARGETS*100, (mae_mp_all<5).sum()/N_TARGETS*100]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, m0_vals, w, label='M0', color='#FF7777')
ax.bar(x + w/2, mp_vals, w, label='M_phys', color='#77AAFF')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel('Value (% or MAE%)')
ax.set_title('Aggregate Comparison')
ax.legend()

plt.suptitle('Inverse Design: M0 vs M_phys (RCWA-validated)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{ckpt_dir}/inverse_design_results.png", dpi=150, bbox_inches='tight')
plt.savefig(f"{ckpt_dir}/inverse_design_results.pdf", bbox_inches='tight')
print(f"\nPlots saved to {ckpt_dir}/inverse_design_results.png/.pdf")
print("\nDONE!")
