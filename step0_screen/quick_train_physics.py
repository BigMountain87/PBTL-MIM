#!/usr/bin/env python3
"""
Physics-feature augmented NN vs baseline ANN for Structure A MIM absorber.

Instead of TMM (which fails with EMA), we compute ANALYTICAL PHYSICS FEATURES
that capture the essential cavity physics of MIM absorbers:
  - Fabry-Perot cavity resonance phases
  - Fill fractions (coupling strength)
  - Sub-wavelength parameters
  - Metal skin depth ratios (absorption)
  - Inter-cavity coupling
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params
from src.simulation.materials import (get_sio2_permittivity, get_tio2_permittivity,
                                       get_metal_permittivity)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def compute_physics_features(params, wavelengths_nm, metal="Cr"):
    """
    Compute physics-derived features for each (sample, wavelength) pair.

    Parameters (10D): P, Wx, Wy, W2, t1, t2, t_mid, d1, d2, theta

    Returns: [N, Nlam, n_features] array of physics features
    """
    N = len(params)
    Nlam = len(wavelengths_nm)

    P     = params[:, 0]   # period [nm]
    Wx    = params[:, 1]   # top width x
    Wy    = params[:, 2]   # top width y
    W2    = params[:, 3]   # bottom width
    t1    = params[:, 4]   # top metal thickness
    t2    = params[:, 5]   # bottom metal thickness
    t_mid = params[:, 6]   # mid metal thickness
    d1    = params[:, 7]   # SiO2 cavity thickness
    d2    = params[:, 8]   # TiO2 cavity thickness
    theta = params[:, 9]   # incidence angle [deg]

    theta_rad = np.deg2rad(theta)

    # Material properties at each wavelength
    eps_sio2 = get_sio2_permittivity(wavelengths_nm)      # [Nlam], real
    eps_tio2 = get_tio2_permittivity(wavelengths_nm)      # [Nlam], real
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)  # [Nlam], complex

    n_sio2 = np.sqrt(np.real(eps_sio2))   # [Nlam]
    n_tio2 = np.sqrt(np.real(eps_tio2))   # [Nlam]
    n_metal = np.real(np.sqrt(eps_metal))  # [Nlam]
    k_metal = np.imag(np.sqrt(eps_metal))  # [Nlam]

    # Skin depth: delta = lambda / (4*pi*k)  [nm]
    skin_depth = wavelengths_nm / (4 * np.pi * k_metal)   # [Nlam]

    # Build features array [N, Nlam, n_feat]
    features_list = []

    # --- 1-2: Cavity resonance phases (F-P round-trip) ---
    # phase = 4*pi*n*d / lambda * cos(theta_internal)
    # At normal: cos(theta_internal) ≈ 1
    # At oblique: Snell's law gives sin(theta_int) = sin(theta) / n_cav
    for cav_idx, (n_cav, d_cav) in enumerate([(n_sio2, d1), (n_tio2, d2)]):
        # n_cav: [Nlam], d_cav: [N]
        # cos(theta_internal) for each sample
        sin_theta_int = np.sin(theta_rad[:, None]) / n_cav[None, :]  # [N, Nlam]
        sin_theta_int = np.clip(sin_theta_int, -1, 1)
        cos_theta_int = np.sqrt(1 - sin_theta_int**2)  # [N, Nlam]

        phase = 4 * np.pi * n_cav[None, :] * d_cav[:, None] * cos_theta_int / wavelengths_nm[None, :]
        features_list.append(np.cos(phase))   # resonance at cos=-1
        features_list.append(np.sin(phase))   # phase information

    # --- 3-4: Fill fractions ---
    f1 = (Wx * Wy) / (P ** 2)          # [N]
    f2 = (W2 ** 2) / (P ** 2)          # [N]
    features_list.append(np.tile(f1[:, None], (1, Nlam)))
    features_list.append(np.tile(f2[:, None], (1, Nlam)))

    # --- 5-7: Sub-wavelength parameters ---
    P_over_lam = P[:, None] / wavelengths_nm[None, :]      # [N, Nlam]
    Wx_over_lam = Wx[:, None] / wavelengths_nm[None, :]
    W2_over_lam = W2[:, None] / wavelengths_nm[None, :]
    features_list.append(P_over_lam)
    features_list.append(Wx_over_lam)
    features_list.append(W2_over_lam)

    # --- 8-10: Skin depth ratios (metal absorption) ---
    t1_over_skin = t1[:, None] / skin_depth[None, :]    # [N, Nlam]
    t2_over_skin = t2[:, None] / skin_depth[None, :]
    tmid_over_skin = t_mid[:, None] / skin_depth[None, :]
    features_list.append(t1_over_skin)
    features_list.append(t2_over_skin)
    features_list.append(tmid_over_skin)    # inter-cavity coupling

    # --- 11: Angle factor ---
    cos_theta = np.tile(np.cos(theta_rad[:, None]), (1, Nlam))  # [N, Nlam]
    features_list.append(cos_theta)

    # --- 12-13: Normalized optical path lengths ---
    # OPL / lambda (how many wavelengths fit in cavity)
    opl1 = n_sio2[None, :] * d1[:, None] / wavelengths_nm[None, :]  # [N, Nlam]
    opl2 = n_tio2[None, :] * d2[:, None] / wavelengths_nm[None, :]
    features_list.append(opl1)
    features_list.append(opl2)

    # --- 14: Aspect ratios (geometry shape) ---
    aspect = Wy / (Wx + 1e-10)  # [N], asymmetry of top pattern
    features_list.append(np.tile(aspect[:, None], (1, Nlam)))

    # --- 15: Metal absorption coefficient at each wavelength ---
    alpha_metal = 4 * np.pi * k_metal / wavelengths_nm  # [Nlam], absorption coefficient
    features_list.append(np.tile(alpha_metal[None, :], (N, 1)))

    # Stack all features: [N, Nlam, n_features]
    features = np.stack(features_list, axis=-1).astype(np.float32)
    print(f"  Physics features: {features.shape[-1]} features per (sample, wavelength)")

    return features


# ========= Load data =========
data = np.load("data/raw/struct_A_vis_100.npz",
               allow_pickle=True)
params_all = data["params"].astype(np.float32)
A_all = data["A"].astype(np.float32)
R_all = data["R"].astype(np.float32)
wavelengths = data["wavelengths"].astype(np.float32)
N_orig = len(params_all)
Nlam = len(wavelengths)
print(f"Loaded {N_orig} samples, {Nlam} wavelengths ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm)")

# Filter bad samples
good_mask = np.all((R_all >= 0) & (R_all <= 1) & (A_all >= 0) & (A_all <= 1), axis=1)
good_idx = np.where(good_mask)[0]
N = len(good_idx)
print(f"Filtered: {N_orig} -> {N} good samples ({N_orig - N} removed)")

params = params_all[good_idx]
A_arr = A_all[good_idx]
R_arr = R_all[good_idx]

# Compute physics features
print("\nComputing physics features...")
phys_feats = compute_physics_features(params, wavelengths, metal="Cr")
n_phys = phys_feats.shape[-1]

# Check feature statistics
print("\nPhysics feature statistics:")
feat_names = [
    "cos_phase_SiO2", "sin_phase_SiO2", "cos_phase_TiO2", "sin_phase_TiO2",
    "f1_fill", "f2_fill",
    "P/lam", "Wx/lam", "W2/lam",
    "t1/skin", "t2/skin", "tmid/skin",
    "cos_theta",
    "OPL1/lam", "OPL2/lam",
    "aspect_ratio", "alpha_metal"
]
for i, name in enumerate(feat_names[:n_phys]):
    vals = phys_feats[:, :, i].ravel()
    # Correlation with absorption
    corr = np.corrcoef(vals, A_arr.ravel())[0, 1]
    print(f"  {name:18s}: mean={vals.mean():.3f}, std={vals.std():.3f}, corr(A)={corr:.3f}")

# Normalize params
params_norm = normalize_params(params, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

n_params = params.shape[1]  # 10
geo_dim = 1 + n_params  # 11

# Build input tensors
params_rep = np.repeat(params_norm[:, np.newaxis, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[np.newaxis, :, np.newaxis], (N, 1, 1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys = phys_feats.reshape(-1, n_phys).astype(np.float32)

# Normalize physics features
phys_mean = X_phys.mean(axis=0, keepdims=True)
phys_std = X_phys.std(axis=0, keepdims=True) + 1e-8
X_phys_norm = ((X_phys - phys_mean) / phys_std).astype(np.float32)

A_flat = A_arr.reshape(-1).astype(np.float32)
R_flat = R_arr.reshape(-1).astype(np.float32)

# Train/Val/Test split (sample-level, same as before)
rng = np.random.default_rng(42)
idx = rng.permutation(N)
n_train = int(N * 0.70)
n_val = int(N * 0.15)
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

def get_rows(sample_idx):
    return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in sample_idx])

train_rows = get_rows(train_idx)
val_rows = get_rows(val_idx)
test_rows = get_rows(test_idx)
print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


# ========= Model definitions =========
class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            ) for _ in range(n_blocks)
        ])
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for block in self.blocks:
            h = h + self.act(block(h))
        return h


class M0Model(nn.Module):
    """Baseline: ANN only"""
    def __init__(self, in_dim):
        super().__init__()
        self.backbone = BaseResNet(in_dim, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x, **kwargs):
        h = self.backbone(x)
        R = self.head(h).squeeze(-1)
        return {"A": 1.0 - R, "R": R}


class MPhysModel(nn.Module):
    """Physics-feature augmented NN: geo + physics features as input"""
    def __init__(self, geo_dim, phys_dim):
        super().__init__()
        self.backbone = BaseResNet(geo_dim + phys_dim, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x_geo, x_phys=None, **kwargs):
        x = torch.cat([x_geo, x_phys], dim=-1)
        h = self.backbone(x)
        R = self.head(h).squeeze(-1)
        return {"A": 1.0 - R, "R": R}


class MPhysResModel(nn.Module):
    """Physics-feature + residual: predict base from physics, refine with NN"""
    def __init__(self, geo_dim, phys_dim):
        super().__init__()
        # Physics branch: estimate rough R from physics features
        self.phys_net = nn.Sequential(
            nn.Linear(phys_dim, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # Refinement branch: full features -> residual
        self.backbone = BaseResNet(geo_dim + phys_dim + 1, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Tanh()
        )

    def forward(self, x_geo, x_phys=None, **kwargs):
        R_phys = self.phys_net(x_phys).squeeze(-1)
        x = torch.cat([x_geo, x_phys, R_phys.unsqueeze(-1)], dim=-1)
        h = self.backbone(x)
        delta = self.head(h).squeeze(-1)
        R = torch.clamp(R_phys + delta, 0, 1)
        return {"A": 1.0 - R, "R": R, "R_phys": R_phys}


def train_model(model, train_loader, val_loader, n_epochs=5000, lr=1e-3, model_name=""):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if len(batch) == 4:  # geo, A, R, phys
                x, a, r, p = batch
                out = model(x, x_phys=p)
            else:
                x, a, r = batch
                out = model(x)

            loss = criterion(out["A"], a) + criterion(out["R"], r)
            # Optional: add physics branch regularization
            if "R_phys" in out:
                loss += 0.1 * criterion(out["R_phys"], r)  # encourage physics branch accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_n = 0
                for batch in val_loader:
                    if len(batch) == 4:
                        x, a, r, p = batch
                        out = model(x, x_phys=p)
                    else:
                        x, a, r = batch
                        out = model(x)
                    val_loss += (nn.functional.l1_loss(out["A"], a, reduction="sum") +
                                 nn.functional.l1_loss(out["R"], r, reduction="sum")).item()
                    val_n += len(a) * 2
                val_mae = val_loss / val_n
                if val_mae < best_val_loss:
                    best_val_loss = val_mae
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  [{model_name}] Ep {epoch+1}: val_MAE={val_mae*100:.3f}%, best={best_val_loss*100:.3f}%")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ========= Run training =========
SEEDS = [42, 123, 777]
EPOCHS = 5000

MODEL_DEFS = {
    "M0_baseline": ("m0", None),
    "M_phys_feat": ("phys", None),       # Physics features as extra input
    "M_phys_res":  ("phys_res", None),    # Physics + residual architecture
}

results = {name: [] for name in MODEL_DEFS}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    X_train_t = torch.tensor(X_geo[train_rows]).to(device)
    A_train_t = torch.tensor(A_flat[train_rows]).to(device)
    R_train_t = torch.tensor(R_flat[train_rows]).to(device)
    P_train_t = torch.tensor(X_phys_norm[train_rows]).to(device)

    X_val_t = torch.tensor(X_geo[val_rows]).to(device)
    A_val_t = torch.tensor(A_flat[val_rows]).to(device)
    R_val_t = torch.tensor(R_flat[val_rows]).to(device)
    P_val_t = torch.tensor(X_phys_norm[val_rows]).to(device)

    X_test_t = torch.tensor(X_geo[test_rows]).to(device)
    A_test_t = torch.tensor(A_flat[test_rows]).to(device)
    R_test_t = torch.tensor(R_flat[test_rows]).to(device)
    P_test_t = torch.tensor(X_phys_norm[test_rows]).to(device)

    ds_m0_tr = TensorDataset(X_train_t, A_train_t, R_train_t)
    ds_m0_vl = TensorDataset(X_val_t, A_val_t, R_val_t)
    dl_m0_tr = DataLoader(ds_m0_tr, batch_size=512, shuffle=True)
    dl_m0_vl = DataLoader(ds_m0_vl, batch_size=1024)

    ds_ph_tr = TensorDataset(X_train_t, A_train_t, R_train_t, P_train_t)
    ds_ph_vl = TensorDataset(X_val_t, A_val_t, R_val_t, P_val_t)
    dl_ph_tr = DataLoader(ds_ph_tr, batch_size=512, shuffle=True)
    dl_ph_vl = DataLoader(ds_ph_vl, batch_size=1024)

    for model_name, (model_type, _) in MODEL_DEFS.items():
        print(f"\n--- {model_name} ---")
        set_seed(seed)

        if model_type == "m0":
            model = M0Model(geo_dim).to(device)
            dl_tr, dl_vl = dl_m0_tr, dl_m0_vl
        elif model_type == "phys":
            model = MPhysModel(geo_dim, n_phys).to(device)
            dl_tr, dl_vl = dl_ph_tr, dl_ph_vl
        elif model_type == "phys_res":
            model = MPhysResModel(geo_dim, n_phys).to(device)
            dl_tr, dl_vl = dl_ph_tr, dl_ph_vl

        model, _ = train_model(model, dl_tr, dl_vl, EPOCHS, lr=1e-3, model_name=model_name)

        model.eval()
        with torch.no_grad():
            if model_type == "m0":
                out = model(X_test_t)
            else:
                out = model(X_test_t, x_phys=P_test_t)

            mae_A = torch.mean(torch.abs(out["A"] - A_test_t)).item()
            mae_R = torch.mean(torch.abs(out["R"] - R_test_t)).item()

        results[model_name].append(mae_A)
        print(f"  Test MAE: A={mae_A*100:.3f}%, R={mae_R*100:.3f}%")

        if "R_phys" in out:
            phys_mae = torch.mean(torch.abs(out["R_phys"] - R_test_t)).item()
            print(f"  Physics-only branch MAE(R): {phys_mae*100:.3f}%")

# ========= Summary =========
print("\n" + "="*60)
print("SUMMARY: Structure A (Visible, Physics Features)")
print("="*60)
print(f"Samples: {N} (filtered from {N_orig})")
print(f"Wavelength: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm")
print(f"Physics features: {n_phys}")
print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print()

m0_mean = np.mean(results["M0_baseline"])
for model_name in MODEL_DEFS:
    vals = results[model_name]
    mean_v = np.mean(vals) * 100
    std_v = np.std(vals) * 100
    ratio = np.mean(vals) / m0_mean
    seeds_str = ", ".join([f"{v*100:.3f}" for v in vals])
    print(f"{model_name:15s}: {mean_v:.3f} +/- {std_v:.3f}%  ratio={ratio:.3f}  [{seeds_str}]")

print()
best_name = min(MODEL_DEFS.keys(), key=lambda k: np.mean(results[k]))
best_ratio = np.mean(results[best_name]) / m0_mean
print(f"Best: {best_name} (ratio={best_ratio:.3f})")
if best_ratio < 0.85:
    print(">> STRONG physics backbone effect!")
elif best_ratio < 0.95:
    print(">> Moderate physics backbone effect")
else:
    print(">> Weak/no physics backbone effect")
