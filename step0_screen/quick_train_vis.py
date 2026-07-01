#!/usr/bin/env python3
"""
Quick M0 vs M7 training on visible-range Structure A data.
Filters R>1 samples, trains 5000 epochs.
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, BOUNDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
data = np.load("data/raw/struct_A_vis_100.npz",
               allow_pickle=True)
params_all = data["params"].astype(np.float32)
A_all = data["A"].astype(np.float32)
R_all = data["R"].astype(np.float32)
T_all = data["T"].astype(np.float32)
wavelengths = data["wavelengths"].astype(np.float32)
N_orig = len(params_all)
Nlam = len(wavelengths)

print(f"Loaded {N_orig} samples, {Nlam} wavelengths ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm)")

# Filter bad samples (R > 1 at any wavelength)
good_mask = np.all((R_all >= 0) & (R_all <= 1) & (A_all >= 0) & (A_all <= 1), axis=1)
good_idx = np.where(good_mask)[0]
N = len(good_idx)
print(f"Filtered: {N_orig} -> {N} good samples ({N_orig - N} removed)")

params = params_all[good_idx]
A_arr = A_all[good_idx]
R_arr = R_all[good_idx]

# Compute TMM
from src.simulation.tmm_struct_a import compute_tmm_batch
tmm_out = compute_tmm_batch(params, wavelengths, metal="Cr")
A_tmm = tmm_out["A_tmm"].astype(np.float32)
R_tmm = tmm_out["R_tmm"].astype(np.float32)
T_tmm = tmm_out["T_tmm"].astype(np.float32)

tmm_mae = np.mean(np.abs(A_tmm - A_arr))
print(f"TMM MAE on filtered data: {tmm_mae*100:.2f}%")

# Normalize params
params_norm = normalize_params(params, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

n_params = params.shape[1]  # 10
geo_dim = 1 + n_params  # 11

# Build input tensors
params_rep = np.repeat(params_norm[:, np.newaxis, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[np.newaxis, :, np.newaxis], (N, 1, 1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)

# Flatten targets: [N*Nlam]
A_flat = A_arr.reshape(-1).astype(np.float32)
R_flat = R_arr.reshape(-1).astype(np.float32)
A_tmm_flat = A_tmm.reshape(-1).astype(np.float32)
R_tmm_flat = R_tmm.reshape(-1).astype(np.float32)
T_tmm_flat = T_tmm.reshape(-1).astype(np.float32)

# Train/Val/Test split (sample-level)
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

print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


# ========= Simple model definitions =========
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
        self.out_dim = hidden

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for block in self.blocks:
            h = h + self.act(block(h))
        return h


class M0Model(nn.Module):
    """Baseline: ANN only, no TMM"""
    def __init__(self, geo_dim):
        super().__init__()
        self.backbone = BaseResNet(geo_dim, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_geo, **kwargs):
        h = self.backbone(x_geo)
        R = self.head(h).squeeze(-1)
        A = 1.0 - R
        return {"A": A, "R": R}


class M7Model(nn.Module):
    """TMM backbone + residual NN"""
    def __init__(self, geo_dim):
        super().__init__()
        self.backbone = BaseResNet(geo_dim + 3, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x_geo, A_tmm=None, R_tmm=None, T_tmm=None, **kwargs):
        x = torch.cat([x_geo, A_tmm.unsqueeze(-1), R_tmm.unsqueeze(-1), T_tmm.unsqueeze(-1)], dim=-1)
        h = self.backbone(x)
        delta_R = self.head(h).squeeze(-1) * 0.5
        R = torch.clamp(R_tmm + delta_R, 0, 1)
        A = 1.0 - R
        return {"A": A, "R": R}


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
            if len(batch) == 5:
                x, a, r, a_tmm, r_tmm = batch
                t_tmm = torch.zeros_like(r_tmm)
                out = model(x, A_tmm=a_tmm, R_tmm=r_tmm, T_tmm=t_tmm)
            else:
                x, a, r = batch
                out = model(x)

            loss = criterion(out["A"], a) + criterion(out["R"], r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 500 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_n = 0
                for batch in val_loader:
                    if len(batch) == 5:
                        x, a, r, a_tmm, r_tmm = batch
                        t_tmm = torch.zeros_like(r_tmm)
                        out = model(x, A_tmm=a_tmm, R_tmm=r_tmm, T_tmm=t_tmm)
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
                print(f"  [{model_name}] Epoch {epoch+1}: train_loss={train_loss/n_batches:.6f}, "
                      f"val_MAE={val_mae*100:.3f}%, best={best_val_loss*100:.3f}%")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ========= Run training =========
SEEDS = [42, 123, 777]
EPOCHS = 5000

results = {"M0": [], "M7": []}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    set_seed(seed)

    X_train_t = torch.tensor(X_geo[train_rows]).to(device)
    A_train_t = torch.tensor(A_flat[train_rows]).to(device)
    R_train_t = torch.tensor(R_flat[train_rows]).to(device)
    A_tmm_train_t = torch.tensor(A_tmm_flat[train_rows]).to(device)
    R_tmm_train_t = torch.tensor(R_tmm_flat[train_rows]).to(device)

    X_val_t = torch.tensor(X_geo[val_rows]).to(device)
    A_val_t = torch.tensor(A_flat[val_rows]).to(device)
    R_val_t = torch.tensor(R_flat[val_rows]).to(device)
    A_tmm_val_t = torch.tensor(A_tmm_flat[val_rows]).to(device)
    R_tmm_val_t = torch.tensor(R_tmm_flat[val_rows]).to(device)

    X_test_t = torch.tensor(X_geo[test_rows]).to(device)
    A_test_t = torch.tensor(A_flat[test_rows]).to(device)
    R_test_t = torch.tensor(R_flat[test_rows]).to(device)
    A_tmm_test_t = torch.tensor(A_tmm_flat[test_rows]).to(device)
    R_tmm_test_t = torch.tensor(R_tmm_flat[test_rows]).to(device)

    # M0 datasets
    ds_train_m0 = TensorDataset(X_train_t, A_train_t, R_train_t)
    ds_val_m0 = TensorDataset(X_val_t, A_val_t, R_val_t)
    dl_train_m0 = DataLoader(ds_train_m0, batch_size=512, shuffle=True)
    dl_val_m0 = DataLoader(ds_val_m0, batch_size=1024)

    # M7 datasets
    ds_train_m7 = TensorDataset(X_train_t, A_train_t, R_train_t, A_tmm_train_t, R_tmm_train_t)
    ds_val_m7 = TensorDataset(X_val_t, A_val_t, R_val_t, A_tmm_val_t, R_tmm_val_t)
    dl_train_m7 = DataLoader(ds_train_m7, batch_size=512, shuffle=True)
    dl_val_m7 = DataLoader(ds_val_m7, batch_size=1024)

    # --- M0 ---
    print("\n--- M0 (Baseline ANN) ---")
    m0 = M0Model(geo_dim).to(device)
    m0, _ = train_model(m0, dl_train_m0, dl_val_m0, EPOCHS, lr=1e-3, model_name="M0")

    m0.eval()
    with torch.no_grad():
        out_m0 = m0(X_test_t)
        mae_m0_A = torch.mean(torch.abs(out_m0["A"] - A_test_t)).item()
        mae_m0_R = torch.mean(torch.abs(out_m0["R"] - R_test_t)).item()
    results["M0"].append(mae_m0_A)
    print(f"  M0 Test MAE: A={mae_m0_A*100:.3f}%, R={mae_m0_R*100:.3f}%")

    # --- M7 ---
    print("\n--- M7 (TMM + Residual NN) ---")
    m7 = M7Model(geo_dim).to(device)
    m7, _ = train_model(m7, dl_train_m7, dl_val_m7, EPOCHS, lr=1e-3, model_name="M7")

    m7.eval()
    with torch.no_grad():
        t_tmm_test = torch.zeros_like(R_tmm_test_t)
        out_m7 = m7(X_test_t, A_tmm=A_tmm_test_t, R_tmm=R_tmm_test_t, T_tmm=t_tmm_test)
        mae_m7_A = torch.mean(torch.abs(out_m7["A"] - A_test_t)).item()
        mae_m7_R = torch.mean(torch.abs(out_m7["R"] - R_test_t)).item()
    results["M7"].append(mae_m7_A)
    print(f"  M7 Test MAE: A={mae_m7_A*100:.3f}%, R={mae_m7_R*100:.3f}%")

# ========= Summary =========
print("\n" + "="*60)
print("SUMMARY: Structure A (Visible, Filtered)")
print("="*60)
print(f"Samples: {N} (filtered from {N_orig})")
print(f"Wavelength: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm")
print(f"TMM MAE: {tmm_mae*100:.2f}%")
print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print()

for model_name in ["M0", "M7"]:
    vals = results[model_name]
    mean_v = np.mean(vals) * 100
    std_v = np.std(vals) * 100
    seeds_str = ", ".join([f"{v*100:.3f}" for v in vals])
    print(f"{model_name}: {mean_v:.3f} +/- {std_v:.3f}% (seeds: [{seeds_str}])")

m0_mean = np.mean(results["M0"])
m7_mean = np.mean(results["M7"])
ratio = m7_mean / m0_mean
print(f"\nM7/M0 ratio: {ratio:.3f}")
if ratio < 0.85:
    print(">> STRONG TMM backbone effect!")
elif ratio < 0.95:
    print(">> Moderate TMM backbone effect")
else:
    print(">> Weak/no TMM backbone effect")
