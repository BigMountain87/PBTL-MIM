#!/usr/bin/env python3
"""
Quick M0 vs M7 training on visible-range Structure A data.
V2: M7 uses TMM as feature input (not residual), with 3 variants.
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

# Filter bad samples
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

# Also check TMM correlation with true values
corr = np.corrcoef(A_tmm.ravel(), A_arr.ravel())[0, 1]
print(f"TMM-RCWA correlation (A): {corr:.4f}")

# Normalize params
params_norm = normalize_params(params, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

n_params = params.shape[1]  # 10
geo_dim = 1 + n_params  # 11

# Build input tensors
params_rep = np.repeat(params_norm[:, np.newaxis, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[np.newaxis, :, np.newaxis], (N, 1, 1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)

# Flatten targets
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
    """Baseline: ANN only, no TMM"""
    def __init__(self, geo_dim):
        super().__init__()
        self.backbone = BaseResNet(geo_dim, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x_geo, **kwargs):
        h = self.backbone(x_geo)
        R = self.head(h).squeeze(-1)
        A = 1.0 - R
        return {"A": A, "R": R}


class M7aModel(nn.Module):
    """TMM as feature input (not residual). NN predicts R freely."""
    def __init__(self, geo_dim):
        super().__init__()
        # geo_dim + 3 (A_tmm, R_tmm, T_tmm)
        self.backbone = BaseResNet(geo_dim + 3, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Sigmoid()  # free prediction [0,1]
        )

    def forward(self, x_geo, A_tmm=None, R_tmm=None, T_tmm=None, **kwargs):
        x = torch.cat([x_geo, A_tmm.unsqueeze(-1), R_tmm.unsqueeze(-1),
                        T_tmm.unsqueeze(-1)], dim=-1)
        h = self.backbone(x)
        R = self.head(h).squeeze(-1)
        A = 1.0 - R
        return {"A": A, "R": R}


class M7bModel(nn.Module):
    """TMM gated: R = alpha*R_tmm + (1-alpha)*R_nn, alpha learned."""
    def __init__(self, geo_dim):
        super().__init__()
        self.backbone = BaseResNet(geo_dim + 3, hidden=256, n_blocks=4)
        self.head_R = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.head_alpha = nn.Sequential(
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x_geo, A_tmm=None, R_tmm=None, T_tmm=None, **kwargs):
        x = torch.cat([x_geo, A_tmm.unsqueeze(-1), R_tmm.unsqueeze(-1),
                        T_tmm.unsqueeze(-1)], dim=-1)
        h = self.backbone(x)
        R_nn = self.head_R(h).squeeze(-1)
        alpha = self.head_alpha(h).squeeze(-1)
        R = alpha * R_tmm + (1.0 - alpha) * R_nn
        A = 1.0 - R
        return {"A": A, "R": R, "alpha": alpha}


class M7cModel(nn.Module):
    """Full residual with wider range: R = clamp(R_tmm + delta), delta in [-1, 1]"""
    def __init__(self, geo_dim):
        super().__init__()
        self.backbone = BaseResNet(geo_dim + 3, hidden=256, n_blocks=4)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1), nn.Tanh()  # residual in [-1, 1]
        )

    def forward(self, x_geo, A_tmm=None, R_tmm=None, T_tmm=None, **kwargs):
        x = torch.cat([x_geo, A_tmm.unsqueeze(-1), R_tmm.unsqueeze(-1),
                        T_tmm.unsqueeze(-1)], dim=-1)
        h = self.backbone(x)
        delta_R = self.head(h).squeeze(-1)  # full [-1, 1] range
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

        if (epoch + 1) % 1000 == 0 or epoch == 0:
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
                print(f"  [{model_name}] Ep {epoch+1}: val_MAE={val_mae*100:.3f}%, best={best_val_loss*100:.3f}%")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ========= Run training =========
SEEDS = [42, 123, 777]
EPOCHS = 5000

MODEL_CLASSES = {
    "M0": M0Model,
    "M7a_feat": M7aModel,    # TMM as feature input
    "M7b_gate": M7bModel,    # TMM gated
    "M7c_resid": M7cModel,   # Full range residual
}

results = {name: [] for name in MODEL_CLASSES}

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

    # M7 datasets (with TMM features)
    ds_train_m7 = TensorDataset(X_train_t, A_train_t, R_train_t, A_tmm_train_t, R_tmm_train_t)
    ds_val_m7 = TensorDataset(X_val_t, A_val_t, R_val_t, A_tmm_val_t, R_tmm_val_t)
    dl_train_m7 = DataLoader(ds_train_m7, batch_size=512, shuffle=True)
    dl_val_m7 = DataLoader(ds_val_m7, batch_size=1024)

    for model_name, ModelClass in MODEL_CLASSES.items():
        print(f"\n--- {model_name} ---")
        set_seed(seed)  # reset seed for fair comparison

        is_m0 = (model_name == "M0")
        model = ModelClass(geo_dim).to(device)
        dl_train = dl_train_m0 if is_m0 else dl_train_m7
        dl_val = dl_val_m0 if is_m0 else dl_val_m7

        model, _ = train_model(model, dl_train, dl_val, EPOCHS, lr=1e-3, model_name=model_name)

        model.eval()
        with torch.no_grad():
            if is_m0:
                out = model(X_test_t)
            else:
                t_tmm = torch.zeros_like(R_tmm_test_t)
                out = model(X_test_t, A_tmm=A_tmm_test_t, R_tmm=R_tmm_test_t, T_tmm=t_tmm)

            mae_A = torch.mean(torch.abs(out["A"] - A_test_t)).item()
            mae_R = torch.mean(torch.abs(out["R"] - R_test_t)).item()

        results[model_name].append(mae_A)
        print(f"  Test MAE: A={mae_A*100:.3f}%, R={mae_R*100:.3f}%")

        if "alpha" in out:
            alpha_mean = out["alpha"].mean().item()
            print(f"  Mean alpha (TMM weight): {alpha_mean:.3f}")


# ========= Summary =========
print("\n" + "="*60)
print("SUMMARY: Structure A (Visible, Filtered)")
print("="*60)
print(f"Samples: {N} (filtered from {N_orig})")
print(f"Wavelength: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm")
print(f"TMM MAE: {tmm_mae*100:.2f}%")
print(f"TMM-RCWA corr: {corr:.4f}")
print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
print()

m0_mean = np.mean(results["M0"])
for model_name in MODEL_CLASSES:
    vals = results[model_name]
    mean_v = np.mean(vals) * 100
    std_v = np.std(vals) * 100
    ratio = np.mean(vals) / m0_mean
    seeds_str = ", ".join([f"{v*100:.3f}" for v in vals])
    print(f"{model_name:12s}: {mean_v:.3f} +/- {std_v:.3f}%  ratio={ratio:.3f}  [{seeds_str}]")

print()
best_model = min(MODEL_CLASSES.keys(), key=lambda k: np.mean(results[k]))
best_ratio = np.mean(results[best_model]) / m0_mean
print(f"Best model: {best_model} (ratio={best_ratio:.3f})")
if best_ratio < 0.85:
    print(">> STRONG TMM backbone effect!")
elif best_ratio < 0.95:
    print(">> Moderate TMM backbone effect")
else:
    print(">> Weak/no TMM backbone effect")
