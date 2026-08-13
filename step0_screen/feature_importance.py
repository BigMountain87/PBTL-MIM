#!/usr/bin/env python3
"""
Permutation importance analysis for physics features in Structure A.

For both M_phys and M_TL+phys models:
  - Train/load model with n=350, seed=42
  - For each of the 17 physics features, shuffle that feature in the test set
  - Measure increase in MAE over 10 repeats
  - Report individual and category-grouped importances
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ========= Model definitions (same as pbtl_A_10seed.py) =========
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

class M0(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1 - R, "R": R}

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd + pd)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x, p], -1))
        R = self.head(h).squeeze(-1)
        return {"A": 1 - R, "R": R}

def train_model(model, dl_tr, dl_vl, epochs, lr, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if has_phys:
                x, a, r, p = batch
                out = model(x, p=p)
            else:
                x, a, r = batch
                out = model(x)
            loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                for batch in dl_vl:
                    if has_phys:
                        x, a, r, p = batch
                        out = model(x, p=p)
                    else:
                        x, a, r = batch
                        out = model(x)
                    vl += (nn.functional.l1_loss(out["A"], a, reduction="sum") +
                           nn.functional.l1_loss(out["R"], r, reduction="sum")).item()
                    vn += len(a) * 2
                vm = vl / vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k: v.clone() for k, v in model.state_dict().items()}
    if best_st:
        model.load_state_dict(best_st)
    return model

def eval_mae(model, X_geo, A_true, phys=None):
    """Evaluate MAE on absorption. All inputs are tensors on device."""
    model.eval()
    with torch.no_grad():
        if phys is not None:
            out = model(X_geo, p=phys)
        else:
            out = model(X_geo)
        mae = nn.functional.l1_loss(out["A"], A_true).item()
    return mae

# ========= Feature names and categories =========
FEATURE_NAMES = [
    "cos(phase_SiO2)", "sin(phase_SiO2)", "cos(phase_TiO2)", "sin(phase_TiO2)",
    "ff_rect (Wx*Wy/P^2)", "ff_square (W2^2/P^2)",
    "P/lambda", "Wx/lambda", "W2/lambda",
    "t1/delta", "t2/delta", "t_mid/delta",
    "n_SiO2*d1/lambda", "n_TiO2*d2/lambda",
    "cos(theta)", "Wy/Wx", "alpha_metal",
]

CATEGORIES = {
    "Cavity resonance":    [0, 1, 2, 3],
    "Fill fraction":       [4, 5],
    "Sub-wavelength ratio":[6, 7, 8],
    "Skin depth ratio":    [9, 10, 11],
    "Optical path":        [12, 13],
    "Angle & geometry":    [14, 15, 16],
}

N_PERMUTATIONS = 10

# ========= Step 1: Generate TMM data for pre-training stats =========
print("\n=== Step 1: Generate TMM data (for normalization stats) ===", flush=True)
N_TMM = 5000
wavelengths_rcwa = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths_rcwa)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths_rcwa, "Cr")
print(f"TMM generation: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)

phys_tmm = compute_physics_features_A(params_tmm, wavelengths_rcwa, "Cr")
n_phys = phys_tmm.shape[-1]
assert n_phys == 17

params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths_rcwa - wavelengths_rcwa.min()) / (wavelengths_rcwa.max() - wavelengths_rcwa.min())
geo_dim = 1 + 10

params_rep_tmm = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep, params_rep_tmm], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_tmm = phys_tmm.reshape(-1, n_phys).astype(np.float32)

phys_mean_tmm = X_phys_tmm.mean(0, keepdims=True)
phys_std_tmm = X_phys_tmm.std(0, keepdims=True) + 1e-8
X_phys_tmm_n = ((X_phys_tmm - phys_mean_tmm) / phys_std_tmm).astype(np.float32)

A_tmm_flat = A_tmm.reshape(-1)
R_tmm_flat = R_tmm.reshape(-1)

# TMM dataloaders for pre-training
n_tmm_train = int(N_TMM * 0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_idx[:n_tmm_train]])
tmm_vl_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_idx[n_tmm_train:]])

def to_dl(rows, X_geo, A_flat, R_flat, X_phys_n, has_phys, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo[rows]).to(device)
    a = torch.tensor(A_flat[rows]).to(device)
    r = torch.tensor(R_flat[rows]).to(device)
    if has_phys:
        p = torch.tensor(X_phys_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_tmm_tr_ph = to_dl(tmm_tr_rows, X_geo_tmm, A_tmm_flat, R_tmm_flat, X_phys_tmm_n, True, shuffle=True)
dl_tmm_vl_ph = to_dl(tmm_vl_rows, X_geo_tmm, A_tmm_flat, R_tmm_flat, X_phys_tmm_n, True)

# ========= Step 2: Load RCWA data =========
print("\n=== Step 2: Load RCWA data ===", flush=True)
data = np.load("data/raw/struct_A_vis_500.npz", allow_pickle=True)
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)

good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_rcwa = params_rcwa[gi]
A_rcwa = A_rcwa[gi]
R_rcwa = R_rcwa[gi]
N_rcwa = len(gi)
print(f"RCWA data: {N_rcwa} good samples", flush=True)

phys_rcwa = compute_physics_features_A(params_rcwa, wavelengths_rcwa, "Cr")
params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_rcwa = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep_rcwa = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep_rcwa, params_rep_rcwa], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_rcwa = phys_rcwa.reshape(-1, n_phys).astype(np.float32)
X_phys_rcwa_n = ((X_phys_rcwa - phys_mean_tmm) / phys_std_tmm).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)

def get_rows(si):
    return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in si])

# Fixed test/val split (same as pbtl_A_10seed.py)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 50
N_VAL = 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST + N_VAL)]

test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)

def make_dl(rows, has_phys, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    if has_phys:
        p = torch.tensor(X_phys_rcwa_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

# ========= Step 3: Train M_phys and M_TL+phys with n=350, seed=42 =========
print("\n=== Step 3: Train models (n=350, seed=42) ===", flush=True)
n_train = 350
seed = 42

rng2 = np.random.default_rng(seed)
perm = rng2.permutation(len(remaining))
tr_idx = remaining[perm[:n_train]]
tr_rows = get_rows(tr_idx)

dl_tr_ph = make_dl(tr_rows, True, bs=512, shuffle=True)
dl_vl_ph = make_dl(val_rows, True)
dl_te_ph = make_dl(test_rows, True)

FINETUNE_EPOCHS = 1000
FINETUNE_LR = 1e-3
FINETUNE_LR_TL = 3e-4

# --- M_phys: from scratch with physics features ---
print("Training M_phys (from scratch)...", flush=True)
set_seed(seed)
m_phys = MPhys(geo_dim, n_phys).to(device)
t0 = time.time()
m_phys = train_model(m_phys, dl_tr_ph, dl_vl_ph, FINETUNE_EPOCHS, FINETUNE_LR, has_phys=True)
print(f"  M_phys trained in {time.time()-t0:.0f}s", flush=True)

# --- M_TL+phys: pre-trained, fine-tuned ---
print("Training M_TL+phys (pre-trained + fine-tuned)...", flush=True)
set_seed(seed)
m_tlphys = MPhys(geo_dim, n_phys).to(device)
m_tlphys.load_state_dict(torch.load(
    "results/pretrained_mphys_tmm.pt",
    map_location=device, weights_only=True))
t0 = time.time()
m_tlphys = train_model(m_tlphys, dl_tr_ph, dl_vl_ph, FINETUNE_EPOCHS, FINETUNE_LR_TL, has_phys=True)
print(f"  M_TL+phys trained in {time.time()-t0:.0f}s", flush=True)

# ========= Step 4: Permutation importance =========
print("\n=== Step 4: Permutation importance analysis ===", flush=True)

# Prepare test tensors (full, not in dataloader)
X_geo_test = torch.tensor(X_geo_rcwa[test_rows]).to(device)
X_phys_test = torch.tensor(X_phys_rcwa_n[test_rows]).to(device)
A_test = torch.tensor(A_rcwa_flat[test_rows]).to(device)

def permutation_importance(model, X_geo, X_phys, A_true, n_features=17, n_repeats=10):
    """
    Compute permutation importance for each physics feature.
    Returns: (n_features, n_repeats) array of MAE increases.
    """
    # Baseline MAE
    baseline_mae = eval_mae(model, X_geo, A_true, phys=X_phys)
    print(f"  Baseline test MAE: {baseline_mae*100:.4f}%", flush=True)

    importance = np.zeros((n_features, n_repeats))

    for feat_idx in range(n_features):
        for rep in range(n_repeats):
            # Copy physics features and shuffle one column
            X_phys_perm = X_phys.clone()
            perm_idx = torch.randperm(X_phys_perm.shape[0], device=device)
            X_phys_perm[:, feat_idx] = X_phys_perm[perm_idx, feat_idx]

            perm_mae = eval_mae(model, X_geo, A_true, phys=X_phys_perm)
            importance[feat_idx, rep] = perm_mae - baseline_mae

        mean_imp = importance[feat_idx].mean() * 100
        std_imp = importance[feat_idx].std() * 100
        print(f"  Feature {feat_idx:2d} ({FEATURE_NAMES[feat_idx]:>22s}): "
              f"dMAE = {mean_imp:+.4f} +/- {std_imp:.4f}%", flush=True)

    return importance, baseline_mae

print("\n--- M_phys permutation importance ---", flush=True)
imp_mphys, base_mphys = permutation_importance(m_phys, X_geo_test, X_phys_test, A_test)

print("\n--- M_TL+phys permutation importance ---", flush=True)
imp_mtlphys, base_mtlphys = permutation_importance(m_tlphys, X_geo_test, X_phys_test, A_test)

# ========= Step 5: Report results =========
print("\n" + "=" * 90)
print("PERMUTATION IMPORTANCE: Structure A Physics Features")
print("=" * 90)

for model_name, imp, base in [("M_phys", imp_mphys, base_mphys),
                                ("M_TL+phys", imp_mtlphys, base_mtlphys)]:
    print(f"\n{'=' * 90}")
    print(f"  {model_name}  (baseline MAE = {base*100:.4f}%)")
    print(f"{'=' * 90}")

    means = imp.mean(axis=1) * 100  # convert to %
    stds = imp.std(axis=1) * 100
    ranked = np.argsort(-means)  # descending

    print(f"\n  {'Rank':>4s}  {'Feature':>25s}  {'dMAE (%)':>12s}  {'std (%)':>10s}  {'Category':<22s}")
    print(f"  {'-'*4:>4s}  {'-'*25:>25s}  {'-'*12:>12s}  {'-'*10:>10s}  {'-'*22:<22s}")

    # Build feature->category map
    feat_to_cat = {}
    for cat, idxs in CATEGORIES.items():
        for i in idxs:
            feat_to_cat[i] = cat

    for rank, fi in enumerate(ranked, 1):
        print(f"  {rank:4d}  {FEATURE_NAMES[fi]:>25s}  {means[fi]:+12.4f}  {stds[fi]:10.4f}  {feat_to_cat[fi]:<22s}")

    # Category-level importance
    print(f"\n  Category-level importance:")
    print(f"  {'Category':>25s}  {'Mean dMAE (%)':>14s}  {'Sum dMAE (%)':>14s}")
    print(f"  {'-'*25:>25s}  {'-'*14:>14s}  {'-'*14:>14s}")

    cat_means = {}
    cat_sums = {}
    for cat, idxs in CATEGORIES.items():
        cat_mean = means[idxs].mean()
        cat_sum = means[idxs].sum()
        cat_means[cat] = cat_mean
        cat_sums[cat] = cat_sum

    for cat in sorted(cat_sums, key=cat_sums.get, reverse=True):
        print(f"  {cat:>25s}  {cat_means[cat]:+14.4f}  {cat_sums[cat]:+14.4f}")

# ========= Step 6: Save results =========
savepath = "results/feature_importance_A.npz"
# Build a flat category_id array: feature_idx -> category_idx
cat_id = np.zeros(17, dtype=np.int32)
cat_names_list = list(CATEGORIES.keys())
for ci, (cat, idxs) in enumerate(CATEGORIES.items()):
    for i in idxs:
        cat_id[i] = ci
np.savez(savepath,
         imp_mphys=imp_mphys,
         imp_mtlphys=imp_mtlphys,
         baseline_mphys=base_mphys,
         baseline_mtlphys=base_mtlphys,
         feature_names=FEATURE_NAMES,
         category_names=cat_names_list,
         category_id=cat_id,
         n_permutations=N_PERMUTATIONS)
print(f"\nResults saved to: {savepath}", flush=True)
print("Done!", flush=True)
