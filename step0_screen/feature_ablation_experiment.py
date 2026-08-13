#!/usr/bin/env python3
"""
Feature ablation experiment: Effect of removing low-importance features.

Tests whether removing optical path features (< 2% importance) degrades performance.
If not, the feature set can be simplified.

Feature groups for Structure A (17 total):
  Cavity resonance: idx 0-3  (cos/sin SiO2, cos/sin TiO2)
  Fill fraction:    idx 4-5  (rect, sq)
  Sub-wavelength:   idx 6-8  (P/λ, Wx/λ, W2/λ)
  Skin depth:       idx 9-11 (t1/δ, t2/δ, t_mid/δ)
  Optical path:     idx 12-13 (n·d/λ for SiO2, TiO2)  ← target for removal
  Angle & geometry: idx 14-16 (cosθ, Wy/Wx, α_metal)

Conditions:
  1. All 17 features (baseline)
  2. Remove optical path (15 features)
  3. Only cavity resonance + fill fraction (6 features, top ~58%)
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Feature index masks
FEATURE_SETS = {
    "all_17": list(range(17)),
    "no_optpath_15": [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16],  # remove 12,13
    "top_6_only": [0,1,2,3,4,5],  # cavity resonance + fill fraction
}

# ========= Model =========
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

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd + pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x, p], -1))
        R = self.head(h).squeeze(-1)
        return {"A": 1-R, "R": R}

class M0(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1-R, "R": R}

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
        if (ep+1) % 100 == 0:
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
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te, has_phys=False):
    model.eval()
    with torch.no_grad():
        te_loss = 0; te_n = 0
        for batch in dl_te:
            if has_phys:
                x, a, r, p = batch
                out = model(x, p=p)
            else:
                x, a, r = batch
                out = model(x)
            te_loss += nn.functional.l1_loss(out["A"], a, reduction="sum").item()
            te_n += len(a)
    return te_loss / te_n

# ========= Data =========
print("\n=== Loading data ===", flush=True)
wavelengths = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths)
_, bounds_min, bounds_max = get_bounds("A")
geo_dim = 1 + 10
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

# RCWA data
data = np.load("data/raw/struct_A_vis_500.npz", allow_pickle=True)
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)

good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_rcwa, A_rcwa, R_rcwa = params_rcwa[gi], A_rcwa[gi], R_rcwa[gi]
N_rcwa = len(gi)
print(f"RCWA: {N_rcwa} samples", flush=True)

# Compute all physics features
phys_rcwa_all = compute_physics_features_A(params_rcwa, wavelengths, "Cr")  # (N, Nlam, 17)
phys_rcwa_flat = phys_rcwa_all.reshape(-1, 17)

# Prepare geometry
params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)

# TMM data for pre-training
print("Generating TMM data...", flush=True)
N_TMM = 5000
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)
tmm_out = compute_tmm_batch(params_tmm, wavelengths, "Cr")
A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)

phys_tmm_all = compute_physics_features_A(params_tmm, wavelengths, "Cr")
phys_tmm_flat = phys_tmm_all.reshape(-1, 17)

params_tmm_norm = normalize_params(params_tmm, "A")
params_rep_t = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep_t = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep_t, params_rep_t], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_tmm_flat = A_tmm.reshape(-1)
R_tmm_flat = R_tmm.reshape(-1)

# Split
def get_rows(si):
    return np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in si])

rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
test_idx = all_idx[-50:]
val_idx = all_idx[-100:-50]
remaining = all_idx[:-100]
test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)

# TMM split
n_tmm_train = int(N_TMM * 0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr_rows = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in tmm_idx[:n_tmm_train]])
tmm_vl_rows = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in tmm_idx[n_tmm_train:]])

# ========= Run experiment for each feature set =========
TRAIN_SIZES = [50, 100, 200, 350]
SEEDS = [42, 123, 777, 321, 456]  # 5 seeds
FT_EP = 1000
FT_LR = 1e-3
PRETRAIN_EP = 500; PRETRAIN_LR = 1e-3; FT_LR_TL = 3e-4

all_results = {}

for fs_name, fs_idx in FEATURE_SETS.items():
    n_feat = len(fs_idx)
    print(f"\n{'='*80}", flush=True)
    print(f"Feature set: {fs_name} ({n_feat} features)", flush=True)
    print(f"{'='*80}", flush=True)

    # Select features and compute normalization from TMM data
    phys_tmm_sel = phys_tmm_flat[:, fs_idx]
    phys_mean = phys_tmm_sel.mean(0, keepdims=True)
    phys_std = phys_tmm_sel.std(0, keepdims=True) + 1e-8
    phys_tmm_n = ((phys_tmm_sel - phys_mean) / phys_std).astype(np.float32)

    phys_rcwa_sel = phys_rcwa_flat[:, fs_idx]
    phys_rcwa_n = ((phys_rcwa_sel - phys_mean) / phys_std).astype(np.float32)

    # TMM dataloaders
    def tmm_dl(rows, bs=2048, shuffle=False):
        xg = torch.tensor(X_geo_tmm[rows]).to(device)
        a = torch.tensor(A_tmm_flat[rows]).to(device)
        r = torch.tensor(R_tmm_flat[rows]).to(device)
        p = torch.tensor(phys_tmm_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)

    dl_tmm_tr = tmm_dl(tmm_tr_rows, shuffle=True)
    dl_tmm_vl = tmm_dl(tmm_vl_rows)

    # Pre-train M_TL+phys
    print("Pre-training...", flush=True)
    set_seed(42)
    pt_model = MPhys(geo_dim, n_feat).to(device)
    pt_model = train_model(pt_model, dl_tmm_tr, dl_tmm_vl, PRETRAIN_EP, PRETRAIN_LR, has_phys=True)
    pt_path = f"results/pt_{fs_name}.pt"
    torch.save(pt_model.state_dict(), pt_path)
    mae_pt = eval_model(pt_model, dl_tmm_vl, has_phys=True)
    print(f"Pre-train val MAE: {mae_pt*100:.2f}%", flush=True)

    # RCWA dataloaders
    def rcwa_dl(rows, bs=2048, shuffle=False):
        xg = torch.tensor(X_geo_rcwa[rows]).to(device)
        a = torch.tensor(A_rcwa_flat[rows]).to(device)
        r = torch.tensor(R_rcwa_flat[rows]).to(device)
        p = torch.tensor(phys_rcwa_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)

    dl_te_ph = rcwa_dl(test_rows)
    dl_vl_ph = rcwa_dl(val_rows)

    results_fs = {sz: {"M_phys": [], "M_TL+phys": []} for sz in TRAIN_SIZES}

    for n_train in TRAIN_SIZES:
        if n_train > len(remaining): continue
        for seed in SEEDS:
            rng2 = np.random.default_rng(seed)
            tr_idx = remaining[rng2.permutation(len(remaining))[:n_train]]
            tr_rows = get_rows(tr_idx)
            dl_tr_ph = rcwa_dl(tr_rows, bs=512, shuffle=True)

            # M_phys (scratch)
            set_seed(seed)
            mp = MPhys(geo_dim, n_feat).to(device)
            mp = train_model(mp, dl_tr_ph, dl_vl_ph, FT_EP, FT_LR, has_phys=True)
            mae = eval_model(mp, dl_te_ph, has_phys=True)
            results_fs[n_train]["M_phys"].append(mae)

            # M_TL+phys (pre-trained)
            set_seed(seed)
            mtp = MPhys(geo_dim, n_feat).to(device)
            mtp.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
            mtp = train_model(mtp, dl_tr_ph, dl_vl_ph, FT_EP, FT_LR_TL, has_phys=True)
            mae = eval_model(mtp, dl_te_ph, has_phys=True)
            results_fs[n_train]["M_TL+phys"].append(mae)

            print(f"  n={n_train}, seed={seed}: M_phys={results_fs[n_train]['M_phys'][-1]*100:.2f}%, M_TL+phys={results_fs[n_train]['M_TL+phys'][-1]*100:.2f}%", flush=True)

    all_results[fs_name] = results_fs

# ========= Summary =========
print("\n" + "="*90, flush=True)
print("FEATURE ABLATION RESULTS", flush=True)
print("="*90, flush=True)

# Also include M0 baseline (no physics features)
print("\n--- M0 baseline (no features) for reference ---", flush=True)
m0_results = {sz: [] for sz in TRAIN_SIZES}

def make_dl_m0(rows, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_te_m0 = make_dl_m0(test_rows)
dl_vl_m0 = make_dl_m0(val_rows)

for n_train in TRAIN_SIZES:
    if n_train > len(remaining): continue
    for seed in SEEDS:
        rng2 = np.random.default_rng(seed)
        tr_idx = remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows = get_rows(tr_idx)
        dl_tr_m0 = make_dl_m0(tr_rows, bs=512, shuffle=True)
        set_seed(seed)
        m0 = M0(geo_dim).to(device)
        m0 = train_model(m0, dl_tr_m0, dl_vl_m0, FT_EP, FT_LR, has_phys=False)
        mae = eval_model(m0, dl_te_m0, has_phys=False)
        m0_results[n_train].append(mae)
    v = np.array(m0_results[n_train]) * 100
    print(f"  n={n_train}: M0 = {v.mean():.2f}±{v.std():.2f}%", flush=True)

print("\n--- Feature set comparison ---", flush=True)
for fs_name in FEATURE_SETS:
    print(f"\n{fs_name} ({len(FEATURE_SETS[fs_name])} features):", flush=True)
    print(f"  {'n':>6} | {'M_phys':>14} | {'M_TL+phys':>14} | {'vs M0':>10}", flush=True)
    for sz in TRAIN_SIZES:
        mp = np.array(all_results[fs_name][sz]["M_phys"]) * 100
        mt = np.array(all_results[fs_name][sz]["M_TL+phys"]) * 100
        m0 = np.array(m0_results[sz]) * 100
        best = min(mp.mean(), mt.mean())
        impr = (1 - best / m0.mean()) * 100
        print(f"  {sz:>6} | {mp.mean():.2f}±{mp.std():.2f} | {mt.mean():.2f}±{mt.std():.2f} | {impr:>+.1f}%", flush=True)

# Save
savepath = "results/feature_ablation.npz"
save_dict = {"train_sizes": TRAIN_SIZES, "seeds": SEEDS}
for fs_name in all_results:
    for sz in TRAIN_SIZES:
        for k in all_results[fs_name][sz]:
            save_dict[f"{fs_name}_{sz}_{k}"] = all_results[fs_name][sz][k]
for sz in TRAIN_SIZES:
    save_dict[f"M0_{sz}"] = m0_results[sz]
np.savez(savepath, **save_dict)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
