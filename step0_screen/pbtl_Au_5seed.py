#!/usr/bin/env python3
"""
PBTL 4-way comparison for Au/SiO2 Structure A.
Same methodology as pbtl_A_10seed.py but with Au optical constants.

Key differences from Cr version:
  - TMM uses metal="Au" (Drude-Lorentz for gold)
  - Physics features computed with Au optical constants
  - 5 seeds (instead of 10) due to smaller dataset
  - Au RCWA dataset: struct_A_Au_350.npz
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"Au/SiO2 PBTL experiment", flush=True)

# ========= Model definitions (identical to Cr version) =========
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
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1-R, "R": R}

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1))
        R = self.head(h).squeeze(-1)
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
    if best_st:
        model.load_state_dict(best_st)
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

# ========= Step 1: Generate Au TMM data =========
print("\n=== Step 1: Generate Au TMM training data ===", flush=True)
METAL = "Au"
N_TMM = 2000
wavelengths = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths, METAL)
print(f"Au TMM generation: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)
print(f"Au TMM A range: [{A_tmm.min():.3f}, {A_tmm.max():.3f}]", flush=True)

phys_tmm = compute_physics_features_A(params_tmm, wavelengths, METAL)
n_phys = phys_tmm.shape[-1]
print(f"Physics features: {n_phys}", flush=True)

# Prepare TMM tensors
params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
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

# TMM dataloaders
n_tmm_train = int(N_TMM * 0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr_rows = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in tmm_idx[:n_tmm_train]])
tmm_vl_rows = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in tmm_idx[n_tmm_train:]])

def to_dl(rows, has_phys, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_tmm[rows]).to(device)
    a = torch.tensor(A_tmm_flat[rows]).to(device)
    r = torch.tensor(R_tmm_flat[rows]).to(device)
    if has_phys:
        p = torch.tensor(X_phys_tmm_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_tmm_tr_m0 = to_dl(tmm_tr_rows, False, shuffle=True)
dl_tmm_vl_m0 = to_dl(tmm_vl_rows, False)
dl_tmm_tr_ph = to_dl(tmm_tr_rows, True, shuffle=True)
dl_tmm_vl_ph = to_dl(tmm_vl_rows, True)

# ========= Step 2: Pre-train on Au TMM =========
print("\n=== Step 2: Pre-train on Au TMM data ===", flush=True)
PRETRAIN_EPOCHS = 500
PRETRAIN_LR = 1e-3

set_seed(42)
pretrained_m0 = M0(geo_dim).to(device)
t0 = time.time()
pretrained_m0 = train_model(pretrained_m0, dl_tmm_tr_m0, dl_tmm_vl_m0,
                            PRETRAIN_EPOCHS, PRETRAIN_LR, has_phys=False)
mae_pretrain_m0 = eval_model(pretrained_m0, dl_tmm_vl_m0, has_phys=False)
print(f"Pre-trained M0 on Au TMM: val MAE={mae_pretrain_m0*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)

set_seed(42)
pretrained_mphys = MPhys(geo_dim, n_phys).to(device)
t0 = time.time()
pretrained_mphys = train_model(pretrained_mphys, dl_tmm_tr_ph, dl_tmm_vl_ph,
                               PRETRAIN_EPOCHS, PRETRAIN_LR, has_phys=True)
mae_pretrain_ph = eval_model(pretrained_mphys, dl_tmm_vl_ph, has_phys=True)
print(f"Pre-trained MPhys on Au TMM: val MAE={mae_pretrain_ph*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)

torch.save(pretrained_m0.state_dict(), "results/pretrained_m0_au_tmm.pt")
torch.save(pretrained_mphys.state_dict(), "results/pretrained_mphys_au_tmm.pt")
print("Saved Au pre-trained weights", flush=True)

# ========= Step 3: Load Au RCWA data =========
print("\n=== Step 3: Load Au RCWA data ===", flush=True)
data = np.load("data/raw/struct_A_Au_350.npz", allow_pickle=True)
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)

good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_rcwa = params_rcwa[gi]
A_rcwa = A_rcwa[gi]
R_rcwa = R_rcwa[gi]
N_rcwa = len(gi)
print(f"Au RCWA data: {N_rcwa} good samples (out of {len(good)})", flush=True)

# Physics features with Au optical constants
phys_rcwa = compute_physics_features_A(params_rcwa, wavelengths, METAL)
params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_rcwa = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep_rcwa = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep_rcwa, params_rep_rcwa], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_rcwa = phys_rcwa.reshape(-1, n_phys).astype(np.float32)
X_phys_rcwa_n = ((X_phys_rcwa - phys_mean_tmm) / phys_std_tmm).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)

def get_rows(si):
    return np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in si])

# Fixed test/val split (smaller due to 350 total samples)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 40
N_VAL = 40
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST+N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST+N_VAL)]
print(f"Split: {len(remaining)} train pool, {N_VAL} val, {N_TEST} test", flush=True)

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

dl_te_m0 = make_dl(test_rows, False)
dl_te_ph = make_dl(test_rows, True)
dl_vl_m0 = make_dl(val_rows, False)
dl_vl_ph = make_dl(val_rows, True)

# ========= Step 4: 4-way comparison =========
print("\n=== Step 4: Au 4-way comparison ===", flush=True)
max_train = len(remaining)
TRAIN_SIZES = [sz for sz in [50, 100, 200] if sz <= max_train]
# Also add max available if > 200
if max_train > 200:
    TRAIN_SIZES.append(max_train)
print(f"Training sizes: {TRAIN_SIZES} (max available: {max_train})", flush=True)

SEEDS = [42, 123, 777, 321, 456]
FINETUNE_EPOCHS = 1000
FINETUNE_LR = 1e-3
FINETUNE_LR_TL = 3e-4

results = {sz: {"M0": [], "M_phys": [], "M_TL": [], "M_TL+phys": []} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    for seed in SEEDS:
        print(f"\n--- Au n_train={n_train}, seed={seed} ---", flush=True)

        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:n_train]]
        tr_rows = get_rows(tr_idx)

        dl_tr_m0 = make_dl(tr_rows, False, bs=512, shuffle=True)
        dl_tr_ph = make_dl(tr_rows, True, bs=512, shuffle=True)

        # --- M0: from scratch ---
        set_seed(seed)
        m0 = M0(geo_dim).to(device)
        m0 = train_model(m0, dl_tr_m0, dl_vl_m0, FINETUNE_EPOCHS, FINETUNE_LR, has_phys=False)
        mae = eval_model(m0, dl_te_m0, has_phys=False)
        results[n_train]["M0"].append(mae)
        print(f"  M0:        {mae*100:.3f}%", flush=True)

        # --- M_phys: from scratch ---
        set_seed(seed)
        mp = MPhys(geo_dim, n_phys).to(device)
        mp = train_model(mp, dl_tr_ph, dl_vl_ph, FINETUNE_EPOCHS, FINETUNE_LR, has_phys=True)
        mae = eval_model(mp, dl_te_ph, has_phys=True)
        results[n_train]["M_phys"].append(mae)
        print(f"  M_phys:    {mae*100:.3f}%", flush=True)

        # --- M_TL: pre-trained M0 on Au TMM ---
        set_seed(seed)
        m_tl = M0(geo_dim).to(device)
        m_tl.load_state_dict(torch.load("results/pretrained_m0_au_tmm.pt",
                                         map_location=device, weights_only=True))
        m_tl = train_model(m_tl, dl_tr_m0, dl_vl_m0, FINETUNE_EPOCHS, FINETUNE_LR_TL, has_phys=False)
        mae = eval_model(m_tl, dl_te_m0, has_phys=False)
        results[n_train]["M_TL"].append(mae)
        print(f"  M_TL:      {mae*100:.3f}%", flush=True)

        # --- M_TL+phys: pre-trained MPhys on Au TMM ---
        set_seed(seed)
        m_tlp = MPhys(geo_dim, n_phys).to(device)
        m_tlp.load_state_dict(torch.load("results/pretrained_mphys_au_tmm.pt",
                                          map_location=device, weights_only=True))
        m_tlp = train_model(m_tlp, dl_tr_ph, dl_vl_ph, FINETUNE_EPOCHS, FINETUNE_LR_TL, has_phys=True)
        mae = eval_model(m_tlp, dl_te_ph, has_phys=True)
        results[n_train]["M_TL+phys"].append(mae)
        print(f"  M_TL+phys: {mae*100:.3f}%", flush=True)

# ========= Also compute TMM-RCWA fidelity for Au =========
print("\n=== TMM-RCWA fidelity (Au) ===", flush=True)
au_tmm_eval = compute_tmm_batch(params_rcwa[:min(200, N_rcwa)], wavelengths, METAL)
A_tmm_eval = np.clip(au_tmm_eval["A_tmm"], 0, 1)
A_rcwa_eval = A_rcwa[:min(200, N_rcwa)]

# Per-sample Pearson correlation
correlations = []
for i in range(len(A_tmm_eval)):
    r = np.corrcoef(A_tmm_eval[i], A_rcwa_eval[i])[0, 1]
    if not np.isnan(r):
        correlations.append(r)
correlations = np.array(correlations)
tmm_rcwa_mae = np.mean(np.abs(A_tmm_eval - A_rcwa_eval)) * 100

print(f"Au TMM-RCWA spectral MAE: {tmm_rcwa_mae:.1f}%", flush=True)
print(f"Au TMM-RCWA mean r: {correlations.mean():.3f}", flush=True)
print(f"Au TMM-RCWA median r: {np.median(correlations):.3f}", flush=True)

# ========= Summary =========
print("\n" + "="*80, flush=True)
print("Au PBTL 4-WAY COMPARISON: Structure A", flush=True)
print("="*80, flush=True)
print(f"{'n':>6} | {'M0':>14} | {'M_phys':>14} | {'M_TL':>14} | {'M_TL+phys':>14}", flush=True)
print("-"*80, flush=True)

for sz in TRAIN_SIZES:
    vals = {}
    for key in ["M0", "M_phys", "M_TL", "M_TL+phys"]:
        v = np.array(results[sz][key]) * 100
        vals[key] = f"{v.mean():.2f}±{v.std():.2f}"
    print(f"{sz:>6} | {vals['M0']:>14} | {vals['M_phys']:>14} | {vals['M_TL']:>14} | {vals['M_TL+phys']:>14}", flush=True)

print("\n--- Relative improvement over M0 ---", flush=True)
for sz in TRAIN_SIZES:
    m0_mean = np.mean(results[sz]["M0"])
    tlp_mean = np.mean(results[sz]["M_TL+phys"])
    impr = (1 - tlp_mean/m0_mean) * 100
    print(f"  n={sz}: M_TL+phys improvement over M0 = {impr:.1f}%", flush=True)

# Save
savepath = "results/pbtl_Au_5seed.npz"
save_dict = {f"{sz}_{k}": results[sz][k] for sz in TRAIN_SIZES for k in results[sz]}
save_dict["train_sizes"] = TRAIN_SIZES
save_dict["seeds"] = SEEDS
save_dict["tmm_rcwa_mae"] = tmm_rcwa_mae
save_dict["tmm_rcwa_mean_r"] = correlations.mean()
save_dict["tmm_rcwa_median_r"] = np.median(correlations)
np.savez(savepath, **save_dict)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
