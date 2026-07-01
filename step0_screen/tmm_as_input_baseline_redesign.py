#!/usr/bin/env python3
"""
TMM-as-Input Baseline Experiment.

REDESIGN variant: corrected RCWA data (struct_A_500_redesign.npz, unified 400-1800nm,
jc materials, adaptive order, c64). Changes vs tmm_as_input_baseline.py:
  (1) RCWA data -> struct_A_500_redesign.npz;
  (2) wavelength grid LOADED from the data file (not hardcoded) so the NN wavelength
      input, the TMM-as-input feature, the physics features, and the labels all align
      with the redesign label grid (unified 400-1800nm, 100pt);
  (3) MATERIAL_MODEL='jc' set explicitly BEFORE importing tmm modules (matches the
      measured Johnson&Christy constants used to generate the RCWA labels);
  (4) sample filter uses the disclosed `reliable` mask; A,R clipped to [0,1];
  (5) output -> results/tmm_as_input_baseline_redesign.npz (old results preserved),
      and output/sys-path corrected to ...
The METHOD (the 4-way M0 / M_phys / M_TMM_input / M_TL+phys comparison with TMM directly
appended as one extra input feature) is IDENTICAL to tmm_as_input_baseline.py and uses
compute_physics_features_A / compute_tmm_batch exactly as pbtl_A_redesign.py -- only
data/material/grid/filter/paths change.

Compares 4 approaches for incorporating TMM information:
  1. M0:          geometry + wavelength only (baseline)
  2. M_phys:      geometry + wavelength + 17 physics features
  3. M_TMM_input: geometry + wavelength + A_TMM(x, lambda) as 1 extra input
  4. M_TL+phys:   geometry + wavelength + 17 physics features, TMM pre-trained (PBTL)

The TMM-as-input approach is the simplest possible way to use TMM: directly
append the scalar TMM prediction as one additional input feature, trained
from scratch without any pre-training.

Structure A, n in {50, 100, 200, 350}, 5 seeds per condition.
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

# REDESIGN: select measured Johnson&Christy material constants BEFORE any tmm import.
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ========= Model =========
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

class SurrogateModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1 - R, "R": R}

def train_model(model, dl_tr, dl_vl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None
    # GPU-resident chunk loop (no per-sample DataLoader dispatch overhead).
    _ts = dl_tr.dataset.tensors; _N = _ts[0].shape[0]; bs = dl_tr.batch_size
    _vts = dl_vl.dataset.tensors; _vN = _vts[0].shape[0]; _vbs = dl_vl.batch_size
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(_N, device=_ts[0].device)
        for _i in range(0, _N, bs):
            idx = perm[_i:_i + bs]
            x, a, r = (t[idx] for t in _ts)
            out = model(x)
            loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0.0; vn = 0
                for _i in range(0, _vN, _vbs):
                    x, a, r = (t[_i:_i + _vbs] for t in _vts)
                    vl += nn.functional.l1_loss(model(x)["A"], a, reduction="sum").item()
                    vn += len(a)
                vm = vl / vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k: v.clone() for k, v in model.state_dict().items()}
    if best_st:
        model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te):
    model.eval()
    with torch.no_grad():
        te = 0.0; tn = 0
        _ts = dl_te.dataset.tensors; _N = _ts[0].shape[0]; bs = dl_te.batch_size
        for _i in range(0, _N, bs):
            x, a, r = (t[_i:_i + bs] for t in _ts)
            te += nn.functional.l1_loss(model(x)["A"], a, reduction="sum").item()
            tn += len(a)
    return te / tn

# ========= Data (REDESIGN: corrected struct_A_500_redesign.npz; grid loaded from file) =========
print("\n=== Loading RCWA data ===", flush=True)
RCWA_PATH = "data/raw/struct_A_500_redesign.npz"
data = np.load(RCWA_PATH, allow_pickle=True)
params_all = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)
wavelengths = data["wavelengths"].astype(np.float32)   # loaded grid: 400-1800nm, 100pt
Nlam = len(wavelengths)

# Disclosed reliability filter: keep samples physically reliable at ALL wavelengths;
# fall back to [0,1] box if the flag is absent. Clip A,R into [0,1] for training labels.
if "reliable" in data.files:
    good = data["reliable"].all(axis=1)
else:
    good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_all = params_all[gi]
A_rcwa = np.clip(A_rcwa[gi], 0, 1).astype(np.float32)
R_rcwa = np.clip(R_rcwa[gi], 0, 1).astype(np.float32)
N = len(gi)
print(f"Valid RCWA: {N} reliable samples (of {len(good)})", flush=True)
print(f"Grid (from data): {wavelengths.min():.0f}-{wavelengths.max():.0f}nm, {Nlam}pts", flush=True)

# Normalize
params_norm = normalize_params(params_all, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + 10  # wavelength + 10 params

# Build geometry input [N, Nlam, 11]
params_rep = np.repeat(params_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N, 1, 1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).astype(np.float32)  # (N, Nlam, 11)

# Compute TMM for ALL RCWA samples
print("Computing TMM spectra for all RCWA samples...", flush=True)
tmm_out = compute_tmm_batch(params_all, wavelengths, "Cr")
A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)  # (N, Nlam)

# Compute physics features (RAW; normalization is deferred until the TMM pretraining
# distribution stats are available -- see "Z-score normalize ..." below).
print("Computing physics features...", flush=True)
phys_feats = compute_physics_features_A(params_all, wavelengths, "Cr")  # (N, Nlam, 17)

# NOTE: the physics-feature z-score must NOT be fit on the held-out RCWA samples (that
# leaks test/val statistics into M_phys / M_TL+phys). Matching pbtl_A_redesign.py, the
# normalization stats are computed from the TMM pretraining distribution (phys_mean /
# phys_std, defined in the "TMM Pre-training" section) and applied to BOTH the RCWA and
# TMM physics features. We therefore build X_phys / X_phys_flat AFTER those stats exist.

# Build the TMM-as-input variant now (it needs no physics normalization):
# M0: X_geo (11-dim)
# M_TMM_input: X_geo + A_tmm (12-dim)
X_tmm_input = np.concatenate([X_geo, A_tmm[:, :, None]], axis=-1).astype(np.float32)  # (N, Nlam, 12)

# Flatten (X_phys_flat is built later, once TMM phys stats are known)
X_geo_flat = X_geo.reshape(-1, geo_dim)
X_tmm_flat = X_tmm_input.reshape(-1, geo_dim + 1)
A_flat = A_rcwa.reshape(-1)
R_flat = R_rcwa.reshape(-1)

# Fixed test/val split
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N)
N_TEST, N_VAL = 50, 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST + N_VAL)]

def get_rows(si):
    return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in si])

test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)

def make_dl(X_flat, rows, bs=16384, shuffle=False):
    x = torch.tensor(X_flat[rows]).to(device)
    a = torch.tensor(A_flat[rows]).to(device)
    r = torch.tensor(R_flat[rows]).to(device)
    return DataLoader(TensorDataset(x, a, r), batch_size=bs, shuffle=shuffle)

# ========= TMM Pre-training for M_TL+phys =========
print("\n=== TMM Pre-training ===", flush=True)
N_TMM = 5000
rng_tmm = np.random.default_rng(99)
_, bounds_min, bounds_max = get_bounds("A")
params_tmm = rng_tmm.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)
tmm_train_out = compute_tmm_batch(params_tmm, wavelengths, "Cr")
A_tmm_train = np.clip(tmm_train_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm_train = np.clip(tmm_train_out["R_tmm"], 0, 1).astype(np.float32)

# Physics features for TMM params. The z-score normalization stats are fit HERE on the
# TMM pretraining distribution (NOT on the held-out RCWA samples), matching how
# pbtl_A_redesign.py derives phys_mean_tmm / phys_std_tmm. The SAME stats are then applied
# to both the TMM physics features and the RCWA physics features (built just below), so no
# test/val statistics leak into M_phys / M_TL+phys.
phys_tmm = compute_physics_features_A(params_tmm, wavelengths, "Cr")
phys_mean = phys_tmm.mean(axis=(0, 1), keepdims=True)
phys_std = phys_tmm.std(axis=(0, 1), keepdims=True) + 1e-8
phys_tmm_norm = ((phys_tmm - phys_mean) / phys_std).astype(np.float32)

# Now normalize the RCWA physics features with the TMM stats and build the M_phys inputs.
phys_feats_norm = ((phys_feats - phys_mean) / phys_std).astype(np.float32)  # (N, Nlam, 17)
X_phys = np.concatenate([X_geo, phys_feats_norm], axis=-1).astype(np.float32)  # (N, Nlam, 28)
X_phys_flat = X_phys.reshape(-1, geo_dim + 17)

params_tmm_norm = normalize_params(params_tmm, "A")
params_rep_tmm = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep_tmm = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep_tmm, params_rep_tmm], axis=-1).astype(np.float32)
X_phys_tmm = np.concatenate([X_geo_tmm, phys_tmm_norm], axis=-1).reshape(-1, geo_dim + 17).astype(np.float32)

A_tmm_flat_tr = A_tmm_train.reshape(-1)
R_tmm_flat_tr = R_tmm_train.reshape(-1)

n_tmm_tr = int(N_TMM * 0.9)
tmm_idx = rng_tmm.permutation(N_TMM)
tmm_tr_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_idx[:n_tmm_tr]])
tmm_vl_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_idx[n_tmm_tr:]])

def make_tmm_dl(rows, bs=16384, shuffle=False):
    x = torch.tensor(X_phys_tmm[rows]).to(device)
    a = torch.tensor(A_tmm_flat_tr[rows]).to(device)
    r = torch.tensor(R_tmm_flat_tr[rows]).to(device)
    return DataLoader(TensorDataset(x, a, r), batch_size=bs, shuffle=shuffle)

dl_tmm_tr = make_tmm_dl(tmm_tr_rows, shuffle=True)
dl_tmm_vl = make_tmm_dl(tmm_vl_rows)

set_seed(42)
pretrained = SurrogateModel(geo_dim + 17).to(device)
pretrained = train_model(pretrained, dl_tmm_tr, dl_tmm_vl, 500, 1e-3)
pretrained_state = {k: v.clone() for k, v in pretrained.state_dict().items()}
print("Pre-training done.", flush=True)

# ========= Main Experiment =========
SEEDS = [42, 123, 777, 321, 456]
N_TRAINS = [50, 100, 200, 350]
EPOCHS = 1000

print("\n=== Main Experiment ===", flush=True)
results = {m: {n: [] for n in N_TRAINS}
           for m in ["M0", "M_phys", "M_TMM_input", "M_TL+phys"]}

for n_train in N_TRAINS:
    for seed in SEEDS:
        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:n_train]]
        tr_rows = get_rows(tr_idx)

        for model_name, X_flat, in_dim, use_pretrain, lr in [
            ("M0",          X_geo_flat,  geo_dim,      False, 1e-3),
            ("M_phys",      X_phys_flat, geo_dim + 17, False, 1e-3),
            ("M_TMM_input", X_tmm_flat,  geo_dim + 1,  False, 1e-3),
            ("M_TL+phys",   X_phys_flat, geo_dim + 17, True,  3e-4),
        ]:
            dl_tr = make_dl(X_flat, tr_rows, bs=4096, shuffle=True)
            dl_vl = make_dl(X_flat, val_rows)
            dl_te = make_dl(X_flat, test_rows)

            set_seed(seed)
            model = SurrogateModel(in_dim).to(device)
            if use_pretrain:
                model.load_state_dict(pretrained_state)
            model = train_model(model, dl_tr, dl_vl, EPOCHS, lr)
            mae = eval_model(model, dl_te)
            results[model_name][n_train].append(mae)

        print(f"  n={n_train}, seed={seed}: "
              f"M0={results['M0'][n_train][-1]*100:.2f}%, "
              f"Mphys={results['M_phys'][n_train][-1]*100:.2f}%, "
              f"TMM_input={results['M_TMM_input'][n_train][-1]*100:.2f}%, "
              f"TL+phys={results['M_TL+phys'][n_train][-1]*100:.2f}%",
              flush=True)

# ========= Summary =========
print(f"\n{'='*100}", flush=True)
print("TMM-AS-INPUT BASELINE COMPARISON: Structure A (REDESIGN), 5 seeds", flush=True)
print(f"{'='*100}", flush=True)
print(f"{'n':>5} | {'M0':>16} | {'M_phys':>16} | {'M_TMM_input':>16} | {'M_TL+phys':>16}", flush=True)
print("-" * 100, flush=True)

for n_train in N_TRAINS:
    row = f"{n_train:>5} |"
    for m in ["M0", "M_phys", "M_TMM_input", "M_TL+phys"]:
        vals = np.array(results[m][n_train]) * 100
        row += f" {vals.mean():>6.2f} +/- {vals.std():>4.2f} |"
    print(row, flush=True)

# TMM_input vs M0 improvement
print(f"\nRelative improvement of M_TMM_input over M0:", flush=True)
for n_train in N_TRAINS:
    m0 = np.mean(results["M0"][n_train])
    mtmm = np.mean(results["M_TMM_input"][n_train])
    imp = (m0 - mtmm) / m0 * 100
    print(f"  n={n_train}: {imp:.1f}%", flush=True)

# Save
savepath = "results/tmm_as_input_baseline_redesign.npz"
np.savez(savepath,
         n_trains=np.array(N_TRAINS),
         seeds=np.array(SEEDS),
         **{f"{m}_n{n}": np.array(results[m][n]) for m in results for n in N_TRAINS})
print(f"\nSaved: {savepath}")
print("Done!", flush=True)
