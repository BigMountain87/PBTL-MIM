#!/usr/bin/env python3
"""
PBTL (Physics-Based Transfer Learning) experiment for Structure A.

4-way comparison:
  M0:        geometry only, no pre-train
  M_phys:    geometry + physics features, no pre-train  
  M_TL:      geometry only, TMM pre-trained
  M_TL+phys: geometry + physics features, TMM pre-trained

Procedure:
  1. Generate 5000 TMM samples (cheap)
  2. Pre-train NN on TMM data
  3. Fine-tune on RCWA data (50, 100, 200, 350)
  4. Compare test MAE across all 4 methods
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

# REDESIGN variant: corrected RCWA data (struct_A_500_redesign.npz, unified 400-1800nm,
# jc materials, adaptive order, c64). Changes vs pbtl_A_10seed.py:
#  (1) wavelength grid LOADED from the data file (not hardcoded 380-780) so NN inputs,
#      TMM pretraining, and physics features all align with the redesign label grid;
#  (2) MATERIAL_MODEL='jc' set explicitly (matches the RCWA labels' materials);
#  (3) RCWA data -> *_redesign.npz; (4) sample filter uses the disclosed `reliable` mask;
#  (5) output -> *_redesign_10seed.npz (old results preserved).
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ========= Model definitions =========
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
    """Train model, return best val state dict."""
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
    """Evaluate model, return MAE on absorption."""
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


# ========= Step 1: Generate TMM data =========
print("\n=== Step 1: Generate TMM training data ===", flush=True)
N_TMM = 5000
# Grid from the REDESIGN data file so TMM-pretrain / physics-features / NN wavelength
# input all match the RCWA label grid exactly (unified 400-1800nm, 100pt).
RCWA_PATH = "data/raw/struct_A_500_redesign.npz"
wavelengths_rcwa = np.load(RCWA_PATH, allow_pickle=True)["wavelengths"].astype(np.float32)
Nlam = len(wavelengths_rcwa)
print(f"Grid (from data): {wavelengths_rcwa.min():.0f}-{wavelengths_rcwa.max():.0f}nm, {Nlam}pts", flush=True)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths_rcwa, "Cr")
print(f"TMM generation: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)
print(f"TMM A range: [{A_tmm.min():.3f}, {A_tmm.max():.3f}]", flush=True)

# Physics features for TMM data
phys_tmm = compute_physics_features_A(params_tmm, wavelengths_rcwa, "Cr")
n_phys = phys_tmm.shape[-1]

# Prepare TMM tensors
params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths_rcwa - wavelengths_rcwa.min()) / (wavelengths_rcwa.max() - wavelengths_rcwa.min())
geo_dim = 1 + 10  # wavelength + 10 params

params_rep_tmm = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep, params_rep_tmm], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_tmm = phys_tmm.reshape(-1, n_phys).astype(np.float32)

# Save TMM stats for normalization (will be used for RCWA data too)
phys_mean_tmm = X_phys_tmm.mean(0, keepdims=True)
phys_std_tmm = X_phys_tmm.std(0, keepdims=True) + 1e-8
X_phys_tmm_n = ((X_phys_tmm - phys_mean_tmm) / phys_std_tmm).astype(np.float32)

A_tmm_flat = A_tmm.reshape(-1)
R_tmm_flat = R_tmm.reshape(-1)

# TMM dataloaders (use 90% train, 10% val)
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

# ========= Step 2: SKIPPED (lr-control needs no pretrain; M0 is from scratch) =========
# Pretraining + pretrained_*.pt save are intentionally removed here: the control only
# trains M0 from scratch at lr=3e-4 and compares to the already-saved M_TL. (Step 1's
# TMM physics stats phys_mean_tmm/phys_std_tmm are still defined above for Step 3.)

# ========= Step 3: Load RCWA data =========
print("\n=== Step 3: Load RCWA data ===", flush=True)
data = np.load(RCWA_PATH, allow_pickle=True)
assert np.allclose(data["wavelengths"].astype(np.float32), wavelengths_rcwa), "grid mismatch!"
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)

# Disclosed filtering protocol: keep samples whose A is physically reliable at ALL
# wavelengths (reliable flag = A in [-TOL, 1+TOL]); fall back to [0,1] box if absent.
if "reliable" in data.files:
    good = data["reliable"].all(axis=1)
else:
    good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_rcwa = params_rcwa[gi]
A_rcwa = A_rcwa[gi]
R_rcwa = R_rcwa[gi]
N_rcwa = len(gi)
print(f"RCWA data: {N_rcwa} reliable samples (of {len(good)})", flush=True)

# Prepare RCWA tensors (using TMM phys stats for consistent normalization)
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
    return np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in si])

# Fixed test/val split
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 50
N_VAL = 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST+N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST+N_VAL)]

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
print("\n=== Step 4: lr-CONTROL — M0 at fine-tune lr=3e-4 (matched to M_TL) ===", flush=True)
# Codex flagged: M0 used lr=1e-3 while M_TL used lr=3e-4, confounding the TL benefit.
# This control re-trains ONLY M0 at lr=3e-4 (same as M_TL), reusing the IDENTICAL data
# and seed-42 split. We then compare against the already-saved M_TL (also 3e-4):
# if M_TL still beats M0@3e-4, the transfer benefit is NOT an lr artifact.
TRAIN_SIZES = [50, 100, 200, 350]
SEEDS = [42, 123, 777, 321, 456, 654, 999, 111, 222, 333]
FINETUNE_EPOCHS = 1000
FINETUNE_LR_TL = 3e-4

results = {sz: {"M0_lr3e4": []} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    if n_train > len(remaining):
        continue
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:n_train]]
        tr_rows = get_rows(tr_idx)
        dl_tr_m0 = make_dl(tr_rows, False, bs=512, shuffle=True)

        # M0 from scratch, but at lr=3e-4 (MATCHED to M_TL) — the control
        set_seed(seed)
        m0 = M0(geo_dim).to(device)
        m0 = train_model(m0, dl_tr_m0, dl_vl_m0, FINETUNE_EPOCHS, FINETUNE_LR_TL, has_phys=False)
        mae = eval_model(m0, dl_te_m0, has_phys=False)
        results[n_train]["M0_lr3e4"].append(mae)
        print(f"  M0_lr3e4:  {mae*100:.3f}%", flush=True)

# ========= Summary: M_TL(3e-4) vs M0(3e-4, control) — lr-matched =========
print("\n" + "="*80, flush=True)
print("lr-CONTROL: M_TL(3e-4) vs M0(3e-4) — Structure A", flush=True)
print("="*80, flush=True)
mtl = m0orig = None
try:
    saved = np.load("results/pbtl_A_redesign_10seed.npz", allow_pickle=True)
    mtl = {sz: np.array(saved[f"{sz}_M_TL"]) for sz in TRAIN_SIZES}
    m0orig = {sz: np.array(saved[f"{sz}_M0"]) for sz in TRAIN_SIZES}
except Exception as e:
    print(f"(saved M_TL load failed: {e})", flush=True)
print(f"{'n':>6} | {'M0@1e-3':>9} | {'M0@3e-4':>9} | {'M_TL@3e-4':>10} | {'lr-matched benefit':>19}", flush=True)
print("-"*80, flush=True)
for sz in TRAIN_SIZES:
    m0c = np.mean(results[sz]["M0_lr3e4"]) * 100
    if mtl is not None:
        m0o = np.mean(m0orig[sz]) * 100; mt = np.mean(mtl[sz]) * 100
        ben = (1 - mt / m0c) * 100   # benefit with lr held equal
        print(f"{sz:>6} | {m0o:>8.2f}% | {m0c:>8.2f}% | {mt:>9.2f}% | {ben:>+17.1f}%", flush=True)
    else:
        print(f"{sz:>6} | {'?':>9} | {m0c:>8.2f}% | {'?':>10} |", flush=True)

savepath = "results/pbtl_A_lrctrl.npz"
np.savez(savepath, **{f"{sz}_M0_lr3e4": results[sz]["M0_lr3e4"] for sz in TRAIN_SIZES},
         train_sizes=TRAIN_SIZES, seeds=SEEDS)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
print("Done!", flush=True)
