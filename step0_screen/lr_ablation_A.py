#!/usr/bin/env python3
"""
Learning Rate Ablation for PBTL paper - Structure A.

Tests whether the TL benefit is simply from using a different learning rate.

4 conditions:
  1. M0       lr=1e-3  (original scratch setting)
  2. M0       lr=3e-4  (same lr as pre-trained fine-tuning)
  3. M_TL     lr=3e-4  (original TL setting)
  4. M_TL     lr=1e-3  (same lr as scratch)

Structure A only, n=50,100,200,350, seeds=42,123,777,321,456.
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

device = torch.device("mps")
print(f"Device: {device}", flush=True)

# ========= Model definitions =========
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


def train_model(model, dl_tr, dl_vl, epochs, lr):
    """Train model, return best val state dict."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None

    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
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


def eval_model(model, dl_te):
    """Evaluate model, return MAE on absorption."""
    model.eval()
    with torch.no_grad():
        te_loss = 0; te_n = 0
        for batch in dl_te:
            x, a, r = batch
            out = model(x)
            te_loss += nn.functional.l1_loss(out["A"], a, reduction="sum").item()
            te_n += len(a)
    return te_loss / te_n


# ========= Step 1: Generate TMM data (needed for pre-training normalization stats) =========
print("\n=== Step 1: Generate TMM data (for normalization) ===", flush=True)
N_TMM = 5000
wavelengths_rcwa = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths_rcwa)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

# We need to generate TMM data to get consistent normalization with the pre-trained model
t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths_rcwa, "Cr")
print(f"TMM generation: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

# Prepare normalization
params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths_rcwa - wavelengths_rcwa.min()) / (wavelengths_rcwa.max() - wavelengths_rcwa.min())
geo_dim = 1 + 10  # wavelength + 10 params

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

# Prepare RCWA tensors
params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_rcwa = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep_rcwa = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep_rcwa, params_rep_rcwa], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)

def get_rows(si):
    return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in si])

# Fixed test/val split (same as original)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 50
N_VAL = 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST + N_VAL)]

test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)

def make_dl(rows, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_te = make_dl(test_rows)
dl_vl = make_dl(val_rows)

# ========= Step 3: LR Ablation =========
print("\n=== Step 3: LR Ablation Experiment ===", flush=True)
TRAIN_SIZES = [50, 100, 200, 350]
SEEDS = [42, 123, 777, 321, 456]
FINETUNE_EPOCHS = 1000
PRETRAINED_PATH = "results/pretrained_m0_tmm.pt"

# 4 conditions
CONDITIONS = [
    ("M0_lr1e-3",  False, 1e-3),   # original scratch
    ("M0_lr3e-4",  False, 3e-4),   # scratch with TL's lr
    ("MTL_lr3e-4", True,  3e-4),   # original TL
    ("MTL_lr1e-3", True,  1e-3),   # TL with scratch's lr
]

# results[n_train][condition_name] = list of MAEs
results = {sz: {c[0]: [] for c in CONDITIONS} for sz in TRAIN_SIZES}

total_runs = len(TRAIN_SIZES) * len(SEEDS) * len(CONDITIONS)
run_count = 0

for n_train in TRAIN_SIZES:
    if n_train > len(remaining):
        print(f"Skipping n={n_train}, not enough data", flush=True)
        continue
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)

        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:n_train]]
        tr_rows = get_rows(tr_idx)
        dl_tr = make_dl(tr_rows, bs=512, shuffle=True)

        for cond_name, use_pretrained, lr in CONDITIONS:
            run_count += 1
            set_seed(seed)
            model = M0(geo_dim).to(device)

            if use_pretrained:
                model.load_state_dict(torch.load(PRETRAINED_PATH,
                                                  map_location=device, weights_only=True))

            model = train_model(model, dl_tr, dl_vl, FINETUNE_EPOCHS, lr)
            mae = eval_model(model, dl_te)
            results[n_train][cond_name].append(mae)
            print(f"  [{run_count}/{total_runs}] {cond_name:>14s}: {mae*100:.3f}%", flush=True)

# ========= Summary =========
print("\n" + "=" * 90, flush=True)
print("LR ABLATION: Structure A  (test MAE %)", flush=True)
print("=" * 90, flush=True)
header = f"{'n':>6}"
for c, _, _ in CONDITIONS:
    header += f" | {c:>16s}"
print(header, flush=True)
print("-" * 90, flush=True)

for sz in TRAIN_SIZES:
    row = f"{sz:>6}"
    for cond_name, _, _ in CONDITIONS:
        v = np.array(results[sz][cond_name]) * 100
        row += f" | {v.mean():>7.2f}+-{v.std():>5.2f}"
    print(row, flush=True)

# Interpretation
print("\n--- Key comparisons ---", flush=True)
for sz in TRAIN_SIZES:
    m0_orig = np.mean(results[sz]["M0_lr1e-3"]) * 100
    m0_low  = np.mean(results[sz]["M0_lr3e-4"]) * 100
    tl_orig = np.mean(results[sz]["MTL_lr3e-4"]) * 100
    tl_high = np.mean(results[sz]["MTL_lr1e-3"]) * 100
    print(f"  n={sz}:", flush=True)
    print(f"    M0  lr=1e-3: {m0_orig:.2f}%  vs  M0  lr=3e-4: {m0_low:.2f}%  (LR effect on scratch: {m0_orig-m0_low:+.2f}%)", flush=True)
    print(f"    MTL lr=3e-4: {tl_orig:.2f}%  vs  MTL lr=1e-3: {tl_high:.2f}%  (LR effect on TL: {tl_orig-tl_high:+.2f}%)", flush=True)
    print(f"    TL benefit (same lr=1e-3): M0={m0_orig:.2f}% vs MTL={tl_high:.2f}%  ({m0_orig-tl_high:+.2f}%)", flush=True)
    print(f"    TL benefit (same lr=3e-4): M0={m0_low:.2f}% vs MTL={tl_orig:.2f}%  ({m0_low-tl_orig:+.2f}%)", flush=True)

# Save
savepath = "results/lr_ablation_A.npz"
save_dict = {}
for sz in TRAIN_SIZES:
    for cond_name, _, _ in CONDITIONS:
        save_dict[f"{sz}_{cond_name}"] = results[sz][cond_name]
save_dict["train_sizes"] = TRAIN_SIZES
save_dict["seeds"] = SEEDS
save_dict["conditions"] = [c[0] for c in CONDITIONS]
np.savez(savepath, **save_dict)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
