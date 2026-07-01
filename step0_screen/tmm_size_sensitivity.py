#!/usr/bin/env python3
"""
TMM Pre-training Data Size Sensitivity Experiment (PBTL Paper – W4).

Tests how the number of TMM pre-training samples affects transfer-learning
performance for M0 on Structure A.

Conditions:
  TMM sizes : [500, 1000, 2000, 5000, 10000]
  RCWA train: n=100 and n=350
  Seeds     : [42, 123, 777]
  Baseline  : M0 trained from scratch (no pre-training)
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
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("mps")
print(f"Device: {device}", flush=True)

# ======================== Model definitions ========================
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
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(),
                                  nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1 - R, "R": R}


# ======================== Training helpers ========================
def train_model(model, dl_tr, dl_vl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None

    for ep in range(epochs):
        model.train()
        for x, a, r in dl_tr:
            out = model(x)
            loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                for x, a, r in dl_vl:
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
    return model, best_vl


def eval_model(model, dl_te):
    model.eval()
    with torch.no_grad():
        te_loss = 0; te_n = 0
        for x, a, r in dl_te:
            out = model(x)
            te_loss += nn.functional.l1_loss(out["A"], a, reduction="sum").item()
            te_n += len(a)
    return te_loss / te_n


# ======================== Constants ========================
wavelengths = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths)
wl_norm = ((wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())).astype(np.float32)

_, bounds_min, bounds_max = get_bounds("A")
geo_dim = 1 + 10  # wavelength + 10 params

TMM_SIZES = [500, 1000, 2000, 5000, 10000]
RCWA_TRAIN_SIZES = [100, 350]
SEEDS = [42, 123, 777]

PRETRAIN_EPOCHS = 500
PRETRAIN_LR = 1e-3
FINETUNE_EPOCHS = 1000
FINETUNE_LR = 1e-3
FINETUNE_LR_TL = 3e-4


# ======================== Load RCWA data ========================
print("\n=== Loading RCWA data ===", flush=True)
data = np.load("data/raw/struct_A_vis_500.npz",
               allow_pickle=True)
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

params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_rcwa = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep_rcwa = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep_rcwa, params_rep_rcwa], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)

# Fixed test/val split (same as pbtl_A_10seed.py)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 50
N_VAL = 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST + N_VAL)]


def get_rows(si):
    return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in si])


test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)


def make_rcwa_dl(rows, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)


dl_te = make_rcwa_dl(test_rows)
dl_vl = make_rcwa_dl(val_rows)

# ======================== Baseline: M0 from scratch ========================
print("\n=== Baseline: M0 from scratch ===", flush=True)
baseline_results = {}  # key: (n_rcwa, seed) -> MAE

for n_train in RCWA_TRAIN_SIZES:
    for seed in SEEDS:
        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:n_train]]
        tr_rows = get_rows(tr_idx)
        dl_tr = make_rcwa_dl(tr_rows, bs=512, shuffle=True)

        set_seed(seed)
        m0 = M0(geo_dim).to(device)
        m0, _ = train_model(m0, dl_tr, dl_vl, FINETUNE_EPOCHS, FINETUNE_LR)
        mae = eval_model(m0, dl_te)
        baseline_results[(n_train, seed)] = mae
        print(f"  Baseline n={n_train} seed={seed}: MAE={mae*100:.3f}%", flush=True)
        del m0; torch.mps.empty_cache()

# ======================== TMM size sweep ========================
print("\n=== TMM Size Sensitivity Sweep ===", flush=True)

# Storage
tmm_gen_times = {}
pretrain_times = {}
tmm_val_maes = {}
# tl_results[(n_tmm, n_rcwa, seed)] -> MAE
tl_results = {}

for n_tmm in TMM_SIZES:
    print(f"\n--- N_TMM = {n_tmm} ---", flush=True)

    # Generate TMM data
    rng_tmm = np.random.default_rng(99)
    params_tmm = rng_tmm.uniform(bounds_min, bounds_max,
                                  (n_tmm, 10)).astype(np.float32)
    t0 = time.time()
    tmm_out = compute_tmm_batch(params_tmm, wavelengths, "Cr")
    tmm_gen_time = time.time() - t0
    tmm_gen_times[n_tmm] = tmm_gen_time
    print(f"  TMM gen: {n_tmm} samples in {tmm_gen_time:.1f}s", flush=True)

    A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
    R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)

    # Prepare TMM tensors
    params_tmm_norm = normalize_params(params_tmm, "A")
    params_rep_tmm = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
    wl_rep_tmm = np.tile(wl_norm[None, :, None], (n_tmm, 1, 1))
    X_geo_tmm = np.concatenate([wl_rep_tmm, params_rep_tmm],
                                axis=-1).reshape(-1, geo_dim).astype(np.float32)
    A_tmm_flat = A_tmm.reshape(-1)
    R_tmm_flat = R_tmm.reshape(-1)

    # TMM train/val split (90/10)
    rng_tmm2 = np.random.default_rng(99)
    n_tmm_train = int(n_tmm * 0.9)
    tmm_idx = rng_tmm2.permutation(n_tmm)
    tmm_tr_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam)
                                   for i in tmm_idx[:n_tmm_train]])
    tmm_vl_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam)
                                   for i in tmm_idx[n_tmm_train:]])

    def make_tmm_dl(rows, bs=2048, shuffle=False):
        xg = torch.tensor(X_geo_tmm[rows]).to(device)
        a = torch.tensor(A_tmm_flat[rows]).to(device)
        r = torch.tensor(R_tmm_flat[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

    dl_tmm_tr = make_tmm_dl(tmm_tr_rows, shuffle=True)
    dl_tmm_vl = make_tmm_dl(tmm_vl_rows)

    # Pre-train M0 on TMM
    set_seed(42)
    pretrained = M0(geo_dim).to(device)
    t0 = time.time()
    pretrained, _ = train_model(pretrained, dl_tmm_tr, dl_tmm_vl,
                                PRETRAIN_EPOCHS, PRETRAIN_LR)
    pt_time = time.time() - t0
    pretrain_times[n_tmm] = pt_time

    tmm_val_mae = eval_model(pretrained, dl_tmm_vl)
    tmm_val_maes[n_tmm] = tmm_val_mae
    print(f"  Pretrain: {pt_time:.0f}s  |  TMM val MAE: {tmm_val_mae*100:.3f}%",
          flush=True)

    # Save pretrained weights for this TMM size
    pt_weights = {k: v.clone().cpu() for k, v in pretrained.state_dict().items()}
    del pretrained; torch.mps.empty_cache()

    # Fine-tune on each RCWA size x seed
    for n_train in RCWA_TRAIN_SIZES:
        for seed in SEEDS:
            rng2 = np.random.default_rng(seed)
            perm = rng2.permutation(len(remaining))
            tr_idx = remaining[perm[:n_train]]
            tr_rows = get_rows(tr_idx)
            dl_tr = make_rcwa_dl(tr_rows, bs=512, shuffle=True)

            set_seed(seed)
            m_tl = M0(geo_dim).to(device)
            m_tl.load_state_dict({k: v.to(device) for k, v in pt_weights.items()})
            m_tl, _ = train_model(m_tl, dl_tr, dl_vl,
                                  FINETUNE_EPOCHS, FINETUNE_LR_TL)
            mae = eval_model(m_tl, dl_te)
            tl_results[(n_tmm, n_train, seed)] = mae
            print(f"    TL n_tmm={n_tmm} n_rcwa={n_train} seed={seed}: "
                  f"MAE={mae*100:.3f}%", flush=True)
            del m_tl; torch.mps.empty_cache()


# ======================== Summary ========================
print("\n" + "=" * 100, flush=True)
print("TMM PRE-TRAINING DATA SIZE SENSITIVITY", flush=True)
print("=" * 100, flush=True)

# Baseline summary
print("\nBaseline (no pre-training):", flush=True)
for n_train in RCWA_TRAIN_SIZES:
    vals = np.array([baseline_results[(n_train, s)] for s in SEEDS]) * 100
    print(f"  M0 from scratch, n_rcwa={n_train}: "
          f"{vals.mean():.3f} +/- {vals.std():.3f}%", flush=True)

# Main table
hdr = (f"{'N_TMM':>7} | {'TMM gen(s)':>10} | {'PT time(s)':>10} | "
       f"{'TMM val MAE':>12} | {'RCWA n=100':>14} | {'RCWA n=350':>14}")
print(f"\n{hdr}", flush=True)
print("-" * len(hdr), flush=True)

# Baseline row
bl_100 = np.array([baseline_results[(100, s)] for s in SEEDS]) * 100
bl_350 = np.array([baseline_results[(350, s)] for s in SEEDS]) * 100
print(f"{'scratch':>7} | {'---':>10} | {'---':>10} | {'---':>12} | "
      f"{bl_100.mean():.3f}+/-{bl_100.std():.3f}{'':>1} | "
      f"{bl_350.mean():.3f}+/-{bl_350.std():.3f}{'':>1}", flush=True)

for n_tmm in TMM_SIZES:
    gt = tmm_gen_times[n_tmm]
    pt = pretrain_times[n_tmm]
    tv = tmm_val_maes[n_tmm] * 100

    r100 = np.array([tl_results[(n_tmm, 100, s)] for s in SEEDS]) * 100
    r350 = np.array([tl_results[(n_tmm, 350, s)] for s in SEEDS]) * 100

    print(f"{n_tmm:>7} | {gt:>10.1f} | {pt:>10.0f} | "
          f"{tv:>11.3f}% | "
          f"{r100.mean():.3f}+/-{r100.std():.3f}{'':>1} | "
          f"{r350.mean():.3f}+/-{r350.std():.3f}{'':>1}", flush=True)

# Improvement over baseline
print("\nImprovement over M0-scratch (% relative MAE reduction):", flush=True)
for n_train in RCWA_TRAIN_SIZES:
    bl_mean = np.mean([baseline_results[(n_train, s)] for s in SEEDS])
    best_ntmm = None
    best_impr = -999
    for n_tmm in TMM_SIZES:
        tl_mean = np.mean([tl_results[(n_tmm, n_train, s)] for s in SEEDS])
        impr = (1 - tl_mean / bl_mean) * 100
        if impr > best_impr:
            best_impr = impr
            best_ntmm = n_tmm
        print(f"  n_rcwa={n_train}, n_tmm={n_tmm:>5}: {impr:+.1f}%", flush=True)
    print(f"  >>> Best for n_rcwa={n_train}: n_tmm={best_ntmm} ({best_impr:+.1f}%)",
          flush=True)

# ======================== Save ========================
save_data = {
    "tmm_sizes": np.array(TMM_SIZES),
    "rcwa_train_sizes": np.array(RCWA_TRAIN_SIZES),
    "seeds": np.array(SEEDS),
    "tmm_gen_times": np.array([tmm_gen_times[s] for s in TMM_SIZES]),
    "pretrain_times": np.array([pretrain_times[s] for s in TMM_SIZES]),
    "tmm_val_maes": np.array([tmm_val_maes[s] for s in TMM_SIZES]),
}

# Baseline MAEs: shape (len(RCWA_TRAIN_SIZES), len(SEEDS))
for i, n_train in enumerate(RCWA_TRAIN_SIZES):
    save_data[f"baseline_n{n_train}"] = np.array(
        [baseline_results[(n_train, s)] for s in SEEDS])

# TL MAEs: shape (len(TMM_SIZES), len(SEEDS)) per RCWA size
for n_train in RCWA_TRAIN_SIZES:
    for n_tmm in TMM_SIZES:
        save_data[f"tl_tmm{n_tmm}_rcwa{n_train}"] = np.array(
            [tl_results[(n_tmm, n_train, s)] for s in SEEDS])

savepath = "results/tmm_size_sensitivity.npz"
np.savez(savepath, **save_data)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
