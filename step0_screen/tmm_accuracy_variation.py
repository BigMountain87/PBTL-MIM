#!/usr/bin/env python3
"""
TMM Accuracy Variation Experiment for PBTL Paper.

Addresses the N=3 correlation weakness by artificially degrading TMM accuracy
for Structure A and showing the correlation holds within a single structure.

Noise levels applied to TMM spectra:
  Level 0: Original TMM (no noise) - baseline
  Level 1: Gaussian noise sigma=0.05
  Level 2: Gaussian noise sigma=0.10
  Level 3: Gaussian noise sigma=0.15
  Level 4: Gaussian noise sigma=0.20
  Level 5: Random spectra (sigma->inf, equivalent to M_rand)

For each noise level:
  - Compute TMM accuracy as Pearson correlation with RCWA ground truth
  - Pre-train model on 5000 noisy TMM samples (epochs=500, lr=1e-3)
  - Fine-tune on RCWA n=100 (3 seeds: 42, 123, 777)
  - Report test MAE

Plots TMM accuracy (x-axis) vs TL benefit (y-axis) with 6 data points.
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from scipy.stats import pearsonr

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
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


# ========= Step 1: Generate TMM data =========
print("\n=== Step 1: Generate TMM training data ===", flush=True)
N_TMM = 5000
wavelengths_rcwa = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths_rcwa)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths_rcwa, "Cr")
print(f"TMM generation: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

A_tmm_clean = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm_clean = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)
print(f"TMM A range: [{A_tmm_clean.min():.3f}, {A_tmm_clean.max():.3f}]", flush=True)

# Prepare geometry input features
params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths_rcwa - wavelengths_rcwa.min()) / (wavelengths_rcwa.max() - wavelengths_rcwa.min())
geo_dim = 1 + 10  # wavelength + 10 params

params_rep_tmm = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep, params_rep_tmm], axis=-1).reshape(-1, geo_dim).astype(np.float32)

# TMM train/val split
n_tmm_train = int(N_TMM * 0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr_sample_idx = tmm_idx[:n_tmm_train]
tmm_vl_sample_idx = tmm_idx[n_tmm_train:]
tmm_tr_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_tr_sample_idx])
tmm_vl_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_vl_sample_idx])

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

def make_rcwa_dl(rows, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_te = make_rcwa_dl(test_rows)
dl_vl = make_rcwa_dl(val_rows)

# ========= Step 3: Compute TMM predictions for test set (for accuracy measurement) =========
print("\n=== Step 3: Compute TMM for test set parameters ===", flush=True)
params_test = params_rcwa[test_idx]
A_rcwa_test = A_rcwa[test_idx]  # (50, 100) ground truth

tmm_test_out = compute_tmm_batch(params_test, wavelengths_rcwa, "Cr")
A_tmm_test_clean = np.clip(tmm_test_out["A_tmm"], 0, 1).astype(np.float32)
print(f"Test TMM computed: {A_tmm_test_clean.shape}", flush=True)

# ========= Step 4: Define noise levels =========
NOISE_LEVELS = {
    "Level 0 (sigma=0.00)": 0.00,
    "Level 1 (sigma=0.05)": 0.05,
    "Level 2 (sigma=0.10)": 0.10,
    "Level 3 (sigma=0.15)": 0.15,
    "Level 4 (sigma=0.20)": 0.20,
    "Level 5 (random)": None,  # fully random
}

SEEDS = [42, 123, 777]
N_TRAIN_RCWA = 100
PRETRAIN_EPOCHS = 500
PRETRAIN_LR = 1e-3
FINETUNE_EPOCHS = 1000
FINETUNE_LR = 1e-3
FINETUNE_LR_TL = 3e-4

# ========= Step 4b: M0 baseline (no pre-training) =========
print("\n=== Step 4b: M0 baseline (no pre-training) ===", flush=True)
m0_maes = []
for seed in SEEDS:
    rng2 = np.random.default_rng(seed)
    perm = rng2.permutation(len(remaining))
    tr_idx = remaining[perm[:N_TRAIN_RCWA]]
    tr_rows = get_rows(tr_idx)
    dl_tr = make_rcwa_dl(tr_rows, bs=512, shuffle=True)

    set_seed(seed)
    m0 = M0(geo_dim).to(device)
    m0 = train_model(m0, dl_tr, dl_vl, FINETUNE_EPOCHS, FINETUNE_LR)
    mae = eval_model(m0, dl_te)
    m0_maes.append(mae)
    print(f"  M0 seed={seed}: MAE={mae*100:.3f}%", flush=True)

m0_mean_mae = np.mean(m0_maes)
print(f"M0 baseline mean MAE: {m0_mean_mae*100:.3f}%", flush=True)

# ========= Step 5: Loop over noise levels =========
print("\n=== Step 5: Noise level experiments ===", flush=True)
results = {}

for level_name, sigma in NOISE_LEVELS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"  {level_name}", flush=True)
    print(f"{'='*60}", flush=True)

    noise_rng = np.random.default_rng(2024)

    # --- Create noisy TMM training data ---
    if sigma is None:
        # Fully random spectra in [0, 1]
        A_tmm_noisy = noise_rng.uniform(0, 1, A_tmm_clean.shape).astype(np.float32)
        R_tmm_noisy = noise_rng.uniform(0, 1, R_tmm_clean.shape).astype(np.float32)
    elif sigma == 0.0:
        A_tmm_noisy = A_tmm_clean.copy()
        R_tmm_noisy = R_tmm_clean.copy()
    else:
        A_noise = noise_rng.normal(0, sigma, A_tmm_clean.shape).astype(np.float32)
        R_noise = noise_rng.normal(0, sigma, R_tmm_clean.shape).astype(np.float32)
        A_tmm_noisy = np.clip(A_tmm_clean + A_noise, 0, 1)
        R_tmm_noisy = np.clip(R_tmm_clean + R_noise, 0, 1)

    # --- Compute TMM accuracy: Pearson correlation on test set ---
    if sigma is None:
        # Random spectra for test set too
        A_tmm_test_noisy = noise_rng.uniform(0, 1, A_tmm_test_clean.shape).astype(np.float32)
    elif sigma == 0.0:
        A_tmm_test_noisy = A_tmm_test_clean.copy()
    else:
        A_test_noise = noise_rng.normal(0, sigma, A_tmm_test_clean.shape).astype(np.float32)
        A_tmm_test_noisy = np.clip(A_tmm_test_clean + A_test_noise, 0, 1)

    corr, _ = pearsonr(A_tmm_test_noisy.flatten(), A_rcwa_test.flatten())
    print(f"  TMM accuracy (Pearson r): {corr:.4f}", flush=True)

    # --- Create TMM dataloaders with noisy data ---
    A_flat = A_tmm_noisy.reshape(-1)
    R_flat = R_tmm_noisy.reshape(-1)

    def make_tmm_dl(rows, bs=2048, shuffle=False):
        xg = torch.tensor(X_geo_tmm[rows]).to(device)
        a = torch.tensor(A_flat[rows]).to(device)
        r = torch.tensor(R_flat[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

    dl_tmm_tr = make_tmm_dl(tmm_tr_rows, shuffle=True)
    dl_tmm_vl = make_tmm_dl(tmm_vl_rows)

    # --- Pre-train on noisy TMM ---
    print(f"  Pre-training on noisy TMM...", flush=True)
    set_seed(42)
    pretrained = M0(geo_dim).to(device)
    t0 = time.time()
    pretrained = train_model(pretrained, dl_tmm_tr, dl_tmm_vl, PRETRAIN_EPOCHS, PRETRAIN_LR)
    pretrain_mae = eval_model(pretrained, dl_tmm_vl)
    print(f"  Pre-train val MAE: {pretrain_mae*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)

    pretrained_state = {k: v.clone() for k, v in pretrained.state_dict().items()}

    # --- Fine-tune on RCWA n=100 with 3 seeds ---
    tl_maes = []
    for seed in SEEDS:
        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:N_TRAIN_RCWA]]
        tr_rows = get_rows(tr_idx)
        dl_tr = make_rcwa_dl(tr_rows, bs=512, shuffle=True)

        set_seed(seed)
        m_tl = M0(geo_dim).to(device)
        m_tl.load_state_dict(pretrained_state)
        m_tl = train_model(m_tl, dl_tr, dl_vl, FINETUNE_EPOCHS, FINETUNE_LR_TL)
        mae = eval_model(m_tl, dl_te)
        tl_maes.append(mae)
        print(f"    Fine-tune seed={seed}: MAE={mae*100:.3f}%", flush=True)

    tl_mean_mae = np.mean(tl_maes)
    tl_benefit = (m0_mean_mae - tl_mean_mae) / m0_mean_mae * 100

    results[level_name] = {
        "sigma": sigma if sigma is not None else float('inf'),
        "tmm_accuracy": corr,
        "tl_maes": tl_maes,
        "tl_mean_mae": tl_mean_mae,
        "tl_benefit": tl_benefit,
    }
    print(f"  TL mean MAE: {tl_mean_mae*100:.3f}%, TL benefit: {tl_benefit:.1f}%", flush=True)

# ========= Summary =========
print("\n" + "=" * 90, flush=True)
print("TMM ACCURACY VARIATION EXPERIMENT: Structure A", flush=True)
print("=" * 90, flush=True)
print(f"M0 baseline (no pre-training, n=100): {m0_mean_mae*100:.3f}%", flush=True)
print(f"\n{'Level':<25} | {'Sigma':>8} | {'TMM Acc (r)':>12} | {'TL MAE (%)':>12} | {'TL Benefit (%)':>15}", flush=True)
print("-" * 90, flush=True)

tmm_accs = []
tl_benefits = []

for level_name, res in results.items():
    sigma_str = "inf" if res["sigma"] == float('inf') else f"{res['sigma']:.2f}"
    tl_mae_str = f"{res['tl_mean_mae']*100:.3f}"
    tl_maes_arr = np.array(res['tl_maes']) * 100
    tl_std_str = f"{tl_maes_arr.std():.3f}"
    print(f"{level_name:<25} | {sigma_str:>8} | {res['tmm_accuracy']:>12.4f} | {tl_mae_str:>5}+/-{tl_std_str:<5} | {res['tl_benefit']:>14.1f}%", flush=True)
    tmm_accs.append(res["tmm_accuracy"])
    tl_benefits.append(res["tl_benefit"])

# Pearson correlation across noise levels
tmm_accs = np.array(tmm_accs)
tl_benefits = np.array(tl_benefits)
meta_corr, meta_p = pearsonr(tmm_accs, tl_benefits)
print(f"\nPearson correlation (TMM accuracy vs TL benefit): r={meta_corr:.4f}, p={meta_p:.4f}", flush=True)
print(f"Number of data points: {len(tmm_accs)}", flush=True)

# ========= Save results =========
savepath = "results/tmm_accuracy_variation.npz"
np.savez(savepath,
         noise_sigmas=np.array([res["sigma"] for res in results.values()]),
         tmm_accuracies=tmm_accs,
         tl_benefits=tl_benefits,
         tl_maes=np.array([res["tl_maes"] for res in results.values()]),
         m0_maes=np.array(m0_maes),
         m0_mean_mae=m0_mean_mae,
         level_names=np.array(list(results.keys())),
         seeds=np.array(SEEDS),
         meta_pearson_r=meta_corr,
         meta_pearson_p=meta_p)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
