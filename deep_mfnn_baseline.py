"""Deep Multi-Fidelity Neural Network Baseline (composite, Meng et al. 2020 style).

Differs from PBTL in HOW TMM information enters the model:
  - PBTL  : weight-level transfer (pretrain on TMM, fine-tune on RCWA;
            inputs are geometry only).
  - MF-NN : output-level transfer (LF network freezes after TMM pretraining;
            HF prediction is a *composite* of a linear term on the LF output
            and a nonlinear term on (geometry, LF output)).

Composite form used here (closely follows Meng, Babaee, Karniadakis 2020):
    y_H(x, lambda) = alpha(lambda) * y_L(x, lambda) + beta(lambda)
                   +  F_NL(x_norm, lambda, y_L(x, lambda))

where:
  * y_L is a frozen ResNet-256-4 pretrained on 5000 TMM samples
  * alpha, beta are per-wavelength learnable scalars (200 params total)
  * F_NL is a fresh ResNet-256-4 trained only on RCWA fine-tuning data

All other knobs (optimizer, schedule, batch, epochs) mirror the PBTL
fine-tuning protocol so the comparison isolates the multi-fidelity design.

Reduced config (3 seeds, 4 sizes) keeps wall-clock tractable while
providing standard-error estimates.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/home/bigmountain87/mim_novel')
sys.path.insert(0, '/home/bigmountain87/PINN2/mim_novel')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
RCWA_FILE = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_500.npz'
da = np.load(RCWA_FILE, allow_pickle=True)
rcwa_params = da['params'].astype(np.float32)        # [N, 10]
rcwa_spectra = da['A'].astype(np.float32)            # [N, 100]
wavelengths = da['wavelengths'].astype(np.float32)   # [100]
N_WL = rcwa_spectra.shape[1]

print(f'RCWA: {rcwa_params.shape} spectra={rcwa_spectra.shape}', flush=True)

print('Generating TMM companion spectra for the 500 RCWA params...', flush=True)
tmm_result = compute_tmm_batch(rcwa_params, wavelengths)
tmm_spectra_all = tmm_result['A_tmm'].astype(np.float32)  # [N, 100]
print(f'  TMM companion spectra: {tmm_spectra_all.shape}', flush=True)

BOUNDS_A = np.array([
    [300, 600], [50, 540], [50, 540], [50, 540],
    [10, 80],   [10, 80],  [5, 30],
    [30, 200],  [30, 200],
    [0, 45]
], dtype=np.float32)


def norm_params(p):
    return (p - BOUNDS_A[:, 0]) / (BOUNDS_A[:, 1] - BOUNDS_A[:, 0])


# --------------------------------------------------------------------------
# ResNet backbone (identical to PBTL)
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim); self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc1(self.ln1(x)))
        h = self.fc2(self.ln2(h))
        return x + self.act(h)


class ResNet256(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


# --------------------------------------------------------------------------
# Build training flat samples (per-wavelength row vectors)
# --------------------------------------------------------------------------
NORM_WL = np.linspace(0, 1, N_WL, dtype=np.float32)


def flatten_geom(params, tmm_spec):
    """Return flat inputs for LF eval and HF training.
    LF input: [norm_wl, norm_params] (11-D)
    HF input: [norm_wl, norm_params, y_L] (12-D)
    """
    n = len(params)
    np_norm = norm_params(params)
    lf_in = np.empty((n * N_WL, 11), dtype=np.float32)
    tmm_flat = np.empty(n * N_WL, dtype=np.float32)
    for i in range(n):
        base = i * N_WL
        lf_in[base:base + N_WL, 0] = NORM_WL
        lf_in[base:base + N_WL, 1:] = np_norm[i][None, :]
        tmm_flat[base:base + N_WL] = tmm_spec[i]
    return lf_in, tmm_flat


# --------------------------------------------------------------------------
# Phase 1 — pretrain a shared LF network on 5000 TMM samples
# (generate fresh TMM library with same bounds as PBTL; 1 seed is enough)
# --------------------------------------------------------------------------
PRETRAINED_LF = '/home/bigmountain87/PINN2/mim_novel/results/mfnn_lf.pt'

def pretrain_lf(n_samples=5000, epochs=500):
    print('\n[Phase 1] Pretraining LF network on fresh TMM library ...', flush=True)
    set_global_seed(0)
    lb = BOUNDS_A[:, 0]; ub = BOUNDS_A[:, 1]
    params = np.random.uniform(lb, ub, size=(n_samples, 10)).astype(np.float32)
    spec = compute_tmm_batch(params, wavelengths)['A_tmm'].astype(np.float32)
    lf_in, _ = flatten_geom(params, spec)
    y = spec.reshape(-1, 1)

    X = torch.from_numpy(lf_in).to(DEVICE)
    Y = torch.from_numpy(y).to(DEVICE)
    model = ResNet256(in_dim=11).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X, Y), batch_size=4096, shuffle=True)
    lossfn = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = lossfn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                model.eval()
                pred = model(X)
                mae = (pred - Y).abs().mean().item()
            print(f'    epoch {ep+1:3d}  train MAE={mae*100:.3f}%  ({time.time()-t0:.0f}s)', flush=True)
    torch.save(model.state_dict(), PRETRAINED_LF)
    print(f'  Saved LF weights -> {PRETRAINED_LF}', flush=True)
    return model


if os.path.exists(PRETRAINED_LF):
    lf = ResNet256(in_dim=11).to(DEVICE)
    lf.load_state_dict(torch.load(PRETRAINED_LF, map_location=DEVICE))
    print(f'[Phase 1] loaded pretrained LF from {PRETRAINED_LF}', flush=True)
else:
    lf = pretrain_lf()
lf.eval()
for p in lf.parameters():
    p.requires_grad_(False)

# --------------------------------------------------------------------------
# Phase 2 — composite HF training across (n, seed) grid
# --------------------------------------------------------------------------
SEEDS = [42, 123, 777]
SIZES = [50, 100, 200, 350]


class CompositeMFNN(nn.Module):
    """Composite MF-NN head.
        y_H = alpha[w] * y_L + beta[w] + NN(x_norm, norm_wl, y_L)
    """
    def __init__(self, n_wl=N_WL):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(n_wl))   # per-wavelength scale
        self.beta  = nn.Parameter(torch.zeros(n_wl))  # per-wavelength offset
        self.nl    = ResNet256(in_dim=12)             # nonlinear correction

    def forward(self, hf_in, wl_idx, y_L):
        lin = self.alpha[wl_idx] * y_L.squeeze(-1) + self.beta[wl_idx]
        nl  = self.nl(hf_in).squeeze(-1)
        return (lin + nl).unsqueeze(-1)


def run_hf(seed, n_train):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rcwa_params))
    test_idx = perm[:50]
    val_idx  = perm[50:100]
    tr_idx   = perm[100:100 + n_train]

    # Build LF outputs on entire subset
    def build_inputs(idx):
        p = rcwa_params[idx]; t = tmm_spectra_all[idx]
        lf_in, _ = flatten_geom(p, t)
        lf_in_t = torch.from_numpy(lf_in).to(DEVICE)
        with torch.no_grad():
            y_L = lf(lf_in_t).cpu().numpy().flatten()
        # HF input: [norm_wl, norm_params(10), y_L] = 12-D
        hf_in = np.concatenate([lf_in, y_L[:, None]], axis=1).astype(np.float32)
        wl_idx = np.tile(np.arange(N_WL), len(idx)).astype(np.int64)
        y_target = rcwa_spectra[idx].reshape(-1, 1).astype(np.float32)
        return (torch.from_numpy(hf_in).to(DEVICE),
                torch.from_numpy(wl_idx).to(DEVICE),
                torch.from_numpy(y_L).to(DEVICE),
                torch.from_numpy(y_target).to(DEVICE))

    tr = build_inputs(tr_idx)
    te = build_inputs(test_idx)

    model = CompositeMFNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    lossfn = nn.MSELoss()

    # batch over flat rows
    dataset = TensorDataset(tr[0], tr[1], tr[2], tr[3])
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    for ep in range(1000):
        model.train()
        for hf_b, wl_b, yL_b, yH_b in loader:
            pred = model(hf_b, wl_b, yL_b.unsqueeze(-1))
            loss = lossfn(pred, yH_b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(te[0], te[1], te[2].unsqueeze(-1))
        mae = (pred_te - te[3]).abs().mean().item()
    return mae


results = np.zeros((len(SIZES), len(SEEDS)))
t0 = time.time()
for ni, n in enumerate(SIZES):
    for si, sd in enumerate(SEEDS):
        tic = time.time()
        mae = run_hf(sd, n)
        results[ni, si] = mae
        print(f'  n={n:3d} seed={sd}: MAE={mae*100:.3f}%  ({time.time()-tic:.0f}s)', flush=True)

print('\n\n=== DEEP COMPOSITE MF-NN BASELINE (Structure A) ===')
print(f'{"n":>5} | {"MF-NN MAE":>22}')
print('-' * 35)
for ni, n in enumerate(SIZES):
    m = results[ni].mean() * 100
    s = results[ni].std() * 100
    print(f'{n:>5} | {m:>8.2f}+/-{s:.2f}% (seeds={len(SEEDS)})')

np.savez('/home/bigmountain87/PINN2/mim_novel/results/deep_mfnn_baseline.npz',
         sizes=np.array(SIZES), seeds=np.array(SEEDS), mae=results)
print('\nSaved: deep_mfnn_baseline.npz')
print(f'Total time: {(time.time()-t0)/60:.1f} min')
