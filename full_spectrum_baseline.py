"""Full-Spectrum ResNet baseline (Structure A) -- direct test of the
"per-wavelength architecture" choice flagged in the paper's Limitations.

Defends Major weakness #3:
  Reviewer concern: PBTL's win could come from implicit cross-wavelength
  regularization (different wavelengths share weights during pre-training)
  rather than from physics-informed weight transfer per se. To isolate the
  effect we re-run the M0 / M_TL comparison with a Full-Spectrum (FS)
  ResNet whose output is the entire 100-wavelength spectrum at once -- so
  cross-wavelength information is consumed by the same network without the
  wavelength index appearing in the input.

Architecture:
  per-wave   :  input = [norm_wl, norm_params(10)]   -> output = scalar A(lambda)
  full-spec  :  input = [norm_params(10)]            -> output = (100,) full spectrum

Both use ResNet-256 with 4 residual blocks; only the I/O dimensionality differs.
For pretraining, the LF network is trained on N_TMM=5000 TMM full spectra.
For fine-tune, n in {50, 100, 200, 350} RCWA samples, 3 seeds.

Outputs:
  results/full_spectrum_baseline.npz
  logs/fs_baseline.log
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/home/bigmountain87/mim_novel')
sys.path.insert(0, '/home/bigmountain87/PINN2/mim_novel')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
RCWA_FILE = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_500.npz'
da = np.load(RCWA_FILE, allow_pickle=True)
rcwa_params  = da['params'].astype(np.float32)
rcwa_spectra = da['A'].astype(np.float32)
wavelengths  = da['wavelengths'].astype(np.float32)
N_WL = rcwa_spectra.shape[1]
print(f'RCWA: {rcwa_params.shape}, spectra {rcwa_spectra.shape}', flush=True)

BOUNDS = np.array([[300, 600], [50, 540], [50, 540], [50, 540],
                   [10, 80],  [10, 80],  [5, 30],
                   [30, 200], [30, 200], [0, 45]], dtype=np.float32)
def norm_params(p):
    return (p - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.fc1 = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.fc2 = nn.Linear(d, d)
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc1(self.ln1(x)))
        h = self.fc2(self.ln2(h))
        return x + self.act(h)


class FullSpectrumResNet(nn.Module):
    """Input: [norm_params(10)]   ->   Output: (100,) full spectrum."""
    def __init__(self, in_dim=10, hidden=256, n_blocks=4, out_dim=100):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


# --------------------------------------------------------------------------
# TMM library generation (5000 samples to match PBTL pre-training budget)
# --------------------------------------------------------------------------
def make_tmm_library(n=5000, seed=0):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)
    lb, ub = BOUNDS[:, 0], BOUNDS[:, 1]
    params = rng.uniform(lb, ub, size=(n, 10)).astype(np.float32)
    spec = compute_tmm_batch(params, wavelengths)['A_tmm'].astype(np.float32)
    return params, spec


# --------------------------------------------------------------------------
# Pre-train FS network on TMM full spectra
# --------------------------------------------------------------------------
LF_WEIGHTS = '/home/bigmountain87/PINN2/mim_novel/results/fs_lf_5000.pt'

def pretrain_fs(epochs=400, lr=1e-3, batch=512):
    print('\n[Phase 1] Pre-training FS network on TMM library (5000 samples) ...',
          flush=True)
    params, spec = make_tmm_library(n=5000, seed=0)
    X = torch.from_numpy(norm_params(params)).to(DEVICE)
    Y = torch.from_numpy(spec).to(DEVICE)
    model = FullSpectrumResNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)
    lossfn = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                model.eval()
                mae = (model(X) - Y).abs().mean().item()
            print(f'    epoch {ep+1:3d}  TMM val MAE={mae*100:.3f}%  '
                  f'({time.time()-t0:.0f}s)', flush=True)
    torch.save(model.state_dict(), LF_WEIGHTS)
    print(f'  Saved LF weights -> {LF_WEIGHTS}', flush=True)
    return model


# --------------------------------------------------------------------------
# Fine-tune helpers
# --------------------------------------------------------------------------
def fine_tune(model, X_tr, Y_tr, X_te, Y_te, epochs=1000, lr=3e-4, batch=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=min(batch, X_tr.size(0)), shuffle=True)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return (model(X_te) - Y_te).abs().mean().item()


# --------------------------------------------------------------------------
# Phase 1: pre-train (load if cached)
# --------------------------------------------------------------------------
if os.path.exists(LF_WEIGHTS):
    base_lf = FullSpectrumResNet().to(DEVICE)
    base_lf.load_state_dict(torch.load(LF_WEIGHTS, map_location=DEVICE))
    print(f'[Phase 1] loaded FS LF weights from {LF_WEIGHTS}', flush=True)
else:
    base_lf = pretrain_fs()


# --------------------------------------------------------------------------
# Phase 2: 4-way training grid (M0_FS, M_TL_FS) x (3 seeds) x (4 sizes)
# --------------------------------------------------------------------------
SEEDS = [42, 123, 777]
SIZES = [50, 100, 200, 350]

results = {'M0_FS': np.zeros((len(SIZES), len(SEEDS))),
           'M_TL_FS': np.zeros((len(SIZES), len(SEEDS)))}

t_total = time.time()
for ni, n_train in enumerate(SIZES):
    for si, sd in enumerate(SEEDS):
        set_global_seed(sd)
        rng = np.random.RandomState(sd)
        perm = rng.permutation(len(rcwa_params))
        test_idx = perm[:50]
        tr_idx   = perm[100:100 + n_train]

        X_tr = torch.from_numpy(norm_params(rcwa_params[tr_idx])).to(DEVICE)
        Y_tr = torch.from_numpy(rcwa_spectra[tr_idx]).to(DEVICE)
        X_te = torch.from_numpy(norm_params(rcwa_params[test_idx])).to(DEVICE)
        Y_te = torch.from_numpy(rcwa_spectra[test_idx]).to(DEVICE)

        # M0_FS: scratch
        t0 = time.time()
        m0 = FullSpectrumResNet().to(DEVICE)
        mae_m0 = fine_tune(m0, X_tr, Y_tr, X_te, Y_te, epochs=1000, lr=1e-3)
        results['M0_FS'][ni, si] = mae_m0

        # M_TL_FS: pre-trained on TMM, fine-tune on RCWA
        mtl = deepcopy(base_lf)
        mae_mtl = fine_tune(mtl, X_tr, Y_tr, X_te, Y_te, epochs=1000, lr=3e-4)
        results['M_TL_FS'][ni, si] = mae_mtl

        tl_b = (1 - mae_mtl / mae_m0) * 100
        print(f'  n={n_train:>3d} seed={sd}: '
              f'M0_FS={mae_m0*100:.3f}%  M_TL_FS={mae_mtl*100:.3f}%  '
              f'[TL benefit {tl_b:+.1f}%]  ({time.time()-t0:.0f}s)', flush=True)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print('\n\n===== FULL-SPECTRUM RESNET vs PER-WAVELENGTH (Structure A) =====',
      flush=True)
print(f'{"n":>5} | {"M0_FS":>16} | {"M_TL_FS":>16} | {"TL benefit":>12}', flush=True)
print('-' * 65, flush=True)
for ni, n in enumerate(SIZES):
    m0a  = results['M0_FS'][ni]
    mtla = results['M_TL_FS'][ni]
    tlb  = (1 - mtla / m0a) * 100
    print(f'{n:>5} | {m0a.mean()*100:>5.2f}+/-{m0a.std()*100:.2f}% | '
          f'{mtla.mean()*100:>5.2f}+/-{mtla.std()*100:.2f}% | '
          f'{tlb.mean():>+5.1f}+/-{tlb.std():.1f}%', flush=True)

# Reference per-wavelength M0 / M_TL from main paper for direct comparison
print('\nReference (per-wavelength, paper Table 1):', flush=True)
print(f"  n=50:  M0=12.78+/-0.80%, M_TL= 9.23+/-0.55%  [TL benefit ~28%]", flush=True)
print(f"  n=100: M0=10.62+/-0.51%, M_TL= 7.82+/-0.23%  [TL benefit ~26%]", flush=True)
print(f"  n=200: M0= 8.19+/-0.39%, M_TL= 6.39+/-0.35%  [TL benefit ~22%]", flush=True)
print(f"  n=350: M0= 6.76+/-0.40%, M_TL= 4.92+/-0.15%  [TL benefit ~27%]", flush=True)

np.savez('/home/bigmountain87/PINN2/mim_novel/results/full_spectrum_baseline.npz',
         sizes=np.array(SIZES), seeds=np.array(SEEDS),
         m0_fs=results['M0_FS'], m_tl_fs=results['M_TL_FS'])
print(f'\nSaved: full_spectrum_baseline.npz', flush=True)
print(f'Total time: {(time.time()-t_total)/60:.1f} min', flush=True)
