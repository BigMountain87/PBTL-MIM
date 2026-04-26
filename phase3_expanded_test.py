"""Phase 3 -- Expanded test-set evaluation (Structure A, Cr).

Defends Major weakness #5:
  Original test set was 50 samples per seed -> bootstrap 95% CI on MAE
  is +/- 0.3-0.5 pp at ~5% MAE, comparable to some inter-model differences.
  We re-evaluate the key M0 vs M_TL+phys comparison on a 250-sample test
  set (50 original + 200 freshly generated independent RCWA samples),
  shrinking CI by sqrt(5) ~ 2.24x to roughly +/- 0.13-0.22 pp.

Protocol:
  - Combine struct_A_vis_500.npz + struct_A_vis_extra200.npz -> 700 RCWA total.
  - Per seed: fix the same 250-sample test set (50 from original + all 200 new),
    use the remaining 450 as the training pool (300 train candidates after
    excluding 50 val + 100 idx_pool offset to match paper splits).
  - Train M0 and M_TL+phys at n in {100, 350} with 3 seeds.
  - For M_TL+phys, pre-train ResNet on 5,000 fresh Cr TMM samples (matching
    paper Section sec:tmm_size).
  - Report mean +/- std MAE and bootstrap 95% CI on each.

Outputs:
  results/phase3_expanded_test.npz
  logs/phase3.log
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
from scipy import stats

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch
from src.utils.physics_features import compute_physics_features_A

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)

# --------------------------------------------------------------------------
# Combine original + extra RCWA data
# --------------------------------------------------------------------------
ORIG = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_500.npz'
EXTRA = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_extra200.npz'
da_o = np.load(ORIG, allow_pickle=True)
da_e = np.load(EXTRA, allow_pickle=True)
params_orig  = da_o['params'].astype(np.float32)
A_orig       = da_o['A'].astype(np.float32)
params_extra = da_e['params'].astype(np.float32)
A_extra      = da_e['A'].astype(np.float32)
wavelengths  = da_o['wavelengths'].astype(np.float32)
N_WL = A_orig.shape[1]

print(f'Original: {params_orig.shape}, Extra: {params_extra.shape}', flush=True)

# Quality filter on extra (same protocol as paper)
bad = np.any(A_extra < -0.01, axis=1) | np.any(A_extra > 1.01, axis=1)
print(f'Extra quality filter: {(~bad).sum()}/{len(A_extra)} kept '
      f'({100*(~bad).sum()/len(A_extra):.1f}%)', flush=True)
params_extra = params_extra[~bad]
A_extra = A_extra[~bad]
N_extra = len(params_extra)

# Concatenate -> 700 (or fewer after filter) total
params_all = np.concatenate([params_orig, params_extra], axis=0)
A_all = np.concatenate([A_orig, A_extra], axis=0)
N_total = len(params_all)
print(f'Combined: {N_total} samples '
      f'(original 500 + extra {N_extra})', flush=True)

# --------------------------------------------------------------------------
# Bounds + normalisation
# --------------------------------------------------------------------------
BOUNDS = np.array([[300, 600], [50, 540], [50, 540], [50, 540],
                   [10, 80],  [10, 80],  [5, 30],
                   [30, 200], [30, 200], [0, 45]], dtype=np.float32)
def norm_params(p):
    return (p - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])

# --------------------------------------------------------------------------
# Architecture (per-wavelength, identical to paper main)
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


class ResNet256(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


NORM_WL = np.linspace(0, 1, N_WL, dtype=np.float32)
PHYS_DIM = None  # set after first feature compute


def build_inputs(params_norm_arr, phys_arr=None):
    n = len(params_norm_arr)
    if phys_arr is None:
        in_dim = 1 + 10
        X = np.empty((n * N_WL, in_dim), dtype=np.float32)
        for i in range(n):
            base = i * N_WL
            X[base:base + N_WL, 0] = NORM_WL
            X[base:base + N_WL, 1:] = params_norm_arr[i][None, :]
    else:
        in_dim = 1 + 10 + phys_arr.shape[-1]
        X = np.empty((n * N_WL, in_dim), dtype=np.float32)
        for i in range(n):
            base = i * N_WL
            X[base:base + N_WL, 0] = NORM_WL
            X[base:base + N_WL, 1:1 + 10] = params_norm_arr[i][None, :]
            X[base:base + N_WL, 1 + 10:] = phys_arr[i]
    return X


# --------------------------------------------------------------------------
# Pre-train M0 and M_phys on 5000 Cr TMM samples (cached)
# --------------------------------------------------------------------------
PT_M0   = '/home/bigmountain87/PINN2/mim_novel/results/phase3_pt_m0.pt'
PT_PHYS = '/home/bigmountain87/PINN2/mim_novel/results/phase3_pt_mphys.pt'


def make_tmm_library(n=5000, seed=0):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)
    lb, ub = BOUNDS[:, 0], BOUNDS[:, 1]
    p = rng.uniform(lb, ub, size=(n, 10)).astype(np.float32)
    spec = compute_tmm_batch(p, wavelengths)['A_tmm'].astype(np.float32)
    return p, spec


def pretrain(in_dim, X, y, epochs=400, lr=1e-3, batch=4096):
    model = ResNet256(in_dim=in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


print('\n[Phase 1] Building 5000-sample Cr TMM library and pre-training...', flush=True)
t_pre = time.time()
tmm_params, tmm_spec = make_tmm_library(n=5000, seed=0)
phys_tmm = compute_physics_features_A(tmm_params, wavelengths).astype(np.float32)
PHYS_DIM = phys_tmm.shape[-1]
print(f'  TMM library {tmm_spec.shape}, phys {phys_tmm.shape} (PHYS_DIM={PHYS_DIM})', flush=True)

X_tmm_geo  = torch.from_numpy(build_inputs(norm_params(tmm_params))).to(DEVICE)
X_tmm_phys = torch.from_numpy(build_inputs(norm_params(tmm_params), phys_tmm)).to(DEVICE)
y_tmm      = torch.from_numpy(tmm_spec.reshape(-1, 1)).to(DEVICE)

if os.path.exists(PT_M0) and os.path.exists(PT_PHYS):
    base_m0   = ResNet256(in_dim=1 + 10).to(DEVICE)
    base_m0.load_state_dict(torch.load(PT_M0, map_location=DEVICE))
    base_phys = ResNet256(in_dim=1 + 10 + PHYS_DIM).to(DEVICE)
    base_phys.load_state_dict(torch.load(PT_PHYS, map_location=DEVICE))
    print(f'  Loaded cached pre-trained weights', flush=True)
else:
    base_m0   = pretrain(in_dim=1 + 10,             X=X_tmm_geo,  y=y_tmm)
    base_phys = pretrain(in_dim=1 + 10 + PHYS_DIM, X=X_tmm_phys, y=y_tmm)
    torch.save(base_m0.state_dict(),   PT_M0)
    torch.save(base_phys.state_dict(), PT_PHYS)
    print(f'  Pre-training done in {time.time()-t_pre:.0f}s', flush=True)

with torch.no_grad():
    base_m0.eval(); base_phys.eval()
    pt_mae_m0   = (base_m0(X_tmm_geo)  - y_tmm).abs().mean().item()
    pt_mae_phys = (base_phys(X_tmm_phys) - y_tmm).abs().mean().item()
print(f'  TMM val MAE: M0={pt_mae_m0*100:.3f}%, M_phys={pt_mae_phys*100:.3f}%', flush=True)


# --------------------------------------------------------------------------
# Phase 2: M0 vs M_TL+phys at n in {100, 350} on 250-sample test set
# --------------------------------------------------------------------------
SEEDS = [42, 123, 777]
SIZES = [100, 350]

# Test set is fixed: 50 from original (idx 0-49 in seed permutation) + all 200 extra
# Build per-seed using original RCWA permutation but pinned to 50 specific original samples
EXTRA_TEST_IDX = np.arange(500, N_total)   # all extra samples in concat order


def train_and_eval(model_init, X_tr, y_tr, X_te, y_te, lr, epochs=1000, batch=2048):
    model = deepcopy(model_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=min(batch, X_tr.size(0)),
                        shuffle=True)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        # Per-test-sample MAE for bootstrap CI later
        N_te = X_te.size(0) // N_WL
        per_sample_abs = (model(X_te) - y_te).abs().view(N_te, N_WL).mean(dim=1).cpu().numpy()
    return float(per_sample_abs.mean()), per_sample_abs


# Init scratch model template (no pre-trained weights)
def fresh_m0():    return ResNet256(in_dim=1 + 10).to(DEVICE)
def fresh_mphys(): return ResNet256(in_dim=1 + 10 + PHYS_DIM).to(DEVICE)


results = {'M0': {n: [] for n in SIZES},
           'M_TL+phys': {n: [] for n in SIZES}}
per_sample = {'M0': {n: [] for n in SIZES},
              'M_TL+phys': {n: [] for n in SIZES}}

t_train = time.time()
for seed in SEEDS:
    print(f'\n[Seed] {seed}', flush=True)
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    perm_orig = rng.permutation(500)
    test_orig_idx = perm_orig[:50]                    # 50 from original (paper-style)
    test_full_idx = np.concatenate([test_orig_idx, EXTRA_TEST_IDX])  # 250 total
    val_idx       = perm_orig[50:100]
    pool_idx      = perm_orig[100:]                   # 400 candidates from original

    # Build test set tensors (250 samples)
    test_params = params_all[test_full_idx]
    test_specs  = A_all[test_full_idx]
    phys_test   = compute_physics_features_A(test_params, wavelengths).astype(np.float32)
    X_te_geo  = torch.from_numpy(build_inputs(norm_params(test_params))).to(DEVICE)
    X_te_phys = torch.from_numpy(build_inputs(norm_params(test_params), phys_test)).to(DEVICE)
    y_te      = torch.from_numpy(test_specs.reshape(-1, 1)).to(DEVICE)

    for n in SIZES:
        tr_idx = pool_idx[:n]
        tr_params = params_all[tr_idx]
        tr_specs  = A_all[tr_idx]
        phys_tr   = compute_physics_features_A(tr_params, wavelengths).astype(np.float32)
        X_tr_geo  = torch.from_numpy(build_inputs(norm_params(tr_params))).to(DEVICE)
        X_tr_phys = torch.from_numpy(build_inputs(norm_params(tr_params), phys_tr)).to(DEVICE)
        y_tr      = torch.from_numpy(tr_specs.reshape(-1, 1)).to(DEVICE)

        t0 = time.time()
        # M0: scratch geometry-only
        mae_m0, ps_m0 = train_and_eval(fresh_m0(), X_tr_geo, y_tr, X_te_geo, y_te,
                                        lr=1e-3, epochs=1000)
        # M_TL+phys: TMM pre-trained physics-augmented, fine-tune at lr=3e-4
        mae_mtlp, ps_mtlp = train_and_eval(base_phys, X_tr_phys, y_tr,
                                            X_te_phys, y_te, lr=3e-4, epochs=1000)

        results['M0'][n].append(mae_m0)
        results['M_TL+phys'][n].append(mae_mtlp)
        per_sample['M0'][n].append(ps_m0)
        per_sample['M_TL+phys'][n].append(ps_mtlp)
        delta = (1 - mae_mtlp / mae_m0) * 100
        print(f'  n={n:>3d}: M0={mae_m0*100:.3f}%, '
              f'M_TL+phys={mae_mtlp*100:.3f}%  [improvement {delta:+.1f}%]  '
              f'({time.time()-t0:.0f}s)', flush=True)


# --------------------------------------------------------------------------
# Bootstrap CIs and summary
# --------------------------------------------------------------------------
print('\n\n===== PHASE 3: 250-SAMPLE TEST SET RESULTS (Structure A, Cr) =====',
      flush=True)
print(f'{"n":>5} | {"M0 (mean+/-std)":>22} | {"M_TL+phys":>22} | '
      f'{"95% CI (M_TL+phys)":>22} | {"improv":>10}', flush=True)
print('-' * 100, flush=True)

summary = {}
for n in SIZES:
    m0  = np.array(results['M0'][n])
    mtl = np.array(results['M_TL+phys'][n])
    # Meta-bootstrap on flattened per-sample errors over 3 seeds * 250 samples
    flat_m0  = np.concatenate(per_sample['M0'][n])
    flat_mtl = np.concatenate(per_sample['M_TL+phys'][n])

    rng_b = np.random.default_rng(0)
    boots_m0  = np.array([flat_m0[rng_b.integers(0, len(flat_m0),  len(flat_m0))].mean()
                          for _ in range(5000)])
    boots_mtl = np.array([flat_mtl[rng_b.integers(0, len(flat_mtl), len(flat_mtl))].mean()
                          for _ in range(5000)])
    ci_m0  = (np.percentile(boots_m0, 2.5),  np.percentile(boots_m0, 97.5))
    ci_mtl = (np.percentile(boots_mtl, 2.5), np.percentile(boots_mtl, 97.5))

    delta_per_seed = (1 - mtl / m0) * 100
    print(f'{n:>5} | {m0.mean()*100:>5.2f}+/-{m0.std()*100:.2f}%      | '
          f'{mtl.mean()*100:>5.2f}+/-{mtl.std()*100:.2f}%      | '
          f'[{ci_mtl[0]*100:.2f}, {ci_mtl[1]*100:.2f}]%   | '
          f'{delta_per_seed.mean():>+5.1f}+/-{delta_per_seed.std():.1f}%', flush=True)
    summary[n] = {'m0_mean': m0.mean(), 'm0_std': m0.std(),
                  'mtl_mean': mtl.mean(), 'mtl_std': mtl.std(),
                  'ci_m0': ci_m0, 'ci_mtl': ci_mtl}

# Reference per-seed bootstrap CI on original 50-sample test (for direct comparison)
print('\n--- CI width comparison (95% bootstrap CI for M_TL+phys MAE) ---', flush=True)
for n in SIZES:
    s = summary[n]
    width = (s['ci_mtl'][1] - s['ci_mtl'][0]) * 100
    print(f'  n={n}: 250-sample test  -> CI width = {width:.3f} pp', flush=True)
print('  (paper original, 50-sample test, expected width ~ 0.6-1.0 pp)', flush=True)

np.savez('/home/bigmountain87/PINN2/mim_novel/results/phase3_expanded_test.npz',
         seeds=np.array(SEEDS), sizes=np.array(SIZES),
         m0_per_seed={n: np.array(results['M0'][n]) for n in SIZES},
         mtl_per_seed={n: np.array(results['M_TL+phys'][n]) for n in SIZES},
         per_sample_m0={n: np.array(per_sample['M0'][n]) for n in SIZES},
         per_sample_mtl={n: np.array(per_sample['M_TL+phys'][n]) for n in SIZES})
print(f'\nSaved: phase3_expanded_test.npz', flush=True)
print(f'Total time: {(time.time()-t_train)/60:.1f} min', flush=True)
