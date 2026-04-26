"""Structure C replica of the TMM-noise injection experiment (W3).

Motivation:
  The main W3 result (Section sec:tmm_accuracy) was produced on Structure A only.
  A reviewer can ask: "is the fidelity--benefit monotonicity a Structure-A
  artefact, or does it hold for other structures too?"

Design (reduced for wall-clock tractability):
  * Structure: C (dual-polarization, isotropic TMM), which is the harder,
    intermediate-fidelity case (median r ~0.34 in Table tab:tmm_fidelity).
  * Noise levels: sigma in {0, 0.10, 0.20, inf (random)} -- 4 points.
  * Fine-tune RCWA n=100 (matches W3 protocol).
  * 3 seeds per condition.

For each level we:
  1. Generate 2000 TMM samples with noise injected into the target spectra,
  2. Pretrain the ResNet-256-4 on those noisy samples,
  3. Fine-tune on 100 Structure-C RCWA samples (3 seeds),
  4. Compute TL benefit vs scratch M0 baseline (already measured).

Outputs: noise_injection_C.npz, log: logs/noise_C.log
"""
from __future__ import annotations
import os, sys, time, math
sys.path.insert(0, '/home/bigmountain87/mim_novel')
sys.path.insert(0, '/home/bigmountain87/PINN2/mim_novel')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_c import compute_tmm_batch as tmm_C  # isotropic variant
from src.utils.physics_features import compute_physics_features_C

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)

# --------------------------------------------------------------------------
# Structure C data / bounds
# --------------------------------------------------------------------------
RCWA_FILE = '/home/bigmountain87/mim_novel/data/raw/struct_C_500.npz'
da = np.load(RCWA_FILE, allow_pickle=True)
rcwa_params = da['params'].astype(np.float32)
rcwa_spectra = da['A'].astype(np.float32)
wavelengths = da['wavelengths'].astype(np.float32)
N_WL = rcwa_spectra.shape[1]
print(f'Structure-C RCWA: {rcwa_params.shape}, spectra {rcwa_spectra.shape}', flush=True)

# Bounds for Structure C (7 params): period, wx, wy, tm, tox, tcr, angle
# Infer bounds directly from RCWA data -- safe since TMM is evaluated on the same params.
BOUNDS_C = np.stack([rcwa_params.min(axis=0), rcwa_params.max(axis=0)], axis=1).astype(np.float32)
print(f'Bounds C:\n{BOUNDS_C}', flush=True)
N_PAR = BOUNDS_C.shape[0]


def norm_params(p):
    return (p - BOUNDS_C[:, 0]) / (BOUNDS_C[:, 1] - BOUNDS_C[:, 0] + 1e-8)


# --------------------------------------------------------------------------
# ResNet (same as PBTL)
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


# --------------------------------------------------------------------------
# Physics-feature-augmented input builder (17 physics features for C)
# --------------------------------------------------------------------------
NORM_WL = np.linspace(0, 1, N_WL, dtype=np.float32)


def build_flat_inputs(params, phys, target):
    """Return (X, y) flattened per-wavelength."""
    n = len(params)
    np_norm = norm_params(params)
    # feature dim = norm_wl (1) + norm_params (N_PAR) + phys_features (17)
    f_dim = 1 + N_PAR + phys.shape[-1]
    X = np.empty((n * N_WL, f_dim), dtype=np.float32)
    y = target.reshape(-1, 1).astype(np.float32)
    for i in range(n):
        base = i * N_WL
        X[base:base + N_WL, 0] = NORM_WL
        X[base:base + N_WL, 1:1 + N_PAR] = np_norm[i][None, :]
        X[base:base + N_WL, 1 + N_PAR:] = phys[i]
    return X, y


# --------------------------------------------------------------------------
# Helpers: seeded noise and per-sample correlation
# --------------------------------------------------------------------------
def inject_noise(spec, sigma, rng):
    """Additive gaussian noise, clipped to [0, 1]."""
    if not np.isfinite(sigma):
        return rng.uniform(0.0, 1.0, size=spec.shape).astype(np.float32)
    noisy = spec + rng.normal(0.0, sigma, size=spec.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def pearson_per_sample(a, b):
    rs = []
    for i in range(len(a)):
        r, _ = stats.pearsonr(a[i], b[i])
        rs.append(r)
    return float(np.mean(rs))


# --------------------------------------------------------------------------
# TMM library generation (fresh 2000 samples in bounds)
# --------------------------------------------------------------------------
def make_tmm_library(n=2000, seed=0):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)
    lb, ub = BOUNDS_C[:, 0], BOUNDS_C[:, 1]
    params = rng.uniform(lb, ub, size=(n, N_PAR)).astype(np.float32)
    result = tmm_C(params, wavelengths)
    spec = result['A_tmm'].astype(np.float32)
    return params, spec


def pretrain(noisy_spec, params, phys_feats, epochs=400):
    X, y = build_flat_inputs(params, phys_feats, noisy_spec)
    X = torch.from_numpy(X).to(DEVICE)
    y = torch.from_numpy(y).to(DEVICE)
    model = ResNet256(in_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(X, y), batch_size=4096, shuffle=True)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    # Final val MAE
    model.eval()
    with torch.no_grad():
        mae = (model(X) - y).abs().mean().item()
    return model, mae


def finetune(pt_state, n_train, seed):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rcwa_params))
    test_idx = perm[:50]
    val_idx  = perm[50:100]
    tr_idx   = perm[100:100 + n_train]

    # Compute physics features for each subset
    phys_tr = compute_physics_features_C(rcwa_params[tr_idx], wavelengths).astype(np.float32)
    phys_te = compute_physics_features_C(rcwa_params[test_idx], wavelengths).astype(np.float32)

    X_tr, y_tr = build_flat_inputs(rcwa_params[tr_idx], phys_tr, rcwa_spectra[tr_idx])
    X_te, y_te = build_flat_inputs(rcwa_params[test_idx], phys_te, rcwa_spectra[test_idx])

    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_te_t = torch.from_numpy(X_te).to(DEVICE)
    y_te_t = torch.from_numpy(y_te).to(DEVICE)

    model = ResNet256(in_dim=X_tr.shape[1]).to(DEVICE)
    model.load_state_dict(pt_state)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=2048, shuffle=True)
    lossfn = nn.MSELoss()
    for _ in range(1000):
        model.train()
        for xb, yb in loader:
            loss = lossfn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return (model(X_te_t) - y_te_t).abs().mean().item()


# --------------------------------------------------------------------------
# Baseline M0: already measured in Table pbtl_c; we retrieve from disk if possible
# --------------------------------------------------------------------------
M0_MAE_REF = None
try:
    ref = np.load('/home/bigmountain87/mim_novel/results/pbtl_C_v2_10seed.npz')
    if '100_M0' in ref.files:
        M0_MAE_REF = float(ref['100_M0'].mean())
        print(f'M0 (C, n=100, 10-seed mean) = {M0_MAE_REF*100:.3f}%', flush=True)
except Exception as e:
    print(f'warn: could not load M0 reference ({e})', flush=True)


# --------------------------------------------------------------------------
# Main loop: 4 noise levels x 3 seeds
# --------------------------------------------------------------------------
LEVELS = [
    ('L0_sigma0.00',  0.00),
    ('L2_sigma0.10',  0.10),
    ('L4_sigma0.20',  0.20),
    ('L5_random',     math.inf),
]
SEEDS = [42, 123, 777]
N_TRAIN = 100  # matches W3 primary protocol

print('\nGenerating fresh Structure-C TMM library (2000 samples)...', flush=True)
t0 = time.time()
tmm_params, tmm_clean = make_tmm_library(n=2000, seed=0)
print(f'  TMM library ready in {time.time()-t0:.0f}s | shape {tmm_clean.shape}', flush=True)

# Physics features for TMM params (used both pretraining input + phys aug)
phys_tmm = compute_physics_features_C(tmm_params, wavelengths).astype(np.float32)

# Also compute clean TMM on RCWA params for per-sample fidelity diagnostic
rcwa_tmm_clean = tmm_C(rcwa_params, wavelengths)['A_tmm'].astype(np.float32)

results = {}
for (lname, sigma) in LEVELS:
    rng = np.random.RandomState(hash(lname) % (2**32))
    noisy = inject_noise(tmm_clean, sigma, rng)
    # Fidelity metric on full TMM library
    if np.isfinite(sigma):
        r_mean = pearson_per_sample(noisy, tmm_clean)  # corruption to clean (self-reference)
    else:
        r_mean = 0.0
    # Per-sample correlation vs RCWA (on a 300-sample subset of RCWA for speed)
    sub = np.random.RandomState(0).choice(len(rcwa_params), size=min(300, len(rcwa_params)), replace=False)
    r_tmm_rcwa = pearson_per_sample(
        inject_noise(rcwa_tmm_clean[sub], sigma, np.random.RandomState(1)),
        rcwa_spectra[sub]
    )
    print(f'\n=== {lname} (sigma={sigma}, r_tmm_vs_rcwa={r_tmm_rcwa:+.3f}) ===', flush=True)

    t_pre = time.time()
    model, pt_mae = pretrain(noisy, tmm_params, phys_tmm, epochs=400)
    print(f'  pretrain done in {time.time()-t_pre:.0f}s | TMM val MAE {pt_mae*100:.2f}%', flush=True)

    seed_results = []
    for sd in SEEDS:
        t_ft = time.time()
        pt_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        mae = finetune(pt_state, N_TRAIN, sd)
        seed_results.append(mae)
        print(f'    finetune seed={sd}: MAE={mae*100:.3f}%  ({time.time()-t_ft:.0f}s)', flush=True)
    seed_arr = np.array(seed_results)
    results[lname] = {
        'sigma': sigma,
        'r_tmm_vs_rcwa': r_tmm_rcwa,
        'tl_mae': seed_arr,
        'mean': float(seed_arr.mean()),
        'std':  float(seed_arr.std()),
    }

print('\n\n===== STRUCTURE-C NOISE INJECTION REPLICA =====')
print(f'{"Level":<16} | {"sigma":>7} | {"r_tmm_vs_rcwa":>15} | {"TL MAE (%)":>18} | {"TL benefit":>12}')
print('-' * 85)
rows = []
for (lname, _sigma) in LEVELS:
    r = results[lname]
    tl_pct = r['mean'] * 100
    tl_std = r['std'] * 100
    if M0_MAE_REF is not None:
        benefit = (1 - r['mean'] / M0_MAE_REF) * 100
    else:
        benefit = float('nan')
    print(f'{lname:<16} | {r["sigma"]:>7.2f} | {r["r_tmm_vs_rcwa"]:>+15.3f} | '
          f'{tl_pct:>7.2f}+/-{tl_std:.2f}% | {benefit:>+10.1f}%')
    rows.append((lname, r['sigma'], r['r_tmm_vs_rcwa'], tl_pct, tl_std, benefit))

# Overall correlation r(TMM vs RCWA) vs TL benefit
if M0_MAE_REF is not None:
    rs = np.array([row[2] for row in rows])
    bens = np.array([row[5] for row in rows])
    r_corr, p_corr = stats.pearsonr(rs, bens)
    print(f'\nPearson correlation (r_TMM-vs-RCWA vs TL benefit) on Structure C: '
          f'r={r_corr:+.3f}, p={p_corr:.3f}  (N={len(rows)})')
    rho, p_rho = stats.spearmanr(rs, bens)
    print(f'Spearman rho={rho:+.3f}, p={p_rho:.3f}')

np.savez('/home/bigmountain87/PINN2/mim_novel/results/noise_injection_C.npz',
         level_names=np.array([l[0] for l in LEVELS]),
         sigmas=np.array([l[1] if np.isfinite(l[1]) else -1.0 for l in LEVELS]),
         r_tmm_vs_rcwa=np.array([results[l[0]]['r_tmm_vs_rcwa'] for l in LEVELS]),
         tl_mae_per_seed=np.array([results[l[0]]['tl_mae'] for l in LEVELS]),
         seeds=np.array(SEEDS),
         m0_reference=np.array([M0_MAE_REF if M0_MAE_REF is not None else -1.0]))
print('\nSaved: noise_injection_C.npz')
