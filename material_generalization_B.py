"""Material generalisation experiment for Structure B (Ring--disk Fano) with Au.

Defends critical weakness #2: Au cross-material was previously validated only
on Structure A. A reviewer can argue the result is structure-specific. We
replicate the Cr-vs-Au comparison on Structure B, where Cr already shows the
"missing physics" (Fano) failure mode of TMM pre-training. If Au reproduces
the same pattern, the failure mechanism is confirmed structure-driven, not
material-driven.

Two-stage workflow:
  Stage 1 (ML env, torcwa available):  generate Au RCWA + Au TMM data on
    Structure B (350 RCWA + 2000 TMM samples).
  Stage 2 (pytorch env): train 4-way comparison (M0, M_phys, M_TL, M_TL+phys)
    over 3 seeds and 4 RCWA training sizes (50, 100, 200, 350).

Auto-detects which stage to run from torcwa availability + presence of the
RCWA data file, mirroring the workflow of `material_generalization_experiment.py`.

Outputs:
  data/raw/struct_B_Au_350.npz
  results/material_generalization_B.npz
  logs/material_gen_B.log
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/home/bigmountain87/mim_novel')
sys.path.insert(0, '/home/bigmountain87/PINN2/mim_novel')

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
RCWA_DATA_PATH = '/home/bigmountain87/mim_novel/data/raw/struct_B_Au_350.npz'
RESULTS_PATH   = '/home/bigmountain87/PINN2/mim_novel/results/material_generalization_B.npz'
METAL          = 'Au'
N_RCWA_TOTAL   = 350
N_TMM_PRETRAIN = 2000
TRAIN_SIZES    = [50, 100, 200, 350]
SEEDS          = [42, 123, 777]
WAVELENGTHS    = np.linspace(380, 780, 100)


# ============================================================================
# Stage 1 — RCWA data generation (requires torcwa / conda env "ML")
# ============================================================================
def stage1_generate_rcwa():
    print(f'\n=== STAGE 1: Generating {METAL} RCWA data for Structure B '
          f'(n={N_RCWA_TOTAL}) ===', flush=True)
    print(f'Estimated time: {N_RCWA_TOTAL * 17 / 3600:.1f}h '
          f'(Structure B is faster than A)', flush=True)

    try:
        import torcwa  # noqa: F401
    except ImportError:
        print('ERROR: torcwa not found. Run with: conda activate ML', flush=True)
        sys.exit(1)

    import torch
    from src.simulation.rcwa_struct_b import generate_dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    t0 = time.time()
    data = generate_dataset(N_RCWA_TOTAL, WAVELENGTHS, metal=METAL,
                            seed=42, device=device)
    dt = time.time() - t0
    print(f'\nRCWA done: {dt/3600:.2f}h ({dt/N_RCWA_TOTAL:.1f}s/sample)', flush=True)

    os.makedirs(os.path.dirname(RCWA_DATA_PATH), exist_ok=True)
    np.savez(RCWA_DATA_PATH,
             params=data['params'], A=data['A'], R=data['R'], T=data['T'],
             wavelengths=WAVELENGTHS, metal=METAL)
    print(f'Saved: {RCWA_DATA_PATH}', flush=True)


# ============================================================================
# Stage 2 — PBTL Training (requires pytorch env)
# ============================================================================
def stage2_train():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from copy import deepcopy

    from src.utils.seed_utils import set_global_seed as set_seed
    from src.utils.data_utils import normalize_params, get_bounds
    from src.utils.physics_features import compute_physics_features_B
    from src.simulation.tmm_struct_b import compute_tmm_batch

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}', flush=True)

    # ────────────────────── Load RCWA Au data ──────────────────────
    print(f'\nLoading {RCWA_DATA_PATH}...', flush=True)
    d = np.load(RCWA_DATA_PATH, allow_pickle=True)
    params_all = d['params'].astype(np.float32)            # [N, 8]
    A_all      = d['A'].astype(np.float32)                  # [N, 100]
    R_all      = d['R'].astype(np.float32)
    print(f'  params {params_all.shape}, A {A_all.shape}', flush=True)

    # Quality filter: drop obvious RCWA failures
    bad = np.any(A_all < -0.01, axis=1) | np.any(A_all > 1.01, axis=1)
    keep = ~bad
    print(f'  quality filter: kept {keep.sum()}/{len(A_all)} '
          f'({100*keep.sum()/len(A_all):.1f}%)', flush=True)
    params_all = params_all[keep]; A_all = A_all[keep]; R_all = R_all[keep]
    N = len(params_all)

    # ────────────────────── Bounds + normalisation ──────────────────────
    names, p_min, p_max = get_bounds('B')
    p_min = p_min.astype(np.float32); p_max = p_max.astype(np.float32)
    PARAM_DIM = len(names)   # 8

    params_norm = normalize_params(params_all, 'B').astype(np.float32)

    # ────────────────────── Test split fixed across seeds ──────────────────────
    rng_split = np.random.default_rng(0)
    perm = rng_split.permutation(N)
    n_test = 50
    test_idx = perm[:n_test]
    pool_idx = perm[n_test:]                 # the seed-specific train pool

    A_test = A_all[test_idx]
    X_test = params_norm[test_idx]

    phys_test = compute_physics_features_B(params_all[test_idx], WAVELENGTHS, metal=METAL)
    PHYS_DIM = phys_test.shape[-1]
    print(f'  PHYS_DIM = {PHYS_DIM}', flush=True)

    # ────────────────────── Generate Au TMM pre-training data ──────────────────────
    print(f'\nGenerating {METAL} TMM data (n={N_TMM_PRETRAIN}) for Structure B...',
          flush=True)
    rng_tmm = np.random.default_rng(0)
    params_tmm = rng_tmm.uniform(p_min, p_max, size=(N_TMM_PRETRAIN, PARAM_DIM)).astype(np.float32)
    # Enforce structural constraint: R_in < R_out (cols 1, 2 of B)
    # B params: [P, R_out, R_in, R_disk, t_Cr, d_SiO2, theta, phi]
    for i in range(N_TMM_PRETRAIN):
        if params_tmm[i, 2] >= params_tmm[i, 1]:        # R_in >= R_out
            params_tmm[i, 2] = 0.5 * params_tmm[i, 1]
        if params_tmm[i, 3] >= params_tmm[i, 2]:        # R_disk >= R_in
            params_tmm[i, 3] = 0.5 * params_tmm[i, 2]
    t_tmm = time.time()
    tmm_out = compute_tmm_batch(params_tmm, WAVELENGTHS, metal=METAL)
    print(f'  TMM done in {time.time()-t_tmm:.1f}s', flush=True)
    A_tmm = tmm_out['A_tmm'].astype(np.float32)         # [N_tmm, 100]

    params_tmm_norm = normalize_params(params_tmm, 'B').astype(np.float32)
    phys_tmm = compute_physics_features_B(params_tmm, WAVELENGTHS, metal=METAL).astype(np.float32)

    # ────────────────────── Architecture ──────────────────────
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

    NORM_WL = np.linspace(0, 1, len(WAVELENGTHS), dtype=np.float32)

    def build_inputs(params_norm_arr, phys_arr=None):
        """Per-wavelength flat inputs.
        Returns X (N*Nlam, in_dim).  in_dim = 1 + PARAM_DIM (+ PHYS_DIM if phys).
        """
        n = len(params_norm_arr)
        nlam = len(WAVELENGTHS)
        if phys_arr is None:
            in_dim = 1 + PARAM_DIM
            X = np.empty((n * nlam, in_dim), dtype=np.float32)
            for i in range(n):
                base = i * nlam
                X[base:base + nlam, 0] = NORM_WL
                X[base:base + nlam, 1:] = params_norm_arr[i][None, :]
        else:
            in_dim = 1 + PARAM_DIM + PHYS_DIM
            X = np.empty((n * nlam, in_dim), dtype=np.float32)
            for i in range(n):
                base = i * nlam
                X[base:base + nlam, 0] = NORM_WL
                X[base:base + nlam, 1:1 + PARAM_DIM] = params_norm_arr[i][None, :]
                X[base:base + nlam, 1 + PARAM_DIM:] = phys_arr[i]
        return X

    def to_t(arr):
        return torch.from_numpy(arr).to(DEVICE)

    def train_model(model, X, y, epochs=1000, lr=3e-4, batch=2048):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                loss = loss_fn(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        return model

    def eval_mae(model, X, y):
        model.eval()
        with torch.no_grad():
            return (model(X) - y).abs().mean().item()

    def pretrain_on_tmm(in_dim, X, y, epochs=400, lr=1e-3, batch=4096):
        model = ResNet256(in_dim=in_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                loss = loss_fn(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        return model

    # ────────────────────── Pre-train base PBTL models on TMM (per seed) ──────────────────────
    results = {k: {n: [] for n in TRAIN_SIZES}
               for k in ['M0', 'M_phys', 'M_TL', 'M_TL+phys']}

    # Build TMM input tensors once (per architecture variant)
    X_tmm_g = to_t(build_inputs(params_tmm_norm, phys_arr=None))
    y_tmm   = to_t(A_tmm.reshape(-1, 1))
    X_tmm_p = to_t(build_inputs(params_tmm_norm, phys_arr=phys_tmm))

    # Build test set tensors once
    X_test_g = to_t(build_inputs(X_test, phys_arr=None))
    X_test_p = to_t(build_inputs(X_test, phys_arr=phys_test.astype(np.float32)))
    y_test   = to_t(A_test.reshape(-1, 1))

    print('\nPre-training base models on Au TMM data (Structure B)...', flush=True)
    t_pre = time.time()
    base_geo  = pretrain_on_tmm(in_dim=1 + PARAM_DIM, X=X_tmm_g, y=y_tmm)
    base_phys = pretrain_on_tmm(in_dim=1 + PARAM_DIM + PHYS_DIM, X=X_tmm_p, y=y_tmm)
    print(f'  pre-training done in {time.time()-t_pre:.0f}s', flush=True)
    with torch.no_grad():
        pre_mae_geo  = (base_geo(X_tmm_g)  - y_tmm).abs().mean().item()
        pre_mae_phys = (base_phys(X_tmm_p) - y_tmm).abs().mean().item()
    print(f'  TMM pre-train val MAE: geo={pre_mae_geo*100:.2f}%, '
          f'phys={pre_mae_phys*100:.2f}%', flush=True)

    # ────────────────────── 4-way training over seeds and sizes ──────────────────────
    for seed in SEEDS:
        print(f'\n[Seed] {seed}', flush=True)
        set_seed(seed)
        rng = np.random.default_rng(seed)
        idx_pool_perm = rng.permutation(pool_idx)
        idx_train_full = idx_pool_perm[:max(TRAIN_SIZES)]

        for n in TRAIN_SIZES:
            idx_tr = idx_train_full[:n]
            phys_tr = compute_physics_features_B(params_all[idx_tr], WAVELENGTHS, metal=METAL).astype(np.float32)
            X_tr_g  = to_t(build_inputs(params_norm[idx_tr], phys_arr=None))
            X_tr_p  = to_t(build_inputs(params_norm[idx_tr], phys_arr=phys_tr))
            y_tr    = to_t(A_all[idx_tr].reshape(-1, 1))

            t0 = time.time()
            # M0: scratch geometry-only
            m0 = ResNet256(in_dim=1 + PARAM_DIM).to(DEVICE)
            m0 = train_model(m0, X_tr_g, y_tr, epochs=1000, lr=1e-3)
            mae_m0 = eval_mae(m0, X_test_g, y_test) * 100
            results['M0'][n].append(mae_m0)

            # M_phys: scratch physics-augmented
            mp = ResNet256(in_dim=1 + PARAM_DIM + PHYS_DIM).to(DEVICE)
            mp = train_model(mp, X_tr_p, y_tr, epochs=1000, lr=1e-3)
            mae_mp = eval_mae(mp, X_test_p, y_test) * 100
            results['M_phys'][n].append(mae_mp)

            # M_TL: TMM pre-trained + geometry-only fine-tune
            mtl = deepcopy(base_geo)
            mtl = train_model(mtl, X_tr_g, y_tr, epochs=1000, lr=3e-4)
            mae_mtl = eval_mae(mtl, X_test_g, y_test) * 100
            results['M_TL'][n].append(mae_mtl)

            # M_TL+phys: TMM pre-trained + physics-augmented fine-tune
            mtlp = deepcopy(base_phys)
            mtlp = train_model(mtlp, X_tr_p, y_tr, epochs=1000, lr=3e-4)
            mae_mtlp = eval_mae(mtlp, X_test_p, y_test) * 100
            results['M_TL+phys'][n].append(mae_mtlp)

            tl_b = (1 - mae_mtl / mae_m0) * 100
            print(f'  n={n:>3}: M0={mae_m0:.2f}%, M_phys={mae_mp:.2f}%, '
                  f'M_TL={mae_mtl:.2f}%, M_TL+phys={mae_mtlp:.2f}%  '
                  f'[TL benefit {tl_b:+.1f}%]  ({time.time()-t0:.0f}s)', flush=True)

    # ────────────────────── Summary ──────────────────────
    print(f'\n=== STRUCTURE B + {METAL} (3 seeds, n_test=50) ===', flush=True)
    print(f"{'n':>5} | {'M0':>15} | {'M_phys':>15} | {'M_TL':>15} | {'M_TL+phys':>15} | "
          f"{'TL ben.':>10}", flush=True)
    print('-' * 90, flush=True)
    for n in TRAIN_SIZES:
        m0a  = np.array(results['M0'][n])
        mpa  = np.array(results['M_phys'][n])
        mtla = np.array(results['M_TL'][n])
        mtlp = np.array(results['M_TL+phys'][n])
        tlb  = (1 - mtla / m0a) * 100
        print(f"{n:>5} | {m0a.mean():>5.2f}+/-{m0a.std():.2f}% | "
              f"{mpa.mean():>5.2f}+/-{mpa.std():.2f}% | "
              f"{mtla.mean():>5.2f}+/-{mtla.std():.2f}% | "
              f"{mtlp.mean():>5.2f}+/-{mtlp.std():.2f}% | "
              f"{tlb.mean():>+5.1f}+/-{tlb.std():.1f}%", flush=True)

    # ────────────────────── Save ──────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    save = {'metal': np.array([METAL]),
            'train_sizes': np.array(TRAIN_SIZES),
            'seeds': np.array(SEEDS)}
    for k in results:
        for n in TRAIN_SIZES:
            save[f'{n}_{k}'] = np.array(results[k][n])
    np.savez(RESULTS_PATH, **save)
    print(f'\nSaved: {RESULTS_PATH}', flush=True)


# ============================================================================
# Entry
# ============================================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['1', '2', 'auto'], default='auto')
    args = ap.parse_args()

    if args.stage == '1':
        stage1_generate_rcwa()
    elif args.stage == '2':
        stage2_train()
    else:
        try:
            import torcwa  # noqa: F401
            if not os.path.exists(RCWA_DATA_PATH):
                print('Detected ML env (torcwa) and missing RCWA data → Stage 1', flush=True)
                stage1_generate_rcwa()
            else:
                print('Detected ML env but RCWA data exists → switching to Stage 2 '
                      '(reload script under pytorch env)', flush=True)
                stage2_train()
        except ImportError:
            if not os.path.exists(RCWA_DATA_PATH):
                print('ERROR: RCWA data missing and torcwa not available. '
                      'Run with --stage=1 in ML env first.', flush=True)
                sys.exit(1)
            stage2_train()
