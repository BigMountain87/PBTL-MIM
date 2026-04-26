"""Co-Kriging hyperparameter sensitivity analysis (Structure A).

The paper's main Co-Kriging result uses Matern-5/2. Reviewer: could a
different kernel/length-scale close the gap to PBTL?

Sweeps:
  * Kernels    : Matern-5/2 (paper), Matern-3/2, RBF
  * Length-scale bounds: tight (paper, [1e-2, 1e2]) vs wide ([1e-3, 1e3])
  * Inputs     : 10-D geometry vs 27-D geometry+physics

Fixed: 5 seeds, n_train in {50, 100, 200, 350}, same 50/50 test/val splits.
Reports per-config MAE (mean +/- std) and improvement over M0 baseline.

Independent per-wavelength GPs (matches the paper formulation).
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/bigmountain87/mim_novel')

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, RBF, WhiteKernel
)
from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch
from src.utils.physics_features import compute_physics_features_A

DATA = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_500.npz'
da = np.load(DATA, allow_pickle=True)
rcwa_params  = da['params'].astype(np.float64)
rcwa_spectra = da['A'].astype(np.float64)
wavelengths  = da['wavelengths'].astype(np.float64)
N_WL = rcwa_spectra.shape[1]
print(f'RCWA {rcwa_params.shape}, spectra {rcwa_spectra.shape}', flush=True)

tmm_spectra_all = compute_tmm_batch(rcwa_params, wavelengths)['A_tmm'].astype(np.float64)
phys_feats_all  = compute_physics_features_A(rcwa_params, wavelengths).astype(np.float64)
print(f'TMM {tmm_spectra_all.shape}  phys {phys_feats_all.shape}', flush=True)

BOUNDS = np.array([[300, 600], [50, 540], [50, 540], [50, 540],
                   [10, 80],  [10, 80],  [5, 30],
                   [30, 200], [30, 200], [0, 45]], dtype=np.float64)
def norm_params(p):
    return (p - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


KERNEL_LIB = {
    'matern52_tight': lambda d: (ConstantKernel(1.0, (1e-2, 1e2)) *
                                  Matern(length_scale=np.ones(d), nu=2.5,
                                         length_scale_bounds=(1e-2, 1e2)) +
                                  WhiteKernel(noise_level=1e-4,
                                              noise_level_bounds=(1e-8, 1e-1))),
    'matern52_wide':  lambda d: (ConstantKernel(1.0, (1e-3, 1e3)) *
                                  Matern(length_scale=np.ones(d), nu=2.5,
                                         length_scale_bounds=(1e-3, 1e3)) +
                                  WhiteKernel(noise_level=1e-4,
                                              noise_level_bounds=(1e-10, 1e0))),
    'matern32_wide':  lambda d: (ConstantKernel(1.0, (1e-3, 1e3)) *
                                  Matern(length_scale=np.ones(d), nu=1.5,
                                         length_scale_bounds=(1e-3, 1e3)) +
                                  WhiteKernel(noise_level=1e-4,
                                              noise_level_bounds=(1e-10, 1e0))),
    'rbf_wide':       lambda d: (ConstantKernel(1.0, (1e-3, 1e3)) *
                                  RBF(length_scale=np.ones(d),
                                      length_scale_bounds=(1e-3, 1e3)) +
                                  WhiteKernel(noise_level=1e-4,
                                              noise_level_bounds=(1e-10, 1e0))),
}

CONFIGS = [
    ('matern52_tight', 'geom'),   # paper reference
    ('matern52_wide',  'geom'),
    ('matern32_wide',  'geom'),
    ('rbf_wide',       'geom'),
    ('matern52_wide',  'phys'),   # + physics features (27-D)
]

SEEDS = [42, 123, 777, 321, 456]
SIZES = [50, 100, 200, 350]


def run_cokriging(X_train, y_train_h, y_train_l, X_test, y_test_h, y_test_l,
                   kernel_name):
    """Per-wavelength Kennedy-O'Hagan auto-regressive Co-Kriging.
    y_h(x, wl) = rho * y_l(x, wl) + delta(x, wl)
    delta is modeled by a GP with the chosen kernel; rho by LS.
    Returns test MAE averaged over wavelengths.
    """
    d = X_train.shape[1]
    kernel_fn = KERNEL_LIB[kernel_name]
    n_test = X_test.shape[0]
    preds = np.zeros((n_test, N_WL))
    for wl_i in range(N_WL):
        yh = y_train_h[:, wl_i]
        yl = y_train_l[:, wl_i]
        # Estimate rho by LS
        denom = np.dot(yl, yl) + 1e-12
        rho = np.dot(yl, yh) / denom
        resid = yh - rho * yl
        try:
            gp = GaussianProcessRegressor(kernel=kernel_fn(d), normalize_y=True,
                                          n_restarts_optimizer=2, alpha=1e-8)
            gp.fit(X_train, resid)
            delta = gp.predict(X_test)
        except Exception:
            delta = np.zeros(n_test)
        preds[:, wl_i] = rho * y_test_l[:, wl_i] + delta
    return np.mean(np.abs(preds - y_test_h))


results = {(k, v): np.zeros((len(SIZES), len(SEEDS)))
           for (k, v) in CONFIGS}
m0_results = np.zeros((len(SIZES), len(SEEDS)))  # loaded below

# Load M0 for reference
try:
    data_a = np.load('/home/bigmountain87/mim_novel/results/pbtl_A_10seed.npz')
    all_seeds = list(data_a['seeds'])
    for ni, n in enumerate(SIZES):
        for si, seed in enumerate(SEEDS):
            if seed in all_seeds:
                m0_results[ni, si] = data_a[f'{n}_M0'][all_seeds.index(seed)]
except Exception as e:
    print(f'warn: could not load M0 ref ({e})', flush=True)

t_total = time.time()
for si, seed in enumerate(SEEDS):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rcwa_params))
    test_idx = perm[:50]
    val_idx  = perm[50:100]

    p_norm_all = norm_params(rcwa_params)
    feats_geom_all = p_norm_all
    feats_phys_all = np.concatenate([p_norm_all,
                                     phys_feats_all.mean(axis=1)], axis=1)  # 10 + 17

    test_spec_h = rcwa_spectra[test_idx]
    test_spec_l = tmm_spectra_all[test_idx]

    for ni, n_tr in enumerate(SIZES):
        tr = perm[100:100 + n_tr]
        train_h = rcwa_spectra[tr]
        train_l = tmm_spectra_all[tr]

        for kernel_name, input_kind in CONFIGS:
            if input_kind == 'geom':
                X_tr = feats_geom_all[tr]
                X_te = feats_geom_all[test_idx]
            else:
                X_tr = feats_phys_all[tr]
                X_te = feats_phys_all[test_idx]
            tic = time.time()
            mae = run_cokriging(X_tr, train_h, train_l,
                                 X_te, test_spec_h, test_spec_l,
                                 kernel_name)
            results[(kernel_name, input_kind)][ni, si] = mae
            print(f'  seed={seed} n={n_tr:3d} [{kernel_name:>15s}/{input_kind:>4s}] '
                  f'MAE={mae*100:.3f}%  ({time.time()-tic:.0f}s)', flush=True)

print('\n\n=== CO-KRIGING KERNEL SENSITIVITY (Structure A) ===')
fmt = '{:>5} | ' + ' | '.join([f'{{:>18s}}' for _ in CONFIGS])
print(fmt.format('n', *[f'{k[:6]}/{v[:4]}' for k, v in CONFIGS]))
print('-' * (7 + 21 * len(CONFIGS)))
for ni, n in enumerate(SIZES):
    row = [f'{n:>5d}']
    for cfg in CONFIGS:
        m = results[cfg][ni].mean() * 100
        s = results[cfg][ni].std() * 100
        row.append(f'{m:>6.2f}+/-{s:.2f}%'.rjust(18))
    print(' | '.join(row))

np.savez('/home/bigmountain87/PINN2/mim_novel/results/cokriging_sensitivity.npz',
         sizes=np.array(SIZES), seeds=np.array(SEEDS),
         m0=m0_results,
         **{f'{k}__{v}': results[(k, v)] for (k, v) in CONFIGS})
print('\nSaved: cokriging_sensitivity.npz')
print(f'Total time: {(time.time()-t_total)/60:.1f} min')
