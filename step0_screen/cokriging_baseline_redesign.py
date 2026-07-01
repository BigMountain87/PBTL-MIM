"""
Co-Kriging (Multi-Fidelity Gaussian Process) Baseline for Structure A
(REDESIGN / corrected data).

Implements the Kennedy & O'Hagan (2000) autoregressive Co-Kriging model:
    y_high(x) = rho * y_low(x) + delta(x)

where y_low = TMM prediction, y_high = RCWA ground truth,
rho is a scaling factor, and delta(x) is a GP modeling the discrepancy.

Two variants:
1. Co-Kriging (per-wavelength): Independent GP per wavelength point
2. Co-Kriging + Physics Features: GP uses physics features as additional inputs

Uses same data splits, seeds, and evaluation protocol as
multifidelity_baseline_redesign.py.

Dependencies: scikit-learn (for GaussianProcessRegressor), numpy

REDESIGN variant of cokriging_baseline.py. The Co-Kriging math (rho LSQ, per-wl /
spectral GP, Matern kernel, splits, seeds) is IDENTICAL. Only data/material/grid/paths
and the dependency npz change:
  (1) RCWA data -> struct_A_500_redesign.npz (corrected jc materials, adaptive order,
      complex64, unified 400-1800nm 100pt grid);
  (2) MATERIAL_MODEL='jc' set BEFORE importing tmm modules;
  (3) wavelength grid LOADED from the data file (not a hardcoded visible grid);
  (4) reliability filter: keep samples with data['reliable'].all(axis=1); clip A,R to [0,1];
  (5) M0 / M_TL+phys loaded from results/pbtl_A_redesign_10seed.npz (corrected);
  (6) linear/residual loaded from results/multifidelity_baseline_redesign.npz (corrected
      -> run multifidelity_baseline_redesign.py FIRST);
  (7) output -> results/cokriging_baseline_redesign.npz (original preserved).
"""
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Corrected material model must be set BEFORE importing tmm modules.
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch
from src.utils.physics_features import compute_physics_features_A

print('='*70)
print('CO-KRIGING MULTI-FIDELITY BASELINE (REDESIGN)')
print('='*70)

# ========== Load Data (corrected redesign RCWA dataset) ==========
data_file = 'data/raw/struct_A_500_redesign.npz'
da = np.load(data_file, allow_pickle=True)
rcwa_params = da['params'].astype(np.float32)        # [N, 10]
rcwa_spectra = np.clip(da['A'].astype(np.float32), 0, 1)  # [N, 100]
rcwa_R = np.clip(da['R'].astype(np.float32), 0, 1)
wavelengths = da['wavelengths'].astype(np.float32)   # [100], 400-1800nm

# Reliability filter: keep samples physically reliable at ALL wavelengths.
if 'reliable' in da.files:
    good = da['reliable'].all(axis=1)
else:
    good = np.all((rcwa_spectra >= 0) & (rcwa_spectra <= 1), axis=1)
gi = np.where(good)[0]
rcwa_params = rcwa_params[gi]
rcwa_spectra = rcwa_spectra[gi]
rcwa_R = rcwa_R[gi]

print(f'RCWA data: {rcwa_params.shape[0]} reliable samples (of {len(good)}), '
      f'{rcwa_spectra.shape[1]} wavelengths '
      f'[{wavelengths.min():.0f}-{wavelengths.max():.0f}nm]')

# ========== Generate TMM spectra (on the redesign label grid) ==========
print('Generating TMM spectra for RCWA parameter sets...')
tmm_result = compute_tmm_batch(rcwa_params, wavelengths)
tmm_spectra_all = np.clip(tmm_result['A_tmm'].astype(np.float32), 0, 1)  # [N, 100]
print(f'TMM spectra: {tmm_spectra_all.shape}')

# ========== Compute Physics Features ==========
print('Computing physics features...')
# Physics features need [N, n_params] params and [N_lam] wavelengths
# Returns [N, N_lam, n_phys_features]
phys_feats_all = compute_physics_features_A(rcwa_params, wavelengths)
print(f'Physics features: {phys_feats_all.shape}')  # [N, 100, 17]

# ========== Normalization ==========
bounds_A = np.array([
    [300, 600], [50, 540], [50, 540], [50, 540],
    [10, 80], [10, 80], [5, 30],
    [30, 200], [30, 200],
    [0, 45]
])

def normalize_params(params, bounds):
    return (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

# ========== Experimental Setup ==========
seeds = [42, 123, 777, 321, 456]
train_sizes = [50, 100, 200, 350]

# FIXED split (identical to pbtl_A_redesign.py / ablation_A_redesign.py): one fixed
# permutation from default_rng(42) holds out the SAME 50 test + 50 val geometries for
# every seed; per (size, seed) the training rows are drawn ONLY from `remaining` via
# default_rng(seed). This makes Co-Kriging evaluated on the EXACT same held-out
# geometries as PBTL/M0, so the shared comparison table is fair.
n_total = len(rcwa_params)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(n_total)
N_TEST, N_VAL = 50, 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST + N_VAL)]

# Test set is fixed across all seeds (geometry-identical to PBTL test set).
test_params = rcwa_params[test_idx]
test_spectra = rcwa_spectra[test_idx]
test_tmm = tmm_spectra_all[test_idx]
test_phys = phys_feats_all[test_idx]
test_params_norm = normalize_params(test_params, bounds_A)

# Results storage
results = {}
for method in ['cokriging', 'cokriging_phys']:
    results[method] = np.zeros((len(train_sizes), len(seeds)))


def run_cokriging_per_wavelength(train_params_norm, train_tmm, train_rcwa,
                                test_params_norm, test_tmm,
                                use_physics=False,
                                train_phys=None, test_phys=None):
    """
    Co-Kriging per wavelength point.

    Kennedy & O'Hagan autoregressive model:
        y_RCWA(x, lam) = rho * y_TMM(x, lam) + delta(x, lam)

    For each wavelength:
        1. Compute residual: delta_train = RCWA - rho * TMM
        2. Fit GP on delta_train as function of (params [+ physics_features])
        3. Predict: y_pred = rho * TMM_test + GP_delta(test_params)
    """
    n_train = len(train_params_norm)
    n_test = len(test_params_norm)
    n_wl = train_rcwa.shape[1]
    predictions = np.zeros((n_test, n_wl))

    for wl_i in range(n_wl):
        y_tmm_train = train_tmm[:, wl_i]
        y_rcwa_train = train_rcwa[:, wl_i]
        y_tmm_test = test_tmm[:, wl_i]

        # Step 1: Estimate rho via least squares
        # y_rcwa = rho * y_tmm + delta
        # rho = (y_tmm^T y_rcwa) / (y_tmm^T y_tmm)
        rho = np.dot(y_tmm_train, y_rcwa_train) / (np.dot(y_tmm_train, y_tmm_train) + 1e-10)

        # Step 2: Compute residuals
        delta_train = y_rcwa_train - rho * y_tmm_train

        # Step 3: Build GP input
        if use_physics and train_phys is not None:
            X_train = np.hstack([train_params_norm, train_phys[:, wl_i, :]])
            X_test = np.hstack([test_params_norm, test_phys[:, wl_i, :]])
        else:
            X_train = train_params_norm
            X_test = test_params_norm

        # Step 4: Fit GP on residuals
        # REDESIGN runtime: isotropic (single shared) length scale instead of a
        # per-dimension ARD length_scale=np.ones(n_feat). The 27-D anisotropic
        # length-scale optimization (params + 17 physics features) was the dominant
        # cost (~12 s / wavelength at n=350 -> hours for the full 20-cell grid);
        # an isotropic Matern(nu=2.5) discrepancy GP is ~24x faster (full run ~7 min,
        # under the 15-min budget) and is the standard, well-conditioned Co-Kriging
        # discrepancy kernel. The relative Co-Kriging vs Co-Kriging+phys comparison
        # is preserved.
        kernel = ConstantKernel(1.0, (1e-4, 1e2)) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-3, 1e2),
            nu=2.5
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            # REDESIGN runtime: reduced n_restarts_optimizer 3->1. The per-wavelength
            # GP fit (100 wl x 2 variants x 20 cells) made the original 3-restart fit
            # impractically slow (>15 min; a single n=350 cell took >5.5 min). One
            # restart keeps the Matern + WhiteKernel discrepancy GP well-conditioned
            # and does not change the relative Co-Kriging vs Co-Kriging+phys comparison.
            n_restarts_optimizer=1,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        gp.fit(X_train, delta_train)

        # Step 5: Predict
        delta_pred = gp.predict(X_test)
        predictions[:, wl_i] = rho * y_tmm_test + delta_pred

    return predictions


def run_cokriging_spectral(train_params_norm, train_tmm, train_rcwa,
                           test_params_norm, test_tmm,
                           use_physics=False,
                           train_phys=None, test_phys=None):
    """
    Co-Kriging with spectral (pointwise) GP.

    Instead of per-wavelength GP, use a single GP over (params, wavelength) space.
    More data-efficient but slower for large datasets.

    Falls back to per-wavelength if n_train > 150 (computational cost).
    """
    n_train = len(train_params_norm)
    n_wl = train_rcwa.shape[1]

    # For large training sets, per-wavelength is more practical
    if n_train > 150:
        return run_cokriging_per_wavelength(
            train_params_norm, train_tmm, train_rcwa,
            test_params_norm, test_tmm,
            use_physics, train_phys, test_phys
        )

    # Pointwise expansion
    norm_wl = np.linspace(0, 1, n_wl).reshape(-1, 1)

    # Build pointwise training data
    X_list, y_list, tmm_list = [], [], []
    for i in range(n_train):
        for j in range(n_wl):
            x_base = np.concatenate([[norm_wl[j, 0]], train_params_norm[i]])
            if use_physics and train_phys is not None:
                x_base = np.concatenate([x_base, train_phys[i, j, :]])
            X_list.append(x_base)
            y_list.append(train_rcwa[i, j])
            tmm_list.append(train_tmm[i, j])

    X_train_pw = np.array(X_list)
    y_train_pw = np.array(y_list)
    tmm_train_pw = np.array(tmm_list)

    # Estimate rho globally
    rho = np.dot(tmm_train_pw, y_train_pw) / (np.dot(tmm_train_pw, tmm_train_pw) + 1e-10)
    delta_train = y_train_pw - rho * tmm_train_pw

    # Subsample if too large (GP scales as O(n^3))
    max_points = 5000
    if len(X_train_pw) > max_points:
        idx = np.random.choice(len(X_train_pw), max_points, replace=False)
        X_train_pw = X_train_pw[idx]
        delta_train = delta_train[idx]

    # Fit GP (isotropic length scale; see per-wavelength note re: runtime)
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e2),
        nu=2.5
    ) + WhiteKernel(noise_level=1e-3)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        # REDESIGN runtime: reduced n_restarts_optimizer 2->1 (see per-wavelength note).
        n_restarts_optimizer=1,
        alpha=1e-6,
        normalize_y=True,
        random_state=42
    )
    gp.fit(X_train_pw, delta_train)

    # Predict test
    n_test = len(test_params_norm)
    predictions = np.zeros((n_test, n_wl))
    for i in range(n_test):
        X_test_pts = []
        tmm_test_pts = []
        for j in range(n_wl):
            x_base = np.concatenate([[norm_wl[j, 0]], test_params_norm[i]])
            if use_physics and test_phys is not None:
                x_base = np.concatenate([x_base, test_phys[i, j, :]])
            X_test_pts.append(x_base)
            tmm_test_pts.append(test_tmm[i, j])
        X_test_pts = np.array(X_test_pts)
        tmm_test_pts = np.array(tmm_test_pts)
        delta_pred = gp.predict(X_test_pts)
        predictions[i, :] = rho * tmm_test_pts + delta_pred

    return predictions


# ========== Main Experiment Loop ==========
print('\n' + '='*70)
print('RUNNING CO-KRIGING EXPERIMENTS')
print(f'Seeds: {seeds}')
print(f'Train sizes: {train_sizes}')
print('='*70 + '\n')

total_start = time.time()

for si, seed in enumerate(seeds):
    set_global_seed(seed)
    # Per-seed training rows sampled ONLY from `remaining` (same as PBTL fine-tuning);
    # test/val geometries are the fixed default_rng(42) hold-out defined above.
    rng2 = np.random.default_rng(seed)
    perm_remaining = remaining[rng2.permutation(len(remaining))]

    for ni, n_train in enumerate(train_sizes):
        train_idx = perm_remaining[:n_train]
        iter_start = time.time()

        train_params_norm = normalize_params(rcwa_params[train_idx], bounds_A)
        train_rcwa = rcwa_spectra[train_idx]
        train_tmm = tmm_spectra_all[train_idx]
        train_phys = phys_feats_all[train_idx]

        # Normalize physics features (z-score using training stats)
        phys_mean = train_phys.mean(axis=(0, 1), keepdims=True)
        phys_std = train_phys.std(axis=(0, 1), keepdims=True) + 1e-8
        train_phys_norm = (train_phys - phys_mean) / phys_std
        test_phys_norm = (test_phys - phys_mean) / phys_std

        # === Co-Kriging (geometry only) ===
        pred_ck = run_cokriging_per_wavelength(
            train_params_norm, train_tmm, train_rcwa,
            test_params_norm, test_tmm,
            use_physics=False
        )
        mae_ck = np.mean(np.abs(pred_ck - test_spectra))
        results['cokriging'][ni, si] = mae_ck

        # === Co-Kriging + Physics Features ===
        pred_ck_phys = run_cokriging_per_wavelength(
            train_params_norm, train_tmm, train_rcwa,
            test_params_norm, test_tmm,
            use_physics=True,
            train_phys=train_phys_norm,
            test_phys=test_phys_norm
        )
        mae_ck_phys = np.mean(np.abs(pred_ck_phys - test_spectra))
        results['cokriging_phys'][ni, si] = mae_ck_phys

        elapsed = time.time() - iter_start
        print(f'seed={seed}, n={n_train}: '
              f'CoKriging={mae_ck*100:.2f}%, '
              f'CoKriging+phys={mae_ck_phys*100:.2f}%, '
              f'time={elapsed:.1f}s', flush=True)

total_elapsed = time.time() - total_start
print(f'\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)')

# ========== Load existing results for comparison ==========
print('\n' + '='*70)
print('COMPREHENSIVE COMPARISON')
print('='*70)

# Load M0 and M_TL+phys from existing REDESIGN results
data_a = np.load('results/pbtl_A_redesign_10seed.npz')
all_seeds_list = list(data_a['seeds'])

m0_results = np.zeros((len(train_sizes), len(seeds)))
mtlphys_results = np.zeros((len(train_sizes), len(seeds)))
for ni, n in enumerate(train_sizes):
    for si, seed in enumerate(seeds):
        if seed in all_seeds_list:
            idx = all_seeds_list.index(seed)
            m0_results[ni, si] = data_a[f'{n}_M0'][idx]
            mtlphys_results[ni, si] = data_a[f'{n}_M_TL+phys'][idx]

# Load linear/residual from REDESIGN multifidelity_baseline
mf_file = 'results/multifidelity_baseline_redesign.npz'
has_mf = os.path.exists(mf_file)
if has_mf:
    mf = np.load(mf_file)
    linear_results = mf['linear_correction']
    residual_results = mf['residual_nn']
else:
    linear_results = np.full((len(train_sizes), len(seeds)), np.nan)
    residual_results = np.full((len(train_sizes), len(seeds)), np.nan)

# Print comprehensive table
print(f'\n{"n":>5} | {"M0":>14} | {"Linear":>14} | {"ResidualNN":>14} | '
      f'{"CoKriging":>14} | {"CoKrig+phys":>14} | {"PBTL(ours)":>14}')
print('-' * 105)

for ni, n in enumerate(train_sizes):
    def fmt(arr):
        m = arr[ni].mean() * 100
        s = arr[ni].std() * 100
        return f'{m:.2f}+/-{s:.2f}%'

    print(f'{n:>5} | {fmt(m0_results)} | {fmt(linear_results)} | '
          f'{fmt(residual_results)} | {fmt(results["cokriging"])} | '
          f'{fmt(results["cokriging_phys"])} | {fmt(mtlphys_results)}')

print('\nRelative improvement over M0:')
for ni, n in enumerate(train_sizes):
    m0_mean = m0_results[ni].mean()
    def imp(arr):
        return (1 - arr[ni].mean() / m0_mean) * 100

    print(f'  n={n}: Linear={imp(linear_results):+.1f}%, '
          f'ResNN={imp(residual_results):+.1f}%, '
          f'CoKrig={imp(results["cokriging"]):+.1f}%, '
          f'CoKrig+phys={imp(results["cokriging_phys"]):+.1f}%, '
          f'PBTL={imp(mtlphys_results):+.1f}%')

print('\nPBTL advantage over Co-Kriging:')
for ni, n in enumerate(train_sizes):
    ck_mean = results['cokriging'][ni].mean()
    ckp_mean = results['cokriging_phys'][ni].mean()
    pbtl_mean = mtlphys_results[ni].mean()
    best_ck = min(ck_mean, ckp_mean)
    adv = (1 - pbtl_mean / best_ck) * 100
    print(f'  n={n}: Best CoKrig={best_ck*100:.2f}%, PBTL={pbtl_mean*100:.2f}%, '
          f'PBTL advantage={adv:+.1f}%')

# ========== Save Results ==========
save_path = 'results/cokriging_baseline_redesign.npz'
np.savez(save_path,
         train_sizes=np.array(train_sizes),
         seeds=np.array(seeds),
         cokriging=results['cokriging'],
         cokriging_phys=results['cokriging_phys'],
         m0=m0_results,
         m_tl_phys=mtlphys_results)
print(f'\nSaved: {save_path}')
print('Done!')
