"""Sparse Co-Kriging with increased TMM budget -- defense vs. reviewer W3c.

The paper's main Co-Kriging baseline uses 500 TMM samples because full
(dense) per-wavelength GPs scale as O(n^3) per wavelength and become
infeasible at n_tmm = 2000 (~64x the cost).

Reviewer concern: "if you gave Co-Kriging PBTL's 5000-sample pretraining
budget, would it catch up?"

This script answers with a *sparse variational GP* (SVGP) Co-Kriging that
scales O(n * M^2) in training data n with M=128 inducing points, so we can
probe the full N_TMM grid without intractable wall clock.

Protocol:
  * Kernel: Matern-5/2 (matches paper's main Co-Kriging baseline)
  * Inducing points: M=128, k-means initialised on training inputs
  * N_TMM sweep: {500 (paper ref), 1000, 2000}
  * n_rcwa fine-tune sweep: {50, 100, 200, 350}
  * 3 seeds each
  * Per-wavelength independent SVGPs (matches formulation in paper)

At each (seed, n_rcwa, N_TMM):
    y_H(x, lambda) = rho(lambda) * y_L(x, lambda) + delta(x, lambda)
  where y_L is obtained from a global sparse GP fit to N_TMM TMM samples,
  rho is LS on the training subset, and delta is another SVGP on the
  residual. For fairness we use the SAME sparse approximation for LF (on
  TMM) and the HF residual.

Outputs: cokriging_ntmm_sweep.npz, log: logs/cok_ntmm.log
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/bigmountain87/mim_novel')
sys.path.insert(0, '/home/bigmountain87/PINN2/mim_novel')

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from sklearn.cluster import KMeans

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
DATA = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_500.npz'
da = np.load(DATA, allow_pickle=True)
rcwa_params = da['params'].astype(np.float32)
rcwa_spectra = da['A'].astype(np.float32)
wavelengths = da['wavelengths'].astype(np.float32)
N_WL = rcwa_spectra.shape[1]
print(f'RCWA {rcwa_params.shape}, spectra {rcwa_spectra.shape}', flush=True)

tmm_on_rcwa = compute_tmm_batch(rcwa_params, wavelengths)['A_tmm'].astype(np.float32)
print(f'TMM on RCWA params: {tmm_on_rcwa.shape}', flush=True)

BOUNDS = np.array([[300, 600], [50, 540], [50, 540], [50, 540],
                   [10, 80],  [10, 80],  [5, 30],
                   [30, 200], [30, 200], [0, 45]], dtype=np.float32)
def norm_params(p):
    return (p - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


# --------------------------------------------------------------------------
# SVGP (per-wavelength): shared across LF/HF-residual modelling
# --------------------------------------------------------------------------
class SVGP(ApproximateGP):
    def __init__(self, inducing, lengthscale=None):
        vd = CholeskyVariationalDistribution(inducing.size(0))
        vs = VariationalStrategy(self, inducing, vd, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.cov_module   = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing.size(1)))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.cov_module(x))


def fit_svgp(X_train, y_train, M=128, steps=400, lr=5e-2, batch_size=1024):
    """Fit SVGP and return a callable predictor."""
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    n = X_train.size(0)
    # Inducing points via KMeans on training inputs
    km = KMeans(n_clusters=min(M, n), n_init=3, random_state=0)
    km.fit(X_train.detach().cpu().numpy())
    inducing = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(DEVICE)

    model = SVGP(inducing).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
    opt = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}], lr=lr)

    model.train(); likelihood.train()
    perm_idx = torch.randperm(n, device=DEVICE)
    for it in range(steps):
        # mini-batch over training indices
        idx = perm_idx[(it * batch_size) % n : (it * batch_size) % n + batch_size]
        if len(idx) == 0:
            perm_idx = torch.randperm(n, device=DEVICE)
            idx = perm_idx[:batch_size]
        opt.zero_grad()
        out = model(X_train[idx])
        loss = -mll(out, y_train[idx])
        loss.backward()
        opt.step()
    model.eval(); likelihood.eval()

    def predict(X_query):
        X_query = X_query.to(DEVICE)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = likelihood(model(X_query)).mean
        return mean.detach().cpu().numpy()
    return predict


# --------------------------------------------------------------------------
# Fresh TMM library generation
# --------------------------------------------------------------------------
def make_tmm_library(n, seed=0):
    rng = np.random.RandomState(seed)
    lb, ub = BOUNDS[:, 0], BOUNDS[:, 1]
    params = rng.uniform(lb, ub, size=(n, 10)).astype(np.float32)
    A = compute_tmm_batch(params, wavelengths)['A_tmm'].astype(np.float32)
    return params, A


# --------------------------------------------------------------------------
# Sparse Co-Kriging run for one (seed, n_rcwa, N_TMM) cell
# --------------------------------------------------------------------------
def cokriging_sparse(seed, n_rcwa, n_tmm, tmm_params_ext, tmm_spec_ext):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rcwa_params))
    test_idx = perm[:50]
    tr_idx   = perm[100:100 + n_rcwa]

    # Normalise
    X_rcwa_tr_np = norm_params(rcwa_params[tr_idx])
    X_rcwa_te_np = norm_params(rcwa_params[test_idx])
    X_rcwa_tr = torch.from_numpy(X_rcwa_tr_np)
    X_rcwa_te = torch.from_numpy(X_rcwa_te_np)

    # TMM library (external)
    X_tmm_np = norm_params(tmm_params_ext[:n_tmm])
    X_tmm    = torch.from_numpy(X_tmm_np)
    y_tmm    = tmm_spec_ext[:n_tmm]     # [n_tmm, N_WL]

    rcwa_tmm_tr = tmm_on_rcwa[tr_idx]   # [n_rcwa, N_WL]
    rcwa_tmm_te = tmm_on_rcwa[test_idx]  # [50,     N_WL]
    y_rcwa_tr = rcwa_spectra[tr_idx]    # [n_rcwa, N_WL]
    y_rcwa_te = rcwa_spectra[test_idx]  # [50,     N_WL]

    mae_wl = np.zeros(N_WL)
    for wl_i in range(N_WL):
        # Step 1: LF surrogate -- SVGP on N_TMM TMM samples
        lf_predict = fit_svgp(
            X_tmm, torch.from_numpy(y_tmm[:, wl_i]),
            M=128, steps=300, lr=5e-2, batch_size=512)
        y_L_train = lf_predict(X_rcwa_tr)  # predict TMM from SVGP at RCWA train inputs
        y_L_test  = lf_predict(X_rcwa_te)

        # Step 2: autoregressive Kennedy-O'Hagan scale
        yh_tr = y_rcwa_tr[:, wl_i]
        denom = float(np.dot(y_L_train, y_L_train)) + 1e-12
        rho   = float(np.dot(y_L_train, yh_tr)) / denom
        resid = yh_tr - rho * y_L_train

        # Step 3: HF residual SVGP on (X_rcwa_tr, resid)
        M_hf = min(64, len(tr_idx))
        delta_predict = fit_svgp(
            X_rcwa_tr, torch.from_numpy(resid.astype(np.float32)),
            M=M_hf, steps=250, lr=5e-2, batch_size=min(256, len(tr_idx)))
        delta_te = delta_predict(X_rcwa_te)

        pred_te = rho * y_L_test + delta_te
        mae_wl[wl_i] = np.mean(np.abs(pred_te - y_rcwa_te[:, wl_i]))
    return float(mae_wl.mean())


# --------------------------------------------------------------------------
# Main grid
# --------------------------------------------------------------------------
SEEDS  = [42, 123, 777]
SIZES  = [50, 100, 200, 350]
N_TMMS = [500, 1000, 2000]

# Prepare the largest TMM library once (2000 samples) and slice it for smaller budgets
print('\nGenerating TMM library (2000 samples) ...', flush=True)
t0 = time.time()
tmm_params_ext, tmm_spec_ext = make_tmm_library(max(N_TMMS), seed=0)
print(f'  TMM library ready in {time.time()-t0:.0f}s', flush=True)

results = np.zeros((len(SIZES), len(N_TMMS), len(SEEDS)))
m0_ref = np.zeros((len(SIZES), len(SEEDS)))
try:
    ref = np.load('/home/bigmountain87/mim_novel/results/pbtl_A_10seed.npz')
    all_seeds = list(ref['seeds'])
    for ni, n in enumerate(SIZES):
        for si, sd in enumerate(SEEDS):
            if sd in all_seeds:
                m0_ref[ni, si] = ref[f'{n}_M0'][all_seeds.index(sd)]
except Exception as e:
    print(f'warn: no M0 reference ({e})', flush=True)

t_total = time.time()
for ni, n in enumerate(SIZES):
    for ti, n_tmm in enumerate(N_TMMS):
        for si, sd in enumerate(SEEDS):
            tic = time.time()
            set_global_seed(sd)
            mae = cokriging_sparse(sd, n, n_tmm, tmm_params_ext, tmm_spec_ext)
            results[ni, ti, si] = mae
            print(f'  n_rcwa={n:3d} n_tmm={n_tmm:5d} seed={sd}: '
                  f'MAE={mae*100:.3f}%  ({time.time()-tic:.0f}s)', flush=True)

print('\n\n===== SPARSE CO-KRIGING N_TMM SWEEP (Structure A) =====')
print(f'{"n_rcwa":>7} | ' + ' | '.join([f'N_TMM={t:<6d}' for t in N_TMMS]))
print('-' * (9 + 15 * len(N_TMMS)))
for ni, n in enumerate(SIZES):
    row = [f'{n:>7d}']
    for ti, n_tmm in enumerate(N_TMMS):
        m = results[ni, ti].mean() * 100
        s = results[ni, ti].std()  * 100
        row.append(f'{m:>5.2f}+/-{s:.2f}%')
    print(' | '.join(row))

if m0_ref.any():
    print('\nRelative improvement over M0 (geometry-only ResNet baseline):')
    for ni, n in enumerate(SIZES):
        m0m = m0_ref[ni].mean()
        imp_line = [f'{n:>7d}']
        for ti, n_tmm in enumerate(N_TMMS):
            mf = results[ni, ti].mean()
            imp = (1 - mf / m0m) * 100
            imp_line.append(f'{imp:+7.1f}%')
        print(' | '.join(imp_line))

np.savez('/home/bigmountain87/PINN2/mim_novel/results/cokriging_ntmm_sweep.npz',
         sizes=np.array(SIZES),
         n_tmms=np.array(N_TMMS),
         seeds=np.array(SEEDS),
         mae=results,
         m0_ref=m0_ref)
print('\nSaved: cokriging_ntmm_sweep.npz')
print(f'Total time: {(time.time()-t_total)/60:.1f} min')
