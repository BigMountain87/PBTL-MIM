"""
W2 Response: Multi-fidelity baselines for Structure A
1. Linear correction: A_corrected = alpha * A_TMM + beta (per-wavelength)
2. Residual model: NN predicts delta = A_RCWA - A_TMM
3. PBTL (our method) for comparison

All use same RCWA train/val/test splits and 5 seeds
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed_utils import set_global_seed
from src.simulation.tmm_struct_a import compute_tmm_batch
from src.utils.physics_features import compute_physics_features_A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# Load data
data_file = 'data/raw/struct_A_vis_500.npz'
da = np.load(data_file, allow_pickle=True)
rcwa_params = da['params']
rcwa_spectra = da['A']
wavelengths = da['wavelengths']

print(f'RCWA data: {rcwa_params.shape[0]} samples, {rcwa_spectra.shape[1]} wavelengths', flush=True)

# Generate TMM spectra for all RCWA samples
print('Generating TMM spectra for RCWA parameter sets...', flush=True)
tmm_result = compute_tmm_batch(rcwa_params, wavelengths)
tmm_spectra_all = tmm_result['A_tmm']
print(f'TMM spectra generated: {tmm_spectra_all.shape}', flush=True)

# ResNet architecture (same as paper)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.fc1(self.ln1(x)))
        h = self.fc2(self.ln2(h))
        return x + self.act(h)

class BaseResNet(nn.Module):
    def __init__(self, input_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.head(h)

# Normalization
bounds_A = np.array([
    [300, 600], [50, 540], [50, 540], [50, 540],
    [10, 80], [10, 80], [5, 30],
    [30, 200], [30, 200],
    [0, 45]
])

def normalize_params(params, bounds):
    return (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

seeds = [42, 123, 777, 321, 456]
train_sizes = [50, 100, 200, 350]

results = {}
for method in ['linear_correction', 'residual_nn', 'M_TL+phys']:
    results[method] = np.zeros((len(train_sizes), len(seeds)))

for si, seed in enumerate(seeds):
    set_global_seed(seed)
    rng = np.random.RandomState(seed)

    n_total = len(rcwa_params)
    perm = rng.permutation(n_total)
    test_idx = perm[:50]
    val_idx = perm[50:100]

    for ni, n_train in enumerate(train_sizes):
        train_idx = perm[100:100+n_train]

        test_params = rcwa_params[test_idx]
        test_spectra = rcwa_spectra[test_idx]
        test_tmm = tmm_spectra_all[test_idx]

        train_rcwa = rcwa_spectra[train_idx]
        train_tmm = tmm_spectra_all[train_idx]

        # === Method 1: Linear Correction ===
        corrected_test = np.zeros_like(test_spectra)
        for wl_i in range(100):
            x = train_tmm[:, wl_i]
            y = train_rcwa[:, wl_i]
            A = np.column_stack([x, np.ones(len(x))])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            alpha, beta = coeffs
            corrected_test[:, wl_i] = alpha * test_tmm[:, wl_i] + beta

        mae_linear = np.mean(np.abs(corrected_test - test_spectra))
        results['linear_correction'][ni, si] = mae_linear

        # === Method 2: Residual NN ===
        train_residuals = train_rcwa - train_tmm

        norm_params_train = normalize_params(rcwa_params[train_idx], bounds_A)
        norm_wl = np.linspace(0, 1, 100)

        X_res_train, y_res_train = [], []
        for i in range(n_train):
            for j in range(100):
                x_vec = np.concatenate([[norm_wl[j]], norm_params_train[i], [train_tmm[i, j]]])
                X_res_train.append(x_vec)
                y_res_train.append(train_residuals[i, j])

        X_res_train = torch.tensor(np.array(X_res_train, dtype=np.float32)).to(device)
        y_res_train = torch.tensor(np.array(y_res_train, dtype=np.float32)).unsqueeze(1).to(device)

        res_model = BaseResNet(input_dim=X_res_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(res_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        dataset = TensorDataset(X_res_train, y_res_train)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)

        res_model.train()
        for epoch in range(1000):
            for xb, yb in loader:
                pred = res_model(xb)
                loss = nn.MSELoss()(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        res_model.eval()
        norm_params_test = normalize_params(test_params, bounds_A)
        X_res_test = []
        for i in range(len(test_params)):
            for j in range(100):
                x_vec = np.concatenate([[norm_wl[j]], norm_params_test[i], [test_tmm[i, j]]])
                X_res_test.append(x_vec)
        X_res_test = torch.tensor(np.array(X_res_test, dtype=np.float32)).to(device)

        with torch.no_grad():
            pred_residuals = res_model(X_res_test).cpu().numpy().flatten()
        pred_residuals = pred_residuals.reshape(len(test_params), 100)
        corrected_residual = test_tmm + pred_residuals

        mae_residual = np.mean(np.abs(corrected_residual - test_spectra))
        results['residual_nn'][ni, si] = mae_residual

        # === Method 3: Load M_TL+phys from existing results ===
        data_a = np.load('results/pbtl_A_10seed.npz')
        all_seeds = list(data_a['seeds'])
        if seed in all_seeds:
            seed_idx = all_seeds.index(seed)
            results['M_TL+phys'][ni, si] = data_a[f'{n_train}_M_TL+phys'][seed_idx]

        print(f'seed={seed}, n={n_train}: linear={mae_linear*100:.2f}%, '
              f'residual={mae_residual*100:.2f}%, '
              f'M_TL+phys={results["M_TL+phys"][ni,si]*100:.2f}%', flush=True)

# M0 baseline
data_a = np.load('results/pbtl_A_10seed.npz')
m0_results = np.zeros((len(train_sizes), len(seeds)))
for ni, n in enumerate(train_sizes):
    all_seeds_list = list(data_a['seeds'])
    for si, seed in enumerate(seeds):
        if seed in all_seeds_list:
            idx = all_seeds_list.index(seed)
            m0_results[ni, si] = data_a[f'{n}_M0'][idx]

print('\n\n=== MULTI-FIDELITY BASELINE COMPARISON ===')
fmt = '{:>5} | {:>18} | {:>18} | {:>18} | {:>18}'
print(fmt.format('n', 'M0 (scratch)', 'Linear Corr', 'Residual NN', 'M_TL+phys (ours)'))
print('-' * 85)
for ni, n in enumerate(train_sizes):
    m0_m = m0_results[ni].mean() * 100
    m0_s = m0_results[ni].std() * 100
    lc_m = results['linear_correction'][ni].mean() * 100
    lc_s = results['linear_correction'][ni].std() * 100
    rn_m = results['residual_nn'][ni].mean() * 100
    rn_s = results['residual_nn'][ni].std() * 100
    our_m = results['M_TL+phys'][ni].mean() * 100
    our_s = results['M_TL+phys'][ni].std() * 100
    print(f'{n:>5} | {m0_m:>6.2f}+/-{m0_s:.2f}% | {lc_m:>6.2f}+/-{lc_s:.2f}% | {rn_m:>6.2f}+/-{rn_s:.2f}% | {our_m:>6.2f}+/-{our_s:.2f}%')

print('\nRelative improvement over M0:')
for ni, n in enumerate(train_sizes):
    m0_mean = m0_results[ni].mean()
    lc_imp = (1 - results['linear_correction'][ni].mean() / m0_mean) * 100
    rn_imp = (1 - results['residual_nn'][ni].mean() / m0_mean) * 100
    our_imp = (1 - results['M_TL+phys'][ni].mean() / m0_mean) * 100
    print(f'  n={n}: Linear={lc_imp:+.1f}%, ResidualNN={rn_imp:+.1f}%, M_TL+phys={our_imp:+.1f}%')

np.savez('results/multifidelity_baseline.npz',
         train_sizes=np.array(train_sizes), seeds=np.array(seeds),
         m0=m0_results, linear_correction=results['linear_correction'],
         residual_nn=results['residual_nn'], m_tl_phys=results['M_TL+phys'])
print('\nSaved: multifidelity_baseline.npz')
print('Done!')
