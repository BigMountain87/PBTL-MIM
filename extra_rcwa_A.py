"""Generate 200 additional Cr Structure-A RCWA samples to expand the test set.

Defends Major weakness #5 (test-set size 50 -> bootstrap CI +/-0.3-0.5 pp).
Combining the new 200 samples with the existing 50 test samples per seed
gives an effective 250-sample test set, narrowing CI by ~sqrt(5) = 2.24x.

The new samples use a *different RNG seed* than the original 500, so they
explore the same design space but are statistically independent. We save
under a new file name so the original struct_A_vis_500.npz stays
untouched. Combined evaluation script will load both and concatenate.

Outputs:
  data/raw/struct_A_vis_extra200.npz
  logs/extra_rcwa.log
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/home/bigmountain87/mim_novel')

import numpy as np

OUT_PATH = '/home/bigmountain87/mim_novel/data/raw/struct_A_vis_extra200.npz'
N_NEW    = 200
WAVELENGTHS = np.linspace(380, 780, 100)

print(f'Generating {N_NEW} extra Cr Structure-A RCWA samples...', flush=True)
print(f'Estimated time: {N_NEW * 30 / 3600:.1f}h '
      f'(at ~30s/sample on RTX 4070 Ti SUPER)', flush=True)

try:
    import torcwa  # noqa
except ImportError:
    print('ERROR: torcwa not found. Run with: conda activate ML', flush=True)
    sys.exit(1)

import torch
from src.simulation.rcwa_struct_a import generate_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

t0 = time.time()
# Use seed=2026 to be statistically independent from the original (seed=42)
data = generate_dataset(N_NEW, WAVELENGTHS, metal='Cr', seed=2026, device=device)
dt = time.time() - t0
print(f'\nRCWA done: {dt/3600:.2f}h ({dt/N_NEW:.1f}s/sample)', flush=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.savez(OUT_PATH,
         params=data['params'], A=data['A'], R=data['R'], T=data['T'],
         wavelengths=WAVELENGTHS, metal='Cr')
print(f'Saved: {OUT_PATH}', flush=True)
