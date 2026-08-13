"""
Phase 1: compare Structure A r/MAE on both grids
- 380-780 nm: struct_A_vis_500.npz (n=500)
- 400-1800 nm: struct_A_100.npz (n=100)
A is a single structure, so the difference is purely a grid effect
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.simulation.tmm_struct_a import compute_tmm_batch

data_root = 'data/raw'

def evaluate(npz_path, label):
    d = np.load(npz_path, allow_pickle=True)
    params = d['params']
    rcwa = d['A']
    wl = d['wavelengths'].astype(np.float64)
    tmm = compute_tmm_batch(params, wl)['A_tmm']
    mae = np.mean(np.abs(tmm - rcwa))
    corrs = []
    for i in range(len(rcwa)):
        c = np.corrcoef(tmm[i], rcwa[i])[0,1]
        if not np.isnan(c): corrs.append(c)
    corrs = np.array(corrs)
    pct_neg = np.mean(corrs < 0) * 100
    pct_high = np.mean(corrs > 0.5) * 100
    print(f'--- {label} ---')
    print(f'  wavelengths: {wl.min():.1f}-{wl.max():.1f} nm, step={(wl.max()-wl.min())/(wl.size-1):.2f} nm, n_samples={len(params)}')
    print(f'  MAE: {mae*100:.2f}%')
    print(f'  Pearson r: mean={corrs.mean():.4f}, median={np.median(corrs):.4f}, std={corrs.std():.4f}')
    print(f'  r range: [{corrs.min():.4f}, {corrs.max():.4f}]')
    print(f'  pct r<0: {pct_neg:.1f}%   pct r>0.5: {pct_high:.1f}%')
    # Quartiles
    q = np.percentile(corrs, [10, 25, 50, 75, 90])
    print(f'  r quartiles (10/25/50/75/90): {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f} / {q[4]:.3f}')
    return {'wl_min': wl.min(), 'wl_max': wl.max(), 'n': len(params), 'mae': mae, 'corrs': corrs, 'params': params}

print('=== Phase 1: Structure A at both wavelength grids ===')
print()
res_vis = evaluate(os.path.join(data_root, 'struct_A_vis_500.npz'), 'A @ 380-780 nm (visible, n=500)')
print()
res_ir = evaluate(os.path.join(data_root, 'struct_A_100.npz'), 'A @ 400-1800 nm (broadband, n=100)')

print()
print('=== Compare (same structure, different grid) ===')
print(f'  median r: visible={np.median(res_vis["corrs"]):+.4f}  vs  broadband={np.median(res_ir["corrs"]):+.4f}')
print(f'  MAE:      visible={res_vis["mae"]*100:.2f}%  vs  broadband={res_ir["mae"]*100:.2f}%')

np.savez('results/phase1_A_both_grids.npz',
         corrs_vis=res_vis['corrs'], mae_vis=res_vis['mae'],
         corrs_ir=res_ir['corrs'], mae_ir=res_ir['mae'])
print()
print('Saved: phase1_A_both_grids.npz')
