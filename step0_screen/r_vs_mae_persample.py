"""
Compute per-sample r vs per-sample MAE (corrected grid).
New narrative: is r alone insufficient, requiring joint MAE evaluation?
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

data_root = 'data/raw'

def analyze(npz, tmm_func, label, dual_pol=False):
    d = np.load(npz, allow_pickle=True)
    params = d['params']; wl = d['wavelengths'].astype(np.float64)
    if dual_pol:
        rcwa_te = d['A_TE']; rcwa_tm = d['A_TM']
        tmm_out = tmm_func(params, wl)
        # Try TE/TM keys
        keys = list(tmm_out.keys())
        if 'A_tmm_te' in tmm_out: tmm_te, tmm_tm = tmm_out['A_tmm_te'], tmm_out['A_tmm_tm']
        elif 'A_tmm_TE' in tmm_out: tmm_te, tmm_tm = tmm_out['A_tmm_TE'], tmm_out['A_tmm_TM']
        elif 'A_TE_tmm' in tmm_out: tmm_te, tmm_tm = tmm_out['A_TE_tmm'], tmm_out['A_TM_tmm']
        else: raise KeyError(f'TE/TM key not found, keys: {keys}')
        rs = []; maes = []
        for i in range(len(params)):
            t = np.concatenate([tmm_te[i], tmm_tm[i]])
            r = np.concatenate([rcwa_te[i], rcwa_tm[i]])
            mae = np.mean(np.abs(t - r))
            c = np.corrcoef(t, r)[0,1]
            if not np.isnan(c):
                rs.append(c); maes.append(mae)
    else:
        rcwa = d['A']
        tmm = tmm_func(params, wl)['A_tmm']
        rs = []; maes = []
        for i in range(len(params)):
            mae = np.mean(np.abs(tmm[i] - rcwa[i]))
            c = np.corrcoef(tmm[i], rcwa[i])[0,1]
            if not np.isnan(c):
                rs.append(c); maes.append(mae)
    rs = np.array(rs); maes = np.array(maes)
    print(f'--- {label} ---')
    print(f'  n_valid = {len(rs)}')
    print(f'  r: median = {np.median(rs):+.4f}, mean = {rs.mean():+.4f}')
    print(f'  MAE: median = {np.median(maes)*100:.2f}%, mean = {maes.mean()*100:.2f}%')
    # Correlation between per-sample r and per-sample MAE
    rho = np.corrcoef(rs, maes)[0,1]
    print(f'  Pearson corr (r vs MAE): {rho:+.4f}')
    # Quadrant analysis: high-r vs low-r, high-MAE vs low-MAE
    r_med, mae_med = np.median(rs), np.median(maes)
    q_hr_lm = ((rs >= r_med) & (maes < mae_med)).sum()
    q_hr_hm = ((rs >= r_med) & (maes >= mae_med)).sum()
    q_lr_lm = ((rs < r_med) & (maes < mae_med)).sum()
    q_lr_hm = ((rs < r_med) & (maes >= mae_med)).sum()
    print(f'  quadrants (high-r=above {r_med:.2f}, high-MAE=above {mae_med*100:.1f}%):')
    print(f'    high-r low-MAE: {q_hr_lm} ({q_hr_lm/len(rs)*100:.1f}%) ← ideal (good source for TL)')
    print(f'    high-r high-MAE: {q_hr_hm} ({q_hr_hm/len(rs)*100:.1f}%) ← shape good, amplitude off (pretender)')
    print(f'    low-r low-MAE: {q_lr_lm} ({q_lr_lm/len(rs)*100:.1f}%)')
    print(f'    low-r high-MAE: {q_lr_hm} ({q_lr_hm/len(rs)*100:.1f}%) ← shape and amplitude both wrong')
    print()
    return rs, maes

from src.simulation.tmm_struct_a import compute_tmm_batch as tmm_a
from src.simulation.tmm_struct_b import compute_tmm_batch as tmm_b
from src.simulation.tmm_struct_c_aniso import compute_tmm_batch as tmm_c

rs_a, mae_a = analyze(os.path.join(data_root, 'struct_A_vis_500.npz'), tmm_a, 'A @ 380-780 nm')
rs_b, mae_b = analyze(os.path.join(data_root, 'struct_B_500.npz'), tmm_b, 'B @ 400-1800 nm')
rs_c, mae_c = analyze(os.path.join(data_root, 'struct_C_500.npz'), tmm_c, 'C @ 400-1800 nm (aniso)', dual_pol=True)

np.savez('results/r_vs_mae_persample.npz',
         rs_a=rs_a, mae_a=mae_a, rs_b=rs_b, mae_b=mae_b, rs_c=rs_c, mae_c=mae_c)
print('Saved: r_vs_mae_persample.npz')

# Cross-structure: what predicts TL benefit better, median r or median MAE?
print()
print('=== Cross-structure: TL benefit (paper) vs metric ===')
tl_benefit = {'A': 29.4, 'B': 5.9, 'C': 4.2}
median_r = {'A': float(np.median(rs_a)), 'B': float(np.median(rs_b)), 'C': float(np.median(rs_c))}
median_mae = {'A': float(np.median(mae_a))*100, 'B': float(np.median(mae_b))*100, 'C': float(np.median(mae_c))*100}
print(f'  median r:   A={median_r["A"]:.3f}  B={median_r["B"]:.3f}  C={median_r["C"]:.3f}')
print(f'  median MAE: A={median_mae["A"]:.2f}%  B={median_mae["B"]:.2f}%  C={median_mae["C"]:.2f}%')
print(f'  TL benefit: A={tl_benefit["A"]:.1f}%  B={tl_benefit["B"]:.1f}%  C={tl_benefit["C"]:.1f}%')

r_vals = np.array([median_r['A'], median_r['B'], median_r['C']])
mae_vals = np.array([median_mae['A'], median_mae['B'], median_mae['C']])
tl_vals = np.array([tl_benefit['A'], tl_benefit['B'], tl_benefit['C']])
print(f'  Pearson(median_r, TL): {np.corrcoef(r_vals, tl_vals)[0,1]:+.4f}')
print(f'  Pearson(median_MAE, TL): {np.corrcoef(mae_vals, tl_vals)[0,1]:+.4f}  (expect negative)')
