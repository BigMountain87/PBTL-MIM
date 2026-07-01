"""
FIXED version: each structure uses its own npz wavelengths grid
(was buggy: TMM at 380-780nm for ALL structures, RCWA at native grid)
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

data_root = 'data/raw'

print('=== TMM-RCWA Accuracy Per Structure (FIXED grid) ===')
print()

# Structure A
print('--- Structure A ---')
da = np.load(os.path.join(data_root, 'struct_A_vis_500.npz'), allow_pickle=True)
rcwa_params_a = da['params']
rcwa_spectra_a = da['A']
wl_a = da['wavelengths'].astype(np.float64)
print(f'  wavelengths: {wl_a.min():.1f}-{wl_a.max():.1f} nm, step={(wl_a.max()-wl_a.min())/(wl_a.size-1):.2f} nm')

from src.simulation.tmm_struct_a import compute_tmm_batch
tmm_spectra_a = compute_tmm_batch(rcwa_params_a, wl_a)['A_tmm']

mae_a = np.mean(np.abs(tmm_spectra_a - rcwa_spectra_a))
corrs_a = []
for i in range(len(rcwa_spectra_a)):
    c = np.corrcoef(tmm_spectra_a[i], rcwa_spectra_a[i])[0,1]
    if not np.isnan(c): corrs_a.append(c)
corrs_a = np.array(corrs_a)
print(f'  MAE: {mae_a*100:.2f}%')
print(f'  Pearson r: mean={np.mean(corrs_a):.4f}, median={np.median(corrs_a):.4f}, std={np.std(corrs_a):.4f}')
print(f'  range: [{corrs_a.min():.4f}, {corrs_a.max():.4f}]')

# Structure B
print()
print('--- Structure B ---')
db = np.load(os.path.join(data_root, 'struct_B_500.npz'), allow_pickle=True)
rcwa_params_b = db['params']
rcwa_spectra_b = db['A']
wl_b = db['wavelengths'].astype(np.float64)
print(f'  wavelengths: {wl_b.min():.1f}-{wl_b.max():.1f} nm, step={(wl_b.max()-wl_b.min())/(wl_b.size-1):.2f} nm')

from src.simulation.tmm_struct_b import compute_tmm_batch as compute_tmm_batch_b
tmm_spectra_b = compute_tmm_batch_b(rcwa_params_b, wl_b)['A_tmm']

mae_b = np.mean(np.abs(tmm_spectra_b - rcwa_spectra_b))
corrs_b = []
for i in range(len(rcwa_spectra_b)):
    c = np.corrcoef(tmm_spectra_b[i], rcwa_spectra_b[i])[0,1]
    if not np.isnan(c): corrs_b.append(c)
corrs_b = np.array(corrs_b)
print(f'  MAE: {mae_b*100:.2f}%')
print(f'  Pearson r: mean={np.mean(corrs_b):.4f}, median={np.median(corrs_b):.4f}, std={np.std(corrs_b):.4f}')
print(f'  range: [{corrs_b.min():.4f}, {corrs_b.max():.4f}]')

# Structure C (aniso TMM, both polarizations concatenated, same as original script)
print()
print('--- Structure C (anisotropic TMM, TE+TM concat) ---')
dc = np.load(os.path.join(data_root, 'struct_C_500.npz'), allow_pickle=True)
rcwa_params_c = dc['params']
rcwa_A_TE = dc['A_TE']; rcwa_A_TM = dc['A_TM']
wl_c = dc['wavelengths'].astype(np.float64)
print(f'  wavelengths: {wl_c.min():.1f}-{wl_c.max():.1f} nm, step={(wl_c.max()-wl_c.min())/(wl_c.size-1):.2f} nm')

try:
    from src.simulation.tmm_struct_c_aniso import compute_tmm_batch as compute_tmm_batch_c
    tmm_c = compute_tmm_batch_c(rcwa_params_c, wl_c)
    tmm_A_TE = tmm_c.get('A_tmm_te', tmm_c.get('A_TE_tmm', tmm_c.get('A_tmm_TE')))
    tmm_A_TM = tmm_c.get('A_tmm_tm', tmm_c.get('A_TM_tmm', tmm_c.get('A_tmm_TM')))
    if tmm_A_TE is None:
        print(f'  aniso keys: {list(tmm_c.keys())} - cannot find TE/TM')
        raise KeyError('aniso TE/TM keys missing')
    label = 'aniso'
except Exception as e:
    print(f'  aniso failed: {e} -> using iso')
    from src.simulation.tmm_struct_c import compute_tmm_batch as compute_tmm_batch_c_iso
    tmm_c = compute_tmm_batch_c_iso(rcwa_params_c, wl_c)
    tmm_A_TE = tmm_c['A_tmm']; tmm_A_TM = tmm_c['A_tmm']
    label = 'iso'

mae_c_te = np.mean(np.abs(tmm_A_TE - rcwa_A_TE))
mae_c_tm = np.mean(np.abs(tmm_A_TM - rcwa_A_TM))
mae_c = (mae_c_te + mae_c_tm) / 2
corrs_c = []
for i in range(len(rcwa_params_c)):
    t = np.concatenate([tmm_A_TE[i], tmm_A_TM[i]])
    r = np.concatenate([rcwa_A_TE[i], rcwa_A_TM[i]])
    c = np.corrcoef(t, r)[0,1]
    if not np.isnan(c): corrs_c.append(c)
corrs_c = np.array(corrs_c)
print(f'  TMM mode: {label}')
print(f'  MAE TE/TM/avg: {mae_c_te*100:.2f}% / {mae_c_tm*100:.2f}% / {mae_c*100:.2f}%')
print(f'  Pearson r (TE+TM concat): mean={np.mean(corrs_c):.4f}, median={np.median(corrs_c):.4f}, std={np.std(corrs_c):.4f}')
print(f'  range: [{corrs_c.min():.4f}, {corrs_c.max():.4f}]')

print()
print('=== SUMMARY (corrected grid) ===')
print(f'  A:  MAE={mae_a*100:6.2f}%   mean r={np.mean(corrs_a):+.4f}   median r={np.median(corrs_a):+.4f}')
print(f'  B:  MAE={mae_b*100:6.2f}%   mean r={np.mean(corrs_b):+.4f}   median r={np.median(corrs_b):+.4f}')
print(f'  C:  MAE={mae_c*100:6.2f}%   mean r={np.mean(corrs_c):+.4f}   median r={np.median(corrs_c):+.4f}')

np.savez('results/tmm_rcwa_accuracy_fixed.npz',
         mae_a=mae_a, corrs_a=corrs_a, wl_a=wl_a,
         mae_b=mae_b, corrs_b=corrs_b, wl_b=wl_b,
         mae_c=mae_c, corrs_c=corrs_c, wl_c=wl_c)
print()
print('Saved: tmm_rcwa_accuracy_fixed.npz')
