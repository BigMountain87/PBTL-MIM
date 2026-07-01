"""R2 Q4: representative TMM-vs-RCWA absorptance spectra (the 'cavity-like responses').
Plots a high-fidelity example for Structures B and C from the corrected datasets, showing
where the planar TMM tracks the full-wave RCWA cavity resonance."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import src.simulation.materials as _mat; _mat.MATERIAL_MODEL = 'jc'

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular', 'axes.linewidth': 0.8,
    'axes.labelsize': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def rep_sample(fn, tmm_mod, dual):
    tmm = __import__(tmm_mod, fromlist=['compute_tmm_batch']).compute_tmm_batch
    d = np.load(f'data/raw/{fn}.npz', allow_pickle=True); wl = d['wavelengths']
    out = tmm(d['params'], wl, str(d['metal']))
    if dual:
        At = out.get('A_tmm_te', out.get('A_tmm')); Ar = d['A_TE']; rel = (d['reliable_TE'].astype(bool) & d['reliable_TM'].astype(bool))
    else:
        At = out['A_tmm']; Ar = d['A']; rel = d['reliable'].astype(bool)
    rm = rel.all(axis=1)
    rs = np.full(len(Ar), -2.0)
    for i in np.where(rm)[0]:
        if np.all(np.isfinite(At[i])) and At[i].std() > 0: rs[i] = stats.pearsonr(At[i], Ar[i])[0]
    i = int(np.argmax(rs))
    return wl, Ar[i], At[i], rs[i]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, (fn, mod, dual, title) in zip(axes, [
        ('struct_B_500_redesign', 'src.simulation.tmm_struct_b', False, '(a) Structure B (ring--disk)'),
        ('struct_C_500_redesign', 'src.simulation.tmm_struct_c_aniso', True, '(b) Structure C (TE)')]):
    wl, Arc, Atm, r = rep_sample(fn, mod, dual)
    ax.plot(wl, Arc, color='#0d47a1', lw=2.0, label=f'RCWA ($r={r:+.2f}$)')
    ax.plot(wl, Atm, color='#0d47a1', lw=1.6, ls='--', alpha=0.85, label='TMM')
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Absorptance'); ax.set_xlim(400, 1800)
    ax.set_title(title, loc='left', fontweight='bold'); ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig('figures/Figure_S4.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_S4.png', bbox_inches='tight', dpi=250)
print('saved Figure_S4 (representative cavity-like TMM vs RCWA, B and C)')
