"""R2 Q1: M_TL+phys surrogate predicted spectra vs RCWA truth (and the TMM pre-train
source) for representative test samples spanning high / median / low TMM-RCWA fidelity.
Shows that the fine-tuned surrogate tracks the full-wave RCWA even where the TMM source
(its pre-training signal) diverges. Reads results/persample_A_r2.npz (n=350, seed 42)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular', 'axes.linewidth': 0.8,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})

d = np.load('results/persample_A_r2.npz', allow_pickle=True)
A_rcwa = d['A_rcwa']; A_tmm = d['A_tmm']; A_pred = d['A_pred_tlp']; r = d['ps_r']; wl = d['wl']
mae_tlp = d['ps_mae_tlp']

order = np.argsort(r)
pick = {'(a) High TMM fidelity': order[-1],
        '(b) Median TMM fidelity': order[len(order) // 2],
        '(c) Low TMM fidelity': order[0]}

fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
for ax, (title, i) in zip(axes, pick.items()):
    ax.plot(wl, A_rcwa[i], color='#0d47a1', lw=2.0, label='RCWA (truth)')
    ax.plot(wl, A_pred[i], color='#c62828', lw=1.6, ls='-', alpha=0.9,
            label=f'$M_{{TL+phys}}$ pred (MAE {mae_tlp[i]*100:.1f}%)')
    ax.plot(wl, A_tmm[i], color='0.45', lw=1.4, ls='--', alpha=0.85,
            label=f'TMM source ($r={r[i]:+.2f}$)')
    ax.set_xlabel('Wavelength (nm)'); ax.set_xlim(400, 1800)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.grid(True, alpha=0.3, lw=0.5); ax.legend(fontsize=6.8, loc='best', framealpha=0.9)
axes[0].set_ylabel('Absorptance')
plt.tight_layout()
plt.savefig('figures/Figure_S1.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_S1.png', bbox_inches='tight', dpi=250)
print('saved Figure_S1; picked r =', {k: round(float(r[i]), 3) for k, i in pick.items()},
      'MAE_TLphys =', {k: round(float(mae_tlp[i]*100), 2) for k, i in pick.items()})
