"""R2 Q2: per-sample surrogate test error vs per-sample TMM-RCWA shape correlation r.
Tests whether the FINE-TUNED surrogate's per-sample accuracy depends on the source-TMM
fidelity of that sample. Because both M0 and M_TL+phys are trained on RCWA truth, the
per-sample test error should be largely independent of the per-sample source r (the
fidelity dependence lives in the *aggregate* weight-transfer benefit, not per test point).
Reads results/persample_A_r2.npz (n=350, seed 42)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular', 'axes.linewidth': 0.8,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})

d = np.load('results/persample_A_r2.npz', allow_pickle=True)
r = d['ps_r']; mae_tlp = d['ps_mae_tlp'] * 100; mae_m0 = d['ps_mae_m0'] * 100

rho_tlp = stats.spearmanr(r, mae_tlp)[0]
rho_m0 = stats.spearmanr(r, mae_m0)[0]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.scatter(r, mae_m0, s=34, facecolors='none', edgecolors='0.5', lw=0.9,
           label=f'$M_0$ (from scratch), $\\rho={rho_m0:+.2f}$')
ax.scatter(r, mae_tlp, s=40, color='#c62828', alpha=0.85,
           label=f'$M_{{TL+phys}}$, $\\rho={rho_tlp:+.2f}$')
ax.set_xlabel('Per-sample TMM--RCWA shape correlation $r$')
ax.set_ylabel('Per-sample test MAE (%)')
ax.grid(True, alpha=0.3, lw=0.5)
ax.legend(fontsize=8, framealpha=0.9, loc='best')
ax.set_title('Surrogate error vs. per-sample source fidelity', fontsize=10, loc='left', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/Figure_S2.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_S2.png', bbox_inches='tight', dpi=250)
print(f'saved Figure_S2; n={len(r)}  Spearman rho(r, MAE): M_TL+phys={rho_tlp:+.3f}  M0={rho_m0:+.3f}')
print(f'  MAE_TLphys median={np.median(mae_tlp):.2f}%  MAE_M0 median={np.median(mae_m0):.2f}%')
