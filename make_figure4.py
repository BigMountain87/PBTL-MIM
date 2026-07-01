"""Regenerate Figure 4: TMM-RCWA fidelity vs. weight-transfer benefit, two panels
(benefit vs shape correlation r, and benefit vs operating-band MAE). Values are the
published Structure-A noise-injection conditions (Table tab:tmm_noise) and the
cross-structure points at n=350 (Table tab:tmm_fidelity); this script makes the figure
reproducible from those reported numbers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular',
    'axes.linewidth': 0.8, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 7.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# Structure-A noise conditions (RCWA n=100): r, MAE(%), benefit(%)  -- Table tab:tmm_noise
noise_r   = [0.77, 0.50, 0.35, 0.26, 0.21]
noise_mae = [7.4,  8.0,  10.8, 14.0, 16.7]
noise_ben = [47.3, 40.8, 33.0, 32.3, 22.6]
rand_r, rand_mae, rand_ben = -0.03, 33.0, -57.9            # zero-fidelity control
# Cross-structure points at n=350: r, MAE(%), benefit(%)  -- Table tab:tmm_fidelity (M_TL vs M0)
cs = {'A': (0.83, 7.9, 31.3), 'B': (0.96, 8.9, 13.5), 'C': (0.65, 16.9, 9.7)}
cs_col = {'A': '#0d47a1', 'B': '#00838f', 'C': '#c62828'}

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, (xs_n, xs_r, xlab, panel) in zip(
        axes,
        [(noise_r, rand_r, 'TMM--RCWA shape correlation $r$', '(a)'),
         (noise_mae, rand_mae, 'TMM--RCWA operating-band MAE (%)', '(b)')]):
    ax.scatter(xs_n, noise_ben, s=42, facecolors='none', edgecolors='0.45',
               label='Structure A, noise $\\sigma{=}0$--$0.20$', zorder=3)
    ax.scatter([xs_r], [rand_ben], marker='x', s=55, color='k',
               label='Random (zero fidelity)', zorder=3)
    for s, (r, mae, ben) in cs.items():
        ax.scatter([r if panel == '(a)' else mae], [ben], marker='*', s=150,
                   color=cs_col[s], edgecolors='k', linewidths=0.4,
                   label=f'Structure {s} ($n{{=}}350$)', zorder=4)
    ax.axhline(0, color='0.7', lw=0.7, ls='--')
    ax.set_xlabel(xlab); ax.set_title(panel, loc='left', fontweight='bold')
    ax.grid(True, alpha=0.3, lw=0.5)
axes[0].set_ylabel('Weight-transfer benefit (%)')
# descriptive Pearson over the six Structure-A conditions (panel labels)
allr = noise_r + [rand_r]; allm = noise_mae + [rand_mae]; allb = noise_ben + [rand_ben]
pr = abs(stats.pearsonr(allr, allb)[0]); pm = abs(stats.pearsonr(allm, allb)[0])
# descriptive |Pearson| of benefit vs. the panel's x-quantity (boxed, placed in the
# empty lower-centre so it reads as an annotation and clears the data + random marker)
_bb = dict(boxstyle='round,pad=0.35', fc='white', ec='0.65', lw=0.6, alpha=0.92)
axes[0].text(0.5, 0.13, f'benefit vs. $r$:  $|\\mathrm{{Pearson}}|={pr:.2f}$',
             transform=axes[0].transAxes, fontsize=8, ha='center', bbox=_bb)
axes[1].text(0.5, 0.13, f'benefit vs. MAE:  $|\\mathrm{{Pearson}}|={pm:.2f}$',
             transform=axes[1].transAxes, fontsize=8, ha='center', bbox=_bb)
# single shared legend below both panels (keeps it off the data -- the panel-(b)
# Structure-C star sits where an in-axes legend would occlude it)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5,
           framealpha=0.9, fontsize=7, handletextpad=0.3, columnspacing=1.0)
plt.tight_layout()
plt.savefig('figures/Figure_4.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_4.png', bbox_inches='tight', dpi=250)
print(f'saved Figure_4  |r|(shape)={pr:.2f}  |r|(MAE)={pm:.2f}')
