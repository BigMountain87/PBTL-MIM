"""Regenerate Figure 5: category-level permutation feature importance (Structure A,
M_phys, n=350) from the CORRECTED redesign archive (replaces the stale obsolete plot)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular',
    'axes.linewidth': 0.8, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

d = np.load('results/feature_importance_A_redesign.npz', allow_pickle=True)
cats = d['category_names']; cid = d['category_id']; imp = d['imp_mphys_mean']
labels, vals = [], []
for c in range(len(cats)):
    m = (cid == c)
    if m.sum() > 0:
        labels.append(str(cats[c])); vals.append(float(imp[m].sum()))
vals = np.array(vals); pct = 100 * vals / vals.sum()
order = np.argsort(-pct)
labels = [labels[i] for i in order]; pct = pct[order]

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.barh(range(len(labels)), pct, color='#0d47a1', height=0.66)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Category permutation importance (%)')
ax.set_xlim(0, max(pct) * 1.18)
for i, p in enumerate(pct):
    ax.text(p + 0.6, i, f'{p:.0f}%', va='center', fontsize=8)
ax.grid(True, axis='x', alpha=0.3, lw=0.5)
plt.tight_layout()
plt.savefig('figures/Figure_5.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_5.png', bbox_inches='tight', dpi=250)
print('saved Figure_5:', list(zip(labels, pct.round(1))))
print(f'cavity+fill = {pct[0]+pct[1]:.1f}%')
