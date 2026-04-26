"""Regenerate Figures 3, 4, 5 in the unified house style.

Style choices (consistent with Figures 1 and 2):
  - sans-serif (DejaVu Sans), Type 42 fonts
  - axis label 11, tick 10, title 12 bold, legend 9-10
  - palette: M0 slate, M_phys orange, M_TL teal, M_TL+phys crimson
  - structure palette: A navy, B teal, C orange
  - grid alpha 0.30, lw 0.5
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/Users/sbchoi129/PINN2/mim_novel')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── Unified rcParams (consistent with Figure 2; print-size matched) ───
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Model colours (used in Figures 3 + 5 categories)
C_M0     = '#5b6770'
C_MPHYS  = '#ef6c00'
C_MTL    = '#00838f'
C_MTLP   = '#c62828'

# Structure-level palette (for Figures 1, 2, 4)
C_STRUCT_A = '#0d47a1'
C_STRUCT_B = '#00838f'
C_STRUCT_C = '#ef6c00'

LBL_FS, TICK_FS, TITLE_FS, LEG_FS = 10, 9, 11, 8

# ════════════════════════════════════════════════════════════════════
# Figure 3 — Learning curves (test MAE vs RCWA training samples)
# ════════════════════════════════════════════════════════════════════

def load_A():
    d = np.load('results/pbtl_A_10seed.npz', allow_pickle=True)
    sizes = d['train_sizes']
    out = {}
    for k in ['M0', 'M_phys', 'M_TL', 'M_TL+phys']:
        rows = [d[f'{n}_{k}'] for n in sizes]      # list of length-10 arrays
        out[k] = np.array(rows) * 100              # (4, 10) in %
    return sizes, out

def load_BC(name):
    d = np.load(f'results/{name}.npz', allow_pickle=True)
    sizes = d['train_sizes']
    return sizes, {'M0': d['M0'] * 100, 'M_phys': d['M_phys'] * 100,
                   'M_TL': d['M_TL'] * 100, 'M_TL+phys': d['M_TL_phys'] * 100}


fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
panels = [
    ('(a) Structure A', *load_A()),
    ('(b) Structure B', *load_BC('pbtl_B_10seed')),
    ('(c) Structure C', *load_BC('pbtl_C_v2_10seed')),
]

models = [('M_0', 'M0', C_M0, 'o'),
          (r'M_{\mathrm{phys}}', 'M_phys', C_MPHYS, 's'),
          (r'M_{\mathrm{TL}}', 'M_TL', C_MTL, '^'),
          (r'M_{\mathrm{TL+phys}}', 'M_TL+phys', C_MTLP, 'D')]

for ax, (title, sizes, data) in zip(axes, panels):
    for label_tex, key, color, marker in models:
        arr = data[key]                            # (4, n_seeds) in %
        mean = arr.mean(axis=1)
        std = arr.std(axis=1)
        ax.errorbar(sizes, mean, yerr=std, label=fr'${label_tex}$',
                    marker=marker, color=color, lw=1.3, ms=4.5,
                    capsize=2.5, capthick=0.8, elinewidth=0.8)
    ax.set_xlabel('RCWA training samples', fontsize=LBL_FS)
    ax.set_ylabel('Test MAE (%)', fontsize=LBL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=0)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.30, lw=0.5)
    ax.legend(fontsize=LEG_FS, loc='best', framealpha=0.92)

plt.tight_layout()
plt.savefig('figures/Figure_3.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_3.png', bbox_inches='tight', dpi=250)
plt.close()
print('Saved Figure_3 (learning curves).')


# ════════════════════════════════════════════════════════════════════
# Figure 4 — TMM fidelity vs Transfer-Learning benefit
# ════════════════════════════════════════════════════════════════════

# Noise-injection data (within Structure A): gray points
d_noise = np.load('results/tmm_accuracy_variation.npz', allow_pickle=True)
r_noise = np.array(d_noise['tmm_accuracies'])      # 6 values
benefit_noise = np.array(d_noise['tl_benefits'])   # 6 values (%) — last is random
sigmas_noise = np.array(d_noise['noise_sigmas'])   # finite for physics, inf for random

# Cross-structure points (Table tab:tmm_fidelity in paper):
# (name, r, gain, color, x_text_offset)
struct_data = [
    ('Structure A', 0.72, 29.4, C_STRUCT_A, +0.04),   # median r ; best-model gain at n=350
    ('Structure B', -0.07, 5.9, C_STRUCT_B, +0.04),
    ('Structure C', 0.34, 4.2, C_STRUCT_C, +0.04),
]

fig, ax = plt.subplots(figsize=(4.5, 3.3))

# Noise points (exclude the random point so the curve does not extrapolate to large negative).
# Random pre-training corresponds to sigma=inf in the noise sweep; physics-based
# conditions have finite sigma. Match the paper's primary N=5 fit.
phys_mask = np.isfinite(sigmas_noise)
ax.scatter(r_noise[phys_mask], benefit_noise[phys_mask],
           s=90, fc='#cfd8dc', ec='#37474f', lw=0.8, zorder=3,
           label='Noise injection (Structure A)')

# Random extreme as black X
rand_mask = ~phys_mask
ax.scatter(r_noise[rand_mask], benefit_noise[rand_mask],
           s=130, marker='X', color='black', lw=0.8, zorder=4,
           label='Random pre-training')

# Linear fit on N = 5 physics-based points (excluding random leverage point)
xv, yv = r_noise[phys_mask], benefit_noise[phys_mask]
if len(xv) >= 2:
    slope, intercept = np.polyfit(xv, yv, 1)
    xfit = np.linspace(min(min(xv), -0.05), max(max(xv), 0.75), 100)
    ax.plot(xfit, slope * xfit + intercept, ls='--', color='#90a4ae', lw=1.4,
            label=f'Linear fit ($r=0.981$, $N=5$)', zorder=2)

# Cross-structure stars
for name, r, gain, color, x_off in struct_data:
    ax.scatter([r], [gain], s=240, marker='*', color=color,
               ec='black', lw=0.8, zorder=5,
               label=name)
    ha = 'right' if x_off < 0 else 'left'
    ax.text(r + x_off, gain, name, fontsize=9, color=color,
            fontweight='bold', va='center', ha=ha)

ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.7)
ax.set_xlabel(r'TMM--RCWA spectral correlation $r$', fontsize=LBL_FS)
ax.set_ylabel('Transfer-learning benefit (%)', fontsize=LBL_FS)
ax.set_title('TMM fidelity vs. transfer-learning benefit',
             fontsize=TITLE_FS, fontweight='bold', loc='center', pad=16)
ax.tick_params(labelsize=TICK_FS)
ax.grid(True, alpha=0.30, lw=0.5)

leg = ax.legend(fontsize=7, loc='lower right', framealpha=0.92, markerscale=0.65)
plt.tight_layout()
plt.savefig('figures/Figure_4.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_4.png', bbox_inches='tight', dpi=250)
plt.close()
print('Saved Figure_4 (fidelity vs benefit).')


# ════════════════════════════════════════════════════════════════════
# Figure 5 — Permutation feature importance (Pareto chart)
# ════════════════════════════════════════════════════════════════════

d = np.load('results/feature_importance_A.npz', allow_pickle=True)
imp = d['imp_mphys_mean'] * 100                    # (17,) in %
cat_id = d['category_id']
cat_names = list(d['category_names'])

# Aggregate by category
cat_totals = np.zeros(len(cat_names))
cat_counts = np.zeros(len(cat_names), dtype=int)
for i, c in enumerate(cat_id):
    cat_totals[c] += imp[i]
    cat_counts[c] += 1
order = np.argsort(-cat_totals)
cat_totals = cat_totals[order]
cat_names_o = [cat_names[i] for i in order]
cat_counts_o = cat_counts[order]

# Cumulative percentage
cum_pct = np.cumsum(cat_totals) / cat_totals.sum() * 100

# Unified palette (sequential for category bars, qualitative)
bar_colors = ['#c62828', '#ef6c00', '#2e7d32', '#1565c0', '#6a1b9a', '#5d4037']

fig, ax1 = plt.subplots(figsize=(5.0, 3.0))
xs = np.arange(len(cat_names_o))
bars = ax1.bar(xs, cat_totals, color=bar_colors[:len(xs)],
               ec='black', lw=0.7, zorder=3)
for i, (b, n) in enumerate(zip(bars, cat_counts_o)):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.20,
             f'({n} feat)', ha='center', va='bottom',
             fontsize=8, color='#37474f')

_LABEL_MAP = {
    'cavity resonance':  'Cavity\nResonance',
    'fill fraction':     'Fill\nFraction',
    'angle & geometry':  'Angle &\nGeometry',
    'sub-wavelength ratio': 'Sub-λ\nRatio',
    'sub-wave ratio':    'Sub-λ\nRatio',
    'skin depth ratio':  'Skin Depth\nRatio',
    'optical path':      'Optical\nPath',
}
def _format_label(n: str) -> str:
    key = n.lower().strip()
    return _LABEL_MAP.get(key, n)
ax1.set_xticks(xs)
ax1.set_xticklabels([_format_label(n) for n in cat_names_o],
                    fontsize=7.5)
ax1.set_ylabel(r'Permutation importance ($\Delta$MAE, %)', fontsize=LBL_FS)
ax1.tick_params(axis='y', labelsize=TICK_FS)
ax1.grid(True, alpha=0.30, lw=0.5, axis='y')
ax1.set_ylim(0, max(cat_totals) * 1.18)

# Cumulative percentage on twin y-axis
ax2 = ax1.twinx()
ax2.plot(xs, cum_pct, color='black', marker='o', lw=1.5, ms=6, zorder=5)
ax2.set_ylabel('Cumulative %', fontsize=LBL_FS)
ax2.tick_params(axis='y', labelsize=TICK_FS)
ax2.set_ylim(0, 105)

ax1.set_title('Physics feature importance by category (Structure A)',
              fontsize=TITLE_FS, fontweight='bold', loc='center', pad=16)

plt.tight_layout()
plt.savefig('figures/Figure_5.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_5.png', bbox_inches='tight', dpi=250)
plt.close()
print('Saved Figure_5 (feature importance).')

print('\nAll three figures regenerated in unified house style.')
