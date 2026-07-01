"""Generate Figure 3: learning curves (test MAE vs. RCWA training samples) for the
4-way model comparison across Structures A, B, and C, from the CORRECTED redesign
archives (results/pbtl_{A,B,C}_redesign_10seed.npz). Replaces the superseded combined
generator; this is the active, release-pipeline generator for Figure 3."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular',
    'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

C_M0, C_MPHYS, C_MTL, C_MTLP = '#5b6770', '#ef6c00', '#00838f', '#c62828'
LBL_FS, TICK_FS, TITLE_FS, LEG_FS = 10, 9, 11, 8


def load_A():
    d = np.load('results/pbtl_A_redesign_10seed.npz', allow_pickle=True)
    sizes = d['train_sizes']
    out = {}
    for k in ['M0', 'M_phys', 'M_TL', 'M_TL+phys']:
        out[k] = np.array([d[f'{n}_{k}'] for n in sizes]) * 100  # (n_sizes, n_seeds) in %
    return sizes, out


def load_BC(name):
    d = np.load(f'results/{name}.npz', allow_pickle=True)
    sizes = d['train_sizes']
    return sizes, {'M0': d['M0'] * 100, 'M_phys': d['M_phys'] * 100,
                   'M_TL': d['M_TL'] * 100, 'M_TL+phys': d['M_TL_phys'] * 100}


fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
panels = [
    ('(a) Structure A', *load_A()),
    ('(b) Structure B', *load_BC('pbtl_B_redesign_10seed')),
    ('(c) Structure C', *load_BC('pbtl_C_v2_redesign_10seed')),
]
models = [('M_0', 'M0', C_M0, 'o'),
          (r'M_{\mathrm{phys}}', 'M_phys', C_MPHYS, 's'),
          (r'M_{\mathrm{TL}}', 'M_TL', C_MTL, '^'),
          (r'M_{\mathrm{TL+phys}}', 'M_TL+phys', C_MTLP, 'D')]

for ax, (title, sizes, data) in zip(axes, panels):
    for label_tex, key, color, marker in models:
        arr = data[key]
        ax.errorbar(sizes, arr.mean(axis=1), yerr=arr.std(axis=1), label=fr'${label_tex}$',
                    marker=marker, color=color, lw=1.3, ms=4.5,
                    capsize=2.5, capthick=0.8, elinewidth=0.8)
    ax.set_xlabel('RCWA training samples', fontsize=LBL_FS)
    ax.set_ylabel('Test MAE (%)', fontsize=LBL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left')
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.tick_params(labelsize=TICK_FS); ax.grid(True, alpha=0.30, lw=0.5)
    ax.legend(fontsize=LEG_FS, loc='best', framealpha=0.92)

plt.tight_layout()
plt.savefig('figures/Figure_3.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_3.png', bbox_inches='tight', dpi=250)
print('saved Figure_3 (learning curves) from corrected redesign archives')
