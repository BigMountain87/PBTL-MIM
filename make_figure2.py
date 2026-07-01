"""Generate Figure 2: TMM vs. RCWA absorptance spectra (Structure A).

Design:
  - Two panels (high-fidelity vs. low-fidelity), each showing TWO sample pairs
  - Each sample = distinct dark hue (sample 1 navy, sample 2 teal)
  - RCWA   : solid thick line (lw=2.0)
  - TMM    : same-hue dashed thinner line (lw=1.6)
  - Single legend per panel groups by sample, eliminating colour-overlap ambiguity
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import src.simulation.materials as _mat; _mat.MATERIAL_MODEL = 'jc'  # corrected J&C constants
from src.simulation.tmm_struct_a import compute_tmm_batch

# Publication-grade rcParams — UNIFIED across Figures 2–5 (print-size matched)
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

# ─── Load RCWA data ───
d = np.load('data/raw/struct_A_500_redesign.npz', allow_pickle=True)
params = d['params'].astype(np.float32)
rcwa_A = np.clip(d['A'].astype(np.float32), 0.0, 1.0)  # match training reliability clip
wavelengths = d['wavelengths'].astype(np.float32)

# ─── Compute TMM for all 500 samples ───
tmm_out = compute_tmm_batch(params, wavelengths)
tmm_A = tmm_out['A_tmm'].astype(np.float32)

# ─── Per-sample Pearson r ───
N = rcwa_A.shape[0]
r_samples = np.array([stats.pearsonr(rcwa_A[i], tmm_A[i])[0] for i in range(N)])

# ─── Pick representative samples ───
# High-fidelity: 2 samples with r closest to 1.0
hi_order = np.argsort(-r_samples)
hi_idx = hi_order[:2]
# Low-fidelity: 2 samples with the most negative r
lo_order = np.argsort(r_samples)
lo_idx = lo_order[:2]

print(f'High-fidelity samples: indices {hi_idx.tolist()}, r = {r_samples[hi_idx].round(3).tolist()}')
print(f'Low-fidelity samples:  indices {lo_idx.tolist()}, r = {r_samples[lo_idx].round(3).tolist()}')

# ─── Sample colours (two hues per panel) ───
COLORS = ['#0d47a1', '#00838f']     # navy, teal
LBL_FS = 10
TICK_FS = 9
TITLE_FS = 11

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))

def plot_panel(ax, sample_idx, title, legend_loc, ymax_pad=0.18):
    ymin_panel, ymax_panel = np.inf, -np.inf
    for k, idx in enumerate(sample_idx):
        c = COLORS[k]
        rc = rcwa_A[idx]
        tm = tmm_A[idx]
        r_i = r_samples[idx]
        ax.plot(wavelengths, rc, color=c, lw=2.0, ls='-',
                label=f'Sample {k+1} RCWA ($r={r_i:+.2f}$)')
        ax.plot(wavelengths, tm, color=c, lw=1.6, ls='--', alpha=0.85,
                label=f'Sample {k+1} TMM')
        ymin_panel = min(ymin_panel, rc.min(), tm.min())
        ymax_panel = max(ymax_panel, rc.max(), tm.max())
    ax.set_xlabel('Wavelength (nm)', fontsize=LBL_FS)
    ax.set_ylabel('Absorptance', fontsize=LBL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left')
    ax.set_xlim(400, 1800)
    span = ymax_panel - ymin_panel
    ax.set_ylim(ymin_panel - 0.05 * span, ymax_panel + ymax_pad * span)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.30, lw=0.5)
    ax.legend(fontsize=7, loc=legend_loc, framealpha=0.92, ncol=1,
              handlelength=1.8, handletextpad=0.4, borderpad=0.35)

plot_panel(axes[0], hi_idx, '(a) High TMM fidelity', legend_loc='upper right',
           ymax_pad=0.45)
plot_panel(axes[1], lo_idx, '(b) Low TMM fidelity',  legend_loc='upper left',
           ymax_pad=0.30)

plt.tight_layout()
plt.savefig('figures/Figure_2.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/Figure_2.png', bbox_inches='tight', dpi=250)
print('Saved figures/Figure_2.pdf and Figure_2.png')
