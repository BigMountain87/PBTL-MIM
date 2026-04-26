"""
논문 Figure 생성 스크립트 (수정판)
- Fig 1: Data Efficiency PBTL 4-way (MAE ×100 적용)
- Fig 2: Negative Transfer (y축 스케일 수정)
- Fig 3: 3-Structure Comparison (가로 레이아웃)
- Fig 4: TMM Accuracy vs TL Benefit (한계점 명시)
- Fig 5: Method Robustness box plot (MAE ×100 적용)
- Fig 6: Random Baseline (이미 올바름, 스타일 통일)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

RESULTS = 'results'

# ── Load data ──
dA = np.load(f'{RESULTS}/pbtl_A_10seed.npz', allow_pickle=True)
dB = np.load(f'{RESULTS}/pbtl_B_10seed.npz', allow_pickle=True)
dC = np.load(f'{RESULTS}/pbtl_C_v2_10seed.npz', allow_pickle=True)
dR = np.load(f'{RESULTS}/random_baseline_10seed.npz', allow_pickle=True)

train_sizes = [50, 100, 200, 350]
methods = ['M0', 'M_phys', 'M_TL', 'M_TL+phys']
method_labels = [r'$M_0$', r'$M_\mathrm{phys}$', r'$M_\mathrm{TL}$', r'$M_\mathrm{TL+phys}$']
method_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']


def get_A_data(d, method, n):
    """Structure A uses keys like '50_M0'."""
    return d[f'{n}_{method}'] * 100  # ×100 fix


def get_BC_data(d, method, n):
    """Structure B/C uses keys like 'M0' with shape (4, 10)."""
    key_map = {'M0': 'M0', 'M_phys': 'M_phys', 'M_TL': 'M_TL', 'M_TL+phys': 'M_TL_phys'}
    idx = train_sizes.index(n)
    return d[key_map[method]][idx] * 100  # ×100 fix


# ═══════════════════════════════════════════════
# Fig 1: Data Efficiency - PBTL 4-way Comparison
# ═══════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

structs = [
    ('A: Dual-Cavity (Accurate TMM)', dA, get_A_data),
    ('B: Ring-Disk (Inaccurate TMM)', dB, get_BC_data),
    ('C: Dual-Pol (No Accuracy Gain)', dC, get_BC_data),
]

for ax, (title, data, get_fn) in zip(axes1, structs):
    for m, label, color in zip(methods, method_labels, method_colors):
        means = []
        stds = []
        for n in train_sizes:
            vals = get_fn(data, m, n)
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(train_sizes, means, 'o-', color=color, label=label, markersize=5, linewidth=1.5)
        ax.fill_between(train_sizes, means - stds, means + stds, alpha=0.15, color=color)
    ax.set_title(title)
    ax.set_xlabel('Training samples (n)')
    ax.set_xticks(train_sizes)
    ax.grid(True, alpha=0.3)

axes1[0].set_ylabel('MAE (%)')
axes1[0].legend(loc='upper right')
fig1.suptitle('Data Efficiency: PBTL 4-way Comparison (10 seeds)', fontsize=13, fontweight='bold')
fig1.tight_layout()
fig1.savefig(f'{RESULTS}/fig1_data_efficiency_fixed.png', bbox_inches='tight')
print("Fig 1 saved")


# ═══════════════════════════════════════════════
# Fig 2: Negative Transfer (y축 자동 스케일)
# ═══════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, data, get_fn, struct_label) in zip(axes2, [
    ('B', dB, get_BC_data, 'Structure B (n=350)'),
    ('C', dC, get_BC_data, 'Structure C v2 (n=350)'),
]):
    m0_vals = get_fn(data, 'M0', 350)
    mtl_vals = get_fn(data, 'M_TL', 350)
    mphys_vals = get_fn(data, 'M_phys', 350)

    x = np.arange(3)
    means = [m0_vals.mean(), mtl_vals.mean(), mphys_vals.mean()]
    stds = [m0_vals.std(), mtl_vals.std(), mphys_vals.std()]
    colors_bar = ['#1f77b4', '#d62728', '#2ca02c']
    labels_bar = [r'$M_0$', r'$M_\mathrm{TL}$', r'$M_\mathrm{phys}$']

    bars = ax.bar(x, means, yerr=stds, width=0.5, color=colors_bar, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel('MAE (%)')

    # 자동 스케일: 데이터 범위에 맞춤
    y_min = min(means) - max(stds) * 2
    y_max = max(means) + max(stds) * 2
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(max(0, y_min - margin), y_max + margin)

    # 값 표시
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + margin * 0.2, f'{m:.2f}%', ha='center', fontsize=9)

    # TL vs M0 비교
    if means[1] > means[0]:
        status = f'$M_{{TL}}$={means[1]:.2f}% > $M_0$={means[0]:.2f}% (negative transfer)'
        color_status = 'red'
    else:
        status = f'$M_{{TL}}$={means[1]:.2f}% < $M_0$={means[0]:.2f}% (positive transfer)'
        color_status = 'green'
    ax.set_title(f'{struct_label}\n{status}', fontsize=10, color=color_status)
    ax.grid(True, alpha=0.3, axis='y')

fig2.suptitle('Negative Transfer: When Physics-Based TL Fails', fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(f'{RESULTS}/fig2_negative_transfer_fixed.png', bbox_inches='tight')
print("Fig 2 saved")


# ═══════════════════════════════════════════════
# Fig 3: 3-Structure Comparison (가로 레이아웃)
# ═══════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))

struct_names = ['A: Dual-Cavity\n(Accurate TMM)', 'B: Ring-Disk\n(Inaccurate TMM)', 'C: Dual-Pol\n(Failed TMM)']
struct_data = [dA, dB, dC]
struct_getters = [get_A_data, get_BC_data, get_BC_data]

for ax, sname, sdata, getter in zip(axes3, struct_names, struct_data, struct_getters):
    m0_350 = getter(sdata, 'M0', 350)
    # Best method at n=350
    best_mae = float('inf')
    best_label = ''
    for m, ml in zip(methods[1:], method_labels[1:]):
        vals = getter(sdata, m, 350)
        if vals.mean() < best_mae:
            best_mae = vals.mean()
            best_label = ml
            best_vals = vals

    m0_mean = m0_350.mean()
    best_mean = best_mae
    improvement = (1 - best_mean / m0_mean) * 100

    x = [0, 1]
    means = [m0_mean, best_mean]
    stds = [m0_350.std(), best_vals.std()]
    colors_b = ['#1f77b4', '#ff7f0e']
    bar_labels = [r'$M_0$ (baseline)', f'{best_label} (best)']

    bars = ax.bar(x, means, yerr=stds, width=0.5, color=colors_b, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel('MAE at n=350 (%)')

    # Improvement annotation
    color_imp = 'green' if improvement > 0 else 'red'
    sign = '+' if improvement < 0 else '-'  # lower MAE = improvement
    ax.annotate(f'{improvement:+.1f}%', xy=(0.5, max(means)),
                fontsize=14, fontweight='bold', ha='center', color=color_imp,
                xytext=(0.5, max(means) + max(stds) * 1.5),
                arrowprops=dict(arrowstyle='->', color=color_imp))

    ax.set_title(sname, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # y축 여유
    ax.set_ylim(0, max(means) + max(stds) * 3)

fig3.suptitle('Physics-Based Transfer Learning Effectiveness\n(10-seed average, n=350)',
              fontsize=13, fontweight='bold')
fig3.tight_layout()
fig3.savefig(f'{RESULTS}/fig3_3structure_comparison_fixed.png', bbox_inches='tight')
print("Fig 3 saved")


# ═══════════════════════════════════════════════
# Fig 4: TMM Accuracy vs TL Benefit (한계점 명시)
# ═══════════════════════════════════════════════
fig4, ax4 = plt.subplots(1, 1, figsize=(7, 5))

# TMM accuracy (correlation with RCWA) - from original analysis
tmm_accuracy = [0.91, 0.60, 0.35]  # A, B, C
struct_labels_4 = ['A: Dual-Cavity', 'B: Ring-Disk', 'C: Dual-Pol']
marker_colors = ['#2ca02c', '#ff7f0e', '#d62728']
marker_sizes = [120, 100, 100]

# TL benefit = (1 - best_TL_MAE / M0_MAE) * 100
tl_benefits = []
for sdata, getter in zip(struct_data, struct_getters):
    m0 = getter(sdata, 'M0', 350).mean()
    best_tl = min(getter(sdata, 'M_TL', 350).mean(), getter(sdata, 'M_TL+phys', 350).mean())
    benefit = (1 - best_tl / m0) * 100
    tl_benefits.append(benefit)

for x, y, label, c, s in zip(tmm_accuracy, tl_benefits, struct_labels_4, marker_colors, marker_sizes):
    ax4.scatter(x, y, s=s, c=c, zorder=5, edgecolors='black', linewidth=0.5)
    ax4.annotate(label, (x, y), textcoords="offset points", xytext=(10, 10), fontsize=10)

# Linear fit
z = np.polyfit(tmm_accuracy, tl_benefits, 1)
p = np.poly1d(z)
x_line = np.linspace(0.2, 1.0, 100)
ax4.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, label=f'Trend: y={z[0]:.1f}x{z[1]:+.1f}')

# Break-even line
ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Break-even')

# Shaded regions
ax4.fill_between(x_line, 0, p(x_line), where=(p(x_line) > 0), alpha=0.08, color='green')
ax4.fill_between(x_line, p(x_line), 0, where=(p(x_line) < 0), alpha=0.08, color='red')

ax4.set_xlabel('TMM Model Accuracy (correlation with RCWA)')
ax4.set_ylabel('TL Benefit vs $M_0$ (%)')
ax4.set_title('Physics Model Quality Determines Transfer Learning Effectiveness', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.set_xlim(0.2, 1.0)
ax4.grid(True, alpha=0.3)

# 한계점 명시 텍스트
ax4.text(0.02, 0.02,
         'Note: N=3 structures; trend is qualitative.\n'
         'Additional structures needed for statistical validation.',
         transform=ax4.transAxes, fontsize=8, fontstyle='italic',
         verticalalignment='bottom', color='gray',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig4.tight_layout()
fig4.savefig(f'{RESULTS}/fig4_tmm_accuracy_correlation_fixed.png', bbox_inches='tight')
print("Fig 4 saved")


# ═══════════════════════════════════════════════
# Fig 5: Method Robustness box plot (MAE ×100)
# ═══════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (title, data, getter) in zip(axes5, [
    ('A: Dual-Cavity', dA, get_A_data),
    ('B: Ring-Disk', dB, get_BC_data),
    ('C: Dual-Pol', dC, get_BC_data),
]):
    box_data = []
    labels_b = []
    colors_b = []
    for m, ml, c in zip(methods, method_labels, method_colors):
        vals = getter(data, m, 350)
        box_data.append(vals)
        labels_b.append(ml)
        colors_b.append(c)

    bp = ax.boxplot(box_data, labels=labels_b, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors_b):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    # Find best
    means_b = [d.mean() for d in box_data]
    best_idx = np.argmin(means_b)
    ax.set_title(f'{title}\n(Best: {labels_b[best_idx]})', fontsize=11)
    ax.set_ylabel('MAE (%)')
    ax.grid(True, alpha=0.3, axis='y')

fig5.suptitle('Method Robustness: Error Distribution Across Seeds (n=350)',
              fontsize=13, fontweight='bold')
fig5.tight_layout()
fig5.savefig(f'{RESULTS}/fig5_method_robustness_fixed.png', bbox_inches='tight')
print("Fig 5 saved")


# ═══════════════════════════════════════════════
# Fig 6: Random Baseline (스타일 통일)
# ═══════════════════════════════════════════════
fig6, ax6 = plt.subplots(1, 1, figsize=(8, 5))

# Random baseline uses Structure A data
rand_key_map = {'M0': 'M0', 'M_phys': 'M_phys', 'M_TL': 'M_TL',
                'M_TL+phys': 'M_TL_phys', 'M_rand': 'M_rand', 'M_rand+phys': 'M_rand_phys'}

mtl_means = []
mrand_means = []
for i, n in enumerate(train_sizes):
    mtl_means.append(dR['M_TL'][i].mean() * 100)
    mrand_means.append(dR['M_rand'][i].mean() * 100)

x = np.arange(len(train_sizes))
width = 0.35

bars1 = ax6.bar(x - width/2, mtl_means, width, label=r'$M_\mathrm{TL}$ (TMM pre-train)',
                color='#2ca02c', alpha=0.85)
bars2 = ax6.bar(x + width/2, mrand_means, width, label=r'$M_\mathrm{rand}$ (Random pre-train)',
                color='#d62728', alpha=0.85)

# Ratio annotations
for i, (tm, rm) in enumerate(zip(mtl_means, mrand_means)):
    ratio = rm / tm
    ax6.text(i, max(tm, rm) + 0.8, f'×{ratio:.2f}', ha='center', fontsize=10,
             fontweight='bold', color='darkred')

ax6.set_xlabel('Training Samples (n)')
ax6.set_ylabel('MAE (%)')
ax6.set_xticks(x)
ax6.set_xticklabels(train_sizes)
ax6.legend(loc='upper right')
ax6.set_title('Random Baseline: TMM Transfers Genuine Physics Knowledge',
              fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

fig6.tight_layout()
fig6.savefig(f'{RESULTS}/fig6_random_baseline_fixed.png', bbox_inches='tight')
print("Fig 6 saved")


# ═══════════════════════════════════════════════
# Summary Table (콘솔 출력)
# ═══════════════════════════════════════════════
print("\n" + "="*80)
print("CORRECTED RESULTS SUMMARY (MAE in %)")
print("="*80)

for sname, sdata, getter in [('Structure A', dA, get_A_data),
                               ('Structure B', dB, get_BC_data),
                               ('Structure C', dC, get_BC_data)]:
    print(f"\n{'─'*60}")
    print(f"  {sname}")
    print(f"{'─'*60}")
    print(f"  {'n':>5}  {'M0':>12}  {'M_phys':>12}  {'M_TL':>12}  {'M_TL+phys':>12}")
    for n in train_sizes:
        vals = {m: getter(sdata, m, n) for m in methods}
        row = f"  {n:>5}"
        for m in methods:
            row += f"  {vals[m].mean():>5.2f}±{vals[m].std():>4.2f}"
        print(row)

print(f"\n{'─'*60}")
print("  Random Baseline (Structure A)")
print(f"{'─'*60}")
print(f"  {'n':>5}  {'M_TL':>12}  {'M_rand':>12}  {'Gap':>8}")
for i, n in enumerate(train_sizes):
    mtl = dR['M_TL'][i].mean() * 100
    mrand = dR['M_rand'][i].mean() * 100
    print(f"  {n:>5}  {mtl:>5.2f}%       {mrand:>5.2f}%       {mrand-mtl:>+5.2f}%")

print("\nAll figures saved to results/ directory with '_fixed' suffix.")
