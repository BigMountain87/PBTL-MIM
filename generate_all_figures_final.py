#!/usr/bin/env python3
"""
Generate all publication-quality figures for the PBTL manuscript.
All MAE values in NPZ files are stored as fractions; multiply by 100 for %.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path

# ── Global settings ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

RESULTS = Path("/Users/sbchoi129/PINN2/mim_novel/results")
TRAIN_SIZES = [50, 100, 200, 350]

# Consistent colour palette
C = {
    "M0":       "#1f77b4",   # blue
    "M_phys":   "#2ca02c",   # green
    "M_TL":     "#d62728",   # red
    "M_TL+phys":"#ff7f0e",   # orange
    "M_rand":   "#9467bd",   # purple
    "M_rand+phys": "#8c564b",# brown
}
METHOD_LABELS = {
    "M0": "$M_0$",
    "M_phys": r"$M_{\mathrm{phys}}$",
    "M_TL": "$M_{TL}$",
    "M_TL+phys": r"$M_{TL+\mathrm{phys}}$",
    "M_rand": "$M_{rand}$",
    "M_rand+phys": r"$M_{rand+\mathrm{phys}}$",
}
METHODS4 = ["M0", "M_phys", "M_TL", "M_TL+phys"]

# ── Helper: load Structure A (flat keys) into (4, 10) arrays ─────────────
def load_A():
    d = np.load(RESULTS / "pbtl_A_10seed.npz")
    out = {}
    for m in METHODS4:
        arr = np.stack([d[f"{n}_{m}"] for n in TRAIN_SIZES])  # (4, 10)
        out[m] = arr
    return out

# ── Helper: load B / C (matrix keys) ────────────────────────────────────
def load_BC(fname):
    d = np.load(RESULTS / fname)
    key_map = {"M0": "M0", "M_phys": "M_phys", "M_TL": "M_TL", "M_TL+phys": "M_TL_phys"}
    return {m: d[key_map[m]] for m in METHODS4}

# ── Helper: paired t-test p-value ────────────────────────────────────────
def paired_pval(a, b):
    """Two-sided paired t-test, returns p-value."""
    _, p = stats.ttest_rel(a, b)
    return p

def sig_marker(p):
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""

# ── Helper: propagated std for difference of means ──────────────────────
def diff_std(a, b):
    """Std of (a - b) for paired observations."""
    return np.std(a - b, ddof=1) / np.sqrt(len(a))

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Data Efficiency (3-panel, one per structure)
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    datasets = {
        "Structure A": load_A(),
        "Structure B": load_BC("pbtl_B_10seed.npz"),
        "Structure C": load_BC("pbtl_C_v2_10seed.npz"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, (title, data) in zip(axes, datasets.items()):
        for m in METHODS4:
            means = data[m].mean(axis=1) * 100
            stds  = data[m].std(axis=1, ddof=1) * 100
            ax.errorbar(TRAIN_SIZES, means, yerr=stds, marker="o", ms=5,
                        capsize=3, lw=1.5, label=METHOD_LABELS[m], color=C[m])
        ax.set_xlabel("Training samples ($n_{train}$)")
        ax.set_ylabel("MAE (%)")
        ax.set_title(title)
        ax.set_xticks(TRAIN_SIZES)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor="0.8")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(RESULTS / "final_fig1_data_efficiency.png")
    fig.savefig(RESULTS / "final_fig1_data_efficiency.pdf")
    plt.close(fig)
    print("  Fig 1 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Negative Transfer (2-panel, B and C)
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (label, fname) in zip(axes, [("Structure B", "pbtl_B_10seed.npz"),
                                          ("Structure C", "pbtl_C_v2_10seed.npz")]):
        data = load_BC(fname)
        x = np.arange(len(TRAIN_SIZES))
        deltas_mean = []
        deltas_err  = []
        colors_bar  = []
        for i in range(len(TRAIN_SIZES)):
            tl = data["M_TL"][i] * 100
            m0 = data["M0"][i] * 100
            delta = tl - m0  # positive = TL hurts
            dm = delta.mean()
            de = diff_std(tl, m0)
            deltas_mean.append(dm)
            deltas_err.append(de)
            colors_bar.append("#d62728" if dm > 0 else "#2ca02c")

        bars = ax.bar(x, deltas_mean, yerr=deltas_err, capsize=4,
                       color=colors_bar, edgecolor="k", lw=0.5, width=0.55)

        # significance markers
        for i in range(len(TRAIN_SIZES)):
            tl = data["M_TL"][i] * 100
            m0 = data["M0"][i] * 100
            p = paired_pval(tl, m0)
            s = sig_marker(p)
            if s:
                ypos = deltas_mean[i] + (deltas_err[i] + 0.15) * np.sign(deltas_mean[i])
                ax.text(i, ypos, s, ha="center", va="bottom" if deltas_mean[i] > 0 else "top",
                        fontsize=11, fontweight="bold")

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in TRAIN_SIZES])
        ax.set_xlabel("Training samples ($n_{train}$)")
        ax.set_title(label)

    axes[0].set_ylabel(r"$\Delta$ MAE (%) : $M_{TL} - M_0$")
    fig.tight_layout()
    fig.savefig(RESULTS / "final_fig2_negative_transfer.png")
    fig.savefig(RESULTS / "final_fig2_negative_transfer.pdf")
    plt.close(fig)
    print("  Fig 2 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – 3-Structure Comparison at n=350 (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    datasets = {
        "Structure A": load_A(),
        "Structure B": load_BC("pbtl_B_10seed.npz"),
        "Structure C": load_BC("pbtl_C_v2_10seed.npz"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    bar_w = 0.18
    x = np.arange(len(METHODS4))

    for ax, (title, data) in zip(axes, datasets.items()):
        vals_350 = {m: data[m][3] * 100 for m in METHODS4}  # index 3 = n=350
        means = [vals_350[m].mean() for m in METHODS4]
        stds  = [vals_350[m].std(ddof=1) for m in METHODS4]
        bars = ax.bar(x, means, yerr=stds, capsize=3, width=0.55,
                      color=[C[m] for m in METHODS4], edgecolor="k", lw=0.5)

        # percentage improvement relative to M0
        m0_mean = means[0]
        for j, m in enumerate(METHODS4):
            if j == 0:
                continue
            pct = (m0_mean - means[j]) / m0_mean * 100
            if pct > 0:
                ax.annotate(f"-{pct:.1f}%", xy=(j, means[j] + stds[j]),
                            ha="center", va="bottom", fontsize=8, color="0.3")

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS4])
        ax.set_ylabel("MAE (%)")
        ax.set_title(f"{title}  ($n = 350$)")

    fig.tight_layout()
    fig.savefig(RESULTS / "final_fig3_structure_comparison_350.png")
    fig.savefig(RESULTS / "final_fig3_structure_comparison_350.pdf")
    plt.close(fig)
    print("  Fig 3 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Method Robustness (box plots at n=350)
# ═══════════════════════════════════════════════════════════════════════════
def fig5():
    datasets = {
        "Structure A": load_A(),
        "Structure B": load_BC("pbtl_B_10seed.npz"),
        "Structure C": load_BC("pbtl_C_v2_10seed.npz"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (title, data) in zip(axes, datasets.items()):
        box_data = [data[m][3] * 100 for m in METHODS4]
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=4, markerfacecolor="grey"))
        for patch, m in zip(bp["boxes"], METHODS4):
            patch.set_facecolor(C[m])
            patch.set_alpha(0.7)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS4])
        ax.set_ylabel("MAE (%)")
        ax.set_title(f"{title}  ($n = 350$)")

    fig.tight_layout()
    fig.savefig(RESULTS / "final_fig5_robustness_boxplots.png")
    fig.savefig(RESULTS / "final_fig5_robustness_boxplots.pdf")
    plt.close(fig)
    print("  Fig 5 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Random Baseline (grouped bars WITH error bars)
# ═══════════════════════════════════════════════════════════════════════════
def fig6():
    d = np.load(RESULTS / "random_baseline_10seed.npz")
    key_map = {"M0": "M0", "M_TL": "M_TL", "M_rand": "M_rand", "M_rand+phys": "M_rand_phys"}
    methods = ["M0", "M_TL", "M_rand", "M_rand+phys"]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_groups = len(TRAIN_SIZES)
    n_bars = len(methods)
    bar_w = 0.18
    x = np.arange(n_groups)

    for j, m in enumerate(methods):
        vals = d[key_map[m]] * 100  # (4, 10)
        means = vals.mean(axis=1)
        stds  = vals.std(axis=1, ddof=1)
        ax.bar(x + j * bar_w, means, bar_w, yerr=stds, capsize=3,
               color=C[m], edgecolor="k", lw=0.5, label=METHOD_LABELS[m])

    ax.set_xticks(x + bar_w * (n_bars - 1) / 2)
    ax.set_xticklabels([str(n) for n in TRAIN_SIZES])
    ax.set_xlabel("Training samples ($n_{train}$)")
    ax.set_ylabel("MAE (%)")
    ax.set_title("Random-Weight Baseline Comparison (Structure A)")
    ax.legend(frameon=True, edgecolor="0.8")
    fig.tight_layout()
    fig.savefig(RESULTS / "final_fig6_random_baseline.png")
    fig.savefig(RESULTS / "final_fig6_random_baseline.pdf")
    plt.close(fig)
    print("  Fig 6 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE W4 – LR Ablation (2x2)
# ═══════════════════════════════════════════════════════════════════════════
def figW4():
    d = np.load(RESULTS / "lr_ablation_A.npz")
    ts = TRAIN_SIZES

    def get(prefix, lr):
        return np.array([d[f"{n}_{prefix}_lr{lr}"] * 100 for n in ts])  # (4, seeds)

    m0_1e3 = get("M0", "1e-3")
    m0_3e4 = get("M0", "3e-4")
    mtl_1e3 = get("MTL", "1e-3")
    mtl_3e4 = get("MTL", "3e-4")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel 1: M0 at two LRs
    ax = axes[0, 0]
    ax.errorbar(ts, m0_1e3.mean(1), m0_1e3.std(1, ddof=1), marker="o", capsize=3,
                label="$M_0$ (lr=1e-3)", color="#1f77b4", lw=1.5)
    ax.errorbar(ts, m0_3e4.mean(1), m0_3e4.std(1, ddof=1), marker="s", capsize=3,
                label="$M_0$ (lr=3e-4)", color="#aec7e8", lw=1.5)
    ax.set_title("$M_0$: Learning Rate Comparison")
    ax.set_xlabel("$n_{train}$"); ax.set_ylabel("MAE (%)")
    ax.set_xticks(ts); ax.legend(frameon=True)

    # Panel 2: M_TL at two LRs
    ax = axes[0, 1]
    ax.errorbar(ts, mtl_1e3.mean(1), mtl_1e3.std(1, ddof=1), marker="o", capsize=3,
                label="$M_{TL}$ (lr=1e-3)", color="#d62728", lw=1.5)
    ax.errorbar(ts, mtl_3e4.mean(1), mtl_3e4.std(1, ddof=1), marker="s", capsize=3,
                label="$M_{TL}$ (lr=3e-4)", color="#ff9896", lw=1.5)
    ax.set_title("$M_{TL}$: Learning Rate Comparison")
    ax.set_xlabel("$n_{train}$"); ax.set_ylabel("MAE (%)")
    ax.set_xticks(ts); ax.legend(frameon=True)

    # Panel 3: TL benefit at matched LRs
    ax = axes[1, 0]
    for lr, ls, alpha in [("1e-3", "-", 1.0), ("3e-4", "--", 0.7)]:
        m0 = get("M0", lr)
        mtl = get("MTL", lr)
        benefit = (m0.mean(1) - mtl.mean(1))  # positive = TL helps
        err = np.sqrt(m0.std(1, ddof=1)**2 + mtl.std(1, ddof=1)**2) / np.sqrt(m0.shape[1])
        ax.errorbar(ts, benefit, err, marker="o", capsize=3, ls=ls, alpha=alpha,
                    label=f"lr={lr}", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("TL Benefit ($M_0 - M_{TL}$)")
    ax.set_xlabel("$n_{train}$"); ax.set_ylabel(r"$\Delta$ MAE (%)")
    ax.set_xticks(ts); ax.legend(frameon=True)

    # Panel 4: All 4 conditions at n=50
    ax = axes[1, 1]
    conds = [("$M_0$ lr=1e-3", m0_1e3[0], "#1f77b4"),
             ("$M_0$ lr=3e-4", m0_3e4[0], "#aec7e8"),
             ("$M_{TL}$ lr=1e-3", mtl_1e3[0], "#d62728"),
             ("$M_{TL}$ lr=3e-4", mtl_3e4[0], "#ff9896")]
    x_pos = np.arange(len(conds))
    for j, (lab, vals, col) in enumerate(conds):
        ax.bar(j, vals.mean(), yerr=vals.std(ddof=1), capsize=4,
               color=col, edgecolor="k", lw=0.5, width=0.55)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in conds], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE (%)")
    ax.set_title("All Conditions at $n = 50$")

    fig.tight_layout()
    fig.savefig(RESULTS / "final_figW4_lr_ablation.png")
    fig.savefig(RESULTS / "final_figW4_lr_ablation.pdf")
    plt.close(fig)
    print("  Fig W4 saved.")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE W6 – Feature Importance (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════════════
def figW6():
    d = np.load(RESULTS / "feature_importance_A.npz", allow_pickle=True)
    feat_names = d["feature_names"]
    cat_ids    = d["category_id"]
    cat_names  = d["category_names"]
    imp_mean   = d["imp_mtlphys_mean"] * 100   # convert to %
    imp_std    = d["imp_mtlphys_std"]  * 100

    # Sort by importance descending
    order = np.argsort(imp_mean)[::-1]
    feat_names = feat_names[order]
    cat_ids    = cat_ids[order]
    imp_mean   = imp_mean[order]
    imp_std    = imp_std[order]

    # Category colours
    cat_colors = plt.cm.Set2(np.linspace(0, 1, len(cat_names)))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(feat_names))
    bars = ax.barh(y, imp_mean, xerr=imp_std, capsize=3, height=0.6,
                    color=[cat_colors[int(c)] for c in cat_ids],
                    edgecolor="k", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (MAE % increase)")
    ax.set_title(r"Feature Importance ($M_{TL+\mathrm{phys}}$, Structure A)")

    # Category legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=cat_colors[i], edgecolor="k", lw=0.4,
                            label=str(cat_names[i])) for i in range(len(cat_names))]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              edgecolor="0.8", fontsize=9, title="Category", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(RESULTS / "final_figW6_feature_importance.png")
    fig.savefig(RESULTS / "final_figW6_feature_importance.pdf")
    plt.close(fig)
    print("  Fig W6 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating all final figures...")
    fig1()
    fig2()
    fig3()
    fig5()
    fig6()
    figW4()
    figW6()
    print("Done. All figures saved to", RESULTS)
