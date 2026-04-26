#!/usr/bin/env python3
"""
Publication-quality figures — Optics Communications (Elsevier) style.

Layout specs (Elsevier author guidelines):
  Single column : 90 mm  = 3.543 in
  1.5 column    : 140 mm = 5.512 in
  Full width    : 190 mm = 7.480 in
  Max height    : 240 mm = 9.449 in

Typography:
  Body font      : Arial (Helvetica fallback)
  Figure labels  : 7 pt  (tick labels, legend)
  Axis labels    : 8 pt
  Panel titles   : 8 pt, bold
  DPI            : 600 (print-ready)

Colour palette  : Wong (2011) — colorblind-safe, distinct in greyscale
  M0            : #0072B2  deep blue
  M_phys        : #E69F00  amber
  M_TL          : #009E73  bluish-green
  M_TL+phys     : #D55E00  vermillion
  M_rand        : #CC79A7  reddish-purple
  M_rand+phys   : #56B4E9  sky-blue
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
RESULTS = Path("/Users/sbchoi129/PINN2/mim_novel/results")
OUT     = RESULTS / "optcomm_figures"
OUT.mkdir(exist_ok=True)

TRAIN_SIZES = [50, 100, 200, 350]

# ─── Optics Communications rcParams ──────────────────────────────────────────
mpl.rcParams.update({
    # font
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":   "custom",
    "mathtext.rm":        "Arial",
    "mathtext.it":        "Arial:italic",
    "mathtext.bf":        "Arial:bold",

    # sizes (pt)
    "font.size":          7,
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.title_fontsize": 7,

    # axes appearance
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,

    # ticks
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.direction":    "in",
    "ytick.direction":    "in",

    # lines & markers
    "lines.linewidth":    1.2,
    "lines.markersize":   4.0,
    "errorbar.capsize":   2.5,

    # legend
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.75",
    "legend.borderpad":   0.4,
    "legend.handlelength":1.5,

    # save
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

# inch conversion helper
def mm(x): return x / 25.4

# ─── Wong colour palette ─────────────────────────────────────────────────────
C = {
    "M0":         "#0072B2",
    "M_phys":     "#E69F00",
    "M_TL":       "#009E73",
    "M_TL+phys":  "#D55E00",
    "M_rand":     "#CC79A7",
    "M_rand+phys":"#56B4E9",
}

# markers per method
MK = {
    "M0":         "o",
    "M_phys":     "s",
    "M_TL":       "^",
    "M_TL+phys":  "D",
    "M_rand":     "v",
    "M_rand+phys":"P",
}

# LaTeX labels
LBL = {
    "M0":         r"$M_0$",
    "M_phys":     r"$M_\mathrm{phys}$",
    "M_TL":       r"$M_\mathrm{TL}$",
    "M_TL+phys":  r"$M_\mathrm{TL+phys}$",
    "M_rand":     r"$M_\mathrm{rand}$",
    "M_rand+phys":r"$M_\mathrm{rand+phys}$",
}

METHODS4 = ["M0", "M_phys", "M_TL", "M_TL+phys"]

# ─── Data loaders ────────────────────────────────────────────────────────────
def load_A():
    d = np.load(RESULTS / "pbtl_A_10seed.npz")
    return {m: np.stack([d[f"{n}_{m}"] for n in TRAIN_SIZES]) for m in METHODS4}

def load_BC(fname):
    d   = np.load(RESULTS / fname)
    km  = {"M0": "M0", "M_phys": "M_phys",
           "M_TL": "M_TL", "M_TL+phys": "M_TL_phys"}
    return {m: d[km[m]] for m in METHODS4}

# ─── Stats helpers ───────────────────────────────────────────────────────────
def paired_p(a, b):
    _, p = stats.ttest_rel(a, b)
    return p

def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def sem_diff(a, b):
    return np.std(a - b, ddof=1) / np.sqrt(len(a))

# ─── Axis helpers ────────────────────────────────────────────────────────────
def set_panel_label(ax, letter):
    """(a), (b), (c) panel label — upper-left inside axes."""
    ax.text(-0.18, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left")

def clean_ax(ax):
    """Remove top/right spines (already global, but call for safety)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Learning curves  (full-width, 3-panel)
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    datasets = [
        ("Structure A",   load_A()),
        ("Structure B",   load_BC("pbtl_B_10seed.npz")),
        ("Structure C",   load_BC("pbtl_C_v2_10seed.npz")),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm(190), mm(65)),
        sharey=False,
        constrained_layout=True,
    )

    for ax, letter, (title, data) in zip(axes, "abc", datasets):
        for m in METHODS4:
            means = data[m].mean(axis=1) * 100
            sems  = data[m].std(axis=1, ddof=1) * 100 / np.sqrt(data[m].shape[1])
            ax.errorbar(
                TRAIN_SIZES, means, yerr=sems,
                marker=MK[m], color=C[m],
                lw=1.2, ms=4, capsize=2.5,
                label=LBL[m],
            )
        ax.set_xlabel(r"Training samples, $n$")
        ax.set_ylabel("MAE (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(TRAIN_SIZES)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        clean_ax(ax)
        set_panel_label(ax, letter)

    # single shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        edgecolor="0.75",
    )

    fig.savefig(OUT / "fig1_learning_curves.pdf")
    fig.savefig(OUT / "fig1_learning_curves.tif", dpi=600)
    plt.close(fig)
    print("  Fig 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Negative transfer  (1.5-column, 2-panel)
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(mm(140), mm(60)),
        sharey=True,
        constrained_layout=True,
    )

    for ax, letter, (title, fname) in zip(
        axes, "ab",
        [("Structure B", "pbtl_B_10seed.npz"),
         ("Structure C", "pbtl_C_v2_10seed.npz")]
    ):
        data = load_BC(fname)
        x    = np.arange(len(TRAIN_SIZES))
        dm_list, de_list, col_list = [], [], []

        for i in range(len(TRAIN_SIZES)):
            tl  = data["M_TL"][i] * 100
            m0  = data["M0"][i]   * 100
            dm  = (tl - m0).mean()
            de  = sem_diff(tl, m0)
            dm_list.append(dm)
            de_list.append(de)
            col_list.append("#D55E00" if dm > 0 else "#009E73")

        ax.bar(x, dm_list, yerr=de_list, capsize=2.5,
               color=col_list, edgecolor="k", lw=0.4, width=0.55,
               error_kw=dict(lw=0.8))

        # significance stars
        for i in range(len(TRAIN_SIZES)):
            tl = data["M_TL"][i] * 100
            m0 = data["M0"][i]   * 100
            s  = sig_star(paired_p(tl, m0))
            if s:
                sign = np.sign(dm_list[i])
                ypos = dm_list[i] + sign * (de_list[i] + 0.05)
                va   = "bottom" if sign > 0 else "top"
                ax.text(i, ypos, s, ha="center", va=va,
                        fontsize=7, fontweight="bold")

        ax.axhline(0, color="0.3", lw=0.6, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in TRAIN_SIZES])
        ax.set_xlabel(r"Training samples, $n$")
        ax.set_title(title, fontweight="bold")
        clean_ax(ax)
        set_panel_label(ax, letter)

    axes[0].set_ylabel(r"$\Delta$MAE (%) : $M_\mathrm{TL} - M_0$")

    # manual legend patches
    legend_elements = [
        Patch(facecolor="#009E73", edgecolor="k", lw=0.4, label="Positive transfer"),
        Patch(facecolor="#D55E00", edgecolor="k", lw=0.4, label="Negative transfer"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right",
                   fontsize=6, frameon=True, edgecolor="0.75")

    fig.savefig(OUT / "fig2_negative_transfer.pdf")
    fig.savefig(OUT / "fig2_negative_transfer.tif", dpi=600)
    plt.close(fig)
    print("  Fig 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — 3-structure bar comparison at n = 350  (full-width, 3-panel)
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    datasets = [
        ("Structure A", load_A()),
        ("Structure B", load_BC("pbtl_B_10seed.npz")),
        ("Structure C", load_BC("pbtl_C_v2_10seed.npz")),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm(190), mm(65)),
        sharey=False,
        constrained_layout=True,
    )
    x = np.arange(len(METHODS4))
    bar_w = 0.55

    for ax, letter, (title, data) in zip(axes, "abc", datasets):
        means = [data[m][3].mean() * 100 for m in METHODS4]   # index 3 = n=350
        sems  = [data[m][3].std(ddof=1) * 100 / np.sqrt(data[m].shape[1])
                 for m in METHODS4]

        bars = ax.bar(
            x, means, yerr=sems,
            width=bar_w,
            color=[C[m] for m in METHODS4],
            edgecolor="k", lw=0.4,
            capsize=2.5,
            error_kw=dict(lw=0.8),
        )

        # improvement labels (skip M0)
        m0_mean = means[0]
        for j in range(1, len(METHODS4)):
            pct = (m0_mean - means[j]) / m0_mean * 100
            sign_str = f"\u2212{pct:.1f}%" if pct > 0 else f"+{abs(pct):.1f}%"
            col_str  = "#009E73" if pct > 0 else "#D55E00"
            yoff = means[j] + sems[j] + 0.05
            ax.text(j, yoff, sign_str, ha="center", va="bottom",
                    fontsize=6, color=col_str, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LBL[m] for m in METHODS4], fontsize=7)
        ax.set_ylabel("MAE (%)")
        ax.set_title(f"{title}  ($n = 350$)", fontweight="bold")

        # y-axis lower bound zero
        ymin = max(0, min(means) - 1.5)
        ax.set_ylim(bottom=ymin)

        clean_ax(ax)
        set_panel_label(ax, letter)

    fig.savefig(OUT / "fig3_structure_comparison.pdf")
    fig.savefig(OUT / "fig3_structure_comparison.tif", dpi=600)
    plt.close(fig)
    print("  Fig 3 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Robustness boxplots  (full-width, 3-panel)
# ═══════════════════════════════════════════════════════════════════════════
def fig4():
    datasets = [
        ("Structure A", load_A()),
        ("Structure B", load_BC("pbtl_B_10seed.npz")),
        ("Structure C", load_BC("pbtl_C_v2_10seed.npz")),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm(190), mm(65)),
        sharey=False,
        constrained_layout=True,
    )

    bp_props = dict(
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="k", lw=1.2),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", ms=2.5, markerfacecolor="0.5",
                        markeredgewidth=0.3, alpha=0.6),
        boxprops=dict(lw=0.6),
    )

    for ax, letter, (title, data) in zip(axes, "abc", datasets):
        box_data = [data[m][3] * 100 for m in METHODS4]
        bp = ax.boxplot(box_data, **bp_props)

        for patch, m in zip(bp["boxes"], METHODS4):
            patch.set_facecolor(C[m])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(METHODS4) + 1))
        ax.set_xticklabels([LBL[m] for m in METHODS4], fontsize=7)
        ax.set_ylabel("MAE (%)")
        ax.set_title(f"{title}  ($n = 350$)", fontweight="bold")
        clean_ax(ax)
        set_panel_label(ax, letter)

    fig.savefig(OUT / "fig4_robustness_boxplots.pdf")
    fig.savefig(OUT / "fig4_robustness_boxplots.tif", dpi=600)
    plt.close(fig)
    print("  Fig 4 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — Random baseline  (single-column, clean grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
def fig5():
    d        = np.load(RESULTS / "random_baseline_10seed.npz")
    key_map  = {"M0": "M0", "M_TL": "M_TL",
                "M_rand": "M_rand", "M_rand+phys": "M_rand_phys"}
    methods  = ["M0", "M_TL", "M_rand", "M_rand+phys"]

    n_grp   = len(TRAIN_SIZES)
    n_bar   = len(methods)
    bar_w   = 0.18
    gap     = 0.06
    total_w = n_bar * bar_w + gap
    x_ctr   = np.arange(n_grp)

    fig, ax = plt.subplots(figsize=(mm(90), mm(70)), constrained_layout=True)

    for j, m in enumerate(methods):
        vals  = d[key_map[m]] * 100     # (4, seeds)
        means = vals.mean(axis=1)
        sems  = vals.std(axis=1, ddof=1) / np.sqrt(vals.shape[1])
        offset = (j - (n_bar - 1) / 2) * (bar_w + gap / n_bar)
        ax.bar(
            x_ctr + offset, means, bar_w,
            yerr=sems, capsize=2,
            color=C[m], edgecolor="k", lw=0.4,
            label=LBL[m],
            error_kw=dict(lw=0.8),
        )

    ax.set_xticks(x_ctr)
    ax.set_xticklabels([str(n) for n in TRAIN_SIZES])
    ax.set_xlabel(r"Training samples, $n$")
    ax.set_ylabel("MAE (%)")
    ax.set_title("Random-weight baseline (Structure A)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6, ncol=1, frameon=True, edgecolor="0.75")
    clean_ax(ax)

    fig.savefig(OUT / "fig5_random_baseline.pdf")
    fig.savefig(OUT / "fig5_random_baseline.tif", dpi=600)
    plt.close(fig)
    print("  Fig 5 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG W4 — LR Ablation  (1.5-column, 2-panel)
# ═══════════════════════════════════════════════════════════════════════════
def figW4():
    d  = np.load(RESULTS / "lr_ablation_A.npz")
    ts = TRAIN_SIZES

    def get(prefix, lr):
        return np.array([d[f"{n}_{prefix}_lr{lr}"] * 100 for n in ts])

    m0_hi  = get("M0",  "1e-3")
    m0_lo  = get("M0",  "3e-4")
    mtl_hi = get("MTL", "1e-3")
    mtl_lo = get("MTL", "3e-4")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(mm(140), mm(60)),
        constrained_layout=True,
    )

    # Panel (a): M0 vs M_TL at same LR
    ax = axes[0]
    kw = dict(lw=1.2, ms=4, capsize=2.5)
    ax.errorbar(ts, m0_hi.mean(1), m0_hi.std(1, ddof=1) / np.sqrt(m0_hi.shape[1]),
                marker="o", color=C["M0"],
                label=r"$M_0$ (lr=10$^{-3}$)", **kw)
    ax.errorbar(ts, m0_lo.mean(1), m0_lo.std(1, ddof=1) / np.sqrt(m0_lo.shape[1]),
                marker="s", color=C["M0"], ls="--", alpha=0.6,
                label=r"$M_0$ (lr=3×10$^{-4}$)", **kw)
    ax.errorbar(ts, mtl_hi.mean(1), mtl_hi.std(1, ddof=1) / np.sqrt(mtl_hi.shape[1]),
                marker="^", color=C["M_TL"],
                label=r"$M_\mathrm{TL}$ (lr=10$^{-3}$)", **kw)
    ax.errorbar(ts, mtl_lo.mean(1), mtl_lo.std(1, ddof=1) / np.sqrt(mtl_lo.shape[1]),
                marker="D", color=C["M_TL"], ls="--", alpha=0.6,
                label=r"$M_\mathrm{TL}$ (lr=3×10$^{-4}$)", **kw)
    ax.set_xlabel(r"Training samples, $n$")
    ax.set_ylabel("MAE (%)")
    ax.set_title("Learning rate comparison", fontweight="bold")
    ax.set_xticks(ts)
    ax.legend(fontsize=6, frameon=True, edgecolor="0.75")
    clean_ax(ax)
    set_panel_label(ax, "a")

    # Panel (b): TL benefit at matched LRs
    ax = axes[1]
    for lr, ls, col, mk, m_m0, m_mtl in [
        ("1e-3", "-",  "#0072B2", "o", m0_hi, mtl_hi),
        ("3e-4", "--", "#D55E00", "s", m0_lo, mtl_lo),
    ]:
        benefit = m_m0.mean(1) - m_mtl.mean(1)
        err = np.sqrt(
            (m_m0.std(1, ddof=1) / np.sqrt(m_m0.shape[1]))**2 +
            (m_mtl.std(1, ddof=1) / np.sqrt(m_mtl.shape[1]))**2
        )
        ax.errorbar(ts, benefit, err,
                    marker=mk, lw=1.2, ms=4, capsize=2.5,
                    ls=ls, color=col,
                    label=r"lr = " + f"{lr}")
    ax.axhline(0, color="0.4", lw=0.6, ls=":")
    ax.set_xlabel(r"Training samples, $n$")
    ax.set_ylabel(r"TL benefit, $\Delta$MAE (%)")
    ax.set_title(r"$M_0 - M_\mathrm{TL}$ (both LRs)", fontweight="bold")
    ax.set_xticks(ts)
    ax.legend(fontsize=6, frameon=True, edgecolor="0.75")
    clean_ax(ax)
    set_panel_label(ax, "b")

    fig.savefig(OUT / "figW4_lr_ablation.pdf")
    fig.savefig(OUT / "figW4_lr_ablation.tif", dpi=600)
    plt.close(fig)
    print("  Fig W4 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIG W6 — Feature Importance  (single-column, horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════
def figW6():
    # Map raw feature strings → publication-quality LaTeX labels
    FEAT_LABEL = {
        r"cos(phase_SiO2)":    r"$\cos\varphi_{\mathrm{SiO_2}}$",
        r"sin(phase_SiO2)":    r"$\sin\varphi_{\mathrm{SiO_2}}$",
        r"cos(phase_TiO2)":    r"$\cos\varphi_{\mathrm{TiO_2}}$",
        r"sin(phase_TiO2)":    r"$\sin\varphi_{\mathrm{TiO_2}}$",
        r"ff_rect (Wx*Wy/P^2)":r"$f_{\mathrm{rect}}=W_xW_y/P^2$",
        r"ff_square (W2^2/P^2)":r"$f_{\mathrm{sq}}=W_2^2/P^2$",
        r"P/lambda":           r"$P/\lambda$",
        r"Wx/lambda":          r"$W_x/\lambda$",
        r"W2/lambda":          r"$W_2/\lambda$",
        r"t1/delta":           r"$t_1/\delta$",
        r"t2/delta":           r"$t_2/\delta$",
        r"t_mid/delta":        r"$t_{\mathrm{mid}}/\delta$",
        r"n_SiO2*d1/lambda":   r"$n_{\mathrm{SiO_2}}\,d_1/\lambda$",
        r"n_TiO2*d2/lambda":   r"$n_{\mathrm{TiO_2}}\,d_2/\lambda$",
        r"cos(theta)":         r"$\cos\theta$",
        r"Wy/Wx":              r"$W_y/W_x$",
        r"alpha_metal":        r"$\alpha_{\mathrm{metal}}$",
    }

    d          = np.load(RESULTS / "feature_importance_A.npz", allow_pickle=True)
    feat_names = d["feature_names"]
    cat_ids    = d["category_id"].astype(int)
    cat_names  = d["category_names"]
    imp_mean   = d["imp_mtlphys_mean"] * 100
    imp_std    = d["imp_mtlphys_std"]  * 100

    order      = np.argsort(imp_mean)[::-1]
    feat_names = feat_names[order]
    feat_names = np.array([FEAT_LABEL.get(f, f) for f in feat_names])
    cat_ids    = cat_ids[order]
    imp_mean   = imp_mean[order]
    imp_std    = imp_std[order]

    # colorblind-safe category palette (6 max)
    cat_pal = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    n_cats  = len(cat_names)

    fig, ax = plt.subplots(figsize=(mm(90), mm(110)), constrained_layout=True)
    y = np.arange(len(feat_names))

    ax.barh(
        y, imp_mean, xerr=imp_std,
        height=0.65,
        color=[cat_pal[c % len(cat_pal)] for c in cat_ids],
        edgecolor="k", lw=0.3,
        capsize=2, error_kw=dict(lw=0.7),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (ΔMAE, %)")
    ax.set_title(r"Feature importance — $M_\mathrm{TL+phys}$, Structure A",
                 fontweight="bold")

    legend_handles = [
        Patch(facecolor=cat_pal[i % len(cat_pal)], edgecolor="k", lw=0.3,
              label=str(cat_names[i]))
        for i in range(n_cats)
    ]
    ax.legend(handles=legend_handles,
              loc="lower right", fontsize=6,
              frameon=True, edgecolor="0.75",
              title="Category", title_fontsize=6)
    clean_ax(ax)

    fig.savefig(OUT / "figW6_feature_importance.pdf")
    fig.savefig(OUT / "figW6_feature_importance.tif", dpi=600)
    plt.close(fig)
    print("  Fig W6 saved.")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUT}\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    figW4()
    figW6()
    print("\nAll figures saved.")
