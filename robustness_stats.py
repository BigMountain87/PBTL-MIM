"""Robustness statistics for Section tmm_accuracy (W3).

Computes:
  (1) Bootstrap 95% CI and permutation-test p-value for Pearson r (N=5 and N=6)
  (2) Fisher-z 95% CI for Pearson r
  (3) Physics-feature benefit t-test per structure (paper.tex 4-19% claim)
  (4) Bonferroni-corrected p-values for the 6-condition family
  (5) ROC-style derivation of the r > 0.3 threshold (sec:practical guidelines)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# -----------------------------------------------------------------------------
# W3 data (from ~/PINN2/mim_novel/tmm_accuracy_log.txt)
# -----------------------------------------------------------------------------
tmm_acc_all = np.array([0.5670, 0.5475, 0.4994, 0.4415, 0.3856, -0.0082])
tl_benefit_all = np.array([27.8, 27.5, 24.2, 19.0, 11.0, -43.7])

# Per-seed TL MAE to derive benefit SE (3 seeds per condition)
tl_mae_seeds = {
    "L0": [7.802 - 0.090, 7.802, 7.802 + 0.090],  # approximated from mean +/- std
    "L1": [7.830 - 0.122, 7.830, 7.830 + 0.122],
    "L2": [7.802 + 0.437, 7.802 + 0.492, 7.802 + 0.251],  # 8.239, 8.294, 8.053
    "L3": [8.866, 8.932, 8.451],
    "L4": [9.736, 9.910, 9.188],
    "L5": [14.746, 17.472, 14.349],
}
m0_baseline = 10.805  # 3-seed baseline


def pearson_ci_fisher(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform 95% CI for Pearson r."""
    if n <= 3:
        return (-1.0, 1.0)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return tuple(np.tanh((z - zcrit * se, z + zcrit * se)))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05) -> tuple[float, float, np.ndarray]:
    n = len(x)
    rs = np.empty(n_boot)
    hit = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            rs[i] = np.nan
            continue
        rs[i] = stats.pearsonr(xb, yb)[0]
        hit += 1
    rs = rs[~np.isnan(rs)]
    return (np.percentile(rs, 100 * alpha / 2),
            np.percentile(rs, 100 * (1 - alpha / 2)),
            rs)


def permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 100000) -> float:
    r_obs = abs(stats.pearsonr(x, y)[0])
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = abs(stats.pearsonr(x, yp)[0])
        if r >= r_obs:
            count += 1
    return (count + 1) / (n_perm + 1)  # add-one smoothed


# -----------------------------------------------------------------------------
# (1) Primary N=5 and supplementary N=6 — r, CI, permutation p
# -----------------------------------------------------------------------------
print("=" * 72)
print("  TMM accuracy vs TL benefit — robustness statistics")
print("=" * 72)
for label, idx in [("N=5 (physics-based only)", slice(0, 5)),
                   ("N=6 (incl. random)", slice(0, 6))]:
    x = tmm_acc_all[idx]
    y = tl_benefit_all[idx]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    fz_lo, fz_hi = pearson_ci_fisher(r, n)
    b_lo, b_hi, _ = bootstrap_ci(x, y, n_boot=20000)
    p_perm = permutation_test(x, y, n_perm=100000)

    print(f"\n[{label}]  n={n}")
    print(f"  Pearson  r = {r:+.4f},  parametric p = {p:.3e}")
    print(f"  Spearman rho = {rho:+.4f},  p = {p_rho:.3e}")
    print(f"  Fisher z 95% CI for r : [{fz_lo:+.3f}, {fz_hi:+.3f}]")
    print(f"  Bootstrap 95% CI for r: [{b_lo:+.3f}, {b_hi:+.3f}]  (20k resamples)")
    print(f"  Permutation-test p-value: {p_perm:.4f}  (100k shuffles)")
    print(f"  Bonferroni (family size 6): p_adj = {min(1.0, p * 6):.3e}")

# -----------------------------------------------------------------------------
# (2) Physics feature benefit per structure — 4-19% claim
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  Physics feature (M_phys) vs baseline (M_0) — per-structure t-test")
print("=" * 72)

# From paper.tex tables 1-3 (10-seed mean +/- std)
# Structure A (tab:pbtl_a)
A = {
    50:  (12.78, 0.80, 10.41, 0.69),
    100: (10.62, 0.51,  8.55, 0.33),
    200: ( 8.19, 0.39,  7.00, 0.19),
    350: ( 6.76, 0.40,  5.70, 0.23),
}
# Structure B (tab:pbtl_b)
B = {
    50:  (8.07, 0.51, 6.63, 0.19),
    100: (6.33, 0.41, 5.62, 0.48),
    200: (4.46, 0.30, 4.13, 0.17),
    350: (3.37, 0.11, 3.17, 0.12),
}
# Structure C (tab:pbtl_c)
C = {
    50:  (9.11, 0.60, 7.57, 0.52),
    100: (6.13, 0.34, 5.51, 0.47),
    200: (4.49, 0.37, 4.17, 0.24),
    350: (3.58, 0.13, 3.43, 0.09),
}

for name, tab in [("Structure A", A), ("Structure B", B), ("Structure C", C)]:
    print(f"\n{name}:")
    for n, (m0, s0, mp, sp) in tab.items():
        # Welch's unpaired t-test (10 seeds each; approx — same seed pairings unknown)
        t, p = stats.ttest_ind_from_stats(m0, s0, 10, mp, sp, 10, equal_var=False)
        # Effect size: Cohen's d (pooled)
        sd = np.sqrt((s0 ** 2 + sp ** 2) / 2)
        d = (m0 - mp) / sd
        improvement = (m0 - mp) / m0 * 100
        print(f"  n={n:3d}: M0={m0:5.2f}±{s0:.2f} -> M_phys={mp:5.2f}±{sp:.2f}"
              f"  | improvement {improvement:+5.1f}%  |  d={d:+.2f}  |  p={p:.1e}")

# -----------------------------------------------------------------------------
# (3) ROC-style r > 0.3 threshold derivation
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  r > 0.3 threshold — ROC-style derivation")
print("=" * 72)
# Event: "positive transfer" (benefit > 0)
# Predictor: TMM accuracy r
# Data from the 6 noise-injection levels (within Structure A)
events = (tl_benefit_all > 0).astype(int)   # 1 if positive transfer
print(f"  Per-level events (benefit > 0): {events.tolist()}")
print(f"  At r = 0.30 threshold: predicted positive = {(tmm_acc_all > 0.3).tolist()}")
# Sweep threshold
ths = np.linspace(-0.1, 0.7, 81)
youden = []
for t in ths:
    pred = (tmm_acc_all > t).astype(int)
    tp = int(((pred == 1) & (events == 1)).sum())
    fp = int(((pred == 1) & (events == 0)).sum())
    fn = int(((pred == 0) & (events == 1)).sum())
    tn = int(((pred == 0) & (events == 0)).sum())
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    youden.append((t, tpr - fpr, tpr, fpr, tp, fp, fn, tn))

youden.sort(key=lambda z: -z[1])
print("  Top 5 Youden-J optimal thresholds on noise-injection data:")
for t, j, tpr, fpr, tp, fp, fn, tn in youden[:5]:
    print(f"    r*>{t:+.3f}  J={j:+.2f}  TPR={tpr:.2f}  FPR={fpr:.2f}"
          f"  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

# Also use benefit > 5% (practically meaningful)
events5 = (tl_benefit_all > 5.0).astype(int)
print(f"\n  Practical event (benefit > 5%): {events5.tolist()}")
youden5 = []
for t in ths:
    pred = (tmm_acc_all > t).astype(int)
    tp = int(((pred == 1) & (events5 == 1)).sum())
    fp = int(((pred == 1) & (events5 == 0)).sum())
    fn = int(((pred == 0) & (events5 == 1)).sum())
    tn = int(((pred == 0) & (events5 == 0)).sum())
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    youden5.append((t, tpr - fpr, tpr, fpr))

youden5.sort(key=lambda z: -z[1])
print("  Top 5 Youden-J optimal thresholds (benefit > 5%):")
for t, j, tpr, fpr in youden5[:5]:
    print(f"    r*>{t:+.3f}  J={j:+.2f}  TPR={tpr:.2f}  FPR={fpr:.2f}")

# -----------------------------------------------------------------------------
# (4) Cross-structure fidelity vs benefit (3 structures from Table tmm_fidelity)
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("  Cross-structure spearman check (N=3)")
print("=" * 72)
# From paper line 374 context
struct_r = np.array([0.72, -0.07, 0.31])    # A, B, C median r (approx from paper)
struct_benefit_tl = np.array([29.5, -5.9, -10.6])  # M_TL improvement at n=50
rho, p_rho = stats.spearmanr(struct_r, struct_benefit_tl)
print(f"  Cross-structure r vs M_TL benefit  —  Spearman rho = {rho:+.3f}, p = {p_rho:.3f}")
print("  (N=3 only — illustrative)")

print("\nDone.\n")
