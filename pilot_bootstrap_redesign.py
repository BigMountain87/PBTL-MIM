#!/usr/bin/env python3
"""Pilot-set bootstrap precision analysis for the joint r-and-MAE transferability
diagnostic (paper, "Pilot-set validation").

Resamples the per-sample TMM-RCWA fidelity values
(results/tmm_rcwa_fidelity_redesign.npz: per-sample Pearson r [ca,cb,cc] and
operating-band MAE [ma,mb,mc] for Structures A/B/C) at pilot sizes
n_pilot in {10,20,50,100} over 1000 repetitions, and reports the median estimate,
scatter, 95% CI, and bias vs the full-pool reference for BOTH the correlation (r)
and amplitude (MAE) components. Reproduces the numbers quoted in the manuscript's
"Pilot-set validation" paragraph. Subsets are drawn without replacement; seed fixed.
"""
import numpy as np

D = np.load("results/tmm_rcwa_fidelity_redesign.npz")
PILOTS = [10, 20, 50, 100]
REPS = 1000
SEED = 42


def bootstrap(vals, is_mae):
    v = vals[np.isfinite(vals)]
    scale = 100.0 if (is_mae and np.median(v) < 1.0) else 1.0   # MAE stored as fraction -> %
    ref = float(np.median(v) * scale)
    rng = np.random.default_rng(SEED)
    rows = []
    for n in PILOTS:
        meds = np.array([np.median(v[rng.choice(len(v), n, replace=False)]) * scale
                         for _ in range(REPS)])
        lo, hi = np.percentile(meds, [2.5, 97.5])
        frac_hi = float((meds > 0.5).mean())   # fraction of replicates with median > 0.5
        rows.append((n, float(meds.mean()), float(meds.std()), float(lo), float(hi),
                     abs(float(meds.mean()) - ref), frac_hi))
    return ref, rows, len(v)


COMPONENTS = [("A", "ma", True), ("B", "mb", True), ("C", "mc", True),
              ("A", "ca", False), ("B", "cb", False), ("C", "cc", False)]

if __name__ == "__main__":
    for struct, key, is_mae in COMPONENTS:
        comp = "MAE(%)" if is_mae else "r"
        ref, rows, n = bootstrap(D[key], is_mae)
        print(f"\n=== Structure {struct} {comp}  (n={n}, full-pool reference={ref:.2f}) ===")
        for npil, mean, std, lo, hi, bias, frac_hi in rows:
            extra = "" if is_mae else f"  frac(med>0.5)={frac_hi:.3f}"
            print(f"  n_pilot={npil:3d}: {mean:6.2f} +/- {std:4.2f}  "
                  f"95% CI [{lo:.2f}, {hi:.2f}]  bias={bias:.3f}{extra}")
