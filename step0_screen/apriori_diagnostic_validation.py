#!/usr/bin/env python3
"""
A Priori Diagnostic Validation Experiment.

Tests whether TMM--RCWA spectral fidelity (median Pearson r) can be reliably
estimated from small pilot sets (20, 50, 100 RCWA samples), compared to the
full 500-sample reference.

This validates the paper's claim that "computing [fidelity] on as few as
20--50 RCWA samples" can indicate whether PBTL is likely to help.

Output:
  - For each subsample size: mean, std, 95% CI of median Pearson r
  - Comparison with 500-sample reference value
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import pearsonr
from src.simulation.tmm_struct_a import compute_tmm_batch

# ========= Load RCWA data =========
print("Loading RCWA data...", flush=True)
data = np.load("data/raw/struct_A_vis_500.npz",
               allow_pickle=True)
params = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
wavelengths = data["wavelengths"].astype(np.float32)

# Filter valid samples
good = np.all((A_rcwa >= 0) & (A_rcwa <= 1), axis=1)
params = params[good]
A_rcwa = A_rcwa[good]
N = len(params)
print(f"Valid RCWA samples: {N}", flush=True)

# ========= Compute TMM for all samples =========
print("Computing TMM spectra for all samples...", flush=True)
tmm_out = compute_tmm_batch(params, wavelengths, "Cr")
A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
print(f"TMM computed: {A_tmm.shape}", flush=True)

# ========= Compute per-sample Pearson r =========
print("\nComputing per-sample Pearson correlations...", flush=True)
per_sample_r = np.zeros(N)
for i in range(N):
    r, _ = pearsonr(A_tmm[i], A_rcwa[i])
    per_sample_r[i] = r

ref_mean = np.mean(per_sample_r)
ref_median = np.median(per_sample_r)
print(f"Reference (N={N}): mean r = {ref_mean:.4f}, median r = {ref_median:.4f}")

# ========= Subsampling experiment =========
SUBSAMPLE_SIZES = [10, 20, 30, 50, 75, 100, 150, 200, 300]
N_REPEATS = 1000
rng = np.random.default_rng(42)

print(f"\nSubsampling experiment: {N_REPEATS} repeats per size")
print(f"{'n_pilot':>8} | {'mean(median r)':>14} | {'std':>8} | {'95% CI':>20} | {'|bias|':>8} | {'P(rank correct)':>16}")
print("-" * 100)

results = {}
for n_sub in SUBSAMPLE_SIZES:
    median_rs = np.zeros(N_REPEATS)
    mean_rs = np.zeros(N_REPEATS)
    for rep in range(N_REPEATS):
        idx = rng.choice(N, size=n_sub, replace=False)
        median_rs[rep] = np.median(per_sample_r[idx])
        mean_rs[rep] = np.mean(per_sample_r[idx])

    m = np.mean(median_rs)
    s = np.std(median_rs)
    ci_lo = np.percentile(median_rs, 2.5)
    ci_hi = np.percentile(median_rs, 97.5)
    bias = abs(m - ref_median)

    # What fraction of time would the diagnostic correctly identify
    # Structure A as "beneficial" (median r > 0.3)?
    p_correct = np.mean(median_rs > 0.3) * 100

    results[n_sub] = {
        "median_r_mean": m, "median_r_std": s,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "bias": bias, "p_above_03": p_correct,
        "mean_r_mean": np.mean(mean_rs), "mean_r_std": np.std(mean_rs),
    }

    print(f"{n_sub:>8} | {m:>14.4f} | {s:>8.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | {bias:>8.4f} | {p_correct:>15.1f}%")

# ========= Summary =========
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Reference median r (N={N}): {ref_median:.4f}")
print(f"Reference mean r (N={N}):   {ref_mean:.4f}")
print(f"\nKey question: Can n=20-50 pilot samples reliably detect median r > 0.3?")

for n_sub in [20, 50]:
    r = results[n_sub]
    print(f"\n  n={n_sub}:")
    print(f"    Estimated median r = {r['median_r_mean']:.4f} +/- {r['median_r_std']:.4f}")
    print(f"    95% CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print(f"    Bias from reference: {r['bias']:.4f}")
    print(f"    P(correctly identifies r > 0.3): {r['p_above_03']:.1f}%")

# ========= Save =========
savepath = "results/apriori_diagnostic_validation.npz"
np.savez(savepath,
         subsample_sizes=np.array(SUBSAMPLE_SIZES),
         ref_median_r=ref_median,
         ref_mean_r=ref_mean,
         per_sample_r=per_sample_r,
         **{f"n{n}_median_r_mean": results[n]["median_r_mean"] for n in SUBSAMPLE_SIZES},
         **{f"n{n}_median_r_std": results[n]["median_r_std"] for n in SUBSAMPLE_SIZES},
         **{f"n{n}_p_above_03": results[n]["p_above_03"] for n in SUBSAMPLE_SIZES},
)
print(f"\nSaved: {savepath}")
print("Done!")
