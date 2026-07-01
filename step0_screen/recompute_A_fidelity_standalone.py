#!/usr/bin/env python3
"""STANDALONE Part-A fidelity recompute (NO training).

Recomputes ONLY the per-level `tmm_accuracies` axis of
results/tmm_accuracy_variation_redesign.npz, switching from the legacy
pooled-flatten Pearson to the MEDIAN per-sample Pearson over wavelengths
(matching compute_fidelity / the B&C noise scripts' pearson_per_sample).

It replicates EXACTLY the data loading and per-level noise-draw ORDER of
step0_screen/tmm_accuracy_variation_redesign.py so the noisy-TMM test
realization is identical, then overwrites only:
    tmm_accuracies, meta_pearson_r, meta_pearson_p
KEEPING tl_benefits, tl_maes, m0_maes, m0_mean_mae, noise_sigmas,
level_names, seeds byte-identical. Trains NOTHING.
"""
import sys, os
HOME = '.'
sys.path.insert(0, HOME)

import numpy as np
from scipy.stats import pearsonr

# Set material model BEFORE importing tmm (matches the original script).
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"
from src.utils.data_utils import get_bounds
from src.simulation.tmm_struct_a import compute_tmm_batch

print(f"MATERIAL_MODEL={_mat.MATERIAL_MODEL}", flush=True)


def pearson_median_per_sample(a, b):
    """Median per-sample Pearson r over wavelengths (skip near-constant rows)."""
    rs = []
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        if np.std(sa) < 1e-9 or np.std(sb) < 1e-9:
            continue
        r, _ = pearsonr(sa, sb)
        rs.append(r)
    return float(np.median(rs)) if rs else 0.0


# ========= Step 1: TMM training data (needed only for noise-draw SHAPES) =========
N_TMM = 5000
RCWA_PATH = os.path.join(HOME, "data/raw/struct_A_500_redesign.npz")
wavelengths_rcwa = np.load(RCWA_PATH, allow_pickle=True)["wavelengths"].astype(np.float32)
Nlam = len(wavelengths_rcwa)
print(f"Grid: {wavelengths_rcwa.min():.0f}-{wavelengths_rcwa.max():.0f}nm, {Nlam}pts", flush=True)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

tmm_out = compute_tmm_batch(params_tmm, wavelengths_rcwa, "Cr")
A_tmm_clean = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm_clean = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)
print(f"A_tmm_clean shape={A_tmm_clean.shape} R_tmm_clean shape={R_tmm_clean.shape}", flush=True)

# NOTE: the original script consumes the SAME rng (seed 99) draws for the
# train/val sample split BEFORE the noise loop, but `noise_rng` is a SEPARATE
# default_rng(2024) reset fresh per level, so the train/val permutation does
# NOT affect noise realizations. We skip the split (no training here).

# ========= Step 2: Load RCWA data (reliable filter) =========
data = np.load(RCWA_PATH, allow_pickle=True)
assert np.allclose(data["wavelengths"].astype(np.float32), wavelengths_rcwa), "grid mismatch!"
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)

if "reliable" in data.files:
    good = data["reliable"].all(axis=1)
else:
    good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_rcwa = params_rcwa[gi]
A_rcwa = A_rcwa[gi]
R_rcwa = R_rcwa[gi]
N_rcwa = len(gi)
print(f"RCWA data: {N_rcwa} reliable samples (of {len(good)})", flush=True)

# Fixed test/val split (identical to original).
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 50
N_VAL = 50
test_idx = all_idx[-N_TEST:]

# ========= Step 3: TMM for the 50 fixed test geometries =========
params_test = params_rcwa[test_idx]
A_rcwa_test = A_rcwa[test_idx]  # (50, 100) ground truth
tmm_test_out = compute_tmm_batch(params_test, wavelengths_rcwa, "Cr")
A_tmm_test_clean = np.clip(tmm_test_out["A_tmm"], 0, 1).astype(np.float32)
print(f"A_tmm_test_clean shape={A_tmm_test_clean.shape}", flush=True)

# ========= Step 4: per-level noise + median fidelity =========
NOISE_LEVELS = {
    "Level 0 (sigma=0.00)": 0.00,
    "Level 1 (sigma=0.05)": 0.05,
    "Level 2 (sigma=0.10)": 0.10,
    "Level 3 (sigma=0.15)": 0.15,
    "Level 4 (sigma=0.20)": 0.20,
    "Level 5 (random)": None,
}

new_accs = []
for level_name, sigma in NOISE_LEVELS.items():
    noise_rng = np.random.default_rng(2024)  # fresh per level (exact match)

    # --- TRAIN noise draws FIRST (to advance the stream identically) ---
    if sigma is None:
        _A_train = noise_rng.uniform(0, 1, A_tmm_clean.shape).astype(np.float32)   # A_tmm_noisy
        _R_train = noise_rng.uniform(0, 1, R_tmm_clean.shape).astype(np.float32)   # R_tmm_noisy
    elif sigma == 0.0:
        pass  # original copies clean -> NO rng draw
    else:
        _A_noise = noise_rng.normal(0, sigma, A_tmm_clean.shape).astype(np.float32)  # A_noise
        _R_noise = noise_rng.normal(0, sigma, R_tmm_clean.shape).astype(np.float32)  # R_noise

    # --- TEST noise draw (after train draws) ---
    if sigma is None:
        A_tmm_test_noisy = noise_rng.uniform(0, 1, A_tmm_test_clean.shape).astype(np.float32)
    elif sigma == 0.0:
        A_tmm_test_noisy = A_tmm_test_clean.copy()
    else:
        A_test_noise = noise_rng.normal(0, sigma, A_tmm_test_clean.shape).astype(np.float32)
        A_tmm_test_noisy = np.clip(A_tmm_test_clean + A_test_noise, 0, 1)

    corr = pearson_median_per_sample(A_tmm_test_noisy, A_rcwa_test)
    new_accs.append(corr)
    print(f"  {level_name}: median-per-sample r = {corr:.6f}", flush=True)

new_accs = np.array(new_accs, dtype=np.float64)

# ========= Step 5: overwrite npz (keep everything else byte-identical) =========
SAVEPATH = os.path.join(HOME, "results/tmm_accuracy_variation_redesign.npz")
old = np.load(SAVEPATH, allow_pickle=True)
old_dict = {k: old[k] for k in old.files}
tl_benefits = old_dict["tl_benefits"]  # UNCHANGED, saved benefits

meta_r, meta_p = pearsonr(new_accs, tl_benefits)
print("\n=== OLD vs NEW ===", flush=True)
print("old tmm_accuracies:", np.array2string(old_dict["tmm_accuracies"], precision=6), flush=True)
print("new tmm_accuracies:", np.array2string(new_accs, precision=6), flush=True)
print(f"old meta_r={float(old_dict['meta_pearson_r']):.6f} p={float(old_dict['meta_pearson_p']):.6g}", flush=True)
print(f"new meta_r={meta_r:.6f} p={meta_p:.6g}", flush=True)

new_dict = dict(old_dict)
new_dict["tmm_accuracies"] = new_accs
new_dict["meta_pearson_r"] = np.array(meta_r)
new_dict["meta_pearson_p"] = np.array(meta_p)

# sanity: untouched fields must be byte-identical
for k in ["tl_benefits", "tl_maes", "m0_maes", "m0_mean_mae",
          "noise_sigmas", "level_names", "seeds"]:
    assert np.array_equal(new_dict[k], old_dict[k]), f"{k} changed!"

np.savez(SAVEPATH, **new_dict)
print(f"\nSaved: {SAVEPATH}", flush=True)
print("Done!", flush=True)
