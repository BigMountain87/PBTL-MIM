#!/usr/bin/env python3
"""
Permutation feature importance for Structure A physics features (REDESIGN variant).

Ported from step0_screen/feature_importance.py to the corrected "redesign" dataset,
mirroring step0_screen/pbtl_A_redesign.py for data/material/grid handling:
  (1) MATERIAL_MODEL='jc' BEFORE importing tmm modules;
  (2) RCWA data from data/raw/struct_A_500_redesign.npz (key 'A'), not legacy *_vis_500;
  (3) wavelength grid LOADED from the npz (400-1800nm, 100pt), not hardcoded 380-780;
  (4) reliable physicality mask data['reliable'].all(axis=1);
  (5) Mac path <repo root>;
  (6) CPU + 2 threads (shared-GPU etiquette).

The M_phys physics-feature model is pre-trained on TMM (redesign grid) in-script so the
TL+phys weights are aligned with the redesign wavelength grid (the on-disk
pretrained_mphys_tmm.pt was trained on the legacy 380-780 grid). Then M_phys (from
scratch) and M_TL+phys (pre-trained, fine-tuned) are trained at n=350, seed=42, and
permutation importance (dMAE over 10 repeats) is computed for each of the 17 features.

Output: results/feature_importance_A_redesign.npz with the SAME keys the figure scripts
read (feature_names, category_id, category_names, imp_mtlphys_mean, imp_mtlphys_std) plus
the task-required imp_mphys_mean / imp_mphys_std and raw arrays.
"""
import sys, os, time
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(2)

# jc materials BEFORE importing tmm modules (matches redesign RCWA labels)
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

# Prefer CPU (one of several parallel agents sharing one Mac GPU); datasets are small.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

ROOT = '.'
RCWA_PATH = f"{ROOT}/data/raw/struct_A_500_redesign.npz"


# ========= Model definitions (same as pbtl_A_redesign.py) =========
class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = h + self.act(b(h))
        return h


class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd + pd)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x, p], -1))
        R = self.head(h).squeeze(-1)
        return {"A": 1 - R, "R": R}


def train_model(model, dl_tr, dl_vl, epochs, lr, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        _ts = dl_tr.dataset.tensors; _N = _ts[0].shape[0]; _perm = torch.randperm(_N, device=_ts[0].device)
        for _i in range(0, _N, 131072):
            batch = tuple(_t[_perm[_i:_i+131072]] for _t in _ts)
            if has_phys:
                x, a, r, p = batch
                out = model(x, p=p)
            else:
                x, a, r = [t.to(device) for t in batch]
                out = model(x)
            loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                _ts = dl_vl.dataset.tensors; _N = _ts[0].shape[0]
                for _i in range(0, _N, 131072):
                    batch = tuple(_t[_i:_i+131072] for _t in _ts)
                    if has_phys:
                        x, a, r, p = batch
                        out = model(x, p=p)
                    else:
                        x, a, r = [t.to(device) for t in batch]
                        out = model(x)
                    vl += (nn.functional.l1_loss(out["A"], a, reduction="sum") +
                           nn.functional.l1_loss(out["R"], r, reduction="sum")).item()
                    vn += len(a) * 2
                vm = vl / vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k: v.clone() for k, v in model.state_dict().items()}
    if best_st:
        model.load_state_dict(best_st)
    return model


def eval_mae(model, X_geo, A_true, phys=None):
    model.eval()
    with torch.no_grad():
        if phys is not None:
            out = model(X_geo, p=phys)
        else:
            out = model(X_geo)
        mae = nn.functional.l1_loss(out["A"], A_true).item()
    return mae


# ========= Feature names and categories =========
FEATURE_NAMES = [
    "cos(phase_SiO2)", "sin(phase_SiO2)", "cos(phase_TiO2)", "sin(phase_TiO2)",
    "ff_rect (Wx*Wy/P^2)", "ff_square (W2^2/P^2)",
    "P/lambda", "Wx/lambda", "W2/lambda",
    "t1/delta", "t2/delta", "t_mid/delta",
    "n_SiO2*d1/lambda", "n_TiO2*d2/lambda",
    "cos(theta)", "Wy/Wx", "alpha_metal",
]

CATEGORIES = {
    "Cavity resonance":     [0, 1, 2, 3],
    "Fill fraction":        [4, 5],
    "Sub-wavelength ratio": [6, 7, 8],
    "Skin depth ratio":     [9, 10, 11],
    "Optical path":         [12, 13],
    "Angle & geometry":     [14, 15, 16],
}

N_PERMUTATIONS = 10

# ========= Step 1: Generate TMM data + normalization stats (redesign grid) =========
print("\n=== Step 1: Generate TMM data (redesign grid, for pretrain + stats) ===", flush=True)
N_TMM = 5000
wavelengths_rcwa = np.load(RCWA_PATH, allow_pickle=True)["wavelengths"].astype(np.float32)
Nlam = len(wavelengths_rcwa)
print(f"Grid (from data): {wavelengths_rcwa.min():.0f}-{wavelengths_rcwa.max():.0f}nm, {Nlam}pts", flush=True)

_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths_rcwa, "Cr")
print(f"TMM generation: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)

phys_tmm = compute_physics_features_A(params_tmm, wavelengths_rcwa, "Cr")
n_phys = phys_tmm.shape[-1]
assert n_phys == 17, f"expected 17 physics features, got {n_phys}"

params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths_rcwa - wavelengths_rcwa.min()) / (wavelengths_rcwa.max() - wavelengths_rcwa.min())
geo_dim = 1 + 10

params_rep_tmm = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep, params_rep_tmm], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_tmm = phys_tmm.reshape(-1, n_phys).astype(np.float32)

phys_mean_tmm = X_phys_tmm.mean(0, keepdims=True)
phys_std_tmm = X_phys_tmm.std(0, keepdims=True) + 1e-8
X_phys_tmm_n = ((X_phys_tmm - phys_mean_tmm) / phys_std_tmm).astype(np.float32)

A_tmm_flat = A_tmm.reshape(-1)
R_tmm_flat = R_tmm.reshape(-1)

n_tmm_train = int(N_TMM * 0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_idx[:n_tmm_train]])
tmm_vl_rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in tmm_idx[n_tmm_train:]])


def to_dl(rows, has_phys, bs=131072, shuffle=False):
    xg = torch.tensor(X_geo_tmm[rows]).to(device)
    a = torch.tensor(A_tmm_flat[rows]).to(device)
    r = torch.tensor(R_tmm_flat[rows]).to(device)
    if has_phys:
        p = torch.tensor(X_phys_tmm_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)


dl_tmm_tr_ph = to_dl(tmm_tr_rows, True, shuffle=True)
dl_tmm_vl_ph = to_dl(tmm_vl_rows, True)

# ========= Step 2: Pre-train MPhys on TMM (redesign grid) =========
print("\n=== Step 2: Pre-train MPhys on TMM (redesign grid) ===", flush=True)
PRETRAIN_EPOCHS = 500
PRETRAIN_LR = 1e-3
set_seed(42)
pretrained_mphys = MPhys(geo_dim, n_phys).to(device)
t0 = time.time()
pretrained_mphys = train_model(pretrained_mphys, dl_tmm_tr_ph, dl_tmm_vl_ph,
                               PRETRAIN_EPOCHS, PRETRAIN_LR, has_phys=True)
pretrained_state = {k: v.clone() for k, v in pretrained_mphys.state_dict().items()}
print(f"Pre-trained MPhys in {time.time()-t0:.0f}s", flush=True)

# ========= Step 3: Load RCWA redesign data =========
print("\n=== Step 3: Load RCWA redesign data ===", flush=True)
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

phys_rcwa = compute_physics_features_A(params_rcwa, wavelengths_rcwa, "Cr")
params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_rcwa = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep_rcwa = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep_rcwa, params_rep_rcwa], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys_rcwa = phys_rcwa.reshape(-1, n_phys).astype(np.float32)
X_phys_rcwa_n = ((X_phys_rcwa - phys_mean_tmm) / phys_std_tmm).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)


def get_rows(si):
    return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in si])


# Fixed test/val split (same protocol as pbtl_A_redesign.py)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
N_TEST = 50
N_VAL = 50
test_idx = all_idx[-N_TEST:]
val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
remaining = all_idx[:-(N_TEST + N_VAL)]

test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)


def make_dl(rows, has_phys, bs=131072, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    if has_phys:
        p = torch.tensor(X_phys_rcwa_n[rows]).to(device)
        return DataLoader(TensorDataset(xg, a, r, p), batch_size=bs, shuffle=shuffle)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)


# ========= Step 4: Train M_phys and M_TL+phys (n=350, seed=42) =========
print("\n=== Step 4: Train models (n=350, seed=42) ===", flush=True)
n_train = 350
seed = 42

rng2 = np.random.default_rng(seed)
perm = rng2.permutation(len(remaining))
tr_idx = remaining[perm[:n_train]]
tr_rows = get_rows(tr_idx)

dl_tr_ph = make_dl(tr_rows, True, bs=131072, shuffle=True)
dl_vl_ph = make_dl(val_rows, True)

FINETUNE_EPOCHS = 1000
FINETUNE_LR = 1e-3
FINETUNE_LR_TL = 3e-4

print("Training M_phys (from scratch)...", flush=True)
set_seed(seed)
m_phys = MPhys(geo_dim, n_phys).to(device)
t0 = time.time()
m_phys = train_model(m_phys, dl_tr_ph, dl_vl_ph, FINETUNE_EPOCHS, FINETUNE_LR, has_phys=True)
print(f"  M_phys trained in {time.time()-t0:.0f}s", flush=True)

print("Training M_TL+phys (pre-trained + fine-tuned)...", flush=True)
set_seed(seed)
m_tlphys = MPhys(geo_dim, n_phys).to(device)
m_tlphys.load_state_dict(pretrained_state)
t0 = time.time()
m_tlphys = train_model(m_tlphys, dl_tr_ph, dl_vl_ph, FINETUNE_EPOCHS, FINETUNE_LR_TL, has_phys=True)
print(f"  M_TL+phys trained in {time.time()-t0:.0f}s", flush=True)

# ========= Step 5: Permutation importance =========
print("\n=== Step 5: Permutation importance ===", flush=True)
X_geo_test = torch.tensor(X_geo_rcwa[test_rows]).to(device)
X_phys_test = torch.tensor(X_phys_rcwa_n[test_rows]).to(device)
A_test = torch.tensor(A_rcwa_flat[test_rows]).to(device)


def permutation_importance(model, X_geo, X_phys, A_true, n_features=17, n_repeats=10):
    baseline_mae = eval_mae(model, X_geo, A_true, phys=X_phys)
    print(f"  Baseline test MAE: {baseline_mae*100:.4f}%", flush=True)
    importance = np.zeros((n_features, n_repeats))
    for feat_idx in range(n_features):
        for rep in range(n_repeats):
            X_phys_perm = X_phys.clone()
            perm_idx = torch.randperm(X_phys_perm.shape[0], device=device)
            X_phys_perm[:, feat_idx] = X_phys_perm[perm_idx, feat_idx]
            perm_mae = eval_mae(model, X_geo, A_true, phys=X_phys_perm)
            importance[feat_idx, rep] = perm_mae - baseline_mae
        mean_imp = importance[feat_idx].mean() * 100
        std_imp = importance[feat_idx].std() * 100
        print(f"  Feature {feat_idx:2d} ({FEATURE_NAMES[feat_idx]:>22s}): "
              f"dMAE = {mean_imp:+.4f} +/- {std_imp:.4f}%", flush=True)
    return importance, baseline_mae


print("\n--- M_phys permutation importance ---", flush=True)
imp_mphys, base_mphys = permutation_importance(m_phys, X_geo_test, X_phys_test, A_test)

print("\n--- M_TL+phys permutation importance ---", flush=True)
imp_mtlphys, base_mtlphys = permutation_importance(m_tlphys, X_geo_test, X_phys_test, A_test)

# ========= Step 6: Report (per-category aggregation + cumulative top-2) =========
print("\n" + "=" * 90)
print("PERMUTATION IMPORTANCE: Structure A (redesign) Physics Features")
print("=" * 90)

cat_names_list = list(CATEGORIES.keys())
feat_to_cat = {}
for cat, idxs in CATEGORIES.items():
    for i in idxs:
        feat_to_cat[i] = cat

for model_name, imp, base in [("M_phys", imp_mphys, base_mphys),
                              ("M_TL+phys", imp_mtlphys, base_mtlphys)]:
    print(f"\n{'=' * 90}")
    print(f"  {model_name}  (baseline MAE = {base*100:.4f}%)")
    print(f"{'=' * 90}")
    means = imp.mean(axis=1) * 100
    ranked = np.argsort(-means)
    print(f"\n  {'Rank':>4s}  {'Feature':>25s}  {'dMAE (%)':>12s}  {'Category':<22s}")
    for rank, fi in enumerate(ranked, 1):
        print(f"  {rank:4d}  {FEATURE_NAMES[fi]:>25s}  {means[fi]:+12.4f}  {feat_to_cat[fi]:<22s}")

    # Category-level aggregation (sum of per-feature dMAE within category)
    cat_sums = {cat: means[idxs].sum() for cat, idxs in CATEGORIES.items()}
    cat_means = {cat: means[idxs].mean() for cat, idxs in CATEGORIES.items()}
    total_pos = sum(max(v, 0.0) for v in cat_sums.values())
    total_all = sum(cat_sums.values())
    print(f"\n  Category-level importance (sum of feature dMAE):")
    print(f"  {'Category':>25s}  {'Sum dMAE(%)':>12s}  {'Mean dMAE(%)':>13s}  {'% of total':>11s}")
    ordered = sorted(cat_sums, key=cat_sums.get, reverse=True)
    for cat in ordered:
        pct = 100 * cat_sums[cat] / total_all if total_all != 0 else float('nan')
        print(f"  {cat:>25s}  {cat_sums[cat]:+12.4f}  {cat_means[cat]:+13.4f}  {pct:10.1f}%")
    top2 = ordered[:2]
    top2_sum = cat_sums[top2[0]] + cat_sums[top2[1]]
    print(f"\n  Top-2 categories: {top2[0]} + {top2[1]}")
    print(f"    cumulative sum dMAE = {top2_sum:+.4f}%")
    if total_all != 0:
        print(f"    cumulative %% of total          = {100*top2_sum/total_all:.1f}%")
    if total_pos != 0:
        print(f"    cumulative %% of positive total = {100*top2_sum/total_pos:.1f}%")

# ========= Step 7: Save (figure-contract keys + task-required keys) =========
imp_mphys_mean = imp_mphys.mean(axis=1)
imp_mphys_std = imp_mphys.std(axis=1)
imp_mtlphys_mean = imp_mtlphys.mean(axis=1)
imp_mtlphys_std = imp_mtlphys.std(axis=1)

cat_id = np.zeros(17, dtype=np.int32)
for ci, (cat, idxs) in enumerate(CATEGORIES.items()):
    for i in idxs:
        cat_id[i] = ci

savepath = f"{ROOT}/results/feature_importance_A_redesign.npz"
np.savez(
    savepath,
    imp_mphys_mean=imp_mphys_mean.astype(np.float32),
    imp_mphys_std=imp_mphys_std.astype(np.float32),
    imp_mtlphys_mean=imp_mtlphys_mean.astype(np.float32),
    imp_mtlphys_std=imp_mtlphys_std.astype(np.float32),
    imp_mphys=imp_mphys.astype(np.float32),
    imp_mtlphys=imp_mtlphys.astype(np.float32),
    baseline_mphys=base_mphys,
    baseline_mtlphys=base_mtlphys,
    feature_names=FEATURE_NAMES,
    category_names=cat_names_list,
    category_id=cat_id,
    n_permutations=N_PERMUTATIONS,
)
print(f"\nResults saved to: {savepath}", flush=True)
print("Done!", flush=True)
