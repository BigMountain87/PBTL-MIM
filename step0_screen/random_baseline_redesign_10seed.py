#!/usr/bin/env python3
"""
Random pre-training baseline for Structure A -- REDESIGN variant (Sec sec:random_baseline + Supp S1).

Ported from random_baseline_10seed.py to the corrected "redesign" dataset.
Question: does TMM pre-training provide *genuine physics transfer*, or is it just
"more pre-training data"? Compare against a control that pre-trains on uniform-random
absorptance spectra with the same param distribution and the same fine-tuning protocol.

Per training size n in {50,100,200,350}, 10 seeds, report test MAE (% on A):
  M0:     no pre-train (from scratch)
  M_TL:   TMM-pretrained, fine-tuned on RCWA
  M_rand: RANDOM-pretrained (uniform absorptance), fine-tuned on RCWA
  gap = M_rand - M_TL (pp)  -> >0 means TMM genuinely helps over random
  M_rand vs M0 (%)          -> negative => random pre-train HURTS vs scratch

REDESIGN changes vs legacy random_baseline_10seed.py (mirrors pbtl_A_redesign.py):
 (1) MATERIAL_MODEL='jc' set BEFORE importing tmm modules;
 (2) RCWA data from struct_A_500_redesign.npz (NOT struct_A_vis_500.npz);
 (3) wavelength grid LOADED from the npz (400-1800nm, 100pt; NOT hardcoded 380-780);
 (4) sample filter uses disclosed `reliable` mask (.all(axis=1));
 (5) all 6 bigmountain87 paths -> <repo root>;
 (6) TMM- and RANDOM-pretrain are both regenerated IN-SCRIPT on the redesign grid (the
     legacy *.pt weights are stale visible-range artifacts), so M_TL and M_rand share an
     identical 5000-sample pretrain budget and identical fine-tune protocol -- the only
     difference is TMM-physics labels vs uniform-random labels;
 (7) output -> results/random_baseline_redesign_10seed.npz;
 (8) device='cpu', torch threads pinned to 2 (shared-GPU parallel-agent etiquette).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_num_threads(2)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

# (1) jc materials BEFORE tmm imports
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.utils.physics_features import compute_physics_features_A
from src.simulation.tmm_struct_a import compute_tmm_batch

ROOT = '.'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # prefer CPU: parallel agents share one Mac GPU
print(f"Device: {device}", flush=True)

# ===== Model definitions (same as pbtl_A_redesign.py) =====
class BaseResNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden,hidden), nn.LayerNorm(hidden), nn.SiLU(),
                          nn.Linear(hidden,hidden), nn.LayerNorm(hidden))
            for _ in range(n_blocks)])
        self.act = nn.SiLU()
    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks: h = h + self.act(b(h))
        return h

class M0(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1); return {"A": 1-R, "R": R}

def train_model(model, dl_tr, dl_vl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss(); best_vl, best_st = float("inf"), None
    for ep in range(epochs):
        model.train()
        _ts = dl_tr.dataset.tensors; _N = _ts[0].shape[0]; _perm = torch.randperm(_N, device=_ts[0].device)
        for _i in range(0, _N, 131072):
            batch = tuple(_t[_perm[_i:_i+131072]] for _t in _ts)
            x, a, r = [t.to(device) for t in batch]; out = model(x)
            loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                _ts = dl_vl.dataset.tensors; _N = _ts[0].shape[0]
                for _i in range(0, _N, 131072):
                    batch = tuple(_t[_i:_i+131072] for _t in _ts)
                    x, a, r = [t.to(device) for t in batch]; out = model(x)
                    vl += (nn.functional.l1_loss(out["A"], a, reduction="sum") +
                           nn.functional.l1_loss(out["R"], r, reduction="sum")).item(); vn += len(a)*2
                vm = vl / vn
                if vm < best_vl:
                    best_vl = vm; best_st = {k: v.clone() for k, v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te):
    model.eval()
    with torch.no_grad():
        te = 0; tn = 0
        _ts = dl_te.dataset.tensors; _N = _ts[0].shape[0]
        for _i in range(0, _N, 131072):
            batch = tuple(_t[_i:_i+131072] for _t in _ts)
            x, a, r = [t.to(device) for t in batch]; out = model(x)
            te += nn.functional.l1_loss(out["A"], a, reduction="sum").item(); tn += len(a)
    return te / tn

# ===== Grid from REDESIGN data file (400-1800nm, 100pt) =====
RCWA_PATH = f"{ROOT}/data/raw/struct_A_500_redesign.npz"
data = np.load(RCWA_PATH, allow_pickle=True)
wavelengths = data["wavelengths"].astype(np.float32)
Nlam = len(wavelengths)
print(f"Grid (from data): {wavelengths.min():.0f}-{wavelengths.max():.0f}nm, {Nlam}pts", flush=True)
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + 10  # wavelength + 10 params

_, bounds_min, bounds_max = get_bounds("A")

# ===== Step 1: Generate TMM pre-train data (5000) on redesign grid =====
print("\n=== Step 1: Generate TMM pre-train data ===", flush=True)
N_PT = 5000
rng = np.random.default_rng(99)
params_pt = rng.uniform(bounds_min, bounds_max, (N_PT, 10)).astype(np.float32)
t0 = time.time()
tmm_out = compute_tmm_batch(params_pt, wavelengths, "Cr")
A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)
print(f"TMM: {N_PT} samples in {time.time()-t0:.1f}s, A range [{A_tmm.min():.3f},{A_tmm.max():.3f}]", flush=True)

# ===== Step 1b: RANDOM pre-train data (5000) -- same params, uniform-random spectra =====
print("\n=== Step 1b: Generate RANDOM pre-train data ===", flush=True)
rng_r = np.random.default_rng(99)
params_rand = rng_r.uniform(bounds_min, bounds_max, (N_PT, 10)).astype(np.float32)
A_rand = rng_r.uniform(0, 1, (N_PT, Nlam)).astype(np.float32)
R_rand = rng_r.uniform(0, 1 - A_rand, (N_PT, Nlam)).astype(np.float32)  # R <= 1-A
print(f"Random: A range [{A_rand.min():.3f},{A_rand.max():.3f}] (no physics)", flush=True)

# ----- shared builder for a pretrain corpus (geometry-only) -----
def build_pt_tensors(params_raw, A_raw, R_raw):
    pn = normalize_params(params_raw, "A")
    prep = np.repeat(pn[:, None, :], Nlam, axis=1)
    wlrep = np.tile(wl_norm[None, :, None], (len(params_raw), 1, 1))
    Xg = np.concatenate([wlrep, prep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
    return Xg, A_raw.reshape(-1).astype(np.float32), R_raw.reshape(-1).astype(np.float32)

Xg_tmm, A_tmm_f, R_tmm_f = build_pt_tensors(params_pt, A_tmm, R_tmm)
Xg_rnd, A_rnd_f, R_rnd_f = build_pt_tensors(params_rand, A_rand, R_rand)

n_pt_tr = int(N_PT * 0.9)
pt_idx = rng.permutation(N_PT)
pt_tr_rows = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in pt_idx[:n_pt_tr]])
pt_vl_rows = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in pt_idx[n_pt_tr:]])

def dl_pt(Xg, Af, Rf, rows, bs=131072, shuffle=False):
    xg = torch.tensor(Xg[rows]).to(device); a = torch.tensor(Af[rows]).to(device); r = torch.tensor(Rf[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

# ===== Step 2: Pre-train M0 on TMM and on RANDOM =====
print("\n=== Step 2: Pre-train (TMM and RANDOM) ===", flush=True)
PT_EPOCHS, PT_LR = 500, 1e-3

set_seed(42)
pt_tmm = M0(geo_dim).to(device)
t0 = time.time()
pt_tmm = train_model(pt_tmm, dl_pt(Xg_tmm, A_tmm_f, R_tmm_f, pt_tr_rows, shuffle=True),
                     dl_pt(Xg_tmm, A_tmm_f, R_tmm_f, pt_vl_rows), PT_EPOCHS, PT_LR)
print(f"Pre-trained M_TL (TMM): val MAE={eval_model(pt_tmm, dl_pt(Xg_tmm,A_tmm_f,R_tmm_f,pt_vl_rows))*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
torch.save(pt_tmm.state_dict(), f"{ROOT}/results/pretrained_m0_tmm_redesign.pt")

set_seed(42)
pt_rnd = M0(geo_dim).to(device)
t0 = time.time()
pt_rnd = train_model(pt_rnd, dl_pt(Xg_rnd, A_rnd_f, R_rnd_f, pt_tr_rows, shuffle=True),
                     dl_pt(Xg_rnd, A_rnd_f, R_rnd_f, pt_vl_rows), PT_EPOCHS, PT_LR)
print(f"Pre-trained M_rand (random): val MAE={eval_model(pt_rnd, dl_pt(Xg_rnd,A_rnd_f,R_rnd_f,pt_vl_rows))*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
torch.save(pt_rnd.state_dict(), f"{ROOT}/results/pretrained_m0_rand_redesign.pt")

# ===== Step 3: Load RCWA data (reliable mask) =====
print("\n=== Step 3: Load RCWA data ===", flush=True)
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)
good = data["reliable"].all(axis=1)
gi = np.where(good)[0]
params_rcwa = params_rcwa[gi]; A_rcwa = A_rcwa[gi]; R_rcwa = R_rcwa[gi]
N_r = len(gi)
print(f"RCWA: {N_r} reliable samples (of {len(good)})", flush=True)

pn_r = normalize_params(params_rcwa, "A")
prep_r = np.repeat(pn_r[:, None, :], Nlam, axis=1)
wlrep_r = np.tile(wl_norm[None, :, None], (N_r, 1, 1))
Xg_r = np.concatenate([wlrep_r, prep_r], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_r_f = A_rcwa.reshape(-1).astype(np.float32); R_r_f = R_rcwa.reshape(-1).astype(np.float32)

def get_rows(si): return np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in si])
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_r)
N_TEST, N_VAL = 50, 50
test_idx = all_idx[-N_TEST:]; val_idx = all_idx[-(N_TEST+N_VAL):-N_TEST]; remaining = all_idx[:-(N_TEST+N_VAL)]
test_rows = get_rows(test_idx); val_rows = get_rows(val_idx)

def make_dl(rows, bs=131072, shuffle=False):
    xg = torch.tensor(Xg_r[rows]).to(device); a = torch.tensor(A_r_f[rows]).to(device); r = torch.tensor(R_r_f[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_te = make_dl(test_rows); dl_vl = make_dl(val_rows)

# ===== Step 4: 3-way comparison (M0 / M_TL / M_rand) =====
print("\n=== Step 4: 3-way comparison ===", flush=True)
TRAIN_SIZES = [50, 100, 200, 350]
SEEDS = [42, 123, 777, 321, 456, 654, 999, 111, 222, 333]
FT_EPOCHS = 1000
FT_LR = 1e-3       # scratch
FT_LR_PT = 3e-4    # pre-trained (TMM or random)
results = {sz: {"M0": [], "M_TL": [], "M_rand": []} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    if n_train > len(remaining): continue
    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
        rng2 = np.random.default_rng(seed)
        tr_idx = remaining[rng2.permutation(len(remaining))[:n_train]]
        dl_tr = make_dl(get_rows(tr_idx), bs=131072, shuffle=True)

        # M0: from scratch
        set_seed(seed); m0 = M0(geo_dim).to(device)
        m0 = train_model(m0, dl_tr, dl_vl, FT_EPOCHS, FT_LR)
        results[n_train]["M0"].append(eval_model(m0, dl_te))
        print(f"  M0:     {results[n_train]['M0'][-1]*100:.3f}%", flush=True)

        # M_TL: TMM pre-train -> fine-tune
        set_seed(seed); m_tl = M0(geo_dim).to(device)
        m_tl.load_state_dict(deepcopy(pt_tmm.state_dict()))
        m_tl = train_model(m_tl, dl_tr, dl_vl, FT_EPOCHS, FT_LR_PT)
        results[n_train]["M_TL"].append(eval_model(m_tl, dl_te))
        print(f"  M_TL:   {results[n_train]['M_TL'][-1]*100:.3f}%", flush=True)

        # M_rand: RANDOM pre-train -> fine-tune (key baseline)
        set_seed(seed); m_rd = M0(geo_dim).to(device)
        m_rd.load_state_dict(deepcopy(pt_rnd.state_dict()))
        m_rd = train_model(m_rd, dl_tr, dl_vl, FT_EPOCHS, FT_LR_PT)
        results[n_train]["M_rand"].append(eval_model(m_rd, dl_te))
        print(f"  M_rand: {results[n_train]['M_rand'][-1]*100:.3f}%", flush=True)

# ===== Summary =====
print("\n" + "="*84, flush=True)
print("RANDOM PRE-TRAINING BASELINE (redesign): Structure A", flush=True)
print("="*84, flush=True)
print(f"{'n':>6} | {'M0':>14} | {'M_TL':>14} | {'M_rand':>14} | {'gap(pp)':>9} | {'rand vs M0':>11}", flush=True)
print("-"*84, flush=True)
for sz in TRAIN_SIZES:
    r = results[sz]
    if not r["M0"]: continue
    m0 = np.array(r["M0"]) * 100; tl = np.array(r["M_TL"]) * 100; rd = np.array(r["M_rand"]) * 100
    gap = rd.mean() - tl.mean()                       # pp; >0 => TMM beats random
    rand_vs_m0 = (rd.mean() / m0.mean() - 1) * 100    # % worse than scratch
    print(f"{sz:>6} | {m0.mean():>6.2f}±{m0.std():>5.2f} | {tl.mean():>6.2f}±{tl.std():>5.2f} | "
          f"{rd.mean():>6.2f}±{rd.std():>5.2f} | {gap:>+8.2f} | {rand_vs_m0:>+10.1f}%", flush=True)

print("\nInterpretation: gap>0 => TMM pre-train is genuine physics transfer, not just data;", flush=True)
print("                rand vs M0 > 0 => random pre-train actively HURTS vs from-scratch.", flush=True)

# ===== Save =====
os.makedirs(f"{ROOT}/results", exist_ok=True)
savepath = f"{ROOT}/results/random_baseline_redesign_10seed.npz"
np.savez(savepath,
         train_sizes=np.array(TRAIN_SIZES), seeds=np.array(SEEDS),
         M0=np.array([results[sz]["M0"] for sz in TRAIN_SIZES]),
         M_TL=np.array([results[sz]["M_TL"] for sz in TRAIN_SIZES]),
         M_rand=np.array([results[sz]["M_rand"] for sz in TRAIN_SIZES]))
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
