#!/usr/bin/env python3
"""
MLP baseline experiment: Does PBTL work with a simple MLP (no residual connections)?

Compares ResNet vs MLP backbone for M0 and M_TL on Structure A.
If PBTL benefits persist with MLP, the framework is architecture-agnostic.

Models:
  M0_resnet:  ResNet backbone, scratch
  M0_mlp:     MLP backbone, scratch
  M_TL_resnet: ResNet backbone, TMM pre-trained
  M_TL_mlp:   MLP backbone, TMM pre-trained
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params, get_bounds
from src.simulation.tmm_struct_a import compute_tmm_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ========= Model definitions =========
class BaseResNet(nn.Module):
    """ResNet backbone (same as paper)."""
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

class BaseMLP(nn.Module):
    """Simple MLP backbone (no residual connections, no LayerNorm).
    Matched parameter count: 4 hidden layers of 256."""
    def __init__(self, in_dim, hidden=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SurrogateModel(nn.Module):
    """Unified surrogate with swappable backbone."""
    def __init__(self, backbone):
        super().__init__()
        self.bb = backbone
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x, **kw):
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1-R, "R": R}


def train_model(model, dl_tr, dl_vl, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None
    for ep in range(epochs):
        model.train()
        for x, a, r in dl_tr:
            out = model(x)
            loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if (ep+1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                for x, a, r in dl_vl:
                    out = model(x)
                    vl += (nn.functional.l1_loss(out["A"], a, reduction="sum") +
                           nn.functional.l1_loss(out["R"], r, reduction="sum")).item()
                    vn += len(a) * 2
                vm = vl / vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k: v.clone() for k, v in model.state_dict().items()}
    if best_st: model.load_state_dict(best_st)
    return model

def eval_model(model, dl_te):
    model.eval()
    with torch.no_grad():
        te_loss = 0; te_n = 0
        for x, a, r in dl_te:
            out = model(x)
            te_loss += nn.functional.l1_loss(out["A"], a, reduction="sum").item()
            te_n += len(a)
    return te_loss / te_n

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ========= Data preparation =========
print("\n=== Generating TMM data ===", flush=True)
N_TMM = 5000
wavelengths = np.linspace(380, 780, 100).astype(np.float32)
Nlam = len(wavelengths)
_, bounds_min, bounds_max = get_bounds("A")
rng = np.random.default_rng(99)
params_tmm = rng.uniform(bounds_min, bounds_max, (N_TMM, 10)).astype(np.float32)

t0 = time.time()
tmm_out = compute_tmm_batch(params_tmm, wavelengths, "Cr")
print(f"TMM: {N_TMM} samples in {time.time()-t0:.1f}s", flush=True)

A_tmm = np.clip(tmm_out["A_tmm"], 0, 1).astype(np.float32)
R_tmm = np.clip(tmm_out["R_tmm"], 0, 1).astype(np.float32)

params_tmm_norm = normalize_params(params_tmm, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + 10

params_rep = np.repeat(params_tmm_norm[:, None, :], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None, :, None], (N_TMM, 1, 1))
X_geo_tmm = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_flat = A_tmm.reshape(-1)
R_flat = R_tmm.reshape(-1)

n_tmm_train = int(N_TMM * 0.9)
tmm_idx = rng.permutation(N_TMM)
tmm_tr = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in tmm_idx[:n_tmm_train]])
tmm_vl = np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in tmm_idx[n_tmm_train:]])

def to_dl(rows, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_tmm[rows]).to(device)
    a = torch.tensor(A_flat[rows]).to(device)
    r = torch.tensor(R_flat[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_tmm_tr = to_dl(tmm_tr, shuffle=True)
dl_tmm_vl = to_dl(tmm_vl)

# ========= Pre-train both architectures =========
print("\n=== Pre-training ===", flush=True)
PRETRAIN_EP = 500; PRETRAIN_LR = 1e-3

set_seed(42)
pt_resnet = SurrogateModel(BaseResNet(geo_dim)).to(device)
print(f"ResNet params: {count_params(pt_resnet):,}")
pt_resnet = train_model(pt_resnet, dl_tmm_tr, dl_tmm_vl, PRETRAIN_EP, PRETRAIN_LR)
mae_res = eval_model(pt_resnet, dl_tmm_vl)
print(f"ResNet TMM val MAE: {mae_res*100:.2f}%", flush=True)

set_seed(42)
pt_mlp = SurrogateModel(BaseMLP(geo_dim)).to(device)
print(f"MLP params: {count_params(pt_mlp):,}")
pt_mlp = train_model(pt_mlp, dl_tmm_tr, dl_tmm_vl, PRETRAIN_EP, PRETRAIN_LR)
mae_mlp = eval_model(pt_mlp, dl_tmm_vl)
print(f"MLP TMM val MAE: {mae_mlp*100:.2f}%", flush=True)

torch.save(pt_resnet.state_dict(), "results/pt_resnet.pt")
torch.save(pt_mlp.state_dict(), "results/pt_mlp.pt")

# ========= Load RCWA data =========
print("\n=== Loading RCWA data ===", flush=True)
data = np.load("data/raw/struct_A_vis_500.npz", allow_pickle=True)
params_rcwa = data["params"].astype(np.float32)
A_rcwa = data["A"].astype(np.float32)
R_rcwa = data["R"].astype(np.float32)

good = np.all((A_rcwa >= 0) & (A_rcwa <= 1) & (R_rcwa >= 0) & (R_rcwa <= 1), axis=1)
gi = np.where(good)[0]
params_rcwa, A_rcwa, R_rcwa = params_rcwa[gi], A_rcwa[gi], R_rcwa[gi]
N_rcwa = len(gi)
print(f"RCWA: {N_rcwa} samples", flush=True)

params_rcwa_norm = normalize_params(params_rcwa, "A")
params_rep_r = np.repeat(params_rcwa_norm[:, None, :], Nlam, axis=1)
wl_rep_r = np.tile(wl_norm[None, :, None], (N_rcwa, 1, 1))
X_geo_rcwa = np.concatenate([wl_rep_r, params_rep_r], axis=-1).reshape(-1, geo_dim).astype(np.float32)
A_rcwa_flat = A_rcwa.reshape(-1)
R_rcwa_flat = R_rcwa.reshape(-1)

def get_rows(si):
    return np.concatenate([np.arange(i*Nlam, (i+1)*Nlam) for i in si])

rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N_rcwa)
test_idx = all_idx[-50:]
val_idx = all_idx[-100:-50]
remaining = all_idx[:-100]

test_rows = get_rows(test_idx)
val_rows = get_rows(val_idx)

def make_dl(rows, bs=2048, shuffle=False):
    xg = torch.tensor(X_geo_rcwa[rows]).to(device)
    a = torch.tensor(A_rcwa_flat[rows]).to(device)
    r = torch.tensor(R_rcwa_flat[rows]).to(device)
    return DataLoader(TensorDataset(xg, a, r), batch_size=bs, shuffle=shuffle)

dl_te = make_dl(test_rows)
dl_vl = make_dl(val_rows)

# ========= Main experiment =========
print("\n=== 4-way architecture comparison ===", flush=True)
TRAIN_SIZES = [50, 100, 200, 350]
SEEDS = [42, 123, 777, 321, 456]  # 5 seeds for efficiency
FT_EP = 1000; FT_LR = 1e-3; FT_LR_TL = 3e-4

results = {sz: {k: [] for k in ["M0_resnet", "M0_mlp", "M_TL_resnet", "M_TL_mlp"]}
           for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    if n_train > len(remaining): continue
    for seed in SEEDS:
        print(f"\n--- n={n_train}, seed={seed} ---", flush=True)
        rng2 = np.random.default_rng(seed)
        tr_idx = remaining[rng2.permutation(len(remaining))[:n_train]]
        tr_rows = get_rows(tr_idx)
        dl_tr = make_dl(tr_rows, bs=512, shuffle=True)

        # M0_resnet
        set_seed(seed)
        m = SurrogateModel(BaseResNet(geo_dim)).to(device)
        m = train_model(m, dl_tr, dl_vl, FT_EP, FT_LR)
        mae = eval_model(m, dl_te)
        results[n_train]["M0_resnet"].append(mae)
        print(f"  M0_resnet:  {mae*100:.2f}%", flush=True)

        # M0_mlp
        set_seed(seed)
        m = SurrogateModel(BaseMLP(geo_dim)).to(device)
        m = train_model(m, dl_tr, dl_vl, FT_EP, FT_LR)
        mae = eval_model(m, dl_te)
        results[n_train]["M0_mlp"].append(mae)
        print(f"  M0_mlp:     {mae*100:.2f}%", flush=True)

        # M_TL_resnet
        set_seed(seed)
        m = SurrogateModel(BaseResNet(geo_dim)).to(device)
        m.load_state_dict(torch.load("results/pt_resnet.pt",
                                      map_location=device, weights_only=True))
        m = train_model(m, dl_tr, dl_vl, FT_EP, FT_LR_TL)
        mae = eval_model(m, dl_te)
        results[n_train]["M_TL_resnet"].append(mae)
        print(f"  M_TL_resnet:{mae*100:.2f}%", flush=True)

        # M_TL_mlp
        set_seed(seed)
        m = SurrogateModel(BaseMLP(geo_dim)).to(device)
        m.load_state_dict(torch.load("results/pt_mlp.pt",
                                      map_location=device, weights_only=True))
        m = train_model(m, dl_tr, dl_vl, FT_EP, FT_LR_TL)
        mae = eval_model(m, dl_te)
        results[n_train]["M_TL_mlp"].append(mae)
        print(f"  M_TL_mlp:   {mae*100:.2f}%", flush=True)

# ========= Summary =========
print("\n" + "="*90, flush=True)
print("MLP vs ResNet PBTL Comparison: Structure A (5 seeds)", flush=True)
print("="*90, flush=True)
header = f"{'n':>6} | {'M0_resnet':>14} | {'M0_mlp':>14} | {'M_TL_resnet':>14} | {'M_TL_mlp':>14}"
print(header, flush=True)
print("-"*90, flush=True)

for sz in TRAIN_SIZES:
    vals = {}
    for key in ["M0_resnet", "M0_mlp", "M_TL_resnet", "M_TL_mlp"]:
        v = np.array(results[sz][key]) * 100
        vals[key] = f"{v.mean():.2f}±{v.std():.2f}"
    print(f"{sz:>6} | {vals['M0_resnet']:>14} | {vals['M0_mlp']:>14} | {vals['M_TL_resnet']:>14} | {vals['M_TL_mlp']:>14}", flush=True)

print("\n--- TL Benefit (%) ---", flush=True)
for sz in TRAIN_SIZES:
    m0r = np.mean(results[sz]["M0_resnet"])
    m0m = np.mean(results[sz]["M0_mlp"])
    tlr = np.mean(results[sz]["M_TL_resnet"])
    tlm = np.mean(results[sz]["M_TL_mlp"])
    print(f"  n={sz}: ResNet TL benefit = {(1-tlr/m0r)*100:.1f}%, MLP TL benefit = {(1-tlm/m0m)*100:.1f}%", flush=True)

# Parameter counts
set_seed(42)
m_res = SurrogateModel(BaseResNet(geo_dim))
m_mlp = SurrogateModel(BaseMLP(geo_dim))
print(f"\nParameter counts: ResNet={count_params(m_res):,}, MLP={count_params(m_mlp):,}", flush=True)

savepath = "results/mlp_vs_resnet.npz"
np.savez(savepath, **{f"{sz}_{k}": results[sz][k] for sz in TRAIN_SIZES for k in results[sz]},
         train_sizes=TRAIN_SIZES, seeds=SEEDS)
print(f"\nSaved: {savepath}", flush=True)
print("Done!", flush=True)
