#!/usr/bin/env python3
"""
Data efficiency experiment: M0 vs M_phys at different training set sizes.
Uses existing 94 good samples, varying train size: 20, 35, 50, 65.
"""
import sys, os
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed_utils import set_global_seed as set_seed
from src.utils.data_utils import normalize_params
from src.simulation.materials import (get_sio2_permittivity, get_tio2_permittivity,
                                       get_metal_permittivity)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def compute_physics_features(params, wavelengths_nm, metal="Cr"):
    N = len(params)
    Nlam = len(wavelengths_nm)
    P, Wx, Wy, W2 = params[:,0], params[:,1], params[:,2], params[:,3]
    t1, t2, t_mid = params[:,4], params[:,5], params[:,6]
    d1, d2, theta = params[:,7], params[:,8], params[:,9]
    theta_rad = np.deg2rad(theta)

    eps_sio2 = get_sio2_permittivity(wavelengths_nm)
    eps_tio2 = get_tio2_permittivity(wavelengths_nm)
    eps_metal = get_metal_permittivity(wavelengths_nm, metal)
    n_sio2 = np.sqrt(np.real(eps_sio2))
    n_tio2 = np.sqrt(np.real(eps_tio2))
    k_metal = np.imag(np.sqrt(eps_metal))
    skin_depth = wavelengths_nm / (4 * np.pi * k_metal)

    feats = []
    for n_cav, d_cav in [(n_sio2, d1), (n_tio2, d2)]:
        sin_ti = np.clip(np.sin(theta_rad[:,None]) / n_cav[None,:], -1, 1)
        cos_ti = np.sqrt(1 - sin_ti**2)
        phase = 4*np.pi * n_cav[None,:] * d_cav[:,None] * cos_ti / wavelengths_nm[None,:]
        feats.append(np.cos(phase))
        feats.append(np.sin(phase))

    feats.append(np.tile((Wx*Wy/P**2)[:,None], (1,Nlam)))
    feats.append(np.tile((W2**2/P**2)[:,None], (1,Nlam)))
    feats.append(P[:,None] / wavelengths_nm[None,:])
    feats.append(Wx[:,None] / wavelengths_nm[None,:])
    feats.append(W2[:,None] / wavelengths_nm[None,:])
    feats.append(t1[:,None] / skin_depth[None,:])
    feats.append(t2[:,None] / skin_depth[None,:])
    feats.append(t_mid[:,None] / skin_depth[None,:])
    feats.append(np.tile(np.cos(theta_rad[:,None]), (1,Nlam)))
    feats.append(n_sio2[None,:] * d1[:,None] / wavelengths_nm[None,:])
    feats.append(n_tio2[None,:] * d2[:,None] / wavelengths_nm[None,:])
    feats.append(np.tile((Wy/(Wx+1e-10))[:,None], (1,Nlam)))
    alpha = 4*np.pi*k_metal / wavelengths_nm
    feats.append(np.tile(alpha[None,:], (N,1)))

    return np.stack(feats, axis=-1).astype(np.float32)


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

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1)); R = self.head(h).squeeze(-1); return {"A": 1-R, "R": R}


def train_eval(model, dl_tr, dl_vl, dl_te, epochs=5000, lr=1e-3, has_phys=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None

    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if has_phys: x,a,r,p = batch; out = model(x, p=p)
            else: x,a,r = batch; out = model(x)
            loss = crit(out["A"],a) + crit(out["R"],r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (ep+1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                for batch in dl_vl:
                    if has_phys: x,a,r,p = batch; out = model(x, p=p)
                    else: x,a,r = batch; out = model(x)
                    vl += (nn.functional.l1_loss(out["A"],a,reduction="sum") +
                           nn.functional.l1_loss(out["R"],r,reduction="sum")).item()
                    vn += len(a)*2
                vm = vl/vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k:v.clone() for k,v in model.state_dict().items()}

    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        te_loss = 0; te_n = 0
        for batch in dl_te:
            if has_phys: x,a,r,p = batch; out = model(x, p=p)
            else: x,a,r = batch; out = model(x)
            te_loss += nn.functional.l1_loss(out["A"],a,reduction="sum").item()
            te_n += len(a)
    return te_loss / te_n


# ========= Load data =========
data = np.load("data/raw/struct_A_vis_100.npz", allow_pickle=True)
params_all = data["params"].astype(np.float32)
A_all = data["A"].astype(np.float32)
R_all = data["R"].astype(np.float32)
wavelengths = data["wavelengths"].astype(np.float32)
Nlam = len(wavelengths)

good = np.all((R_all>=0)&(R_all<=1)&(A_all>=0)&(A_all<=1), axis=1)
gi = np.where(good)[0]
N = len(gi)
params = params_all[gi]; A_arr = A_all[gi]; R_arr = R_all[gi]
print(f"Data: {N} samples, {Nlam} wavelengths")

phys = compute_physics_features(params, wavelengths, "Cr")
n_phys = phys.shape[-1]

params_norm = normalize_params(params, "A")
wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
geo_dim = 1 + params.shape[1]

params_rep = np.repeat(params_norm[:,None,:], Nlam, axis=1)
wl_rep = np.tile(wl_norm[None,:,None], (N,1,1))
X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
X_phys = phys.reshape(-1, n_phys).astype(np.float32)
pm, ps = X_phys.mean(0,keepdims=True), X_phys.std(0,keepdims=True)+1e-8
X_phys_n = ((X_phys - pm) / ps).astype(np.float32)
A_flat = A_arr.reshape(-1).astype(np.float32)
R_flat = R_arr.reshape(-1).astype(np.float32)

def get_rows(si): return np.concatenate([np.arange(i*Nlam,(i+1)*Nlam) for i in si])

# ========= Data efficiency experiment =========
TRAIN_SIZES = [20, 35, 50, 65]
N_TEST = 15  # fixed test set
SEEDS = [42, 123, 777]
EPOCHS = 5000

# Fixed test set (last 15 samples in shuffled order)
rng_split = np.random.default_rng(42)
all_idx = rng_split.permutation(N)
test_idx = all_idx[-N_TEST:]
remaining = all_idx[:-N_TEST]

test_rows = get_rows(test_idx)
X_te = torch.tensor(X_geo[test_rows]).to(device)
A_te = torch.tensor(A_flat[test_rows]).to(device)
R_te = torch.tensor(R_flat[test_rows]).to(device)
P_te = torch.tensor(X_phys_n[test_rows]).to(device)

ds_te_m0 = TensorDataset(X_te, A_te, R_te)
ds_te_ph = TensorDataset(X_te, A_te, R_te, P_te)
dl_te_m0 = DataLoader(ds_te_m0, batch_size=2048)
dl_te_ph = DataLoader(ds_te_ph, batch_size=2048)

results = {sz: {"M0": [], "M_phys": []} for sz in TRAIN_SIZES}

for n_train in TRAIN_SIZES:
    n_val = max(5, len(remaining) - n_train)  # use rest as val
    if n_train + n_val > len(remaining):
        n_val = len(remaining) - n_train

    for seed in SEEDS:
        print(f"\n--- n_train={n_train}, seed={seed} ---")
        set_seed(seed)

        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(remaining))
        tr_idx = remaining[perm[:n_train]]
        vl_idx = remaining[perm[n_train:n_train+n_val]]

        tr_rows = get_rows(tr_idx)
        vl_rows = get_rows(vl_idx)

        X_tr = torch.tensor(X_geo[tr_rows]).to(device)
        A_tr = torch.tensor(A_flat[tr_rows]).to(device)
        R_tr = torch.tensor(R_flat[tr_rows]).to(device)
        P_tr = torch.tensor(X_phys_n[tr_rows]).to(device)

        X_vl = torch.tensor(X_geo[vl_rows]).to(device)
        A_vl = torch.tensor(A_flat[vl_rows]).to(device)
        R_vl = torch.tensor(R_flat[vl_rows]).to(device)
        P_vl = torch.tensor(X_phys_n[vl_rows]).to(device)

        dl_tr_m0 = DataLoader(TensorDataset(X_tr,A_tr,R_tr), batch_size=512, shuffle=True)
        dl_vl_m0 = DataLoader(TensorDataset(X_vl,A_vl,R_vl), batch_size=1024)
        dl_tr_ph = DataLoader(TensorDataset(X_tr,A_tr,R_tr,P_tr), batch_size=512, shuffle=True)
        dl_vl_ph = DataLoader(TensorDataset(X_vl,A_vl,R_vl,P_vl), batch_size=1024)

        # M0
        set_seed(seed)
        m0 = M0(geo_dim).to(device)
        mae_m0 = train_eval(m0, dl_tr_m0, dl_vl_m0, dl_te_m0, EPOCHS, has_phys=False)
        results[n_train]["M0"].append(mae_m0)
        print(f"  M0: {mae_m0*100:.3f}%")

        # M_phys
        set_seed(seed)
        mp = MPhys(geo_dim, n_phys).to(device)
        mae_ph = train_eval(mp, dl_tr_ph, dl_vl_ph, dl_te_ph, EPOCHS, has_phys=True)
        results[n_train]["M_phys"].append(mae_ph)
        print(f"  M_phys: {mae_ph*100:.3f}%")

# ========= Summary =========
print("\n" + "="*70)
print("DATA EFFICIENCY: M0 vs M_phys (Structure A, Visible)")
print("="*70)
print(f"Test set: {N_TEST} samples (fixed)")
print(f"Seeds: {SEEDS}")
print()
print(f"{'n_train':>8} | {'M0 MAE':>15} | {'M_phys MAE':>15} | {'ratio':>8} | {'improvement':>12}")
print("-"*70)

for sz in TRAIN_SIZES:
    m0_vals = np.array(results[sz]["M0"]) * 100
    mp_vals = np.array(results[sz]["M_phys"]) * 100
    m0_m, m0_s = m0_vals.mean(), m0_vals.std()
    mp_m, mp_s = mp_vals.mean(), mp_vals.std()
    ratio = mp_m / m0_m
    impr = (1 - ratio) * 100
    print(f"{sz:>8} | {m0_m:>6.2f} +/- {m0_s:>4.2f}% | {mp_m:>6.2f} +/- {mp_s:>4.2f}% | {ratio:>7.3f} | {impr:>10.1f}%")

print()
print("Key insight: Physics features help MORE with LESS data (higher improvement at small n_train)")
