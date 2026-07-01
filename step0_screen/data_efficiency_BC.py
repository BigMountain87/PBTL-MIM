#!/usr/bin/env python3
"""
Data efficiency experiment for Structure B and C.
Uses shared physics_features module and generalized model classes.
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
from src.utils.physics_features import (
    compute_physics_features_B, compute_physics_features_C, N_PHYS_FEATURES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ========= Model definitions =========
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
        R = self.head(self.bb(x)).squeeze(-1)
        return {"A": 1-R, "R": R}

class MPhys(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1))
        R = self.head(h).squeeze(-1)
        return {"A": 1-R, "R": R}

# For Structure C: predict TE and TM separately
class M0_DualPol(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bb = BaseResNet(d)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,2), nn.Sigmoid())
    def forward(self, x, **kw):
        out = self.head(self.bb(x))  # (B, 2) = R_TE, R_TM
        return {"A_TE": 1-out[:,0], "R_TE": out[:,0], "A_TM": 1-out[:,1], "R_TM": out[:,1]}

class MPhys_DualPol(nn.Module):
    def __init__(self, gd, pd):
        super().__init__()
        self.bb = BaseResNet(gd+pd)
        self.head = nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,2), nn.Sigmoid())
    def forward(self, x, p=None, **kw):
        h = self.bb(torch.cat([x,p],-1))
        out = self.head(h)
        return {"A_TE": 1-out[:,0], "R_TE": out[:,0], "A_TM": 1-out[:,1], "R_TM": out[:,1]}


def train_eval(model, dl_tr, dl_vl, dl_te, epochs=5000, lr=1e-3, has_phys=False, dual_pol=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.MSELoss()
    best_vl, best_st = float('inf'), None

    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            if dual_pol:
                if has_phys:
                    x, a_te, r_te, a_tm, r_tm, p = batch
                    out = model(x, p=p)
                else:
                    x, a_te, r_te, a_tm, r_tm = batch
                    out = model(x)
                loss = crit(out["A_TE"], a_te) + crit(out["R_TE"], r_te) + \
                       crit(out["A_TM"], a_tm) + crit(out["R_TM"], r_tm)
            else:
                if has_phys:
                    x, a, r, p = batch
                    out = model(x, p=p)
                else:
                    x, a, r = batch
                    out = model(x)
                loss = crit(out["A"], a) + crit(out["R"], r)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (ep+1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                vl = 0; vn = 0
                for batch in dl_vl:
                    if dual_pol:
                        if has_phys:
                            x, a_te, r_te, a_tm, r_tm, p = batch
                            out = model(x, p=p)
                        else:
                            x, a_te, r_te, a_tm, r_tm = batch
                            out = model(x)
                        vl += (nn.functional.l1_loss(out["A_TE"], a_te, reduction="sum") +
                               nn.functional.l1_loss(out["A_TM"], a_tm, reduction="sum")).item()
                        vn += len(a_te) * 2
                    else:
                        if has_phys:
                            x, a, r, p = batch
                            out = model(x, p=p)
                        else:
                            x, a, r = batch
                            out = model(x)
                        vl += nn.functional.l1_loss(out["A"], a, reduction="sum").item()
                        vn += len(a)
                vm = vl / vn
                if vm < best_vl:
                    best_vl = vm
                    best_st = {k: v.clone() for k, v in model.state_dict().items()}

    if best_st:
        model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        te_loss = 0; te_n = 0
        for batch in dl_te:
            if dual_pol:
                if has_phys:
                    x, a_te, r_te, a_tm, r_tm, p = batch
                    out = model(x, p=p)
                else:
                    x, a_te, r_te, a_tm, r_tm = batch
                    out = model(x)
                te_loss += (nn.functional.l1_loss(out["A_TE"], a_te, reduction="sum") +
                            nn.functional.l1_loss(out["A_TM"], a_tm, reduction="sum")).item()
                te_n += len(a_te) * 2
            else:
                if has_phys:
                    x, a, r, p = batch
                    out = model(x, p=p)
                else:
                    x, a, r = batch
                    out = model(x)
                te_loss += nn.functional.l1_loss(out["A"], a, reduction="sum").item()
                te_n += len(a)
    return te_loss / te_n


def run_experiment(struct_name):
    print(f"\n{'='*70}", flush=True)
    print(f"DATA EFFICIENCY: Structure {struct_name}", flush=True)
    print(f"{'='*70}", flush=True)

    dual_pol = (struct_name == "C")

    # Load data
    datapath = f"data/raw/struct_{struct_name}_500.npz"
    if not os.path.exists(datapath):
        print(f"ERROR: {datapath} not found! Generate data first.", flush=True)
        return None
    
    data = np.load(datapath, allow_pickle=True)
    params_all = data["params"].astype(np.float32)
    wavelengths = data["wavelengths"].astype(np.float32)
    Nlam = len(wavelengths)

    if dual_pol:
        A_TE_all = data["A_TE"].astype(np.float32)
        R_TE_all = data["R_TE"].astype(np.float32)
        A_TM_all = data["A_TM"].astype(np.float32)
        R_TM_all = data["R_TM"].astype(np.float32)
        # Filter bad samples
        good = (np.all((A_TE_all >= -0.01) & (A_TE_all <= 1.01), axis=1) &
                np.all((A_TM_all >= -0.01) & (A_TM_all <= 1.01), axis=1))
    else:
        A_all = data["A"].astype(np.float32)
        R_all = data["R"].astype(np.float32)
        good = np.all((A_all >= -0.01) & (A_all <= 1.01) & (R_all >= -0.01) & (R_all <= 1.01), axis=1)

    gi = np.where(good)[0]
    N = len(gi)
    params = params_all[gi]
    print(f"Data: {N} good samples out of {len(params_all)}, {Nlam} wavelengths", flush=True)

    if dual_pol:
        A_TE = np.clip(A_TE_all[gi], 0, 1)
        R_TE = np.clip(R_TE_all[gi], 0, 1)
        A_TM = np.clip(A_TM_all[gi], 0, 1)
        R_TM = np.clip(R_TM_all[gi], 0, 1)
    else:
        A_arr = np.clip(A_all[gi], 0, 1)
        R_arr = np.clip(R_all[gi], 0, 1)

    # Physics features
    if struct_name == "B":
        phys = compute_physics_features_B(params, wavelengths, "Cr")
    elif struct_name == "C":
        phys = compute_physics_features_C(params, wavelengths, "Cr")
    n_phys = phys.shape[-1]
    print(f"Physics features: {n_phys}", flush=True)

    # Normalize
    params_norm = normalize_params(params, struct_name)
    wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
    geo_dim = 1 + params.shape[1]

    params_rep = np.repeat(params_norm[:, None, :], Nlam, axis=1)
    wl_rep = np.tile(wl_norm[None, :, None], (N, 1, 1))
    X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, geo_dim).astype(np.float32)
    X_phys = phys.reshape(-1, n_phys).astype(np.float32)
    pm, ps = X_phys.mean(0, keepdims=True), X_phys.std(0, keepdims=True) + 1e-8
    X_phys_n = ((X_phys - pm) / ps).astype(np.float32)

    if dual_pol:
        A_TE_flat = A_TE.reshape(-1).astype(np.float32)
        R_TE_flat = R_TE.reshape(-1).astype(np.float32)
        A_TM_flat = A_TM.reshape(-1).astype(np.float32)
        R_TM_flat = R_TM.reshape(-1).astype(np.float32)
    else:
        A_flat = A_arr.reshape(-1).astype(np.float32)
        R_flat = R_arr.reshape(-1).astype(np.float32)

    def get_rows(si):
        return np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in si])

    # Split
    TRAIN_SIZES = [50, 100, 200, 350]
    N_TEST = 50
    N_VAL = 50
    SEEDS = [42, 123, 777]
    EPOCHS = 5000

    rng_split = np.random.default_rng(42)
    all_idx = rng_split.permutation(N)
    test_idx = all_idx[-N_TEST:]
    val_idx = all_idx[-(N_TEST + N_VAL):-N_TEST]
    remaining = all_idx[:-(N_TEST + N_VAL)]
    print(f"Test: {len(test_idx)}, Val: {len(val_idx)}, Pool: {len(remaining)}", flush=True)

    test_rows = get_rows(test_idx)
    val_rows = get_rows(val_idx)

    def make_tensors(rows):
        xg = torch.tensor(X_geo[rows]).to(device)
        xp = torch.tensor(X_phys_n[rows]).to(device)
        if dual_pol:
            return (xg, torch.tensor(A_TE_flat[rows]).to(device),
                    torch.tensor(R_TE_flat[rows]).to(device),
                    torch.tensor(A_TM_flat[rows]).to(device),
                    torch.tensor(R_TM_flat[rows]).to(device), xp)
        else:
            return (xg, torch.tensor(A_flat[rows]).to(device),
                    torch.tensor(R_flat[rows]).to(device), xp)

    te_tensors = make_tensors(test_rows)
    vl_tensors = make_tensors(val_rows)

    if dual_pol:
        ds_te_m0 = TensorDataset(te_tensors[0], te_tensors[1], te_tensors[2], te_tensors[3], te_tensors[4])
        ds_te_ph = TensorDataset(*te_tensors)
        ds_vl_m0 = TensorDataset(vl_tensors[0], vl_tensors[1], vl_tensors[2], vl_tensors[3], vl_tensors[4])
        ds_vl_ph = TensorDataset(*vl_tensors)
    else:
        ds_te_m0 = TensorDataset(te_tensors[0], te_tensors[1], te_tensors[2])
        ds_te_ph = TensorDataset(*te_tensors)
        ds_vl_m0 = TensorDataset(vl_tensors[0], vl_tensors[1], vl_tensors[2])
        ds_vl_ph = TensorDataset(*vl_tensors)

    dl_te_m0 = DataLoader(ds_te_m0, batch_size=2048)
    dl_te_ph = DataLoader(ds_te_ph, batch_size=2048)
    dl_vl_m0 = DataLoader(ds_vl_m0, batch_size=2048)
    dl_vl_ph = DataLoader(ds_vl_ph, batch_size=2048)

    results = {sz: {"M0": [], "M_phys": []} for sz in TRAIN_SIZES}

    for n_train in TRAIN_SIZES:
        if n_train > len(remaining):
            print(f"Skipping n_train={n_train} (only {len(remaining)} available)", flush=True)
            continue
        for seed in SEEDS:
            print(f"\n--- n_train={n_train}, seed={seed} ---", flush=True)
            set_seed(seed)
            rng2 = np.random.default_rng(seed)
            perm = rng2.permutation(len(remaining))
            tr_idx = remaining[perm[:n_train]]
            tr_rows = get_rows(tr_idx)
            tr_tensors = make_tensors(tr_rows)

            if dual_pol:
                dl_tr_m0 = DataLoader(TensorDataset(tr_tensors[0], tr_tensors[1], tr_tensors[2],
                                                     tr_tensors[3], tr_tensors[4]),
                                      batch_size=512, shuffle=True)
                dl_tr_ph = DataLoader(TensorDataset(*tr_tensors), batch_size=512, shuffle=True)
            else:
                dl_tr_m0 = DataLoader(TensorDataset(tr_tensors[0], tr_tensors[1], tr_tensors[2]),
                                      batch_size=512, shuffle=True)
                dl_tr_ph = DataLoader(TensorDataset(*tr_tensors), batch_size=512, shuffle=True)

            # M0
            set_seed(seed)
            if dual_pol:
                m0 = M0_DualPol(geo_dim).to(device)
            else:
                m0 = M0(geo_dim).to(device)
            mae_m0 = train_eval(m0, dl_tr_m0, dl_vl_m0, dl_te_m0, EPOCHS, has_phys=False, dual_pol=dual_pol)
            results[n_train]["M0"].append(mae_m0)
            print(f"  M0: {mae_m0*100:.3f}%", flush=True)

            # M_phys
            set_seed(seed)
            if dual_pol:
                mp = MPhys_DualPol(geo_dim, n_phys).to(device)
            else:
                mp = MPhys(geo_dim, n_phys).to(device)
            mae_ph = train_eval(mp, dl_tr_ph, dl_vl_ph, dl_te_ph, EPOCHS, has_phys=True, dual_pol=dual_pol)
            results[n_train]["M_phys"].append(mae_ph)
            print(f"  M_phys: {mae_ph*100:.3f}%", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS: Structure {struct_name} (M0 vs M_phys)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'n_train':>8} | {'M0 MAE':>15} | {'M_phys MAE':>15} | {'improvement':>12}", flush=True)
    print("-" * 60, flush=True)

    for sz in TRAIN_SIZES:
        if not results[sz]["M0"]:
            continue
        m0_vals = np.array(results[sz]["M0"]) * 100
        mp_vals = np.array(results[sz]["M_phys"]) * 100
        m0_m, m0_s = m0_vals.mean(), m0_vals.std()
        mp_m, mp_s = mp_vals.mean(), mp_vals.std()
        impr = (1 - mp_m / m0_m) * 100
        print(f"{sz:>8} | {m0_m:>6.2f} +/- {m0_s:>4.2f}% | {mp_m:>6.2f} +/- {mp_s:>4.2f}% | {impr:>10.1f}%", flush=True)

    # Save results
    savepath = f"results/data_efficiency_{struct_name}.npz"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    np.savez(savepath, **{f"{sz}_M0": results[sz]["M0"] for sz in TRAIN_SIZES},
             **{f"{sz}_MPhys": results[sz]["M_phys"] for sz in TRAIN_SIZES},
             train_sizes=TRAIN_SIZES, seeds=SEEDS)
    print(f"\nSaved: {savepath}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct", required=True, choices=["B", "C"])
    args = parser.parse_args()
    run_experiment(args.struct)
