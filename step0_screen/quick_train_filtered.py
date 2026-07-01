#!/usr/bin/env python3
"""
Step 0: Quick M0 vs M7 training with filtered data (remove R>1 samples).
"""
import sys, os, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

EPOCHS = 5000
LR = 1e-3
SEED = 42

from src.utils.seed_utils import set_global_seed as set_seed
from src.models.tmm_nn import create_model


def filter_bad_samples(npz_path, structure):
    """Filter out samples with R > 1 at any wavelength."""
    d = dict(np.load(npz_path, allow_pickle=True))
    
    if structure == "C":
        R = d["R_TE"]
        R2 = d["R_TM"]
        good = ~(np.any(R > 1.0, axis=1) | np.any(R2 > 1.0, axis=1))
    else:
        R = d["R"]
        good = ~np.any(R > 1.0, axis=1)
    
    n_total = len(d["params"])
    n_good = np.sum(good)
    print(f"  Filtered: {n_total} → {n_good} samples ({n_total-n_good} removed)")
    
    # Filter
    d["params"] = d["params"][good]
    if structure == "C":
        for key in ["A_TE", "R_TE", "T_TE", "A_TM", "R_TM", "T_TM"]:
            d[key] = d[key][good]
    else:
        for key in ["A", "R", "T"]:
            d[key] = d[key][good]
    
    return d


def prepare_data(d, structure, metal="Cr"):
    """Prepare data with TMM backbone, normalize, and split."""
    from src.utils.data_utils import normalize_params, BOUNDS
    
    params = d["params"].astype(np.float32)
    wavelengths = d["wavelengths"].astype(np.float32)
    N = len(params)
    Nlam = len(wavelengths)
    
    # Load RCWA targets
    if structure == "C":
        targets = {k: d[k].astype(np.float32) for k in ["A_TE", "R_TE", "T_TE", "A_TM", "R_TM", "T_TM"]}
    else:
        targets = {k: d[k].astype(np.float32) for k in ["A", "R", "T"]}
    
    # Compute TMM
    print(f"  Computing TMM backbone...")
    if structure == "A":
        from src.simulation.tmm_struct_a import compute_tmm_batch
        tmm = compute_tmm_batch(params, wavelengths, metal=metal)
        tmm_data = {f"{k}": v.astype(np.float32) for k, v in tmm.items()}
    elif structure == "B":
        from src.simulation.tmm_struct_b import compute_tmm_batch
        tmm = compute_tmm_batch(params, wavelengths, metal=metal)
        tmm_data = {f"{k}": v.astype(np.float32) for k, v in tmm.items()}
    else:
        from src.simulation.tmm_struct_c import compute_tmm_batch
        tmm = compute_tmm_batch(params, wavelengths, metal=metal)
        tmm_data = {k: v.astype(np.float32) for k, v in tmm.items()}
    
    # Normalize
    params_norm = normalize_params(params, structure)
    wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
    
    n_params = params.shape[1]
    params_rep = np.repeat(params_norm[:, np.newaxis, :], Nlam, axis=1)
    wl_rep = np.tile(wl_norm[np.newaxis, :, np.newaxis], (N, 1, 1))
    X_geo = np.concatenate([wl_rep, params_rep], axis=-1).reshape(-1, 1 + n_params).astype(np.float32)
    
    geo_dim = 1 + n_params
    
    # Split: 70/15/15
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    n_train = int(N * 0.70)
    n_val = int(N * 0.15)
    train_idx = idx[:n_train]
    test_idx = idx[n_train + n_val:]
    
    def _sub(arr_flat, sub_idx):
        rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in sub_idx])
        return arr_flat[rows]
    
    # Build flat arrays
    result = {"geo_dim": geo_dim, "n_channels": 6 if structure == "C" else 3}
    
    for split, sidx in [("train", train_idx), ("test", test_idx)]:
        result[f"X_geo_{split}"] = torch.tensor(_sub(X_geo, sidx))
        
        if structure == "C":
            for ch in ["A_TE", "R_TE", "T_TE", "A_TM", "R_TM", "T_TM"]:
                result[f"{ch}_{split}"] = torch.tensor(_sub(targets[ch].reshape(-1), sidx))
            result[f"A_tmm_te_{split}"] = torch.tensor(_sub(tmm_data["A_tmm_te"].reshape(-1), sidx))
            result[f"R_tmm_te_{split}"] = torch.tensor(_sub(tmm_data["R_tmm_te"].reshape(-1), sidx))
            result[f"T_tmm_te_{split}"] = torch.tensor(_sub(tmm_data["T_tmm_te"].reshape(-1), sidx))
            result[f"A_tmm_tm_{split}"] = torch.tensor(_sub(tmm_data["A_tmm_tm"].reshape(-1), sidx))
            result[f"R_tmm_tm_{split}"] = torch.tensor(_sub(tmm_data["R_tmm_tm"].reshape(-1), sidx))
            result[f"T_tmm_tm_{split}"] = torch.tensor(_sub(tmm_data["T_tmm_tm"].reshape(-1), sidx))
        else:
            for ch in ["A", "R", "T"]:
                result[f"{ch}_{split}"] = torch.tensor(_sub(targets[ch].reshape(-1), sidx))
            result[f"A_tmm_{split}"] = torch.tensor(_sub(tmm_data["A_tmm"].reshape(-1), sidx))
            result[f"R_tmm_{split}"] = torch.tensor(_sub(tmm_data["R_tmm"].reshape(-1), sidx))
            result[f"T_tmm_{split}"] = torch.tensor(_sub(tmm_data["T_tmm"].reshape(-1), sidx))
    
    # TMM baseline MAE
    if structure == "C":
        tmm_mae = (np.mean(np.abs(tmm_data["A_tmm_te"] - targets["A_TE"])) +
                   np.mean(np.abs(tmm_data["A_tmm_tm"] - targets["A_TM"]))) / 2
    else:
        tmm_mae = np.mean(np.abs(tmm_data["A_tmm"] - targets["A"]))
    result["tmm_mae"] = tmm_mae
    
    return result


def train_and_eval(model, data, structure, device):
    """Train and evaluate model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    X_train = data["X_geo_train"].to(device)
    use_tmm = model.cfg.get("use_tmm_backbone", False)
    
    if structure == "C":
        targets_train = {k: data[f"{k}_train"].to(device) for k in ["A_TE", "R_TE", "T_TE", "A_TM", "R_TM", "T_TM"]}
        if use_tmm:
            tmm_train = {k: data[f"{k}_train"].to(device) for k in ["A_tmm_te", "R_tmm_te", "T_tmm_te", "A_tmm_tm", "R_tmm_tm", "T_tmm_tm"]}
    else:
        targets_train = {k: data[f"{k}_train"].to(device) for k in ["A", "R", "T"]}
        if use_tmm:
            tmm_train = {k: data[f"{k}_train"].to(device) for k in ["A_tmm", "R_tmm", "T_tmm"]}
    
    best_loss = float('inf')
    best_state = None
    
    for ep in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        if structure == "C":
            if use_tmm:
                out = model(X_train, tmm_train["A_tmm_te"], tmm_train["R_tmm_te"], tmm_train["T_tmm_te"],
                           tmm_train["A_tmm_tm"], tmm_train["R_tmm_tm"], tmm_train["T_tmm_tm"])
            else:
                out = model(X_train)
            loss = sum(nn.MSELoss()(out[i], targets_train[k]) 
                      for i, k in enumerate(["A_TE", "R_TE", "T_TE", "A_TM", "R_TM", "T_TM"]))
        else:
            if use_tmm:
                out = model(X_train, tmm_train["A_tmm"], tmm_train["R_tmm"], tmm_train["T_tmm"])
            else:
                out = model(X_train)
            loss = sum(nn.MSELoss()(out[i], targets_train[k]) 
                      for i, k in enumerate(["A", "R", "T"]))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    model.eval()
    
    # Evaluate on test
    X_test = data["X_geo_test"].to(device)
    with torch.no_grad():
        if structure == "C":
            if use_tmm:
                out = model(X_test, data["A_tmm_te_test"].to(device), data["R_tmm_te_test"].to(device),
                           data["T_tmm_te_test"].to(device), data["A_tmm_tm_test"].to(device),
                           data["R_tmm_tm_test"].to(device), data["T_tmm_tm_test"].to(device))
            else:
                out = model(X_test)
            mae = (torch.mean(torch.abs(out[0] - data["A_TE_test"].to(device))).item() +
                   torch.mean(torch.abs(out[3] - data["A_TM_test"].to(device))).item()) / 2
        else:
            if use_tmm:
                out = model(X_test, data["A_tmm_test"].to(device), data["R_tmm_test"].to(device),
                           data["T_tmm_test"].to(device))
            else:
                out = model(X_test)
            mae = torch.mean(torch.abs(out[0] - data["A_test"].to(device))).item()
    
    return mae


# ========== Main ==========
structures = ["A", "B", "C"]
results = {}

for struct in structures:
    print(f"\n{'='*70}")
    print(f"Structure {struct}")
    print(f"{'='*70}")
    
    npz_path = f'data/raw/struct_{struct}_100.npz'
    
    # Filter
    d = filter_bad_samples(npz_path, struct)
    data = prepare_data(d, struct)
    
    geo_dim = data["geo_dim"]
    n_channels = data["n_channels"]
    tmm_mae = data["tmm_mae"]
    print(f"  geo_dim={geo_dim}, n_channels={n_channels}")
    print(f"  TMM-only MAE (filtered): {tmm_mae*100:.2f}%")
    
    results[struct] = {"tmm_mae": tmm_mae}
    
    for model_name in ["M0", "M7"]:
        set_seed(SEED)
        model = create_model(model_name, geo_dim=geo_dim, n_channels=n_channels)
        n_p = sum(p.numel() for p in model.parameters())
        
        t0 = time.time()
        mae = train_and_eval(model, data, struct, device)
        dt = time.time() - t0
        
        results[struct][model_name] = mae
        print(f"  {model_name}: MAE(A)={mae*100:.2f}%, params={n_p:,}, time={dt:.0f}s")

# Summary
print(f"\n{'='*70}")
print(f"STEP 0 SCREENING RESULTS (filtered data)")
print(f"{'='*70}")
print(f"\n{'Structure':<15} {'TMM MAE':<12} {'M0 MAE':<12} {'M7 MAE':<12} {'M7/M0':<10} {'TMM helps?'}")
print("-" * 70)
for struct in structures:
    r = results[struct]
    tmm = r["tmm_mae"] * 100
    m0 = r["M0"] * 100
    m7 = r["M7"] * 100
    ratio = m7 / m0 if m0 > 0 else float('inf')
    helps = "YES ✓" if ratio < 0.85 else "MARGINAL" if ratio < 0.95 else "NO ✗"
    print(f"{struct:<15} {tmm:<12.2f} {m0:<12.2f} {m7:<12.2f} {ratio:<10.3f} {helps}")
