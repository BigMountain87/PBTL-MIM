#!/usr/bin/env python3
"""
Step 0: Quick M0 vs M7 training for each structure (5000 epochs).
Uses 70/15/15 split on 100 samples.
Reads from data/raw/struct_{A,B,C}_100.npz
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


def train_model_3ch(model, data, epochs, lr, device):
    """Train 3-channel model (Structure A or B)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    X_train = data["train"]["X_geo"].to(device)
    A_train = data["train"]["A"].to(device)
    R_train = data["train"]["R"].to(device)
    T_train = data["train"]["T"].to(device)

    use_tmm = model.cfg.get("use_tmm_backbone", False)
    if use_tmm:
        A_tmm_train = data["train"]["A_tmm"].to(device)
        R_tmm_train = data["train"]["R_tmm"].to(device)
        T_tmm_train = data["train"]["T_tmm"].to(device)
    
    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if use_tmm:
            A_p, R_p, T_p = model(X_train, A_tmm_train, R_tmm_train, T_tmm_train)
        else:
            A_p, R_p, T_p = model(X_train)
        
        loss = nn.MSELoss()(A_p, A_train) + nn.MSELoss()(R_p, R_train) + nn.MSELoss()(T_p, T_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model


def train_model_6ch(model, data, epochs, lr, device):
    """Train 6-channel model (Structure C)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    X_train = data["train"]["X_geo"].to(device)
    A_te_train = data["train"]["A_TE"].to(device)
    R_te_train = data["train"]["R_TE"].to(device)
    T_te_train = data["train"]["T_TE"].to(device)
    A_tm_train = data["train"]["A_TM"].to(device)
    R_tm_train = data["train"]["R_TM"].to(device)
    T_tm_train = data["train"]["T_TM"].to(device)

    use_tmm = model.cfg.get("use_tmm_backbone", False)
    if use_tmm:
        A_tmm_te = data["train"]["A_tmm_te"].to(device)
        R_tmm_te = data["train"]["R_tmm_te"].to(device)
        T_tmm_te = data["train"]["T_tmm_te"].to(device)
        A_tmm_tm = data["train"]["A_tmm_tm"].to(device)
        R_tmm_tm = data["train"]["R_tmm_tm"].to(device)
        T_tmm_tm = data["train"]["T_tmm_tm"].to(device)
    
    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if use_tmm:
            A_te_p, R_te_p, T_te_p, A_tm_p, R_tm_p, T_tm_p = model(
                X_train, A_tmm_te, R_tmm_te, T_tmm_te,
                A_tmm_tm, R_tmm_tm, T_tmm_tm)
        else:
            A_te_p, R_te_p, T_te_p, A_tm_p, R_tm_p, T_tm_p = model(X_train)
        
        loss = (nn.MSELoss()(A_te_p, A_te_train) + nn.MSELoss()(R_te_p, R_te_train) +
                nn.MSELoss()(T_te_p, T_te_train) + nn.MSELoss()(A_tm_p, A_tm_train) +
                nn.MSELoss()(R_tm_p, R_tm_train) + nn.MSELoss()(T_tm_p, T_tm_train))
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model


def eval_model_3ch(model, data, device):
    """Evaluate 3-channel model on test set."""
    model.eval()
    model = model.to(device)
    
    X_test = data["test"]["X_geo"].to(device)
    A_test = data["test"]["A"].to(device)
    R_test = data["test"]["R"].to(device)
    T_test = data["test"]["T"].to(device)

    use_tmm = model.cfg.get("use_tmm_backbone", False)
    
    with torch.no_grad():
        if use_tmm:
            A_tmm = data["test"]["A_tmm"].to(device)
            R_tmm = data["test"]["R_tmm"].to(device)
            T_tmm = data["test"]["T_tmm"].to(device)
            A_p, R_p, T_p = model(X_test, A_tmm, R_tmm, T_tmm)
        else:
            A_p, R_p, T_p = model(X_test)
    
    mae_a = torch.mean(torch.abs(A_p - A_test)).item()
    mae_r = torch.mean(torch.abs(R_p - R_test)).item()
    mae_t = torch.mean(torch.abs(T_p - T_test)).item()
    mae_avg = (mae_a + mae_r + mae_t) / 3
    return {"MAE_A": mae_a, "MAE_R": mae_r, "MAE_T": mae_t, "MAE_avg": mae_avg}


def eval_model_6ch(model, data, device):
    """Evaluate 6-channel model on test set."""
    model.eval()
    model = model.to(device)
    
    X_test = data["test"]["X_geo"].to(device)
    use_tmm = model.cfg.get("use_tmm_backbone", False)
    
    with torch.no_grad():
        if use_tmm:
            A_te_p, R_te_p, T_te_p, A_tm_p, R_tm_p, T_tm_p = model(
                X_test,
                data["test"]["A_tmm_te"].to(device),
                data["test"]["R_tmm_te"].to(device),
                data["test"]["T_tmm_te"].to(device),
                data["test"]["A_tmm_tm"].to(device),
                data["test"]["R_tmm_tm"].to(device),
                data["test"]["T_tmm_tm"].to(device))
        else:
            A_te_p, R_te_p, T_te_p, A_tm_p, R_tm_p, T_tm_p = model(X_test)
    
    mae_a_te = torch.mean(torch.abs(A_te_p - data["test"]["A_TE"].to(device))).item()
    mae_a_tm = torch.mean(torch.abs(A_tm_p - data["test"]["A_TM"].to(device))).item()
    mae_avg = (mae_a_te + mae_a_tm) / 2
    return {"MAE_A_TE": mae_a_te, "MAE_A_TM": mae_a_tm, "MAE_avg": mae_avg}


# =====================================================
from src.utils.data_utils import load_and_preprocess
from src.models.tmm_nn import create_model
from src.utils.seed_utils import set_seed

structures = ["A", "B", "C"]
results = {}

for struct in structures:
    print(f"\n{'='*70}")
    print(f"Structure {struct}: M0 vs M7 Quick Training ({EPOCHS} epochs)")
    print(f"{'='*70}")
    
    npz_path = f'data/raw/struct_{struct}_100.npz'
    if not os.path.exists(npz_path):
        print(f"  [SKIP] {npz_path} not found")
        continue
    
    set_seed(SEED)
    data = load_and_preprocess(npz_path, structure=struct, metal="Cr", seed=SEED)
    
    geo_dim = data["geo_dim"]
    n_channels = 6 if struct == "C" else 3
    print(f"  geo_dim={geo_dim}, n_channels={n_channels}")
    print(f"  Train: {data['train']['X_geo'].shape[0]} points, "
          f"Test: {data['test']['X_geo'].shape[0]} points")
    
    results[struct] = {}
    
    for model_name in ["M0", "M7"]:
        set_seed(SEED)
        model = create_model(model_name, geo_dim=geo_dim, n_channels=n_channels)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {model_name}: {n_params:,} parameters")
        
        t0 = time.time()
        if n_channels == 6:
            model = train_model_6ch(model, data, EPOCHS, LR, device)
            metrics = eval_model_6ch(model, data, device)
        else:
            model = train_model_3ch(model, data, EPOCHS, LR, device)
            metrics = eval_model_3ch(model, data, device)
        dt = time.time() - t0
        
        results[struct][model_name] = metrics
        print(f"  {model_name}: MAE_avg={metrics['MAE_avg']*100:.2f}% "
              f"(time={dt:.0f}s)")
        for k, v in metrics.items():
            if k != "MAE_avg":
                print(f"    {k}: {v*100:.2f}%")

# =====================================================
print("\n" + "="*70)
print("STEP 0 SCREENING RESULTS")
print("="*70)
print(f"\n{'Structure':<15} {'M0 MAE':<12} {'M7 MAE':<12} {'M7/M0':<10} {'TMM gain'}")
print("-" * 60)
for struct in structures:
    if struct not in results:
        continue
    m0 = results[struct]["M0"]["MAE_avg"] * 100
    m7 = results[struct]["M7"]["MAE_avg"] * 100
    ratio = m7 / m0 if m0 > 0 else float('inf')
    gain = (m0 - m7)
    print(f"{struct:<15} {m0:<12.2f} {m7:<12.2f} {ratio:<10.2f} {gain:<10.2f}")

print("\nSelection criteria:")
print("  1. TMM MAE 10-20% (physics backbone useful but NN correction needed)")
print("  2. M7 << M0 (TMM backbone significantly improves NN)")
print("  3. Spectral complexity (multiple peaks, Fano features)")
