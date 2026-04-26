# src/training/trainer.py

import torch
import torch.optim as optim
import numpy as np
from typing import Optional


def train_model(model, X_geo, A_tmm, R_tmm, T_tmm,
                A_rcwa, R_rcwa, T_rcwa,
                config: dict,
                val_data: Optional[dict] = None,
                device: str = "cuda") -> dict:
    """
    단일 모델 학습.

    X_geo:  [N, 5] normalized geometric+lambda input (torch)
    *_tmm:  [N]    TMM backbone predictions (torch)
    *_rcwa: [N]    RCWA ground truth (torch)
    config: 학습 하이퍼파라미터
    returns: history dict
    """
    model = model.to(device)
    X_geo   = X_geo.to(device)
    A_tmm   = A_tmm.to(device); R_tmm = R_tmm.to(device); T_tmm = T_tmm.to(device)
    A_rcwa  = A_rcwa.to(device); R_rcwa = R_rcwa.to(device); T_rcwa = T_rcwa.to(device)

    epochs     = config.get("epochs", 20000)
    lr         = config.get("lr", 1e-3)
    wd         = config.get("weight_decay", 1e-4)
    w_energy   = config.get("w_energy", 0.0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=epochs,
                                                       eta_min=1e-6)
    history = {"train_loss": [], "train_mae": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state   = None

    N_total = X_geo.shape[0]
    batch_size = min(8192, N_total)

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        perm = torch.randperm(N_total, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_total, batch_size):
            idx = perm[start:start+batch_size]
            optimizer.zero_grad()

            A_pred, R_pred, T_pred = model(
                X_geo[idx], A_tmm[idx], R_tmm[idx], T_tmm[idx])

            # Data loss
            loss_data = (torch.mean((A_pred - A_rcwa[idx])**2) +
                         torch.mean((R_pred - R_rcwa[idx])**2) +
                         torch.mean((T_pred - T_rcwa[idx])**2))

            # Energy penalty loss
            loss_energy = 0.0
            if w_energy > 0:
                viol = torch.abs(A_pred + R_pred + T_pred - 1.0)
                loss_energy = w_energy * torch.mean(viol**2)

            loss = loss_data + loss_energy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Logging
        if epoch % 500 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                model.eval()
                A_pred_all, R_pred_all, T_pred_all = model(
                    X_geo, A_tmm, R_tmm, T_tmm)
                mae_train = torch.mean(torch.abs(A_pred_all - A_rcwa)).item() * 100
                history["train_loss"].append(epoch_loss / max(n_batches, 1))
                history["train_mae"].append(mae_train)

                if val_data is not None:
                    A_vp, R_vp, T_vp = model(
                        val_data["X_geo"].to(device),
                        val_data["A_tmm"].to(device),
                        val_data["R_tmm"].to(device),
                        val_data["T_tmm"].to(device),
                    )
                    mae_val = torch.mean(torch.abs(
                        A_vp - val_data["A_rcwa"].to(device))).item() * 100
                    history["val_mae"].append(mae_val)

                    if mae_val < best_val_mae:
                        best_val_mae = mae_val
                        best_state   = {k: v.cpu().clone()
                                        for k, v in model.state_dict().items()}

                if epoch % 2000 == 0:
                    val_str = f", MAE_val={mae_val:.3f}%" if val_data else ""
                    print(f"  Epoch {epoch:5d}: loss={epoch_loss/max(n_batches,1):.6f}, "
                          f"MAE_train={mae_train:.3f}%{val_str}")

    if best_state:
        model.load_state_dict(best_state)

    return history
