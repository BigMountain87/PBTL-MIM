# src/utils/data_utils.py
"""
Data preprocessing utilities for 3-structure MIM screening.
Supports structures A (10 params), B (8 params), C (7 params, 6-channel output).
"""

import numpy as np
import torch

# Per-structure parameter bounds
BOUNDS = {
    "A": {
        "names": ["P", "Wx", "Wy", "W2", "t1", "t2", "t_mid", "d1", "d2", "theta"],
        "min": np.array([300., 50., 50., 50., 10., 10., 5., 30., 30., 0.]),
        "max": np.array([600., 540., 540., 540., 80., 80., 30., 200., 200., 45.]),
    },
    "B": {
        "names": ["P", "R_out", "R_in", "R_disk", "t_Cr", "d_SiO2", "theta", "phi"],
        "min": np.array([300., 80., 30., 10., 20., 50., 0., 0.]),
        "max": np.array([800., 350., 300., 100., 80., 200., 60., 45.]),
    },
    "C": {
        "names": ["P", "Wx", "Wy", "t_Cr", "d_SiO2", "theta", "phi"],
        "min": np.array([300., 50., 50., 20., 50., 0., 0.]),
        "max": np.array([800., 720., 720., 80., 200., 60., 45.]),
    },
}


def get_bounds(structure: str):
    """Get parameter bounds for a structure."""
    b = BOUNDS[structure]
    return b["names"], b["min"], b["max"]


def normalize_params(params: np.ndarray, structure: str) -> np.ndarray:
    """Normalize params to [0, 1] range."""
    _, mn, mx = get_bounds(structure)
    return (params - mn) / (mx - mn)


def denormalize_params(params_norm, structure: str):
    """Denormalize params from [0, 1] to physical units."""
    _, mn, mx = get_bounds(structure)
    if isinstance(params_norm, torch.Tensor):
        mn_t = torch.tensor(mn, dtype=torch.float32, device=params_norm.device)
        mx_t = torch.tensor(mx, dtype=torch.float32, device=params_norm.device)
        return params_norm * (mx_t - mn_t) + mn_t
    return params_norm * (mx - mn) + mn


def load_and_preprocess(npz_path: str, structure: str, metal: str = "Cr",
                        train_frac: float = 0.70, val_frac: float = 0.15,
                        test_frac: float = 0.15, seed: int = 42) -> dict:
    """
    Load .npz data → compute TMM → normalize → split.
    
    Structure A/B: 3-channel output (A, R, T)
    Structure C: 6-channel output (A_TE, R_TE, T_TE, A_TM, R_TM, T_TM)
    """
    d = np.load(npz_path, allow_pickle=True)
    params = d["params"].astype(np.float32)
    wavelengths = d["wavelengths"].astype(np.float32)
    N = len(params)
    Nlam = len(wavelengths)

    # Load RCWA data
    if structure == "C":
        A_te = d["A_TE"].astype(np.float32)
        R_te = d["R_TE"].astype(np.float32)
        T_te = d["T_TE"].astype(np.float32)
        A_tm = d["A_TM"].astype(np.float32)
        R_tm = d["R_TM"].astype(np.float32)
        T_tm = d["T_TM"].astype(np.float32)
    else:
        A = d["A"].astype(np.float32)
        R = d["R"].astype(np.float32)
        T = d["T"].astype(np.float32)

    # Compute TMM backbone
    print(f"[INFO] Computing TMM for {N} samples (Structure {structure}, {metal})...")
    if structure == "A":
        from src.simulation.tmm_struct_a import compute_tmm_batch
        tmm = compute_tmm_batch(params, wavelengths, metal=metal)
        A_tmm = tmm["A_tmm"].astype(np.float32)
        R_tmm = tmm["R_tmm"].astype(np.float32)
        T_tmm = tmm["T_tmm"].astype(np.float32)
    elif structure == "B":
        from src.simulation.tmm_struct_b import compute_tmm_batch
        tmm = compute_tmm_batch(params, wavelengths, metal=metal)
        A_tmm = tmm["A_tmm"].astype(np.float32)
        R_tmm = tmm["R_tmm"].astype(np.float32)
        T_tmm = tmm["T_tmm"].astype(np.float32)
    elif structure == "C":
        from src.simulation.tmm_struct_c import compute_tmm_batch
        tmm = compute_tmm_batch(params, wavelengths, metal=metal)
        A_tmm_te = tmm["A_tmm_te"].astype(np.float32)
        R_tmm_te = tmm["R_tmm_te"].astype(np.float32)
        T_tmm_te = tmm["T_tmm_te"].astype(np.float32)
        A_tmm_tm = tmm["A_tmm_tm"].astype(np.float32)
        R_tmm_tm = tmm["R_tmm_tm"].astype(np.float32)
        T_tmm_tm = tmm["T_tmm_tm"].astype(np.float32)

    # Normalize params
    params_norm = normalize_params(params, structure)
    wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

    n_params = params.shape[1]
    
    # Build input tensor: [N, Nlam, 1+n_params]
    params_rep = np.repeat(params_norm[:, np.newaxis, :], Nlam, axis=1)
    wl_rep = np.tile(wl_norm[np.newaxis, :, np.newaxis], (N, 1, 1))
    X_geo_full = np.concatenate([wl_rep, params_rep], axis=-1)  # [N, Nlam, 1+n_params]

    geo_dim = 1 + n_params  # e.g., A: 11, B: 9, C: 8

    # Flatten: [N*Nlam, geo_dim]
    X_geo_flat = X_geo_full.reshape(-1, geo_dim).astype(np.float32)

    # Train/Val/Test split (sample level)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def _subset(flat_arr, sub_idx):
        rows = np.concatenate([np.arange(i * Nlam, (i + 1) * Nlam) for i in sub_idx])
        return flat_arr[rows]

    def _build_split(sub_idx):
        result = {
            "X_geo": torch.tensor(_subset(X_geo_flat, sub_idx)),
        }
        if structure == "C":
            for ch, arr in [("A_TE", A_te), ("R_TE", R_te), ("T_TE", T_te),
                            ("A_TM", A_tm), ("R_TM", R_tm), ("T_TM", T_tm)]:
                result[ch] = torch.tensor(_subset(arr.reshape(-1), sub_idx))
            for ch, arr in [("A_tmm_te", A_tmm_te), ("R_tmm_te", R_tmm_te), ("T_tmm_te", T_tmm_te),
                            ("A_tmm_tm", A_tmm_tm), ("R_tmm_tm", R_tmm_tm), ("T_tmm_tm", T_tmm_tm)]:
                result[ch] = torch.tensor(_subset(arr.reshape(-1), sub_idx))
        else:
            for ch, arr in [("A", A), ("R", R), ("T", T)]:
                result[ch] = torch.tensor(_subset(arr.reshape(-1), sub_idx))
            for ch, arr in [("A_tmm", A_tmm), ("R_tmm", R_tmm), ("T_tmm", T_tmm)]:
                result[ch] = torch.tensor(_subset(arr.reshape(-1), sub_idx))
        return result

    result = {
        "train": _build_split(train_idx),
        "test": _build_split(test_idx),
        "wavelengths": wavelengths,
        "params_test": params[test_idx],
        "params_train": params[train_idx],
        "structure": structure,
        "geo_dim": geo_dim,
        "n_params": n_params,
    }
    if n_val > 0:
        result["val"] = _build_split(val_idx)

    return result
