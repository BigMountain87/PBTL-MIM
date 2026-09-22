"""Order x raster convergence ladder for the Au-B converged-truth check.

Each (design, wavelength, grid, order) solve runs in a fresh subprocess, so a CUDA
out-of-memory at one order cannot poison the rest.  Records absorptance, seconds and
peak GPU memory to results/verification_probes/aub_order_raster_ladder.json.  Run from the repository root (needs a CUDA GPU with >= 14 GB for N = 23).

  ORDERS=13,17,19,21,23 GRIDS=64,128 WLS=780,650 N_DESIGNS=3 python verify_au_order_ladder.py
"""
import os, sys, json, subprocess, time
import numpy as np

ORDERS = [int(x) for x in os.environ.get("ORDERS", "13,17,19,21,23").split(",")]
GRIDS = [int(x) for x in os.environ.get("GRIDS", "64,128").split(",")]
WLS = [float(x) for x in os.environ.get("WLS", "780,650").split(",")]
N_DESIGNS = int(os.environ.get("N_DESIGNS", "3"))
PY = sys.executable

d = np.load("data/raw/struct_B_Au_500_jc.npz", allow_pickle=True)
A = d["A"].astype(np.float64); params = d["params"]; names = [str(n) for n in d["param_names"]]
wl = d["wavelengths"].astype(np.float64)
good = ((A >= -0.005) & (A <= 1.005)).all(axis=1); gi = np.where(good)[0]
perm = np.random.default_rng(42).permutation(len(gi)); test_idx = gi[perm[-40:]]
P = params[test_idx, names.index("P")]; oi = np.argsort(-P)
picks = [int(test_idx[k]) for k in oi[:N_DESIGNS - 1]] + [int(test_idx[oi[len(oi) // 2]])]

CHILD = r'''
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.getcwd())
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"
import src.simulation.rcwa_struct_b as mod
di, lam, grid, N = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = np.load("data/raw/struct_B_Au_500_jc.npz", allow_pickle=True)
names = [str(n) for n in d["param_names"]]; p = {n: float(v) for n, v in zip(names, d["params"][di])}
mod.RCWA_SETTINGS["adaptive_order"] = False; mod.RCWA_SETTINGS["order"] = [N, N]
mod.RCWA_SETTINGS["dtype"] = torch.complex64; mod.RCWA_SETTINGS["grid"] = (grid, grid)
dev = torch.device("cuda"); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
a, r, t = mod.simulate_single(p, np.array([lam]), metal="Au", device=dev); torch.cuda.synchronize()
print(json.dumps({"A": float(a[0]), "R": float(r[0]), "T": float(t[0]), "sec": round(time.time() - t0, 2),
                  "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))
'''
open("results/verification_probes/_ladder_child.py", "w").write(CHILD)

out = {"orders": ORDERS, "grids": GRIDS, "wls": WLS, "designs": []}
for di in picks:
    p = {n: float(v) for n, v in zip(names, params[di])}
    rec = {"design": di, "P": p["P"], "cells": {}}
    for lam in WLS:
        j = int(np.argmin(abs(wl - lam))); ad = 17 if p["P"] / wl[j] >= 1.3 else (13 if p["P"] / wl[j] >= 0.8 else 9)
        cell = {"archived": float(A[di, j]), "archived_order": ad, "runs": {}}
        for grid in GRIDS:
            for N in ORDERS:
                key = f"g{grid}_N{N}"
                try:
                    cp = subprocess.run([PY, "results/verification_probes/_ladder_child.py", str(di), str(wl[j]), str(grid), str(N)],
                                        capture_output=True, text=True, timeout=1800)
                    line = [l for l in cp.stdout.splitlines() if l.startswith("{")]
                    if cp.returncode == 0 and line:
                        cell["runs"][key] = json.loads(line[-1])
                        r = cell["runs"][key]
                        print(f"design {di} P={p['P']:.0f} wl={wl[j]:.1f} grid {grid} N={N}: A={r['A']:.4f} (archived {cell['archived']:.4f} @N={ad}) {r['sec']:.1f}s peak {r['peak_GB']:.2f} GB", flush=True)
                    else:
                        err = (cp.stderr.strip().splitlines() or ["?"])[-1][:100]
                        cell["runs"][key] = {"error": err}
                        print(f"design {di} wl={wl[j]:.1f} grid {grid} N={N}: FAILED {err}", flush=True)
                except subprocess.TimeoutExpired:
                    cell["runs"][key] = {"error": "timeout 1800 s"}
                    print(f"design {di} wl={wl[j]:.1f} grid {grid} N={N}: TIMEOUT", flush=True)
        rec["cells"][str(int(wl[j]))] = cell
    out["designs"].append(rec)
    json.dump(out, open("results/verification_probes/aub_order_raster_ladder.json", "w"), indent=1)
print("LADDER DONE", flush=True)
