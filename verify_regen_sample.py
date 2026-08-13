#!/usr/bin/env python3
"""verify_regen_sample.py — archive-vs-code regeneration check (stratified sample).

For each shipped raw RCWA dataset, draw a fixed-seed random sample of designs,
re-run the *current* released simulation code on the archived parameters, and
compare every regenerated (sample, wavelength) cell against the archived value.

This certifies that the shipped .npz archives are the product of the shipped
code path (adaptive Fourier order, jc materials, complex64), i.e. no silent
code/data drift. It complements the independent-solver (grcwa) spot checks and
the SHA-256 file manifest, which cover solver correctness and file integrity.

Run from the repo root (CUDA GPU required):
    python verify_regen_sample.py [--n 50] [--datasets A B C AuA AuB]

Writes results/verify_regen_sample.npz and prints a per-dataset PASS/FAIL
summary. Tolerance: complex64 GPU arithmetic across torch versions is not
bit-stable, so we report max|dA| and flag cells above --tol (default 5e-3,
i.e. 0.5 pp of absorptance).
"""
import argparse, importlib, sys, time
import numpy as np
import torch

DATASETS = {
    # key: (raw npz, module, dual-pol, metal-override or None)
    'A':   ('data/raw/struct_A_500_redesign.npz', 'src.simulation.rcwa_struct_a', False, None),
    'B':   ('data/raw/struct_B_500_redesign.npz', 'src.simulation.rcwa_struct_b', False, None),
    'C':   ('data/raw/struct_C_500_redesign.npz', 'src.simulation.rcwa_struct_c', True,  None),
    'AuA': ('data/raw/struct_A_Au_500_jc.npz',    'src.simulation.rcwa_struct_a', False, None),
    'AuB': ('data/raw/struct_B_Au_500_jc.npz',    'src.simulation.rcwa_struct_b', False, None),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50, help='samples per dataset')
    ap.add_argument('--tol', type=float, default=5e-3, help='per-cell |dA| flag threshold')
    ap.add_argument('--seed', type=int, default=20260807)
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS.keys()))
    ap.add_argument('--wlstride', type=int, default=1,
                    help='compare every k-th wavelength only (adaptive order is per-wavelength, so a subset is exact for those cells)')
    args = ap.parse_args()

    import src.simulation.materials as _mat
    _mat.MATERIAL_MODEL = 'jc'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  torch={torch.__version__}", flush=True)

    out = {}
    overall_ok = True
    for key in args.datasets:
        src, modname, dual, metal_ov = DATASETS[key]
        d = np.load(src, allow_pickle=True)
        params = d['params']; names = list(d['param_names'])
        wl_full = np.asarray(d['wavelengths'], dtype=np.float64)
        wsel = np.arange(0, len(wl_full), args.wlstride)
        wl = wl_full[wsel]
        metal = metal_ov or (str(d['metal']) if 'metal' in d else 'Cr')
        mod = importlib.import_module(modname)
        mod.RCWA_SETTINGS['adaptive_order'] = True
        mod.RCWA_SETTINGS['dtype'] = torch.complex64

        # only compare rows the archive actually finished (finite)
        akey = 'A_TE' if dual else 'A'
        finite = np.isfinite(d[akey]).all(axis=1)
        idx_pool = np.where(finite)[0]
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(idx_pool, size=min(args.n, len(idx_pool)), replace=False)
        pick.sort()

        dmax_all, flagged, per_sample = 0.0, 0, []
        t0 = time.time()
        for c, i in enumerate(pick):
            p = {names[k]: float(params[i, k]) for k in range(len(names))}
            try:
                if dual:
                    ate, rte, tte, atm, rtm, ttm = mod.simulate_single(p, wl, metal=metal, device=device)
                    dA = max(np.nanmax(np.abs(ate - d['A_TE'][i][wsel])),
                             np.nanmax(np.abs(atm - d['A_TM'][i][wsel])))
                else:
                    A, R, T = mod.simulate_single(p, wl, metal=metal, device=device)
                    dA = np.nanmax(np.abs(A - d['A'][i][wsel]))
            except Exception as e:
                print(f"  [{key}] sample {i} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
                dA = np.inf
            per_sample.append((int(i), float(dA)))
            dmax_all = max(dmax_all, dA)
            if dA > args.tol:
                flagged += 1
                print(f"  [{key}] sample {i}: max|dA|={dA:.4g}  > tol", flush=True)
            if (c + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{key}] {c+1}/{len(pick)}  max|dA| so far {dmax_all:.3g}  "
                      f"({el/(c+1):.1f}s/sample)", flush=True)
        ok = flagged == 0 and np.isfinite(dmax_all)
        overall_ok &= ok
        print(f"[{key}] {'PASS' if ok else 'FAIL'}  n={len(pick)}  max|dA|={dmax_all:.4g}  "
              f"flagged(>{args.tol})={flagged}  ({time.time()-t0:.0f}s)", flush=True)
        out[f'{key}_idx'] = np.array([s for s, _ in per_sample])
        out[f'{key}_dmax'] = np.array([v for _, v in per_sample])

    out['tol'] = args.tol; out['seed'] = args.seed; out['wlstride'] = args.wlstride
    np.savez('results/verify_regen_sample.npz', **out)
    print(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}  -> results/verify_regen_sample.npz", flush=True)
    sys.exit(0 if overall_ok else 1)

if __name__ == '__main__':
    main()
