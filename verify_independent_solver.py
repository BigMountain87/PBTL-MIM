#!/usr/bin/env python3
"""verify_independent_solver.py — archive vs. an independent RCWA implementation.

The shipped absorptance archives were produced with torcwa. This script
recomputes selected archived designs with grcwa, a separate open-source RCWA
implementation by different authors, and compares the result to the archived
value. Because the two codes share no implementation, agreement is evidence
that the archived spectra reflect the physics rather than a solver-specific
artifact -- something re-running the same code cannot establish.

Designs are read from the shipped raw archives, so the geometries tested are
the ones actually used in the paper (not synthetic stand-ins). Only
near-normal-incidence designs are used: at normal incidence the two codes'
polarisation conventions coincide unambiguously, which removes the convention
mismatch that a previous audit round showed can masquerade as a physics error.

CPU only (grcwa is NumPy); safe to run alongside GPU jobs.

    python verify_independent_solver.py [--wl 700 1200] [--per-structure 2]

Agreement threshold: |dA| < 0.03 absolute absorptance. The residual is
dominated by Fourier truncation -- the two codes retain different harmonic
sets -- so exact agreement is not expected.
"""
import argparse, sys, time
import numpy as np

sys.path.insert(0, '.')
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = 'jc'
from src.simulation.materials import (get_metal_permittivity, get_sio2_permittivity,
                                      get_tio2_permittivity)
import grcwa

NX = NY = 128
GLASS_EPS = 2.25
THETA_MAX = 1.0      # deg; "near-normal"


def _mask_grid(P, fn):
    x = (np.arange(NX) + 0.5) / NX * P
    y = (np.arange(NY) + 0.5) / NY * P
    X, Y = np.meshgrid(x, y, indexing='ij')
    return fn(X, Y)


def grcwa_absorptance(lam, P, nG, layers, grids):
    """layers: list of ('grid', thickness) or (thickness, eps). grids: list of
    (mask, eps_metal) for each 'grid' layer, in order."""
    o = grcwa.obj(nG, [P, 0.0], [0.0, P], 1.0 / lam, 0.0, 0.0, verbose=0)
    o.Add_LayerUniform(1.0, 1.0)                       # air
    for L in layers:
        if L[0] == 'grid':
            o.Add_LayerGrid(L[1], NX, NY)
        else:
            o.Add_LayerUniform(L[0], L[1])
    o.Add_LayerUniform(1.0, GLASS_EPS)                 # substrate
    o.Init_Setup()
    flat = []
    for mask, eps_m in grids:
        ep = np.ones((NX, NY), complex)
        ep[mask] = eps_m
        flat.append(ep.flatten(order='F'))             # grcwa memory layout
    o.GridLayer_geteps(np.concatenate(flat))
    # polarisation: grcwa 'p' = E_y = TE at normal incidence (convention pinned
    # in the July audit via the y-wire polariser physics check); the archives'
    # single-pol channel is TE (E_y), so excite p, not s.
    o.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    R, T = o.RT_Solve(normalize=1)
    return 1.0 - float(np.real(R)) - float(np.real(T))


# ---- per-structure geometry builders (from archived parameters) -------------

def build_A(p, lam, metal='Cr'):
    P = p['P']
    eCr = complex(get_metal_permittivity(np.array([lam]), metal)[0])
    eSi = float(np.real(get_sio2_permittivity(np.array([lam]))[0]))
    eTi = complex(get_tio2_permittivity(np.array([lam]))[0])
    layers = [('grid', p['t1']), (p['d1'], eSi), (p['t_mid'], eCr),
              (p['d2'], eTi), ('grid', p['t2']), (100.0, eCr)]
    top = _mask_grid(P, lambda X, Y: (np.abs(X - P/2) <= p['Wx']/2) &
                                     (np.abs(Y - P/2) <= p['Wy']/2))
    bot = _mask_grid(P, lambda X, Y: (np.abs(X - P/2) <= p['W2']/2) &
                                     (np.abs(Y - P/2) <= p['W2']/2))
    return P, layers, [(top, eCr), (bot, eCr)]


def build_B(p, lam, metal='Cr'):
    P = p['P']
    eCr = complex(get_metal_permittivity(np.array([lam]), metal)[0])
    eSi = float(np.real(get_sio2_permittivity(np.array([lam]))[0]))
    layers = [('grid', p['t_Cr']), (p['d_SiO2'], eSi), (100.0, eCr)]
    def m(X, Y):
        r = np.sqrt((X - P/2)**2 + (Y - P/2)**2)
        return ((r >= p['R_in']) & (r <= p['R_out'])) | (r <= p['R_disk'])
    return P, layers, [(_mask_grid(P, m), eCr)]


def build_C(p, lam, metal='Cr'):
    P = p['P']
    eCr = complex(get_metal_permittivity(np.array([lam]), metal)[0])
    eSi = float(np.real(get_sio2_permittivity(np.array([lam]))[0]))
    layers = [('grid', p['t_Cr']), (p['d_SiO2'], eSi), (100.0, eCr)]
    m = lambda X, Y: (np.abs(X - P/2) <= p['Wx']/2) & (np.abs(Y - P/2) <= p['Wy']/2)
    return P, layers, [(_mask_grid(P, m), eCr)]


STRUCTS = {
    'A':   ('data/raw/struct_A_500_redesign.npz', build_A, 'A',    'a'),
    'B':   ('data/raw/struct_B_500_redesign.npz', build_B, 'A',    'b'),
    'C':   ('data/raw/struct_C_500_redesign.npz', build_C, 'A_TE', 'c'),
    'AuA': ('data/raw/struct_A_Au_500_jc.npz',    build_A, 'A',    'a'),
    'AuB': ('data/raw/struct_B_Au_500_jc.npz',    build_B, 'A',    'b'),
}


def archive_order(struct, geom, lam):
    """Fourier order the shipped archive used for this design/wavelength."""
    import importlib, inspect
    mod = importlib.import_module(f'src.simulation.rcwa_struct_{struct}')
    ao = mod.adaptive_order
    names = list(inspect.signature(ao).parameters)[1:]
    return int(ao(lam, *[geom[n] for n in names])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wl', type=float, nargs='*', default=[700.0, 1200.0])
    ap.add_argument('--per-structure', type=int, default=2)
    ap.add_argument('--tol', type=float, default=0.03)
    ap.add_argument('--structs', nargs='*', default=['A', 'B', 'C'])
    ap.add_argument('--max-order', type=int, default=9,
                    help='only compare designs whose archive Fourier order is <= this, '
                         'so grcwa can be run at a comparable harmonic count')
    args = ap.parse_args()

    worst_overall, rows, ok_all = 0.0, [], True
    for s in args.structs:
        src, build, akey, modkey = STRUCTS[s]
        d = np.load(src, allow_pickle=True)
        names = list(d['param_names'])
        metal = str(d['metal']) if 'metal' in d.files else 'Cr'
        params, wl_grid = d['params'], np.asarray(d['wavelengths'], float)
        th = params[:, names.index('theta')].astype(float)
        finite = np.isfinite(d[akey]).all(axis=1)
        pool = np.where((th < THETA_MAX) & finite)[0]
        # keep only designs the archive itself resolved at a modest order; a
        # higher-order archive value cannot be checked against a grcwa run we
        # can afford, and comparing across truncation levels is meaningless.
        # A design/wavelength pair is testable only where the archive itself used
        # an order we can afford to match in grcwa; select pairs, not whole designs.
        pairs, seen = [], set()
        for i in pool:
            g = {names[k]: float(params[i, k]) for k in range(len(names))}
            for lam in args.wl:
                j = int(np.argmin(np.abs(wl_grid - lam)))
                if archive_order(modkey, g, wl_grid[j]) <= args.max_order:
                    pairs.append((i, j)); seen.add(i)
            if len(seen) >= args.per_structure and pairs:
                break
        print(f"\n=== Structure {s}: {len(pairs)} testable (design, wavelength) pairs "
              f"from {len(seen)} designs; archive order <= N={args.max_order} "
              f"({len(pool)} near-normal designs in pool) ===", flush=True)
        print(f"{'idx':>4} {'wl':>6} {'theta':>6} {'N':>3} {'nG':>5} | "
              f"{'archive':>8} {'grcwa':>8} {'|dA|':>7}", flush=True)
        for i, j in pairs:
            p = {names[k]: float(params[i, k]) for k in range(len(names))}
            A_arc = float(d[akey][i, j])
            N = archive_order(modkey, p, wl_grid[j])
            nG = (2 * N + 1) ** 2
            t0 = time.time()
            try:
                P, layers, grids = build(p, wl_grid[j], metal)
                A_g = grcwa_absorptance(wl_grid[j], P, nG, layers, grids)
            except Exception as e:
                print(f"{i:4d} {wl_grid[j]:6.0f} {p['theta']:6.2f} {N:3d} {nG:5d} | "
                      f"ERROR {type(e).__name__}: {str(e)[:40]}", flush=True)
                ok_all = False
                continue
            dA = abs(A_arc - A_g)
            worst_overall = max(worst_overall, dA)
            flag = '' if dA < args.tol else '  <-- OVER TOL'
            print(f"{i:4d} {wl_grid[j]:6.0f} {p['theta']:6.2f} {N:3d} {nG:5d} | "
                  f"{A_arc:8.4f} {A_g:8.4f} {dA:7.4f}{flag}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            rows.append((s, int(i), float(wl_grid[j]), A_arc, A_g, dA))
            ok_all &= dA < args.tol

    print(f"\n==== SUMMARY: archive (torcwa) vs grcwa ====")
    for s in args.structs:
        sr = [r[5] for r in rows if r[0] == s]
        if sr:
            print(f"  {s}: n={len(sr)}  worst |dA| = {max(sr):.4f}  "
                  f"{'PASS' if max(sr) < args.tol else 'CHECK'}")
    print(f"  OVERALL worst |dA| = {worst_overall:.4f}  -> "
          f"{'PASS' if ok_all else 'CHECK'}")
    np.savez('results/verify_independent_solver.npz',
             rows=np.array(rows, dtype=object), tol=args.tol)
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
