#!/usr/bin/env python3
"""Re-simulate the Au/SiO2 cross-material RCWA datasets on the CORRECTED Johnson&Christy
Au optical constants with the OPTIMIZED redesign solver settings.

WHY: data/raw/struct_A_Au_350.npz and struct_B_Au_350.npz predate the materials
correction -- they use the mistabulated hand-entered Au k-values (the same error that was
fixed for Cr/Ti/Au in src/simulation/materials.py). The Cr Structures A/B/C were
regenerated on the measured J&C ('jc') model; the Au check was not. This script corrects
that, changing ONLY the material model + the solver optimizations:
  * MATERIAL_MODEL='jc' set BEFORE any tmm/RCWA/material import (measured J&C Au from
    data/ref/Au_JC.txt).
  * adaptive per-wavelength Fourier order (base N=5 raised up to N=17), complex64, and the
    grazing-incidence compute_RT fix -- exactly as generate_redesign.py
    (REDESIGN_VERSION 'v1-jc-metals+tio2siefke+adaptive+grazingfix+c64').

PRESERVED (so the Au check stays directly comparable to the prior Au numbers, modulo the
materials/solver fix):
  * the EXACT original geometries (params loaded from the existing Au npz, frozen),
  * the EXACT original wavelength band (loaded from the npz: 380-780 nm, 100 pts),
  * structure A = struct_A_Au_350 (10 params), structure B = struct_B_Au_350 (8 params).

Emits a per-point RELIABILITY mask (reliable = -TOL <= A <= 1+TOL, TOL=0.005), like the
redesign pipeline. Outputs (NEW files; originals untouched):
  data/raw/struct_A_Au_350_jc.npz
  data/raw/struct_B_Au_350_jc.npz
keys: params, param_names, wavelengths, metal, A, R, T, reliable, redesign_version.

Usage:
  python gen_au_redesign.py --structs A B [--limit N] [--ckpt 10]
Atomic checkpoint + identity-checked resume, per-sample try/except, OOM retry -- so it can
run unattended on the shared GPU.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, time, argparse
import numpy as np

ROOT = '.'
sys.path.insert(0, ROOT)

# MATERIAL_MODEL must be set to 'jc' BEFORE importing any RCWA/material module.
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

import torch

TOL = 0.005
REDESIGN_VERSION = "v1-jc-metals+tio2siefke+adaptive+grazingfix+c64"
OOM_RETRY = 3

# original Au datasets -> (rcwa module, source npz, output npz)
SRC = {
    'A': ('src.simulation.rcwa_struct_a',
          f'{ROOT}/data/raw/struct_A_Au_350.npz',
          f'{ROOT}/data/raw/struct_A_Au_350_jc.npz'),
    'B': ('src.simulation.rcwa_struct_b',
          f'{ROOT}/data/raw/struct_B_Au_350.npz',
          f'{ROOT}/data/raw/struct_B_Au_350_jc.npz'),
}


def reliable(A):
    return (A >= -TOL) & (A <= 1.0 + TOL)


def gen_struct(x, device, ckpt_every, limit=None):
    modname, src_npz, out_npz = SRC[x]
    mod = __import__(modname, fromlist=['simulate_single'])
    # OPTIMIZED redesign solver settings (identical to generate_redesign.py).
    mod.RCWA_SETTINGS['adaptive_order'] = True
    mod.RCWA_SETTINGS['dtype'] = torch.complex64

    d = np.load(src_npz, allow_pickle=True)
    params = d['params'].astype(np.float64)         # frozen original geometries
    names = list(mod.PARAM_NAMES)                    # ordering used at original generation
    assert params.shape[1] == len(names), \
        f"param count {params.shape[1]} != PARAM_NAMES {len(names)} for {x}"
    wl = np.asarray(d['wavelengths'], dtype=np.float64)   # frozen original band (380-780/100)
    metal = str(d['metal']) if 'metal' in d.files else 'Au'
    assert metal == 'Au', f"expected Au, got {metal}"
    n = params.shape[0]
    nwl = len(wl)
    loop_end = min(n, limit) if limit else n

    # resume only if identity (version/params/wl/metal/shape) all match
    resume_ok = False
    if os.path.exists(out_npz):
        try:
            ck = np.load(out_npz, allow_pickle=True)
            ident = ('redesign_version' in ck.files
                     and str(ck['redesign_version']) == REDESIGN_VERSION
                     and len(ck['wavelengths']) == nwl and np.allclose(ck['wavelengths'], wl)
                     and ck['params'].shape == params.shape and np.allclose(ck['params'], params)
                     and str(ck['metal']) == metal and ck['A'].shape == (n, nwl)
                     and 0 <= int(ck['done_count']) <= n)
            if ident:
                store = {k: ck[k].copy() for k in ck.files}
                done = int(ck['done_count']); resume_ok = True
                print(f"[{x}] resuming: {done}/{n} done", flush=True)
            else:
                print(f"[{x}] identity mismatch -> fresh", flush=True)
        except Exception as e:
            print(f"[{x}] corrupt checkpoint ({type(e).__name__}); fresh", flush=True)
    if not resume_ok:
        store = {'params': params, 'param_names': np.array(names), 'wavelengths': wl,
                 'metal': metal, 'structure': x, 'redesign_version': REDESIGN_VERSION,
                 'done_count': 0,
                 'A': np.full((n, nwl), np.nan), 'R': np.full((n, nwl), np.nan),
                 'T': np.full((n, nwl), np.nan), 'reliable': np.zeros((n, nwl), bool)}
        done = 0
        print(f"[{x}] fresh: {n} samples x {nwl} wl ({wl[0]:.0f}-{wl[-1]:.0f}nm), metal={metal}", flush=True)

    t_start = time.time(); times = []
    for i in range(done, loop_end):
        p = {names[c]: float(params[i, c]) for c in range(len(names))}
        t0 = time.time(); ok = False
        for attempt in range(OOM_RETRY + 1):
            try:
                A, R, T = mod.simulate_single(p, wl, metal=metal, device=device)
                store['A'][i] = A; store['R'][i] = R; store['T'][i] = T
                store['reliable'][i] = reliable(A)
                ok = True; break
            except Exception as e:
                is_oom = 'out of memory' in str(e).lower()
                if is_oom and attempt < OOM_RETRY and device.type == 'cuda':
                    torch.cuda.empty_cache(); time.sleep(5 * (attempt + 1)); continue
                print(f"[{x}] sample {i} {'OOM' if is_oom else 'ERROR'} "
                      f"{type(e).__name__}: {str(e)[:60]} -> NaN/unreliable", flush=True)
                break
        dt = time.time() - t0
        if ok:
            times.append(dt)
        store['done_count'] = i + 1
        if (i + 1) % ckpt_every == 0 or (i + 1) == loop_end:
            tmp = out_npz[:-4] + ".tmp.npz"
            np.savez(tmp, **store); os.replace(tmp, out_npz)
            rate = (time.time() - t_start) / (i + 1 - done)
            print(f"[{x}] {i+1}/{loop_end}  last={dt:.1f}s  avg={rate:.1f}s/sample  "
                  f"ETA={rate*(loop_end-i-1)/60:.1f}min", flush=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    A_all = store['A'][:loop_end]
    rel_all = store['reliable'][:loop_end].all(axis=1)
    persample = float(np.mean(times)) if times else float('nan')
    print(f"[{x}] DONE -> {out_npz}", flush=True)
    print(f"[{x}] PER-SAMPLE RCWA time (jc, adaptive, c64): mean={persample:.2f}s "
          f"over {len(times)} solves", flush=True)
    print(f"[{x}] A range [{np.nanmin(A_all):.3f}, {np.nanmax(A_all):.3f}] | "
          f"reliable(all-wl) {int(rel_all.sum())}/{loop_end}", flush=True)
    return persample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structs", nargs="+", default=["A", "B"])
    ap.add_argument("--ckpt", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="cap samples (smoke test)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen_au] MATERIAL_MODEL={_mat.MATERIAL_MODEL} structs={args.structs} "
          f"device={device} version={REDESIGN_VERSION}", flush=True)
    for x in args.structs:
        gen_struct(x, device, args.ckpt, limit=args.limit)
    print("[gen_au] worker finished", flush=True)


if __name__ == "__main__":
    main()
