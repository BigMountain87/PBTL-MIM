#!/usr/bin/env python3
"""PARALLEL/REBALANCED variant of gen_au_redesign.py.

Identical physics/solver/material settings as gen_au_redesign.py, but adds a
SAMPLE-RANGE (--start/--end) + an output --tag so the total Au work can be split
across two balanced GPU-concurrent streams and merged afterwards. The released,
documented serial path remains gen_au_redesign.py; this only parallelizes the
same simulate_single computation. Outputs go to tagged files, e.g.
  data/raw/struct_A_Au_350_jc_s0.npz   (A[start:end] filled, full-350 arrays)
then merge_au_par.py recombines into the standard struct_A/B_Au_350_jc.npz.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"

import torch

# Optional: force the non-Hermitian complex eig onto CPU LAPACK instead of the
# CUDA/MAGMA path. On a single shared GPU, concurrent MAGMA eig calls serialize;
# CPU LAPACK lets concurrent shards run their eigs in parallel across cores.
# Physical outputs (A/R/T) are S-matrix observables and must match the GPU path
# (eigvec phase/order differences cancel) -- validated before any long run.
if os.environ.get("FORCE_CPU_EIG") == "1":
    _orig_eig = torch.linalg.eig

    def _cpu_eig(A, *args, **kwargs):
        if not isinstance(A, torch.Tensor) or A.device.type != "cuda":
            return _orig_eig(A, *args, **kwargs)
        # Large (high adaptive-order) matrices hang in CPU LAPACK on this
        # AMD/torch2.12 setup -> keep those on the GPU/MAGMA path; small ones
        # (the common low-order wavelengths) go to fast CPU LAPACK.
        if A.shape[-1] > 1500:
            return _orig_eig(A, *args, **kwargs)
        r = _orig_eig(A.cpu(), *args, **kwargs)
        # torch structseq (return_types.linalg_eig) must be built from a SINGLE
        # sequence, not two positional args -- type(r)(w, v) raises TypeError.
        return type(r)((r.eigenvalues.to(A.device), r.eigenvectors.to(A.device)))

    torch.linalg.eig = _cpu_eig
    print("[gen_au_par] FORCE_CPU_EIG: torch.linalg.eig patched to CPU LAPACK", flush=True)

TOL = 0.005
REDESIGN_VERSION = "v1-jc-metals+tio2siefke+adaptive+grazingfix+c64"
OOM_RETRY = 3

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


def gen_struct(x, device, ckpt_every, start=0, end=None, tag="", limit=None, geomtag="350"):
    modname, src_npz, out_npz = SRC[x]
    if geomtag != "350":
        src_npz = f'{ROOT}/data/raw/struct_{x}_Au_{geomtag}.npz'
        out_npz = f'{ROOT}/data/raw/struct_{x}_Au_{geomtag}_jc.npz'
    if tag:
        out_npz = out_npz[:-4] + "_" + tag + ".npz"
    mod = __import__(modname, fromlist=['simulate_single'])
    mod.RCWA_SETTINGS['adaptive_order'] = True
    mod.RCWA_SETTINGS['dtype'] = torch.complex64

    d = np.load(src_npz, allow_pickle=True)
    params = d['params'].astype(np.float64)
    names = list(mod.PARAM_NAMES)
    assert params.shape[1] == len(names), \
        f"param count {params.shape[1]} != PARAM_NAMES {len(names)} for {x}"
    wl = np.asarray(d['wavelengths'], dtype=np.float64)
    metal = str(d['metal']) if 'metal' in d.files else 'Au'
    assert metal == 'Au', f"expected Au, got {metal}"
    n = params.shape[0]
    nwl = len(wl)
    end_idx = n if end is None else min(end, n)
    if limit:
        end_idx = min(end_idx, start + limit)

    resume_ok = False
    if os.path.exists(out_npz):
        try:
            ck = np.load(out_npz, allow_pickle=True)
            ident = ('redesign_version' in ck.files
                     and str(ck['redesign_version']) == REDESIGN_VERSION
                     and len(ck['wavelengths']) == nwl and np.allclose(ck['wavelengths'], wl)
                     and ck['params'].shape == params.shape and np.allclose(ck['params'], params)
                     and str(ck['metal']) == metal and ck['A'].shape == (n, nwl))
            if ident:
                store = {k: ck[k].copy() for k in ck.files}
                done = max(int(ck['done_count']), start); resume_ok = True
                print(f"[{x}:{tag}] resuming: at {done} (range {start}:{end_idx})", flush=True)
            else:
                print(f"[{x}:{tag}] identity mismatch -> fresh", flush=True)
        except Exception as e:
            print(f"[{x}:{tag}] corrupt checkpoint ({type(e).__name__}); fresh", flush=True)
    if not resume_ok:
        store = {'params': params, 'param_names': np.array(names), 'wavelengths': wl,
                 'metal': metal, 'structure': x, 'redesign_version': REDESIGN_VERSION,
                 'done_count': start,
                 'A': np.full((n, nwl), np.nan), 'R': np.full((n, nwl), np.nan),
                 'T': np.full((n, nwl), np.nan), 'reliable': np.zeros((n, nwl), bool)}
        done = start
        print(f"[{x}:{tag}] fresh: range {start}:{end_idx} of {n} x {nwl}wl "
              f"({wl[0]:.0f}-{wl[-1]:.0f}nm) metal={metal}", flush=True)

    t_start = time.time(); times = []
    for i in range(start, end_idx):
        if (np.isfinite(store['A'][i]).all()
                and np.isfinite(store['R'][i]).all()
                and np.isfinite(store['T'][i]).all()):
            continue  # already computed (resume or pre-merged Vast results) -> skip
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
                print(f"[{x}:{tag}] sample {i} {'OOM' if is_oom else 'ERROR'} "
                      f"{type(e).__name__}: {str(e)[:60]} -> NaN/unreliable", flush=True)
                break
        dt = time.time() - t0
        if ok:
            times.append(dt)
        store['done_count'] = i + 1
        if (i + 1) % ckpt_every == 0 or (i + 1) == end_idx:
            tmp = out_npz[:-4] + ".tmp.npz"
            np.savez(tmp, **store); os.replace(tmp, out_npz)
            rate = (time.time() - t_start) / max(1, (i + 1 - done))
            print(f"[{x}:{tag}] {i+1}/{end_idx}  last={dt:.1f}s  avg={rate:.1f}s/sample  "
                  f"ETA={rate*(end_idx-i-1)/60:.1f}min", flush=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    A_all = store['A'][start:end_idx]
    rel_all = store['reliable'][start:end_idx].all(axis=1)
    persample = float(np.mean(times)) if times else float('nan')
    print(f"[{x}:{tag}] DONE -> {out_npz}  mean={persample:.2f}s over {len(times)} solves  "
          f"range[{np.nanmin(A_all):.3f},{np.nanmax(A_all):.3f}] reliable {int(rel_all.sum())}/{end_idx-start}",
          flush=True)
    return persample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structs", nargs="+", default=["A", "B"])
    ap.add_argument("--ckpt", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--geomtag", default="350")
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen_au_par] MATERIAL_MODEL={_mat.MATERIAL_MODEL} structs={args.structs} "
          f"start={args.start} end={args.end} tag={args.tag} device={device}", flush=True)
    for x in args.structs:
        gen_struct(x, device, args.ckpt, start=args.start, end=args.end,
                   tag=args.tag, limit=args.limit, geomtag=args.geomtag)
    print("[gen_au_par] worker finished", flush=True)


if __name__ == "__main__":
    main()
