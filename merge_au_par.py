#!/usr/bin/env python3
"""Merge ALL tagged gen_au_par streams into the standard Au jc npz files.

Each gen_au_par run writes struct_{A,B}_Au_350_jc_<tag>.npz with its sample
range filled (the rest NaN). This merges every tagged file found -- any number
of tags from Vast (v0..vN) and/or home (s0/s1/h0..) -- by taking each row from
whichever stream filled it. Overlapping rows are harmless because the physics is
deterministic (same geometry, jc materials, adaptive RCWA). Verifies all 350
rows are finite before writing the standard struct_{A,B}_Au_350_jc.npz consumed
by the trainers; exits nonzero if any row is still NaN.
"""
import numpy as np, glob, sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = f"{ROOT}/data/raw"


def merge_struct(s):
    final = f"{RAW}/struct_{s}_Au_350_jc.npz"
    files = sorted(glob.glob(f"{RAW}/struct_{s}_Au_350_jc_*.npz"))
    if not files:
        print(f"{s}: no tagged files found ({RAW}/struct_{s}_Au_350_jc_*.npz)")
        return False
    out = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if out is None:
            out = {k: np.array(d[k]).copy() for k in d.files}
            n = np.array(d["A"]).shape[0]
            for k in ("A", "R", "T"):
                out[k] = np.full(np.array(d[k]).shape, np.nan)
            out["reliable"] = np.zeros(np.array(d["reliable"]).shape, bool)
        fin = np.isfinite(np.array(d["A"])).all(axis=1)
        for k in ("A", "R", "T", "reliable"):
            out[k][fin] = np.array(d[k])[fin]
    out["done_count"] = n
    fin = (np.isfinite(out["A"]).all(axis=1)
           & np.isfinite(out["R"]).all(axis=1)
           & np.isfinite(out["T"]).all(axis=1))
    ok = int(fin.sum())
    bad = np.where(~fin)[0]
    np.savez(final, **out)
    tags = [os.path.basename(f).split("_jc_")[1][:-4] for f in files]
    print(f"{s} merged from {len(files)} tags {tags} -> finite(A,R,T) {ok}/{n}"
          + (f"  BAD rows {bad.tolist()}" if len(bad) else "  (all good)"))
    return len(bad) == 0


if __name__ == "__main__":
    a_ok = merge_struct("A")
    b_ok = merge_struct("B")
    print("MERGE_OK" if (a_ok and b_ok) else "MERGE_INCOMPLETE (some samples NaN)")
    sys.exit(0 if (a_ok and b_ok) else 1)
