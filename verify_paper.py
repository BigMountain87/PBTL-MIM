#!/usr/bin/env python3
"""
verify_paper.py - mechanical pre-submission gate for the PBTL MIM-absorber paper.

Catches the leak classes that slipped through 11+ LLM review rounds:
  C1 numbers <-> data drift     C5 stale supplementary section numbers
  C2 cross-document mismatch    C6 forbidden correction-minimizing phrases
  C3 figure caption <-> output  C7 retired (pre-correction) numbers as current
  C4 compile / undefined refs   C8 data provenance + bibliography count

The philosophy (from the review post-mortem): RECOMPUTE and GREP, do not "read and
judge". Run from the repo root:  python verify_paper.py
Exits 0 only if every check passes; nonzero otherwise.

Edit the REGISTRY / EXPECTED_SUPP_SECTIONS / FORBIDDEN / RETIRED blocks below when the
paper legitimately changes -- they are the single source of truth.
"""
import os, re, sys, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PDFLATEX = "/Library/TeX/texbin/pdflatex"
TOL = 0.05  # pp tolerance for recomputed-vs-printed numbers
DOCS = ["paper.tex", "supplementary.tex", "cover_letter.tex",
        "reviewer_response_final.md", "reviewer_response_R2_final.md",
        "editor_disclosure_letter.md", "highlights.txt", "highlights.tex", "README.md"]

_results = []
def record(cid, ok, msg):
    _results.append((cid, ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {msg}")

def read(f):
    return open(f, encoding="utf-8").read() if os.path.exists(f) else ""

# ============================================================ registry
# Numbers that must be CONSISTENT across documents, and the pre-correction
# values that must NOT reappear as current claims.
RETIRED = {                      # value-regex : why it is retired
    r"0\.981":          "pre-correction Fisher-CI r",
    r"\b416\b":         "old Au-B reliable count (now 419)",
    r"29\.5\s*\\?%":    "old Au-B benefit (now +28.0)",
    r"\$?308\$?[-~\s]*(?:valid|sample)": "old Au valid-sample pool",
    r"\br\s*=\s*0\.46\b": "old Au fidelity r (now 0.77)",
    r"two distinct (?:negative-transfer )?failure modes": "removed taxonomy",
}
# A retired hit is tolerated only on a line that explicitly frames it as historical:
DISCLOSURE_MARKERS = ("pre-correction", "superseded", "removed", "earlier",
                      "historical", "submitted", "previously", "no longer", "retired",
                      "old ", "before the correction", "change-log", "changelog")

FORBIDDEN = [                    # correction-minimizing / over-claim phrases
    r"unaffected by the correction",
    r"not affected by the forward-model",
    r"the comparison is unaffected",
    r"methodologically unaffected",
    r"qualitatively (?:unchanged|intact)",
    r"carr(?:ies|ying) the predictive weight",
    r"reinforc\w*\b[^.\n]{0,40}\b(?:correction|conclusion|finding|original|taxonomy|unchanged|intact)",
]

# Cross-document headline values: (label, regex, list-of-docs-that-should-agree)
CROSSDOC = [
    ("Au-B reliable count 419", r"\b419\b", ["paper.tex", "supplementary.tex",
        "reviewer_response_final.md", "editor_disclosure_letter.md"]),
]

# Supplementary section order (S1..SN). Update only when sections are added/removed.
EXPECTED_SUPP_SECTIONS = [
    "Random Pre-training Baseline", "TMM Pre-training Data Size", "Learning Rate Ablation",
    "Architecture Ablation", "Permutation Feature Importance", "TMM-as-Input Baseline",
    "Multi-fidelity Baseline", "Architecture Sensitivity", "Material Generalization",
    "Per-Seed Results", "Anisotropic TMM", "Transmittance", "Inverse Design",
    "Sub-band", "Peng", "Convergence", "Geometry-Constraint", "Additional Figures",
    "Correction Change-Log",
]

# ============================================================ C1 numbers <-> data
def parse_table(tex, label):
    """Return list of rows; each row = list of float means parsed from a tabular block."""
    m = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        if "&" not in line or "\\midrule" in line or "toprule" in line:
            continue
        cells = line.split("&")
        vals = []
        for c in cells:
            cc = c.replace(r"\mathbf", "")
            mm = re.search(r"(-?\d+\.\d+)", cc)
            if mm:
                vals.append(float(mm.group(1)))           # decimal cell (mean +/- std)
            else:
                mi = re.search(r"(-?\d+)", cc)
                if mi:
                    vals.append(float(mi.group(1)))       # bare integer (leading n column)
        if vals:
            rows.append(vals)
    return rows

def c1_numbers_vs_data():
    tex = read("paper.tex")
    ok_all = True
    # --- 4-way MAE tables ---
    try:
        a = np.load("results/pbtl_A_redesign_10seed.npz", allow_pickle=True)
        want = {n: [a[f"{n}_{m}"].mean()*100 for m in ["M0","M_phys","M_TL","M_TL+phys"]]
                for n in [50,100,200,350]}
        rows = parse_table(tex, "tab:pbtl_a")
        if rows is None:
            ok_all = False; record("C1", False, "tab:pbtl_a not found in paper.tex")
        # rows: each [n, M0, Mphys, MTL, MTLphys]
        for r in rows or []:
            n = int(r[0]); got = r[1:5]; exp = want.get(n)
            if exp and any(abs(g-e) > TOL for g, e in zip(got, exp)):
                ok_all = False
                record("C1", False, f"Table pbtl_a n={n}: printed {got} != data {[round(x,2) for x in exp]}")
    except Exception as e:
        ok_all = False; record("C1", False, f"Structure-A table check error: {e}")
    for lab, fn in [("tab:pbtl_b", "results/pbtl_B_redesign_10seed.npz"),
                    ("tab:pbtl_c", "results/pbtl_C_v2_redesign_10seed.npz")]:
        try:
            d = np.load(fn, allow_pickle=True); ts = list(d["train_sizes"])
            want = {int(ts[i]): [d[m][i].mean()*100 for m in ["M0","M_phys","M_TL","M_TL_phys"]]
                    for i in range(len(ts))}
            rows_bc = parse_table(tex, lab)
            if rows_bc is None:
                ok_all = False; record("C1", False, f"{lab} not found in paper.tex")
            for r in rows_bc or []:
                n = int(r[0]); got = r[1:5]; exp = want.get(n)
                if exp and any(abs(g-e) > TOL for g, e in zip(got, exp)):
                    ok_all = False
                    record("C1", False, f"Table {lab} n={n}: printed {got} != data {[round(x,2) for x in exp]}")
        except Exception as e:
            ok_all = False; record("C1", False, f"{lab} check error: {e}")
    # --- Table I fidelity ---
    try:
        f = np.load("results/tmm_rcwa_fidelity_redesign.npz", allow_pickle=True)
        med = {s: (np.median(f[c]), np.median(f[m])*100)
               for s, c, m in [("A","ca","ma"), ("B","cb","mb"), ("C","cc","mc")]}
        fb = re.search(r"\\label\{tab:tmm_fidelity\}(.*?)\\end\{tabular\}", tex, re.S)
        fb = fb.group(1) if fb else ""
        if not fb:
            ok_all = False; record("C1b", False, "tab:tmm_fidelity block not found in paper.tex")
        else:
            bad = [s for s, (r_exp, mae_exp) in med.items()
                   if f"{r_exp:.2f}" not in fb or f"{mae_exp:.1f}" not in fb]
            if bad:
                ok_all = False
                record("C1b", False, f"fidelity median-r/MAE not in Table I for {bad} "
                       f"(recomputed A={med['A'][0]:.2f}/{med['A'][1]:.1f} B={med['B'][0]:.2f}/{med['B'][1]:.1f} "
                       f"C={med['C'][0]:.2f}/{med['C'][1]:.1f})")
            else:
                record("C1b", True, f"fidelity median-r+MAE match Table I (A={med['A'][0]:.2f}/{med['A'][1]:.1f} "
                       f"B={med['B'][0]:.2f}/{med['B'][1]:.1f} C={med['C'][0]:.2f}/{med['C'][1]:.1f})")
    except Exception as e:
        ok_all = False; record("C1b", False, f"fidelity check error: {e}")
    if ok_all:
        record("C1", True, "all 4-way table cells reproduce from redesign npz within 0.05pp")

# ============================================================ C2 cross-document
def c2_crossdoc():
    ok = True
    for label, rgx, docs in CROSSDOC:
        present = [d for d in docs if re.search(rgx, read(d))]
        if len(present) < len(docs):
            missing = set(docs) - set(present)
            ok = False
            record("C2", False, f"{label}: not found in {sorted(missing)}")
    if ok:
        record("C2", True, "cross-document headline values present in all expected docs")

# ============================================================ C3 figure caption <-> output
def c3_figures():
    tex_all = read("paper.tex") + read("supplementary.tex")
    checks = [
        ("make_figure4.py", [(r"0\.81", "0.81"), (r"0\.98", "0.98")]),
        ("make_figure5.py", [(r"cavity\+fill = 60\.4", "60.4")]),
        ("make_figure2.py", [(r"0\.99", "high r")]),
        ("make_figure3.py", [(r"saved Figure_3", "Figure 3 learning curves")]),
    ]
    ok = True
    for script, expects in checks:
        if not os.path.exists(script):
            ok = False; record("C3", False, f"{script} missing"); continue
        try:
            out = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=300)
            if out.returncode != 0:
                ok = False; record("C3", False, f"{script} crashed: {out.stderr.strip()[-120:]}"); continue
            for rgx, lab in expects:
                if not re.search(rgx, out.stdout):
                    ok = False; record("C3", False, f"{script} output missing {lab}")
        except Exception as e:
            ok = False; record("C3", False, f"{script} error: {e}")
    if ok:
        record("C3", True, "figure generators run and print expected values")

# ============================================================ C4 compile + refs
def c4_compile():
    ok = True
    for f in ["paper.tex", "supplementary.tex"]:
        try:
            for _ in range(2):
                r = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", f],
                                   capture_output=True, text=True, timeout=300)
            log = read(f.replace(".tex", ".log"))
            errs = log.count("\n! ")
            undef = len(re.findall(r"(?:undefined (?:reference|citation)|multiply.defined)", log, re.I))
            if r.returncode != 0 or errs or undef:
                ok = False; record("C4", False, f"{f}: rc={r.returncode} errors={errs} undefined/dup={undef}")
        except Exception as e:
            ok = False; record("C4", False, f"{f}: {e}")
    # supplementary must not \ref a main-paper label
    paper_labels = set(re.findall(r"\\label\{([^}]+)\}", read("paper.tex")))
    supp = read("supplementary.tex")
    supp_labels = set(re.findall(r"\\label\{([^}]+)\}", supp))
    leaked = [r for r in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", supp)
              if r in paper_labels and r not in supp_labels]
    if leaked:
        ok = False; record("C4", False, f"supplementary \\ref to main-paper labels: {leaked}")
    if ok:
        record("C4", True, "paper+supplementary compile clean; no undefined/dup refs; no cross-doc \\ref leak")

# ============================================================ C5 section-number map
def c5_section_map():
    supp = read("supplementary.tex")
    secs = re.findall(r"\\section\{(.+?)\}", supp)
    ok = True
    if len(secs) != len(EXPECTED_SUPP_SECTIONS):
        ok = False
        record("C5", False, f"supplementary has {len(secs)} sections, expected {len(EXPECTED_SUPP_SECTIONS)}")
    for i, kw in enumerate(EXPECTED_SUPP_SECTIONS):
        if i < len(secs) and kw.lower() not in secs[i].lower():
            ok = False
            record("C5", False, f"S{i+1} expected ~'{kw}' but is '{secs[i][:40]}'")
    # every hardcoded "Supplementary Section~S#" in paper must be <= section count
    nmax = len(secs)
    for m in re.finditer(r"Supplementary[~ ](?:Section[~ ])?S(\d+)", read("paper.tex")):
        if int(m.group(1)) > nmax:
            ok = False; record("C5", False, f"paper references S{m.group(1)} > {nmax} sections")
    if ok:
        record("C5", True, f"supplementary S1..S{nmax} map matches; paper S-refs in range")

# ============================================================ C6 forbidden phrases
def c6_forbidden():
    import glob
    targets = list(DOCS) + [g for g in glob.glob("*.py") + glob.glob("step0_screen/*.py")
                            if "obsolete" not in g and "backup" not in g and "verify_paper" not in g]
    hits = []
    for d in dict.fromkeys(targets):           # dedup, preserve order
        for ln, line in enumerate(read(d).splitlines(), 1):
            for rgx in FORBIDDEN:
                if re.search(rgx, line, re.I):
                    hits.append(f"{d}:{ln} /{rgx}/")
    if hits:
        record("C6", False, "forbidden phrases: " + "; ".join(hits[:6]))
    else:
        record("C6", True, "no forbidden phrases (docs + README + active *.py)")

# ============================================================ C7 retired numbers
def c7_retired():
    # Retired numbers must not appear as CURRENT claims. Scan EVERY root *.tex (manuscript,
    # cover letter, highlights, ANY rebuttal .tex) -- NOT a hardcoded list, which kept
    # omitting files (reviewer_response_R2.tex leaked 29.5%/0.981 because it was never
    # scanned). A file is skipped only if it carries a SUPERSEDED / DO-NOT-SUBMIT banner in
    # its first 12 lines (knowingly-archived stale file). The response/disclosure .md docs
    # legitimately discuss old values (cross-doc consistency is C2's job). The supplementary
    # Correction Change-Log section is exempt.
    import glob
    # Internal working/disclosure notes whose ROLE is to track the correction: they
    # legitimately hold pre-correction values as HISTORY, not as current claims. Anything
    # NOT here (and not banner-marked) is scanned -- so a new masquerading doc fails loud.
    ALLOW_HISTORY = {
        "REVISION_BLUEPRINT.md", "HANDOFF_STATE.md", "github_repo_plan.md",
        "correction_catalog.md", "REVIEW_PROCESS.md", "reviewer_response_R3_draft.md",
        "PROJECT_STATUS_SUMMARY.md", "COMPLETION_CHECKLIST.md",
        "README_COMPLETION_GUIDE.md", "W3_W4_COMPLETION_WORKFLOW.md",
        "CONVERGENCE_C.md", "paper1_critical_review_20260611.md", "verify_paper.py",
    }
    hits = []
    docs = sorted(set(glob.glob("*.tex") + glob.glob("*.md") + glob.glob("*.py"))) + ["highlights.txt"]
    for d in docs:
        if d in ALLOW_HISTORY:
            continue
        lines = read(d).splitlines()
        head = " ".join(lines[:15]).upper()
        if "SUPERSEDED" in head or "DO NOT SUBMIT" in head:
            continue                                          # knowingly-archived stale file
        clog = next((i for i, l in enumerate(lines)
                     if "Correction Change-Log" in l or "Change-Log" in l), len(lines) + 1)
        for ln, line in enumerate(lines):
            if ln >= clog:                                   # exempt change-log section
                continue
            window = " ".join(lines[max(0, ln-3):ln+4]).lower()   # +/- 3 lines
            if any(mk in window for mk in DISCLOSURE_MARKERS):
                continue
            for rgx, why in RETIRED.items():
                if re.search(rgx, line):
                    hits.append(f"{d}:{ln+1} {why}")
    # binary-leak class: a compiled rebuttal PDF in root carries retired numbers with no
    # banner possible. Authoritative responses are the *_final.md; no root rebuttal PDF.
    stray = glob.glob("reviewer_response*.pdf")
    if stray:
        hits.append(f"stale rebuttal PDF(s) in root (binary retired-number leak): {stray}")
    if hits:
        record("C7", False, "retired numbers as current: " + "; ".join(hits[:6]))
    else:
        record("C7", True, "no retired numbers in root *.tex/highlights/README; no stray rebuttal PDFs")

# ============================================================ C8 provenance + bib
def c8_provenance():
    ok = True
    # data files
    for s, exp in [("A", 499), ("B", 498), ("C", 487)]:
        try:
            d = np.load(f"data/raw/struct_{s}_500_redesign.npz", allow_pickle=True)
            wl = d["wavelengths"]
            if "reliable" in d.files:
                rel = d["reliable"].all(axis=1).sum()
            else:
                rel = (d["reliable_TE"].astype(bool) & d["reliable_TM"].astype(bool)).all(axis=1).sum()
            if not (abs(wl.min()-400) < 1 and abs(wl.max()-1800) < 1 and len(wl) == 100 and rel == exp):
                ok = False
                record("C8", False, f"struct_{s}: grid {wl.min():.0f}-{wl.max():.0f}/{len(wl)}, reliable {rel} (exp {exp})")
        except Exception as e:
            ok = False; record("C8", False, f"struct_{s} data check: {e}")
    # Au-B corrected reliable mask must equal the reported 419
    try:
        au = np.load("data/raw/struct_B_Au_500_jc.npz", allow_pickle=True)
        aub = int(au["reliable"].all(axis=1).sum())
        if aub != 419:
            ok = False; record("C8", False, f"struct_B_Au_500_jc reliable {aub} (expected 419)")
    except Exception as e:
        ok = False; record("C8", False, f"Au-B data check: {e}")
    # figure generators must load redesign/jc archives (no stale npz)
    stale = re.compile(r"\.load\(['\"][^'\"]*?(?<!redesign)(?<!_jc)(?<!persample_A_r2)(?<!gap_sweep)\.npz")
    for g in sorted(__import__("glob").glob("make_figure*.py")):
        for ln in read(g).splitlines():
            for mm in re.finditer(r"\.load\(\s*f?['\"]([^'\"]+\.npz)", ln):
                p = mm.group(1)
                if "{" in p:
                    continue                         # variable-driven f-string path; not statically resolvable
                if "redesign" not in p and "_jc" not in p and "persample_A_r2" not in p and "gap_sweep" not in p:
                    ok = False; record("C8", False, f"{g} loads non-corrected npz: {p}")
    # bibliography count
    paper = read("paper.tex")
    m = re.search(r"\\begin\{thebibliography\}\{(\d+)\}", paper)
    nbib = len(re.findall(r"\\bibitem\{", paper))
    if m and int(m.group(1)) != nbib:
        ok = False; record("C8", False, f"thebibliography{{{m.group(1)}}} != {nbib} bibitems")
    if ok:
        record("C8", True, "data grid/reliable correct; figures load corrected npz; bib count matches")

# ============================================================ main
def main():
    print("verify_paper.py - mechanical pre-submission gate\n" + "=" * 60)
    for fn in [c1_numbers_vs_data, c2_crossdoc, c3_figures, c4_compile,
               c5_section_map, c6_forbidden, c7_retired, c8_provenance]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__, False, f"check crashed: {e}")
    print("=" * 60)
    fails = [c for c, ok, _ in _results if not ok]
    if fails:
        print(f"RESULT: FAIL ({len(fails)} check(s): {sorted(set(fails))})")
        sys.exit(1)
    print(f"RESULT: PASS (all {len(_results)} checks)")
    sys.exit(0)

if __name__ == "__main__":
    main()
