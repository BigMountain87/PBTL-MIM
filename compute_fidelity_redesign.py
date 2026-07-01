"""J1 + J2: corrected TMM-RCWA fidelity on REGENERATED data, Mac-analytic (no RCWA).
J1 -> results/tmm_rcwa_fidelity_redesign.npz  (A/B/C full-band median r & MAE; replaces
      the STALE results/tmm_rcwa_accuracy_fixed.npz as the Table-5 source).
J2 -> results/subband_fidelity_redesign.npz   (per sub-band r & MAE for B and C; for S14).
Uses the project's own analytic TMM (jc materials, npz wavelength grid, reliable mask) —
the exact path validated by the M8 correctness gate (A 0.83/7.94, B 0.96/8.93, C 0.65/16.94)."""
import sys, os
ROOT = '.'
sys.path.insert(0, ROOT)
import numpy as np
import src.simulation.materials as _mat
_mat.MATERIAL_MODEL = "jc"
DR = os.path.join(ROOT, 'data', 'raw')
RES = os.path.join(ROOT, 'results')

BANDS = {'full': (400, 1800), 'vis': (400, 780), 'nir1': (780, 1200),
         'nir2': (1200, 1800), 'nir_wide': (1000, 1800)}


def per_sample_r_mae(tmm, rcwa, mask):
    """median per-sample Pearson r and MAE over masked+finite wavelengths."""
    cs, ms = [], []
    for i in range(len(rcwa)):
        m = mask[i] & np.isfinite(tmm[i]) & np.isfinite(rcwa[i])
        if m.sum() < 2:
            continue
        t, r = tmm[i][m], rcwa[i][m]
        ms.append(np.mean(np.abs(t - r)))
        if t.std() > 0 and r.std() > 0:
            c = np.corrcoef(t, r)[0, 1]
            if not np.isnan(c):
                cs.append(c)
    return np.array(cs), np.array(ms)


def band_mask(w, lo, hi):
    return (w >= lo) & (w <= hi)


# ===== A =====
da = np.load(os.path.join(DR, 'struct_A_500_redesign.npz'), allow_pickle=True)
pa, ra, wa = da['params'], da['A'], da['wavelengths'].astype(np.float64)
rela = da['reliable'] if 'reliable' in da.files else np.ones_like(ra, bool)
from src.simulation.tmm_struct_a import compute_tmm_batch as tmm_a
_ta = tmm_a(pa, wa)
ta = _ta['A_tmm'] if 'A_tmm' in _ta else _ta[[k for k in _ta if 'A' in k][0]]
ca, ma = per_sample_r_mae(ta, ra, rela)

# ===== B =====
db = np.load(os.path.join(DR, 'struct_B_500_redesign.npz'), allow_pickle=True)
pb, rb, wb = db['params'], db['A'], db['wavelengths'].astype(np.float64)
relb = db['reliable'] if 'reliable' in db.files else np.ones_like(rb, bool)
from src.simulation.tmm_struct_b import compute_tmm_batch as tmm_b
tb = tmm_b(pb, wb)['A_tmm']
cb, mb = per_sample_r_mae(tb, rb, relb)

# ===== C (aniso, TE+TM concat) =====
dc = np.load(os.path.join(DR, 'struct_C_500_redesign.npz'), allow_pickle=True)
pc, wc = dc['params'], dc['wavelengths'].astype(np.float64)
rte, rtm = dc['A_TE'], dc['A_TM']
relte = dc['reliable_TE'] if 'reliable_TE' in dc.files else np.ones_like(rte, bool)
reltm = dc['reliable_TM'] if 'reliable_TM' in dc.files else np.ones_like(rtm, bool)
try:
    from src.simulation.tmm_struct_c_aniso import compute_tmm_batch as tmm_c
    _tc = tmm_c(pc, wc)
    tte = _tc.get('A_tmm_te', _tc.get('A_TE_tmm', _tc.get('A_tmm_TE')))
    ttm = _tc.get('A_tmm_tm', _tc.get('A_TM_tmm', _tc.get('A_tmm_TM')))
    assert tte is not None and ttm is not None
    cmode = 'aniso'
except Exception as e:
    from src.simulation.tmm_struct_c import compute_tmm_batch as tmm_ci
    _tc = tmm_ci(pc, wc); tte = ttm = _tc['A_tmm']; cmode = f'iso ({e})'


def c_concat_r_mae(band_lo, band_hi):
    bm = band_mask(wc, band_lo, band_hi)
    cs, ms = [], []
    for i in range(len(pc)):
        m = np.concatenate([relte[i] & bm, reltm[i] & bm])
        t = np.concatenate([tte[i], ttm[i]]); r = np.concatenate([rte[i], rtm[i]])
        mm = m & np.isfinite(t) & np.isfinite(r)
        if mm.sum() < 2:
            continue
        ms.append(np.mean(np.abs(t[mm] - r[mm])))
        if t[mm].std() > 0 and r[mm].std() > 0:
            cc_ = np.corrcoef(t[mm], r[mm])[0, 1]
            if not np.isnan(cc_):
                cs.append(cc_)
    return np.array(cs), np.array(ms)


cc, mc = c_concat_r_mae(400, 1800)

# ===== J1 output =====
os.makedirs(RES, exist_ok=True)
np.savez(os.path.join(RES, 'tmm_rcwa_fidelity_redesign.npz'),
         ca=ca, ma=ma, cb=cb, mb=mb, cc=cc, mc=mc)

print("=== J1: Table-5 fidelity (corrected, redesign data) ===")
for name, c, m in [('A', ca, ma), ('B', cb, mb), ('C', cc, mc)]:
    print(f"  {name}: median r = {np.median(c):+.4f}   median MAE = {np.median(m)*100:.2f}%   (n={len(c)})")
print("  targets: A 0.83/7.94  B 0.96/8.93  C 0.65/16.94\n")

# ===== J2: sub-band (B and C) =====
print("=== J2: sub-band fidelity (S14) ===")
sub = {}
for bn, (lo, hi) in BANDS.items():
    bmB = band_mask(wb, lo, hi)
    cB, mB = per_sample_r_mae(tb, rb, relb & bmB[None, :])
    cC, mC = c_concat_r_mae(lo, hi)
    sub[f'B_{bn}_r'] = np.median(cB); sub[f'B_{bn}_mae'] = np.median(mB)
    sub[f'C_{bn}_r'] = np.median(cC); sub[f'C_{bn}_mae'] = np.median(mC)
    print(f"  {bn:9s} ({lo}-{hi}nm):  B r={np.median(cB):+.3f}/MAE={np.median(mB)*100:.1f}%   "
          f"C r={np.median(cC):+.3f}/MAE={np.median(mC)*100:.1f}%")
np.savez(os.path.join(RES, 'subband_fidelity_redesign.npz'), **sub)
print("\nsaved results/tmm_rcwa_fidelity_redesign.npz + results/subband_fidelity_redesign.npz")
