# Post-Acceptance Verification Record

This document summarizes an independent verification pass performed after
manuscript acceptance, covering the reference list, in-text and supplementary
numerical claims, and model checkpoint provenance. It is provided as a
permanent, citable companion to the mechanical gate in
a mechanical gate script, which recomputes every reported table
and figure from the archived result files.

No result, table, figure, or conclusion changed as a result of this pass.
A small number of citation-placement, bibliographic-field, and rounding
corrections were identified and are itemized below; they are also being
submitted as author corrections at the journal proof stage.

## 1. Reference list and citation attribution

- Every work in the manuscript's reference list was checked against the full
  text of the source article (not abstracts or secondary summaries). The
  source-PDF set retrieved for this check comprised 49 articles, covering the
  accepted reference list, the software citation added at proof, and the
  additional tool references carried in this repository's working copy.
- Every citation instance in the body text (53 sites) was checked for
  whether the claim attributed to a reference is actually supported by that
  reference's content, not only whether the bibliographic entry itself is
  correct.
- Findings: 5 bibliographic field errors (page range, volume/year, an author
  surname, a truncated title, an article number), 4 citation-placement
  errors (a claim attributed to a source that does not support it, most
  commonly because the source uses a different simulation method than
  stated), and one missing citation for a software library used to generate
  all full-wave data. All are corrected in the manuscript and are being
  requested as proof corrections on the published version; none required a
  numeric or conclusion change.

## 2. Numerical reproduction

Two passes, independent of the mechanical gate:

**Table values.** Every reported table cell in the main text and in the
19-section supplement was recomputed directly from the
archived `.npz`/`.json` result files in [`results/`](results/) and
[`data/raw/`](data/raw/) — not from the manuscript's own displayed numbers.
One rounding error was found (a convergence-table cell reading `0.761` where
the underlying value `0.76046` rounds to `0.760`) and corrected.

**In-text (prose) numbers.** Approximately 60 numerical claims stated in
running text outside of tables — parameter counts, percentage ranges, timing
and memory figures, grazing-order diffraction fractions, convergence drift
values — were independently recomputed from the same archives and, where
applicable, from closed-form physics (e.g., the grating-equation analysis
behind the reported near-grazing-order exclusion fractions). All ~60 values
reproduced to the stated precision; zero numerical errors were found in this
pass. Three wording-precision issues were identified and corrected: a
sensitivity check described as covering "both low-data regimes" when one of
the two sizes is the maximum training pool; a kernel sweep described as
"four kernel families and two input configurations" when five variants are
enumerated; and a percentage range that needed its reference model named
explicitly.

Both passes were independently cross-checked by a second, separately
prompted review (OpenAI Codex, read-only sandbox, no access to the first
pass's conclusions) that recomputed the same claims from the same archives;
the two passes agreed on every reproduced value.

## 3. Shared upstream material data

The TMM and RCWA solvers share the same optical-constant tables
([`data/ref/`](data/ref/), interpolated in
[`src/simulation/materials.py`](src/simulation/materials.py)), so an error
there would affect both solvers identically and would not be caught by
cross-solver comparison. These tables were therefore checked against their
primary sources directly (2026-08-07):

- **Cr** — all 49 rows match Johnson & Christy 1974 (*Phys. Rev. B* **9**,
  5056), Table II, transcribed from the published table; also bit-identical
  to the refractiveindex.info deposit.
- **Au** — all 49 rows match Johnson & Christy 1972 (*Phys. Rev. B* **6**,
  4370), Table I; also bit-identical to the refractiveindex.info deposit.
- **SiO₂** — all six Sellmeier coefficients match Malitson 1965 verbatim.
- **TiO₂** — identical (max |Δ| ≈ 5×10⁻⁷, text-export rounding) to the
  refractiveindex.info dataset deposited by the original author
  (Siefke 2016; "numerical data kindly provided by Thomas Siefke"), and
  consistent with the published figure (the source article tabulates no
  numeric values, so a direct table comparison is not possible for this
  material).
- **Ti, Cu** — shipped for completeness but not used in any reported
  dataset; bit-identical to the refractiveindex.info deposits.

## 4. Model checkpoint provenance

See [`MODEL_MANIFEST.md`](MODEL_MANIFEST.md) and
[`release_models/README.md`](release_models/README.md) for the full
per-checkpoint record, including which pre-training checkpoints could not be
preserved and why (overwritten filenames, an in-memory-only training path,
and one experiment run on a since-terminated cloud instance).

An independent re-training check confirmed that regenerating the pipeline
from the released driver scripts reproduces the reported numbers: from-scratch
models reproduce bit-exactly given the same seed, and transfer-learned models
agree with the archived results to within ~0.1 percentage point — well
inside the seed-to-seed spread already reported in the paper. This bounds
the impact of TMM pre-training's floating-point non-determinism across runs
and machines.

## 5. Au dataset: wavelength-dependent truncation characteristic

A post-acceptance solver comparison (2026-08-08) identified a material-dependent
truncation characteristic of the Au datasets that users regenerating the data
with an independent, fully converged solver should expect to see.

**What it is.** The adaptive Fourier-order rule is geometry-based (it does not
inspect the material), and the convergence study that calibrated it
(Supplementary Section S16 of the article) covers the Cr structures. Chromium
is strongly lossy (Im ε ≈ 21 in the NIR), which damps resonances and makes
low orders accurate. Gold is low-loss and plasmonic at its band edge
(ε ≈ −22.5 + 1.4i at 780 nm), where RCWA converges much more slowly at fixed
order. Probing archived designs by re-running them at higher order gives, for
cells the archive computed at N = 9:

| Dataset / design | 500 nm | 650 nm | 700 nm | 780 nm |
|---|---|---|---|---|
| Au-B ring-disk (idx 10) | 0.004 | 0.177 | — | 0.098 |
| Au-B ring-disk (idx 28) | 0.018 | — | — | — |
| Au-A dual-patch (idx 50) | — | — | (N=13 cell, exact) | **0.403** |
| Au-A dual-patch (idx 51) | — | — | — | ≈ 0.2–0.3 |

(entries are |A(N=9) − A(N=13)| in absolute absorptance; the N=13 values are
themselves not fully converged for Au, so these are lower bounds — and a
direct check confirms it: on the two worst Au-B designs even the N = 17
reference moves by 0.04-0.10 against N = 21 at 650-780 nm, with the order
series still oscillating at N = 21. The band-edge truncation uncertainty of
Au-B is therefore bounded below, not above, by the table).
Below ≈ 650 nm — where interband absorption makes gold lossy — the same cells
are accurate to ≤ 0.02. A matched control probe on the *Cr* datasets (nine
low-order designs across Structures A/B/C, long-wavelength cells, N = 9 vs
N = 13) gives drifts of ≤ 0.016 with a typical value of ≈ 0.004 — 20-70×
smaller than the Au values above under the identical order rule — confirming
empirically that the characteristic is specific to the low-loss metal, not to
the order rule or the geometries.

**Exposure.** Au-A: 480/500 designs take N = 17 at all wavelengths under the
geometry rule — and a control probe shows N = 17 is converged for gold
(N = 15 → 17 moves a probed design by only 0.002) — so only the 20
mild-geometry designs' λ ≥ 700 nm cells are affected (< 1 % of reliable
cells). Au-B: the structure-B order rule is P/λ-driven, so over the reliable
cells the order distribution is N = 9: 33 %, N = 13: 49 %, N = 17: 18 % —
and the N = 17 cells occur only below ≈ 615 nm (large P/λ), which means the
rule assigns its lowest orders exactly in the long-wavelength band where gold
is plasmonic and needs the most (N = 9 and N = 13 cells occur across the
band, depending on period). Probing both low-order classes shows the
same band-edge behaviour: N = 13 designs move by 0.008-0.25 between N = 13
and N = 17 at 650-780 nm, comparable to the N = 9 class. The honest summary for
Au-B is therefore dataset-wide: any Au-B cell in the plasmonic band
(≈ 650-780 nm) may carry a resonance-position-dependent truncation deviation
of up to ≈ 0.25; cells below ≈ 650 nm are accurate to ≈ 0.02. In every cell probed in
the *order study above*, the archived value matches a same-order re-run to
≤ 0.001 (one resonance-flank cell to 0.0075) — the archives faithfully
implement the stated protocol; the issue is the protocol's accuracy for gold,
not the data's provenance. (The broader regeneration sample below contains
larger same-order deviations on other Au cells; those are cross-library-
version effects, addressed separately.)

**Estimated impact on the reported transfer benefits (estimate, not a
measurement).** The transfer benefit is governed by TMM-RCWA operating-band
agreement, and the article's own noise-injection experiment measures how the
benefit degrades as that agreement is perturbed (benefit falls from +47% to
+33% as the fidelity MAE rises from 7.4% to 10.8%, and its sign does not flip
until the source-target correlation is destroyed). Averaged over the full
band, the probed truncation deviations correspond to a fidelity shift of
roughly ±2-4 pp about the measured Au-B median MAE of 15.4% — a perturbation well
inside the regime where that experiment finds the benefit positive and
substantial. On this basis the *sign* of the reported Au-B transfer benefit
is robust to the truncation characteristic; the *magnitudes* could plausibly
move by several pp under a fully converged target, and the thinnest margin is
the saturated-pool entry (+4.6% at n = 339), which we cannot exclude moving
toward zero. This paragraph extrapolates a Cr-calibrated sensitivity curve to
the Au dataset and is offered as an order-of-magnitude estimate; making it a
measurement would require regenerating the Au-B dataset at high order and
re-training, which has not been done.

**Why the paper's conclusions are unaffected.** All Au results in the article
are within-dataset model comparisons (M₀ vs. M_TL etc.) on the same simulated
ground truth; a truncation bias enters both sides of every comparison
identically. The article presents the Au experiments as a secondary,
descriptive cross-material check and states the order protocol explicitly; it
makes no claim that the Au spectra are converged absolute absorptances of
physical devices. The practical caveat is for reuse: anyone regenerating the
Au datasets with a fully converged solver should expect differences of up to
a few tenths in absorptance for the low-order cells above ≈ 650 nm, largest
near the plasmonic band edge.

Probe records (verbatim logs where retained, transcribed otherwise, with a
provenance note) are in
[`results/verification_probes/`](results/verification_probes/); every probe
is reproducible from the released scripts.

**Control probes (completed 2026-08-09).** (i) The N = 17 majority class is
converged for gold: an N = 17 Au-A design probed at 780 nm moves by only
0.002 between N = 15 and N = 17, and the archived value equals the N = 17
re-run exactly. The characteristic is therefore confined to the low-order
cells described above. (ii) Additional Au-A low-order designs confirm the
class-wide magnitude (N9→N13 shifts of 0.21-0.40 at 780 nm on all three
probed designs), and Au-B designs span 0.02-0.25 at 650-780 nm depending on
where each design's resonance falls.

**Regeneration comparison (sampled, 2026-08-09).** Re-running 20 archived
designs per dataset through the released code in the current environment
(torch 2.9.1; archives generated under torch 2.5.1) reproduces the Cr
archives to ≤ 0.0025 absolute (A ≤ 0.0025, B ≤ 0.00044, C ≤ 0.00125). The Au
archives reproduce to ≤ 0.005 in 36 of 40 sampled designs; three others
deviate by 0.0057-0.019 (spectral inspection of the two largest shows
band-wide ripples near the plasmonic band — cross-library-version numeric
sensitivity of sharp resonances at complex64 precision; each deviation is
bit-stable across repeated runs in the current environment), and one isolated
single-cell anomaly was found and is documented below.

**Known single-cell anomaly.** `struct_B_Au_500_jc.npz`, design index 462,
wavelength 541.6 nm: the archived value 0.1723 is inconsistent with its own
spectral neighbours (0.357 at 537.6 nm, 0.295 at 545.7 nm) and with re-runs
in the current environment at both complex64 (0.342) and complex128 (0.327)
precision, while the ten neighbouring wavelengths reproduce to ≤ 0.0001.
This is an isolated numerical glitch from the original generation run (one
cell out of 49,666 reliable Au-B cells; the value lies inside [0, 1], so
the physicality filter could not flag it). Its effect on training is that of
a single noisy label; no reported quantity changes.

## Scope and limitations

- This pass covers citation attribution, numeric reproduction from archived
  data, and checkpoint/reproducibility provenance. It does not constitute an
  independent re-derivation of the underlying physics or a re-run of the
  full RCWA/TMM simulation pipeline from raw parameters (that pipeline is
  itself the subject of the mechanical gate and the released driver scripts).
- Two items are recorded as read but not independently re-executed due to
  since-terminated compute resources: one worked example in the oblique-
  incidence grazing-order discussion (Supplementary Section) and the
  geometry-constraint sensitivity checkpoints noted in `MODEL_MANIFEST.md`.
