# Trained model checkpoints — PBTL-MIM (PNFA-D-26-00266)

State dicts for the models described in the manuscript. Load with
`torch.load(path, map_location="cpu")` into the matching architecture defined in
the corresponding driver script under `step0_screen/`.

**All checkpoints are TMM pre-training outputs** (the source models that the
RCWA fine-tuning stage starts from), except where noted.

## Contents

| File | Structure / role | Provenance |
|---|---|---|
| `pretrained_m0_tmm_B.pt` | B: M₀ TMM pre-train | saved 2026-06-22, immediately before the Structure-B result file — the run that produced the reported numbers |
| `pretrained_mphys_tmm_B.pt` | B: M_phys TMM pre-train | same run as above |
| `pretrained_m0_tmm.pt` | A: M₀ TMM pre-train | **re-run** (2026-06-29), not the exact file behind the Structure-A table — see *Missing checkpoints* |
| `pretrained_mphys_tmm.pt` | A: M_phys TMM pre-train | **re-run** (2026-06-29), same caveat |
| `pretrained_m0_rand_redesign.pt` | Random-source control (M_rand) | 2026-06-24, as used |
| `pretrained_m0_tmm_redesign.pt` | 10-seed driver M₀ pre-train | 2026-06-24, as used |
| `pretrained_m0_au_tmm_jc.pt` | Au/SiO₂ cross-material: M₀ | 2026-06-29, as used |
| `pretrained_mphys_au_tmm_jc.pt` | Au/SiO₂ cross-material: M_phys | 2026-06-29, as used |
| `mfnn_lf_redesign.pt` | Deep composite MF-NN, low-fidelity branch (Supp. S7) | 2026-06-24, as used |
| `fs_lf_5000_redesign.pt` | Full-Spectrum ResNet variant, LF pre-train (Supp. S8) | 2026-06-24, as used |
| `inverse_design_S13/m0_350.pt` | Supp. S13 inverse-design probe: M₀ | 2026-03-12, **earlier visible-band pipeline** — this is what S13 states it used |
| `inverse_design_S13/mphys_350.pt` | Supp. S13 inverse-design probe: M_TL+phys | same |

`SHA256SUMS.txt` lists checksums for every file.

## Missing checkpoints, and why

Three sets of pre-trained weights behind the paper are not recoverable. The
reason differs in each case and is structural, not accidental:

1. **Structure A** — `pbtl_A_redesign.py` saves its pre-trained weights under the
   same filenames used by the earlier visible-band pipeline
   (`pretrained_m0_tmm.pt`, `pretrained_mphys_tmm.pt`). The Structure-A results
   were produced on a machine where those names were later overwritten. The
   copies shipped here are from a 2026-06-29 re-run of the same script.
2. **Structure C** — `pbtl_C_v2_redesign.py` never writes its pre-trained weights
   to disk; it pre-trains in-process and copies the state dict straight into
   fine-tuning. No file was ever created.
3. **Supplementary S17** (geometry-constraint sensitivity) — the B/C runs were
   executed on a rented cloud GPU instance on 2026-06-27; result archives and
   logs were retrieved (`results/inv3_sens_*_result.npz`), the weights were not,
   and the instance no longer exists. The Structure-A run of the same experiment
   wrote to `/tmp` and was likewise not preserved.

**This does not affect reproducibility of the reported numbers.** Every driver
script is released, and re-running them regenerates the pipeline end to end. Note
that TMM pre-training is not bit-reproducible across runs or machines
(floating-point non-determinism), so regenerated weights differ slightly from the
originals; an independent re-training check (2026-07-22) found from-scratch models
reproduce bit-exactly and transfer-learned models agree to within ~0.1 pp — well
inside the seed-to-seed spread reported in the paper.

## Regenerating any checkpoint

```bash
python step0_screen/pbtl_A_redesign.py      # writes pretrained_m0_tmm.pt, pretrained_mphys_tmm.pt
python step0_screen/pbtl_B_redesign.py      # writes pretrained_{m0,mphys}_tmm_B.pt
python step0_screen/pbtl_C_v2_redesign.py   # pre-trains in memory (no checkpoint file)
```
