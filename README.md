# Physics-Based Transfer Learning for Neural Network Surrogates of Metal–Insulator–Metal Absorbers

This repository contains the source code and data for the paper:

**"Physics-Based Transfer Learning for Neural Network Surrogates of Metal–Insulator–Metal Absorbers"**

Sang-Bae Choi (Independent Researcher), Joonhyub Kim (Pusan National University), and Chang-Mo Kang (Pusan National University). Submitted to *Photonics and Nanostructures – Fundamentals and Applications*.

## Quick start — reproduce a headline result in one minute

```bash
# Table I fidelity from the shipped raw data (pure NumPy, no GPU, ~1 min):
python compute_fidelity_redesign.py
#   -> A: median r = +0.83, MAE = 7.94%   B: +0.96 / 8.93%   C: +0.65 / 16.94%

# Full mechanical gate: every table/figure/number recomputed from the archives
# (needs pdflatex on PATH for the compile check):
python verify_paper.py        # -> RESULT: PASS (all 10 checks)
```

## Overview

We present a systematic study of physics-based transfer learning (PBTL), which combines analytically derived physics features with neural-network pre-training on cheap Transfer Matrix Method (TMM) data, followed by fine-tuning on small Rigorous Coupled-Wave Analysis (RCWA) datasets. Experiments on three structurally distinct MIM absorbers reveal when and why cross-fidelity transfer learning helps or fails, and introduce a joint pilot-set *r*-and-MAE diagnostic that estimates the transfer benefit (within the tested MIM settings) before any pre-training.

## Reproducing the paper

All reported results are produced by the **`*_redesign`** scripts on the corrected pipeline. Run every script **from the repository root** (paths are repo-relative).

```bash
# 1. Cross-structure TMM–RCWA fidelity (Table I): reproduces A 0.83/7.9, B 0.96/8.9, C 0.65/16.9
python compute_fidelity_redesign.py

# 2. 4-way model comparison (Tables 3–5) — Structures A, B, C
python step0_screen/pbtl_A_redesign.py
python step0_screen/pbtl_B_redesign.py
python step0_screen/pbtl_C_v2_redesign.py

# 3. Figures
python make_schematic.py     # Figure 1 (structure schematics)
python make_figure2.py       # Figure 2 (TMM vs RCWA spectra)
python make_figure3.py       # Figure 3 (learning curves, A/B/C)
python make_figure4.py       # Figure 4 (fidelity vs transfer benefit)
python make_figure5.py       # Figure 5 (feature importance)
python make_figureS1.py   # Supplementary figure S1
python make_figureS2.py   # Supplementary figure S2
python make_figureS3.py   # Supplementary figure S3
python make_figureS4.py   # Supplementary figure S4
```

The corrected RCWA datasets (`data/raw/struct_{A,B,C}_500_redesign.npz`) and result archives (`results/*.npz`) are included, so the figure and fidelity scripts run without re-simulating. Regenerating the RCWA data from scratch requires a CUDA GPU and `torcwa`.

## Repository structure

```
mim_novel/
├── src/
│   ├── models/          # NN architectures (the paper model is ResNet-256-4; tmm_nn.py is a legacy baseline)
│   ├── simulation/      # RCWA + TMM engines; tmm_struct_c_aniso.py is the paper's anisotropic Structure-C TMM
│   ├── training/        # trainer.py serves the TMM-as-input baseline only; the *_redesign drivers train the paper models
│   └── utils/           # physics features, data utilities, seed management
├── step0_screen/        # main experiment + ablation drivers (the *_redesign scripts produce the reported numbers)
├── compute_fidelity_redesign.py   # Table I cross-structure fidelity
├── make_figure{2,3,4,5,S1,S2,S3,S4}.py  # publication figures
├── make_schematic.py    # Figure 1 schematics
├── data/raw/            # corrected RCWA datasets (*_redesign.npz)
├── results/             # result archives (*.npz)
├── paper.tex, supplementary.tex
└── obsolete/            # superseded pre-correction scripts (NOT part of the release pipeline)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (the drivers use `torch.load(..., weights_only=True)`)
- NumPy, SciPy, Matplotlib
- [torcwa](https://github.com/kch3782/torcwa) (GPU-accelerated RCWA) — only needed to regenerate RCWA data
- CUDA-capable GPU (only for RCWA simulation; figure/fidelity reproduction is CPU-only)

```bash
pip install torch numpy scipy matplotlib
pip install torcwa   # optional, for RCWA regeneration
```

## Three MIM structures

| Structure | Geometry | Parameters | Dominant physics |
|-----------|----------|------------|------------------|
| A | Asymmetric dual-cavity (SiO₂ + TiO₂) | 10 | Fabry–Pérot interference |
| B | Ring–disk resonator | 8 | Fano resonance (ring–disk near-field coupling) |
| C | Dual-polarization rectangular patch | 7 | Polarization-dependent response |

All three share a 100 nm Cr ground mirror; absorptance is evaluated on the unified 400–1800 nm band (100 points). Optical constants use measured Johnson–Christy data for Cr and Au, Malitson SiO₂, and Siefke TiO₂.

## Key results

- Physics features reduce MAE by 4–30% across all three Cr structures.
- Weight-level TMM pre-training is **positive for every structure and training size** but strongly **fidelity-graded**: +49.7% (Structure A, n=50) down to +9.7% (Structure C, n=350), the gradient tracking operating-band TMM–RCWA MAE.
- A **joint pilot-set diagnostic** (median TMM–RCWA correlation *r* **and** median operating-band MAE) predicts the transfer benefit, with the **absolute MAE component ordering the benefit where shape correlation alone mis-ranks it** (controlled noise: |Pearson| 0.98 for MAE vs 0.81 for *r*). A single-number *r* threshold is insufficient — Structure B has the **highest** *r* (0.96) yet a smaller benefit than A.
- Genuine **negative transfer** is reached only by driving source fidelity low enough — near zero for the high-fidelity Structure A, but already at moderate noise for the lower-fidelity B and C (+47% → −58% in the controlled Structure-A sweep).
- A corrected Au/SiO₂ cross-material check confirms positive weight-level transfer for both Structures A (up to +38%) and B (up to +28%).

## Correction note (2026-06)

A pre-decision audit identified and corrected four methodological issues (Fourier-truncation convergence; Johnson–Christy / Siefke optical constants; a wavelength-grid mismatch in the Table-I fidelity script; and the Structure-C parameter-bounds normalization). The complete RCWA dataset was regenerated and every downstream result re-run by the `*_redesign` scripts; the cross-structure fidelity above is produced by `compute_fidelity_redesign.py`. On the corrected data the weight-transfer benefit is **positive for all three structures** — the earlier B/C natural-negative-transfer taxonomy did not survive and was removed; negative transfer is retained only as a controlled low-fidelity result. Per-correction magnitudes are tabulated in the Supplementary change-log (Section S19). Superseded pre-correction scripts are preserved in `obsolete/` and are not part of the release pipeline.

## Citation

```bibtex
@article{choi2026pbtl,
  title={Physics-Based Transfer Learning for Neural Network Surrogates of Metal--Insulator--Metal Absorbers},
  author={Choi, Sang-Bae and Kim, Joonhyub and Kang, Chang-Mo},
  journal={Photonics and Nanostructures -- Fundamentals and Applications},
  year={2026},
  note={Submitted}
}
```

## License

MIT License
