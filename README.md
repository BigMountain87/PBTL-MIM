# PBTL-MIM: Physics-Based Transfer Learning for MIM Absorber Surrogates

Source code, data, and scripts for the paper:

> **TMM–RCWA Spectral Fidelity Predicts Physics-Based Transfer Learning Success in Surrogate Modeling of Metal–Insulator–Metal Absorbers**
> Sang-Bae Choi¹\*, Joonhyub Kim²,³, Chang-Mo Kang²,³\*
>
> ¹ Independent Researcher, Busan 46264, Republic of Korea
> ² Department of Nanomechatronics Engineering, Pusan National University, Busan 46241, Republic of Korea
> ³ School of Transdisciplinary Engineering, Pusan National University, Busan 46241, Republic of Korea
> \* Corresponding authors: sbchoi129@gmail.com, fd1kcm@pusan.ac.kr

We study when transfer learning from a low-fidelity simulator (TMM with effective-medium approximation) to a high-fidelity simulator (RCWA) improves neural-network surrogates of metamaterial absorbers. Across three structurally distinct MIM absorbers we find that pre-training benefit is **strongly associated** with TMM–RCWA spectral fidelity, and we provide a small-pilot pre-screening procedure (20–50 RCWA samples).

---

## Repository Contents

```
.
├── src/
│   ├── models/         ResNet / MLP architectures
│   ├── simulation/     RCWA (torcwa) + TMM engines for Structures A, B, C
│   ├── training/       Training loop, loss, schedulers
│   └── utils/          Physics features, seed management, data loading
├── data/raw/           RCWA-validated parameter samples (npz)
│   ├── struct_A_*.npz       (Structure A: dual-cavity, 10-param)
│   ├── struct_B_*.npz       (Structure B: ring-disk, 8-param)
│   └── struct_C_*.npz       (Structure C: rect patch, 7-param)
├── figures/            Final figure PDFs/PNGs (Figure_1–5)
├── *.py                Experiment / analysis scripts (see below)
├── *.sh                Convenience launchers
├── requirements.txt
├── LICENSE
└── README.md
```

### Top-level Python scripts (purpose at a glance)

| Script | Purpose |
| --- | --- |
| `extra_rcwa_A.py` | Generate 200 additional Structure-A RCWA samples |
| `phase3_expanded_test.py` | 241-sample expanded test-set evaluation (Sec. 3.8) |
| `noise_injection_C.py` | Structure-C TMM-noise injection replica (Table 5) |
| `compute_noise_C_stats.py` | Statistics for Structure-C noise replica |
| `material_generalization_B.py` | Au cross-material check on Structure B (Sec. 3.7) |
| `full_spectrum_baseline.py` | Full-spectrum ResNet baseline (Sec. S8) |
| `deep_mfnn_baseline.py` | Composite multi-fidelity NN baseline (Sec. S7) |
| `cokriging_ntmm_sweep.py` | Sparse Co-Kriging vs. PBTL data-efficiency sweep |
| `cokriging_sensitivity.py` | Co-Kriging hyperparameter sensitivity |
| `inverse_design_constrained.py` | Inverse-design demonstration |
| `tandem_inverse_design.py` | Tandem-network inverse design |
| `analyze_lr_ablation.py` | Learning-rate ablation analysis |
| `robustness_stats.py` | Permutation/bootstrap robustness statistics |
| `summarize_results.py` | Compile main-paper tables from per-experiment npz outputs |
| `make_schematic.py`, `make_figure2.py`, `make_figures_345.py` | Figure generation |
| `generate_all_figures_final.py`, `generate_figures_optcomm.py`, `generate_figures_fixed.py` | Publication figure compilation |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.13 with CUDA
- `torcwa` (GPU-accelerated RCWA) — https://github.com/kch3782/torcwa
- NumPy, SciPy, Matplotlib, scikit-learn

```bash
pip install -r requirements.txt
pip install torcwa  # follow torcwa repo for CUDA-matched build
```

A CUDA GPU (≥ 8 GB) is recommended; RCWA generation runs at 17–33 s/sample on an RTX 4070 Ti SUPER.

---

## Reproducing the Paper

The headline experiments (Tables 1–3, 5–9 and Figures 3–5) are produced from data in `data/raw/` plus a small number of additional RCWA runs.

### 1. Environment setup

```bash
git clone https://github.com/BigMountain87/PBTL-MIM.git
cd PBTL-MIM
pip install -r requirements.txt
pip install torcwa
```

### 2. Re-train the four model variants (Tables 1–3)

The 4-way comparison runs $\{M_0, M_\text{phys}, M_\text{TL}, M_\text{TL+phys}\}$ on each structure with 10 seeds × 4 training sizes ($n \in \{50, 100, 200, 350\}$). Inputs: `data/raw/struct_*_500.npz` (RCWA targets) and 5,000 TMM pre-training samples generated on the fly.

> **Note:** The original 10-seed orchestration scripts depended on internal cluster paths and are not included in this release. The per-seed training functions are preserved in `src/training/` and can be driven with a short loop. See `summarize_results.py` for the expected `.npz` output schema.

### 3. Headline noise-injection experiment (Sec. 3.5)

Structure-A noise sweep over the TMM source ($\sigma \in \{0, 0.05, 0.10, 0.15, 0.20, \infty\}$) plus the Structure-C replica:

```bash
python noise_injection_C.py        # Structure-C replica
python compute_noise_C_stats.py    # statistics
```

The Structure-A driver follows the same logic with `metal='Cr', struct='A'`.

### 4. Cross-material check (Sec. 3.7)

```bash
python material_generalization_B.py
```

### 5. Expanded test-set validation (Sec. 3.8)

```bash
python extra_rcwa_A.py            # generate 200 extra Structure-A RCWA samples
python phase3_expanded_test.py    # evaluate on 241-sample combined test set
```

### 6. Generate paper figures

```bash
python make_schematic.py          # Figure 1 (structure schematics)
python make_figure2.py            # Figure 2 (TMM-RCWA spectra)
python make_figures_345.py        # Figures 3–5
```

---

## Three MIM Structures

| | Structure A | Structure B | Structure C |
| --- | --- | --- | --- |
| Geometry | Asymmetric dual-dielectric dual-cavity | Concentric ring-and-disk on SiO₂ spacer | Rectangular Cr patch on SiO₂ spacer |
| Parameters | 10 | 8 | 7 |
| Dominant physics | Fabry–Pérot interference | Ring-disk Fano resonance | Polarization-dependent diffraction |
| TMM fidelity (median *r*) | 0.72 | −0.07 | 0.34 |
| Best PBTL gain | 29.5 % @ n=50 (29.4 % @ n=350) | None — physics features alone best | None — physics features alone best |

---

## Headline Results

- Physics-feature augmentation reduces test MAE by **4–19 %** across all three structures.
- TMM pre-training is **conditionally** effective: up to **29.5 %** MAE reduction at n=50 RCWA samples (29.4 % at n=350) for Structure A; negligible or negative for Structures B and C.
- A controlled noise-injection experiment yields **Pearson r = 0.981, p = 3 × 10⁻³ (N = 5)** between TMM–RCWA spectral fidelity and PBTL benefit; replicated on Structure C (descriptive r = −0.973, N = 4).
- Pre-screening with 20–50 RCWA pilot samples reliably classifies high- vs. low-fidelity regimes for Structure A; the r ≳ 0.3 threshold is a working heuristic, not yet externally validated across structure families.

---

## Data Layout

`data/raw/` holds the RCWA ground-truth datasets used throughout the paper:

| File | Description | n_samples |
| --- | --- | --- |
| `struct_A_vis_500.npz` | Structure A, primary 500-sample pool (Cr / SiO₂) | 500 |
| `struct_A_vis_100.npz` | Structure A, 100-sample subset (Au baseline) | 100 |
| `struct_A_100.npz` | Structure A, screening run | 100 |
| `struct_B_500.npz` / `_100.npz` | Structure B | 500 / 100 |
| `struct_C_500.npz` / `_100.npz` | Structure C | 500 / 100 |

Each file contains: `params` (geometry vector), `A` (absorptance at 100 wavelengths 380–780 nm), `R`, `T`, `wavelengths`, and metadata.

Random seeds, train/val/test split indices, and per-experiment sample-index lists are preserved in the corresponding result `.npz` files (one per run) and can be regenerated by passing the same seed to the loader; see `src/utils/seed_utils.py`.

---

## Citation

```bibtex
@article{choi2026pbtl,
  title   = {TMM--RCWA Spectral Fidelity Predicts Physics-Based Transfer
             Learning Success in Surrogate Modeling of Metal--Insulator--Metal Absorbers},
  author  = {Choi, Sang-Bae and Kim, Joonhyub and Kang, Chang-Mo},
  journal = {[journal]},
  year    = {2026}
}
```

The DOI / final journal will be updated upon publication.

---

## License

MIT License — see [LICENSE](LICENSE).
