# Physics-Based Transfer Learning for MIM Absorber Surrogate Modeling

This repository contains the source code and scripts for the paper:

**"When Does Physics-Based Transfer Learning Help? A Systematic Study of Surrogate Modeling for Metal-Insulator-Metal Absorbers"**

Seungbin Choi, Independent Researcher

## Overview

We present a systematic study of physics-based transfer learning (PBTL), which combines analytically derived physics features with neural network pre-training on cheap Transfer Matrix Method (TMM) data, followed by fine-tuning on small Rigorous Coupled-Wave Analysis (RCWA) datasets. Experiments on three structurally distinct MIM absorbers reveal when and why cross-fidelity transfer learning helps or fails.

## Repository Structure

```
mim_novel/
├── src/
│   ├── models/          # Neural network architectures (ResNet, MLP)
│   ├── simulation/      # RCWA and TMM simulation engines
│   ├── training/        # Training loop and utilities
│   └── utils/           # Physics features, data utilities, seed management
├── step0_screen/        # Main experiment scripts (4-way comparison, ablations)
├── validate/            # RCWA validation scripts
├── figures/             # Publication figures (PDF)
├── paper.tex            # Manuscript source (LaTeX)
├── generate_all_figures_final.py    # Figure generation
├── generate_figures_optcomm.py      # Publication-quality figures
├── make_schematic.py                # Structure schematic diagrams
├── inverse_design_constrained.py    # Inverse design experiments
└── tandem_inverse_design.py         # Tandem network inverse design
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, Matplotlib
- [torcwa](https://github.com/kch3782/torcwa) (GPU-accelerated RCWA)
- CUDA-capable GPU (recommended for RCWA simulations)

## Installation

```bash
pip install torch numpy scipy matplotlib
pip install torcwa
```

## Quick Start

### 1. Generate RCWA training data

```bash
cd step0_screen
python pbtl_A_10seed.py  # Structure A, 10-seed experiment
```

### 2. Run 4-way model comparison

The main experiment scripts in `step0_screen/` run all four model variants (M0, M_phys, M_TL, M_TL+phys) and save results as `.npz` files.

### 3. Generate figures

```bash
python generate_all_figures_final.py
```

## Three MIM Structures

| Structure | Geometry | Parameters | Dominant Physics |
|-----------|----------|------------|------------------|
| A | Dual-cavity thin-film | 10 | Fabry-Perot interference |
| B | Ring-disk resonator | 8 | Fano resonance (near-field) |
| C | Rectangular patch | 7 | Polarization-dependent diffraction |

## Key Results

- Physics features universally reduce MAE by 4-19% across all structures
- TMM pre-training provides up to 29.5% MAE reduction when thin-film interference dominates (Structure A)
- TMM pre-training fails when near-field coupling (B) or polarization mismatch (C) governs the response
- TMM-RCWA spectral fidelity quantitatively predicts transfer learning benefit (r=0.981, p=0.003)

## Citation

If you use this code, please cite:

```bibtex
@article{choi2026pbtl,
  title={When Does Physics-Based Transfer Learning Help? A Systematic Study of Surrogate Modeling for Metal--Insulator--Metal Absorbers},
  author={Choi, Seungbin},
  journal={Applied Optics},
  year={2026}
}
```

## License

MIT License
