# Pre-trained Models

TMM-pretrained PyTorch checkpoints for the three MIM absorber structures.
These are the source of weight-level knowledge in the PBTL pipeline:
load these as initial weights, then fine-tune on a small RCWA training set.

| File | Structure | Variant |
|------|-----------|---------|
| `pretrained_m0_tmm.pt`        | A (dual-cavity) | Geometry-only baseline (M0)  |
| `pretrained_mphys_tmm.pt`     | A (dual-cavity) | Physics-feature variant (Mphys) |
| `pretrained_m0_tmm_B.pt`      | B (ring–disk)   | Geometry-only baseline (M0)  |
| `pretrained_mphys_tmm_B.pt`   | B (ring–disk)   | Physics-feature variant (Mphys) |
| `pretrained_m0_tmm_C.pt`      | C (rect patch)  | Geometry-only baseline (M0)  |
| `pretrained_mphys_tmm_C.pt`   | C (rect patch)  | Physics-feature variant (Mphys) |

Each checkpoint contains the ResNet-256-4 backbone weights pre-trained on
≥5,000 TMM samples (see Supplementary Section S2 for sweep). Load with:

```python
import torch
state = torch.load("results/pretrained_mphys_tmm.pt", map_location="cpu")
model.load_state_dict(state)
```
