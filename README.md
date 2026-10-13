# DFCA - Diffusion Framework for Crystal Anisotropy

**DFCA** is an anisotropy-aware diffusion framework for two-dimensional crystal generation and screening. It achieves state-of-the-art performance on 2D materials discovery with explicit modeling of in-plane (XY) and out-of-plane (Z) anisotropy.

## Key Innovation

DFCA introduces **direction-dependent noise scaling** and **dual-stream attention** that separates periodic in-plane bonding from vacuum-dominated out-of-plane geometry. This explicit anisotropy modeling enables stable generation of 2D crystal structures.

## Main Results

On the **2D benchmark**, DFCA achieves:
- **100.00% Compositional Validity** (highest)
- **97.28% Novelty** (highest)
- **71.60% 2D-UN** (highest)
- **0.908 2D DistScore** (highest)

| Metric | DFCA | CrystalDiT | DiffCSP | DiffCSP++ | FlowMM |
|--------|------|------------|---------|-----------|--------|
| Compositional Validity | **100.00%** | 71.80% | 86.52% | 84.00% | 79.98% |
| Novelty | **97.28%** | 71.66% | 81.86% | 44.78% | 78.98% |
| 2D-UN | **71.60%** | 45.00% | 67.87% | 30.00% | 55.76% |
| 2D DistScore | **0.908** | 0.800 | 0.766 | 0.739 | 0.701 |

## Model Specifications

- **Architecture**: 18-layer diffusion transformer (d=512, 8 attention heads)
- **Input**: 46-token sequence with separate XY/Z encoding
- **Training**: Physics-informed loss with 11 constraints (c₁–c₁₁)
- **Screening**: Multi-stage pipeline (S0–S6) with MLIP relaxation

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-repo/DFCA.git
cd DFCA
conda create -n dfca python=3.10
conda activate dfca
pip install -r requirements.txt

# Download dataset (see datasets/DATA_SOURCE.md)
# Prepare data with preprocessing

# Generate crystals
PYTHONPATH=. python generate.py \
    --checkpoint /path/to/checkpoint \
    --config diffusion/config.yaml \
    --num_samples 100 --output_dir ./results

# Train (optional)
PYTHONPATH=. python train.py \
    --config diffusion/config.yaml \
    --data_dir ./datasets --output_dir ./output
```

## Dataset

The training dataset contains curated 2D materials from established databases. See [`datasets/DATA_SOURCE.md`](datasets/DATA_SOURCE.md) for download instructions and citations.

## License

See [`LICENSE`](LICENSE)
