# DFCA - Diffusion Framework for Crystal Anisotropy

**DFCA** is an anisotropy-aware diffusion framework for two-dimensional crystal generation and screening. It achieves state-of-the-art performance on 2D materials discovery with explicit modeling of in-plane (XY) and out-of-plane (Z) anisotropy.

## Key Innovation

Standard crystal diffusion models perturb all spatial directions uniformly, but 2D crystals have fundamentally different geometries: in-plane coordinates describe periodic bonded networks while the out-of-plane direction is largely vacuum.

DFCA addresses this representation mismatch through three coupled design choices:

1. **Direction-Dependent Noise Scaling**: Separate perturbation strengths for out-of-plane atomic coordinates (σ_atom_z = 0.113) and lattice z-components (σ_lattice_z = 0.340)

2. **DualStream Cross-Attention (DSCA)**: Maintains separate XY and Z feature streams with controlled cross-attention (init ≈ 0.78 for main, init ≈ 0.10 for auxiliary)

3. **BondDecoupledMLP**: Limits early nonlinear mixing between XY and Z streams before optional fusion

## Architecture

The model uses an 18-layer DualStream DiT backbone with:

- **CrystalEmbedder**: XY linear embedding + Z MLP with explicit XY/Z separation
- **DualStream Attention**: Separate self-attention for XY and Z tokens with cross-attention gating
- **BondDecoupledMLP**: Concat fusion with learned gating
- **adaLN Modulation**: Adaptive layer normalization with zero initialization

Implementation: [`dfca/models/dit.py`](dfca/models/dit.py)

## Physics-Informed Training

DFCA uses 11 physics constraints (c₁–c₁₁) organized by implementation:

| Category | Constraints | Description |
|----------|-------------|-------------|
| Auxiliary | c₁, c₂ | Invalid atom slots, atom count |
| Chemical | c₃, c₈, c₉ | Row/column validity, element distribution, density |
| Spatial | c₄, c₅ | Slab centering, slab spread |
| Lattice | c₆, c₇ | Lattice length, lattice angle |
| Vacuum | c₁₀, c₁₁ | Vacuum floor/ceiling boundaries |

Constraints use dynamic weighting with noise-dependent decay: high-noise steps apply reduced constraint intensity.

Implementation: [`dfca/training/constraints.py`](dfca/training/constraints.py)

## Generation Pipeline

The generation process consists of four stages:

1. **Noise Seed**: Initialize ε_T ~ N(0,I) with anisotropic scaling S
2. **Guided Denoising**: 18-layer DualStream DiT for T=1000 steps (DDPM)
3. **Atom Feature Decode**: Decode latent to (p, g, x, y, z) predictions via Atom Final Layer
4. **Reconstructed Crystal**: Convert predictions to valid CIF structures

Implementation: [`dfca/generation/generator.py`](dfca/generation/generator.py)

## Main Results

On the **Merged-2D benchmark**, DFCA achieves:

| Metric | DFCA | CrystalDiT | DiffCSP | DiffCSP++ | FlowMM |
|--------|------|------------|---------|-----------|--------|
| Compositional Validity | **100.00%** | 71.80% | 86.52% | 84.00% | 79.98% |
| Novelty | **97.28%** | 71.66% | 81.86% | 44.78% | 78.98% |
| 2D-UN | **71.60%** | 45.00% | 67.87% | 30.00% | 55.76% |
| 2D DistScore | **0.908** | 0.800 | 0.766 | 0.739 | 0.701 |

DFCA achieves the highest scores across all four key metrics.

## Engineering Screening

5,000 generated structures → 49 candidates → **19 dynamically stable** via hierarchical screening:

- **S0**: Composition & novelty filtering
- **S1**: 2D geometry (vacuum gap, slab thickness)
- **S2**: Symmetry detection (spglib)
- **S3**: Deduplication & 2D connectivity
- **S4**: MLIP relaxation (MACE potentials)
- **S5**: Phonon stability (Phonopy)
- **S6**: DFT validation (GPAW)

Implementation: [`dfca/screening/pipeline.py`](dfca/screening/pipeline.py)

## Model Specifications

- **Architecture**: 18-layer DualStream DiT (d=512, 8 attention heads)
- **Input**: 46-token sequence with separate XY/Z encoding
- **Diffusion**: 1000 DDPM steps, linear noise schedule
- **Training**: Physics-informed loss with dynamic constraint weighting

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-repo/DFCA.git
cd DFCA
conda create -n dfca python=3.10
conda activate dfca
pip install -r requirements.txt

# Download dataset (see datasets/DATA_SOURCE.md)

# Generate crystals
PYTHONPATH=. python generate.py \
    --checkpoint /path/to/checkpoint \
    --config diffusion/config.yaml \
    --num_samples 100 --output_dir ./results

# Train (optional)
PYTHONPATH=. python train.py \
    --config diffusion/config.yaml \
    --data_dir ./datasets --output_dir ./output

# Screen candidates
PYTHONPATH=. python screen.py \
    --input ./results --output ./screened
```

## Dataset

The training dataset contains curated 2D materials from established databases. See [`datasets/DATA_SOURCE.md`](datasets/DATA_SOURCE.md) for download instructions and citations.


## Related Work

- [CrystalDiT](https://doi.org/10.1609/aaai.v40i2.37121) – Diffusion Transformer for crystal generation; the foundation architecture extended by DFCA.
- [DiffCSP](https://arxiv.org/abs/2309.04475) – Joint equivariant diffusion for crystal structure prediction.
- [DiffCSP++](https://arxiv.org/search/?query=DiffCSP%2B%2B&searchtype=all) – Symmetry-aware crystal generation using diffusion models.
- [FlowMM](https://proceedings.mlr.press/v235/miller24a.html) – Riemannian flow matching for crystal generation.
- [CDVAE](https://openreview.net/forum?id=03RLpj-tc_) – Crystal diffusion variational autoencoder for periodic materials.


## License

See [`LICENSE`](LICENSE)
