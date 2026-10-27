# DFCA - Diffusion Framework for Crystal Anisotropy

**DFCA** is an anisotropy-aware diffusion framework for two-dimensional crystal generation and screening. Under the reported Merged-2D evaluation protocol, it performs particularly well in compositional validity, formula-based novelty, and 2D-UN, while other evaluated models lead on some geometric-validity and distributional metrics.

## Key Innovation

Standard crystal diffusion models perturb all spatial directions uniformly, but 2D crystals have fundamentally different geometries: in-plane coordinates describe periodic bonded networks while the out-of-plane direction is largely vacuum.

DFCA addresses this representation mismatch through three coupled design choices:

1. **Direction-Dependent Noise Scaling**: Separate perturbation strengths for out-of-plane atomic coordinates (σ_atom_z = 0.113) and lattice z-components (σ_lattice_z = 0.340)

2. **DualStream Cross-Attention (DSCA)**: Maintains separate XY and Z feature streams with bidirectional cross-attention (initial gate values ≈ 0.782 for Z-from-XY and 0.10 for XY-from-Z)

3. **BondDecoupledMLP**: Limits early nonlinear mixing between XY and Z streams before optional fusion

## Architecture

The model uses an 18-layer DualStream DiT backbone with:

- **CrystalEmbedder**: XY linear embedding + Z MLP with explicit XY/Z separation
- **DualStream Attention**: Separate self-attention for XY and Z tokens with cross-attention gating
- **BondDecoupledMLP**: Separate XY/Z MLP processing followed by concatenation and optional fusion MLP
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
2. **Reverse Diffusion Sampling**: 18-layer DualStream DiT for T=1000 steps (DDPM)
3. **Atom Feature Decode**: Decode latent to (p, g, x, y, z) predictions via Atom Final Layer
4. **Reconstructed Crystal**: Convert predictions to valid CIF structures

Implementation: [`dfca/generation/generator.py`](dfca/generation/generator.py)


## Main Results

All five models were evaluated on the Merged-2D benchmark
using 500 generated structures per model under the same
evaluation definitions.

| Metric | CrystalDiT | DFCA | DiffCSP | DiffCSP++ | FlowMM |
|---|---:|---:|---:|---:|---:|
| Structural Validity ↑ | 89.00% | 81.40% | 99.71% | **100.00%** | 93.85% |
| Compositional Validity ↑ | 71.80% | **100.00%** | 86.52% | 84.00% | 79.98% |
| 2D Validity ↑ | **98.60%** | 90.40% | 95.61% | 81.00% | 93.75% |
| Overall Validity ↑ | 63.00% | 73.60% | **82.91%** | 67.00% | 70.61% |
| Uniqueness ↑ | 99.68% | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Formula-based Novelty ↑ | 71.66% | **97.28%** | 81.86% | 44.78% | 78.98% |
| 2D-UN ↑ | 45.00% | **71.60%** | 67.87% | 30.00% | 55.76% |
| 2D DistScore ↑ | 0.799635 | **0.908199** | 0.765719 | 0.739170 | 0.701445 |

**Key findings:**

- **DFCA** achieves the highest compositional validity (100.00%), formula-based novelty (97.28%), 2D-UN (71.60%), and 2D DistScore (0.908199) in this table.
- **DiffCSP** achieves the highest overall validity (82.91%).
- **DiffCSP++** achieves the highest structural validity (100.00%).
- **CrystalDiT** achieves the highest 2D validity (98.60%).
- All models except CrystalDiT report 100.00% uniqueness under this evaluation protocol.

DFCA achieves the strongest composition-based discovery metrics and 2D DistScore among the five evaluated models.

**Evaluation notes:** Novelty measures reduced-formula novelty relative to the training set among unique, overall-valid structures. 2D-UN measures the fraction of all generated structures that are valid, unique, and formula-novel. DistScore combines five IQR-scaled inverse Wasserstein-distance scores with angle-outlier and slab-centering agreement scores. Reported metrics are computed on decoded CIFs after the selected decoding/post-processing steps. The models share the same evaluation definitions, although training and decoding configurations differ.



### Evaluation environment

Install dependencies from `requirements.txt`, including SMACT for compositional-validity evaluation. The evaluator enforces this dependency. Generation uses the element vocabulary collected from the dataset splits supplied through the data directory.

## Engineering Screening

5,000 generated structures were screened to 49 candidates, with 19 subsequently reported as stable under the project's screening criteria. The screening implementation includes:

- Composition and novelty filtering
- 2D geometry checks, symmetry detection, deduplication, and connectivity checks
- ML interatomic-potential relaxation (MACE)
- Elastic-stability and phonon checks
- Thermodynamic and energy-above-hull assessment
- Molecular-dynamics screening

GPAW-based DFT validation is provided through the separate M1–M4 workflow.

Implementation: [`dfca/screening/pipeline.py`](dfca/screening/pipeline.py)

## Model Specifications

- **Architecture**: 18-layer DualStream DiT (d=512, 8 attention heads)
- **Input**: 23 tokens (20 atom slots + 3 lattice tokens), with separate XY/Z feature streams
- **Diffusion**: 1000 DDPM steps, linear noise schedule
- **Training**: Physics-informed loss with dynamic constraint weighting

## Quick Start

The following commands provide the repository's training, generation, and screening entry points.

```bash
# Clone the public repository
git clone https://github.com/NUSstronger/DFCA.git
cd DFCA

# Create an environment (install a CUDA-compatible PyTorch build
# for your system before or when installing the requirements)
conda create -n dfca python=3.10 -y
conda activate dfca
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Inspect supported arguments
python generate.py --help
python train.py --help
python screen.py --help
```

### Generate crystals

Generate crystal structures using a trained checkpoint:

```bash
PYTHONPATH=. python generate.py \
    --checkpoint /path/to/checkpoint.pt \
    --num_samples 100 \
    --batch_size 32 \
    --output_dir ./results
```

Generation supports TopK decoding, element filtering, and 2D lattice projection. Dataset splits are supplied with `--data_dir`.

### Train the model

Train DFCA with physics-informed constraints:

```bash
PYTHONPATH=. python train.py \
    --data_dir ./datasets \
    --output_dir ./output/dfca \
    --epochs 500 \
    --batch_size 256 \
    --seed 42 \
    --use_physical_constraints
```

Training uses a CUDA GPU.

### Screen generated structures

```bash
PYTHONPATH=. python screen.py \
    --input ./results \
    --output ./screened
```

Screening supports YAML configuration through `screen.py --config`. GPAW DFT validation is available through the separate M1–M4 workflow.

## Dataset

The training dataset contains curated 2D materials from established databases. See [`datasets/DATA_SOURCE.md`](datasets/DATA_SOURCE.md) for download instructions and citations.


## Related Work

- [CrystalDiT](https://doi.org/10.1609/aaai.v40i2.37121) – Diffusion Transformer for crystal generation; the foundation architecture extended by DFCA.
- [DiffCSP](https://arxiv.org/abs/2309.04475) – Joint equivariant diffusion for crystal structure prediction.
- [DiffCSP++](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1ba1b2133ac3d3dda0ce5cd67dab8456-Abstract-Conference.html) – Space-group-constrained crystal generation using diffusion models.
- [FlowMM](https://proceedings.mlr.press/v235/miller24a.html) – Riemannian flow matching for crystal generation.
- [CDVAE](https://openreview.net/forum?id=03RLpj-tc_) – Crystal diffusion variational autoencoder for periodic materials.


## License

See [`LICENSE`](LICENSE)
