#!/usr/bin/env python
"""
CrystalDiT Training Script - Entry Point.

This is the main entry point for training the CrystalDiT model with physical constraints
for 2D slab crystal structure generation.

The script handles:
    - Argument parsing and configuration
    - GPU count detection
    - Spawning distributed training workers

All core training logic is delegated to dfca.training.trainer.
"""

# Set deterministic environment variables before any torch/cuda initialization.
# Must be set before 'import torch' so that torch initializes with these settings.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')

import argparse
import torch
import torch.multiprocessing as mp

from dfca.training.trainer import train


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Train Crystal DiT model with physical constraints for 2D slab crystal generation.'
    )

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./datasets/mp_20',
                        help='Dataset directory containing train.csv and val.csv')
    parser.add_argument('--max_atoms', type=int, default=20,
                        help='Maximum number of atoms per structure (determines padding size)')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='Directory for preprocessed data cache (default: <data_dir>/preprocessed)')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='Force re-preprocessing of dataset even if cached files exist')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden dimension size for Transformer layers')
    parser.add_argument('--depth', type=int, default=18,
                        help='Number of Transformer layers (DiT depth)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads per layer')

    # Diffusion parameters
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion timesteps (T)')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                        help='Beta noise schedule type (linear, cosine, etc.)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.000203886,
                        help='Peak learning rate after warmup')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                        help='Minimum learning rate (cosine annealing floor)')
    parser.add_argument('--weight_decay', type=float, default=0.0468414,
                        help='Weight decay (L2 regularization coefficient)')
    parser.add_argument('--clip_grad', type=float, default=2.803345,
                        help='Gradient norm clipping threshold (0 to disable)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps before optimizer update')
    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='[DEPRECATED] Warmup is built into SequentialLR(LinearLR + CosineAnnealingWarmRestarts)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                        help='Epochs without improvement before early stopping')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes (0 = main process)')

    # Checkpoint parameters
    parser.add_argument('--output_dir', type=str, default='./output/crystal_dit',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_every', type=int, default=250,
                        help='Save periodic checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log training metrics every N optimizer steps')

    # Physical constraints parameters
    parser.add_argument('--use_physical_constraints', action='store_true', default=False,
                        help='Enable physical constraint losses (atom count, lattice geometry, density, etc.)')
    parser.add_argument('--slab_center', type=float, default=0.5,
                        help='Expected fractional z-coordinate center of the 2D slab')
    parser.add_argument('--min_angle_deg', type=float, default=60.0,
                        help='Minimum valid lattice angle (gamma, a-b plane)')
    parser.add_argument('--max_angle_deg', type=float, default=125.0,
                        help='Maximum valid lattice angle (gamma, a-b plane)')
    parser.add_argument('--min_angle_ab_deg', type=float, default=80.0,
                        help='Minimum valid angle for alpha/beta (a-c and b-c); must be near 90 for 2D slab')
    parser.add_argument('--max_angle_ab_deg', type=float, default=100.0,
                        help='Maximum valid angle for alpha/beta (a-c and b-c); must be near 90 for 2D slab')
    parser.add_argument('--rowcol_temperature', type=float, default=0.0604,
                        help='Temperature for soft valid/invalid separation in row-column space')
    parser.add_argument('--invalid_slot_weight', type=float, default=0.3,
                        help='Constraint weight: padded invalid atom slots')
    parser.add_argument('--soft_atom_count_weight', type=float, default=0.111,
                        help='Constraint weight: soft atom count alignment')
    parser.add_argument('--rowcol_validity_weight', type=float, default=0.219407,
                        help='Constraint weight: row/column validity (proximity to valid periodic table positions)')
    parser.add_argument('--slab_center_weight', type=float, default=0.1,
                        help='Constraint weight: slab z-center alignment')
    parser.add_argument('--slab_spread_weight', type=float, default=0.108,
                        help='Constraint weight: slab z-direction spread (physical thickness)')
    parser.add_argument('--lattice_len_weight', type=float, default=0.260292,
                        help='Constraint weight: lattice vector length alignment')
    parser.add_argument('--lattice_angle_weight', type=float, default=0.1,
                        help='Constraint weight: lattice angle constraints')
    parser.add_argument('--elem_count_dist_weight', type=float, default=0.15,
                        help='Constraint weight: element count distribution (unique element types per structure)')
    parser.add_argument('--density_weight', type=float, default=0.1,
                        help='Constraint weight: mass density alignment (g/cm3)')
    parser.add_argument('--c_length_floor_weight', type=float, default=0.1,
                        help='Constraint weight: c-axis floor (minimum vacuum layer thickness)')
    parser.add_argument('--c_length_ceiling_weight', type=float, default=0.15,
                        help='Constraint weight: c-axis ceiling (maximum vacuum layer thickness)')
    parser.add_argument('--rowcol_active_threshold', type=float, default=0.403809,
                        help='Timestep fraction threshold above which row/column validity constraint is active')
    parser.add_argument('--lattice_quad_power', type=int, default=4,
                        help='Exponent for quadratic timestep weighting of constraint losses')

    # Noise parameters
    parser.add_argument('--z_noise_scale', type=float, default=0.113001,
                        help='Noise scale for z-direction (c-axis) atom coordinates')
    parser.add_argument('--lattice_c_noise_scale', type=float, default=0.340659,
                        help='Noise scale for lattice c-axis vector components')
    parser.add_argument('--lattice_loss_z_weight', type=float, default=0.706323,
                        help='Relative weight for z-axis components in lattice vector loss')

    # Decay parameters
    parser.add_argument('--decay_mid', type=float, default=0.224396,
                        help='Constraint loss decay factor for mid-stage (progress > 0.30)')
    parser.add_argument('--decay_late', type=float, default=0.122513,
                        help='Constraint loss decay factor for late-stage (progress > 0.70)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    """Main entry point: parse arguments and spawn distributed training."""
    args = parse_args()

    world_size = torch.cuda.device_count()
    assert world_size >= 1, "At least one GPU is required for training"
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
