#!/usr/bin/env python
"""
CrystalDiT Generation Script - Entry Point.

This is the main entry point for generating crystal structures using the CrystalDiT model
with physical constraints for 2D slab crystal generation.

The script handles:
    - Argument parsing and configuration
    - GPU detection
    - Model and checkpoint loading
    - Spawning parallel generation workers

All core generation logic is delegated to dfca.generation.generator.
"""

import argparse
import torch

from dfca.generation.generator import (
    prepare_generation_args,
    run_parallel_generation,
    generate_crystals_single_device,
)


def parse_args():
    """Parse command-line arguments for generation configuration."""
    parser = argparse.ArgumentParser(
        description='Generate crystal structures using Crystal DiT model for 2D slab generation.'
    )

    # Model and checkpoint parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--max_atoms", type=int, default=20,
                        help="Maximum number of atoms per structure (determines padding size)")
    parser.add_argument("--max_length", type=float, default=46.7425,
                        help="Maximum lattice vector length for normalization")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Dataset directory used to infer allowed elements")

    # Model architecture parameters
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden dimension size for Transformer layers")
    parser.add_argument("--depth", type=int, default=18,
                        help="Number of Transformer layers (DiT depth)")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads per layer")

    # Diffusion parameters
    parser.add_argument("--diffusion_steps", type=int, default=1000,
                        help="Number of diffusion timesteps for sampling")

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per generation step")
    parser.add_argument("--output_dir", type=str, default="./generated_crystals",
                        help="Output directory for CIF files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--use_multi_gpu", action="store_true",
                        help="Use multi-GPU parallel generation")

    # Element filtering parameters
    parser.add_argument("--filter_rare_gases", action="store_true", default=True,
                        help="Filter rare gas elements from generated structures")
    parser.add_argument("--disable_dataset_element_filter", action="store_true",
                        help="Do not restrict generated elements to those seen in the dataset")

    # Coordinate and geometry parameters
    parser.add_argument("--coord_tolerance", type=float, default=0.05,
                        help="Allow small coordinate overflow before rejecting an atom")
    parser.add_argument("--slab_center", type=float, default=0.5,
                        help="Expected fractional z center of the 2D slab")
    parser.add_argument("--slab_half_width", type=float, default=0.15,
                        help="Keep only atoms within this wrapped z-distance from slab_center")
    parser.add_argument("--element_sigma", type=float, default=0.05,
                        help="Gaussian sigma for element decoding (sharpness of row/column mapping)")
    parser.add_argument("--min_valid_element_prob", type=float, default=1e-3,
                        help="Minimum decoding probability for accepting an atom")

    # 2D lattice projection parameters
    parser.add_argument("--disable_2d_lattice_projection", action="store_true",
                        help="Do not project the generated lattice back to a 2D-friendly form")
    parser.add_argument("--min_c_length", type=float, default=18.0,
                        help="Minimum c-axis length (vacuum layer thickness) in Angstrom")
    parser.add_argument("--max_c_length", type=float, default=35.0,
                        help="Maximum c-axis length cap to avoid abnormally large vacuum layers")
    parser.add_argument("--gamma_min_deg", type=float, default=60.0,
                        help="Minimum in-plane gamma angle after 2D projection")
    parser.add_argument("--gamma_max_deg", type=float, default=120.0,
                        help="Maximum in-plane gamma angle after 2D projection")

    # Element diversity and forbidden parameters
    parser.add_argument("--max_unique_elements", type=int, default=4,
                        help="Keep at most this many element types per generated structure")
    parser.add_argument("--forbid_atomic_numbers", type=int, nargs="*",
                        default=[84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94],
                        help="Forbidden atomic numbers during decoding")

    # Debug/disable postprocessing parameters
    parser.add_argument("--disable_all_postprocess", action="store_true",
                        help="Disable ALL post-processing to evaluate raw model output")
    parser.add_argument("--coord_oor_penalty", type=float, default=100.0,
                        help="Penalty score for out-of-range coordinates in topk decoder")
    parser.add_argument("--slab_penalty", type=float, default=10.0,
                        help="Penalty score for slab boundary violation in topk decoder")

    # Top-k decoder parameters
    parser.add_argument("--use_topk_decoder", action="store_true",
                        help="Use atom_count prior + occupancy top-k filtering instead of "
                             "per-slot threshold filtering")
    parser.add_argument("--topk_split", type=str, default="train",
                        help="Dataset split used to build the atom_count prior")
    parser.add_argument("--topk_seed", type=int, default=20260504,
                        help="Base seed for sampling target atom counts")

    return parser.parse_args()


def main():
    """Main entry point: parse arguments and run generation."""
    args = parse_args()

    # Apply all postprocessing disable flags if requested
    if args.disable_all_postprocess:
        args.disable_2d_lattice_projection = True
        args.slab_half_width = 1.0
        args.coord_tolerance = 1.0
        args.max_unique_elements = None
        args.min_valid_element_prob = None   # Disable confidence filtering
        args.filter_rare_gases = False
        args.forbid_atomic_numbers = []
        args.coord_oor_penalty = 0.0         # Disable coordinate penalty
        args.slab_penalty = 0.0             # Disable slab penalty
        print("[WARNING] All post-processing DISABLED - evaluating raw model output")

    # Prepare arguments (load from checkpoint, etc.)
    args = prepare_generation_args(args)

    # Run generation (multi-GPU or single device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.use_multi_gpu:
        run_parallel_generation(args)
    else:
        generate_crystals_single_device(args)


if __name__ == "__main__":
    main()
