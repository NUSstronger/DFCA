"""
Crystal Generation Orchestration.

This module handles:
    - Model loading with checkpoint and EMA support
    - Diffusion model initialization
    - Batch generation with sampling loop
    - Multi-GPU parallel generation
    - Argument preparation from checkpoint

IMPORTANT: This code is migrated directly from the original generate_crystals.py.
Scientific behavior must be preserved exactly.
"""

import os
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

# Authoritative imports (Stage 9 tested)
from dfca.models.dit import CrystalDiT
from dfca.diffusion.gaussian import CrystalGaussianDiffusion
from dfca.generation.topk_decoder import (
    build_atom_count_prior,
    sample_target_counts,
    convert_to_structure_topk,
    summarize_pmf,
)

from dfca.generation.structure import (
    convert_to_structure,
    save_structures,
    analyze_generated_structures,
)
from dfca.generation.decoder import load_allowed_atomic_numbers


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path, max_atoms, device, hidden_size=512, depth=18, num_heads=8):
    """
    Load CrystalDiT model from checkpoint.

    Handles both regular checkpoints and EMA (Exponential Moving Average) checkpoints.
    EMA models often produce better quality generations.

    The model is loaded in eval mode with all gradient computation disabled.

    Args:
        checkpoint_path: Path to PyTorch checkpoint file
        max_atoms: Maximum number of atoms (determines model input size)
        device: Target device for model placement
        hidden_size: Model hidden dimension (default 512)
        depth: Number of DiT layers (default 18)
        num_heads: Number of attention heads (default 8)

    Returns:
        Tuple of (model, checkpoint_args)
    """
    model = CrystalDiT(
        max_atoms=max_atoms,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DDP checkpoint format (keys prefixed with 'module.')
    base_dict = checkpoint['model']
    if 'module.' in list(base_dict.keys())[0]:
        clean_base_dict = {k.replace('module.', ''): v for k, v in base_dict.items()}
    else:
        clean_base_dict = base_dict
    model.load_state_dict(clean_base_dict, strict=False)

    # Load EMA shadow weights if available (usually better for generation)
    if 'ema_shadow' in checkpoint:
        shadow_dict = checkpoint['ema_shadow']
        if 'module.' in list(shadow_dict.keys())[0]:
            clean_shadow_dict = {k.replace('module.', ''): v for k, v in shadow_dict.items()}
        else:
            clean_shadow_dict = shadow_dict
        model.load_state_dict(clean_shadow_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model, checkpoint.get('args', None)


# ============================================================================
# Diffusion Model Setup
# ============================================================================

def build_diffusion_from_checkpoint(ckpt_args, diffusion_steps, beta_schedule="linear"):
    """
    Create CrystalGaussianDiffusion model with parameters from checkpoint.

    Preserves training noise scales to ensure generated structures match
    the training distribution. Warns if checkpoint is missing these parameters.

    Args:
        ckpt_args: Arguments from training checkpoint (may be None)
        diffusion_steps: Number of diffusion timesteps for sampling
        beta_schedule: Noise schedule type (default "linear")

    Returns:
        CrystalGaussianDiffusion instance
    """
    # Default values must match training defaults exactly
    fallback_z_noise_scale = 0.113001
    fallback_lattice_c_noise_scale = 0.340659

    if ckpt_args is not None:
        z_noise_scale = getattr(ckpt_args, 'z_noise_scale', fallback_z_noise_scale)
        lattice_c_noise_scale = getattr(ckpt_args, 'lattice_c_noise_scale', fallback_lattice_c_noise_scale)
        if not hasattr(ckpt_args, 'z_noise_scale') or not hasattr(ckpt_args, 'lattice_c_noise_scale'):
            print(
                "[WARNING] checkpoint args missing z_noise_scale / lattice_c_noise_scale; "
                f"falling back to defaults ({fallback_z_noise_scale}, {fallback_lattice_c_noise_scale}). "
                "If these differed from training, generated structures may not match the training distribution."
            )
    else:
        z_noise_scale = fallback_z_noise_scale
        lattice_c_noise_scale = fallback_lattice_c_noise_scale
        print(
            "[WARNING] No training args in checkpoint; cannot verify "
            f"z_noise_scale / lattice_c_noise_scale. Using defaults "
            f"({fallback_z_noise_scale}, {fallback_lattice_c_noise_scale})."
        )

    return CrystalGaussianDiffusion(
        timesteps=diffusion_steps,
        beta_schedule=beta_schedule,
        z_noise_scale=z_noise_scale,
        lattice_c_noise_scale=lattice_c_noise_scale,
    )


# ============================================================================
# Argument Preparation
# ============================================================================

def prepare_generation_args(args):
    """
    Prepare generation arguments from CLI args and checkpoint.

    Extracts dataset information from checkpoint if not provided,
    loads allowed elements from dataset, and builds atom count prior
    for top-k decoding.

    Args:
        args: Namespace with CLI arguments

    Returns:
        Augmented args namespace with additional fields
    """
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", None)

    # Use data_dir from checkpoint if not provided on command line
    if not args.data_dir and checkpoint_args is not None and hasattr(checkpoint_args, "data_dir"):
        args.data_dir = checkpoint_args.data_dir

    # Load allowed elements from dataset
    args.allowed_atomic_numbers = None
    if not args.disable_dataset_element_filter:
        allowed_atomic_numbers = load_allowed_atomic_numbers(args.data_dir)
        if allowed_atomic_numbers:
            args.allowed_atomic_numbers = sorted(allowed_atomic_numbers)
            print(f"Loaded {len(args.allowed_atomic_numbers)} allowed elements from dataset")

    # Build atom count prior for top-k decoder
    args.atom_count_pmf = None
    if args.use_topk_decoder:
        if not args.data_dir:
            raise ValueError("--use_topk_decoder requires --data_dir")
        pmf, raw_counts = build_atom_count_prior(
            args.data_dir,
            max_atoms=args.max_atoms,
            max_length=args.max_length,
            split=args.topk_split,
        )
        args.atom_count_pmf = pmf
        stats = summarize_pmf(pmf, raw_counts)
        print(
            f"[topk] Loaded atom_count prior from '{args.topk_split}' split: "
            f"n={stats['n_samples']}, mean={stats['mean']:.2f}, "
            f"median={stats['median']:.1f}, support={stats['support']}, "
            f"p10={stats['p10']:.0f}, p90={stats['p90']:.0f}"
        )

    return args


# ============================================================================
# GPU Utilities
# ============================================================================

def get_available_gpus():
    """Get list of available CUDA devices."""
    if not torch.cuda.is_available():
        return []
    gpu_count = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(gpu_count)]


# ============================================================================
# Single GPU Generation
# ============================================================================

def generate_crystals_on_gpu(gpu_id, args, samples_per_gpu, start_idx):
    """
    Generate crystal structures on a single GPU.

    This function is called by multiprocessing for parallel generation.
    Each process handles a subset of samples independently.

    Args:
        gpu_id: CUDA device identifier
        args: Generation arguments namespace
        samples_per_gpu: Number of samples to generate on this GPU
        start_idx: Starting index for file naming
    """
    device = torch.device(gpu_id)

    process_id = gpu_id.split(":")[-1]
    print(f"[GPU {process_id}] Starting generation of {samples_per_gpu} crystal structures on {gpu_id}...")

    # Load model and diffusion
    model, ckpt_args = load_model(
        args.checkpoint, args.max_atoms, device,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads
    )
    diffusion = build_diffusion_from_checkpoint(ckpt_args, args.diffusion_steps)

    allowed_atomic_numbers = set(args.allowed_atomic_numbers) if args.allowed_atomic_numbers else None
    all_structures = []
    num_batches = (samples_per_gpu + args.batch_size - 1) // args.batch_size

    # Setup RNG for top-k decoder
    topk_rng = None
    if args.use_topk_decoder and args.atom_count_pmf is not None:
        try:
            gpu_idx = int(process_id)
        except ValueError:
            gpu_idx = 0
        topk_rng = np.random.default_rng(args.topk_seed + gpu_idx * 9973)

    for i in tqdm(range(num_batches), desc=f"GPU {process_id} Generation Progress"):
        current_batch_size = min(args.batch_size, samples_per_gpu - i * args.batch_size)
        if current_batch_size <= 0:
            break

        with torch.no_grad():
            lattice_vectors, atom_features = diffusion.p_sample_loop(
                model,
                batch_size=current_batch_size,
                max_atoms=args.max_atoms,
                device=device,
                progress=False
            )

        # Convert to structures using appropriate decoder
        if args.use_topk_decoder and args.atom_count_pmf is not None:
            target_counts = sample_target_counts(
                args.atom_count_pmf, current_batch_size, rng=topk_rng
            )
            # Convert fractional slab_half_width to Ångström before passing to topk decoder.
            topk_slab_half_width_ang = args.slab_half_width * args.min_c_length
            structures = convert_to_structure_topk(
                lattice_vectors=lattice_vectors,
                atom_features=atom_features,
                target_counts=target_counts,
                max_length=args.max_length,
                filter_rare_gases=args.filter_rare_gases,
                allowed_atomic_numbers=allowed_atomic_numbers,
                element_sigma=args.element_sigma,
                slab_center=args.slab_center,
                slab_half_width_ang=topk_slab_half_width_ang,
                coord_tolerance=args.coord_tolerance,
                project_2d_lattice=not args.disable_2d_lattice_projection,
                min_c_length=args.min_c_length,
                max_c_length=args.max_c_length,
                gamma_min_deg=args.gamma_min_deg,
                gamma_max_deg=args.gamma_max_deg,
                max_unique_elements=args.max_unique_elements,
                forbid_atomic_numbers=args.forbid_atomic_numbers,
                coord_oor_penalty=args.coord_oor_penalty,
                slab_penalty=args.slab_penalty,
                min_valid_element_prob=args.min_valid_element_prob,
            )
        else:
            structures = convert_to_structure(
                lattice_vectors=lattice_vectors,
                atom_features=atom_features,
                max_length=args.max_length,
                filter_rare_gases=args.filter_rare_gases,
                allowed_atomic_numbers=allowed_atomic_numbers,
                coord_tolerance=args.coord_tolerance,
                slab_center=args.slab_center,
                slab_half_width=args.slab_half_width,
                element_sigma=args.element_sigma,
                min_valid_element_prob=args.min_valid_element_prob,
                project_2d_lattice=not args.disable_2d_lattice_projection,
                min_c_length=args.min_c_length,
                max_c_length=args.max_c_length,
                gamma_min_deg=args.gamma_min_deg,
                gamma_max_deg=args.gamma_max_deg,
                max_unique_elements=args.max_unique_elements,
                forbid_atomic_numbers=args.forbid_atomic_numbers,
            )

        all_structures.extend(structures)

    print(f"[GPU {process_id}] Successfully generated {len(all_structures)} crystal structures")

    # Analyze on GPU 0
    if process_id == '0':
        analyze_generated_structures(all_structures)

    # Save all structures
    save_count = save_structures(all_structures, args.output_dir, start_idx=start_idx)

    return save_count


def generate_crystals_single_device(args):
    """
    Generate crystal structures on a single device (GPU or CPU).

    This is the main generation function for single-device operation.
    It handles the complete generation pipeline including model loading,
    batch processing, structure conversion, and output.

    Args:
        args: Generation arguments namespace

    Returns:
        List of generated pymatgen Structure objects
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model, ckpt_args = load_model(
        args.checkpoint, args.max_atoms, device,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads
    )

    diffusion = build_diffusion_from_checkpoint(ckpt_args, args.diffusion_steps)

    allowed_atomic_numbers = set(args.allowed_atomic_numbers) if args.allowed_atomic_numbers else None
    all_structures = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    # Setup RNG for top-k decoder
    topk_rng = np.random.default_rng(args.topk_seed) if args.use_topk_decoder else None

    print(f"Starting generation of {args.num_samples} crystal structures...")
    for i in tqdm(range(num_batches)):
        current_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)

        with torch.no_grad():
            lattice_vectors, atom_features = diffusion.p_sample_loop(
                model,
                batch_size=current_batch_size,
                max_atoms=args.max_atoms,
                device=device,
                progress=True
            )

        # Convert to structures using appropriate decoder
        if args.use_topk_decoder and args.atom_count_pmf is not None:
            target_counts = sample_target_counts(
                args.atom_count_pmf, current_batch_size, rng=topk_rng
            )
            # Convert fractional slab_half_width to Ångström before passing to topk decoder.
            topk_slab_half_width_ang = args.slab_half_width * args.min_c_length
            structures = convert_to_structure_topk(
                lattice_vectors=lattice_vectors,
                atom_features=atom_features,
                target_counts=target_counts,
                max_length=args.max_length,
                filter_rare_gases=args.filter_rare_gases,
                allowed_atomic_numbers=allowed_atomic_numbers,
                element_sigma=args.element_sigma,
                slab_center=args.slab_center,
                slab_half_width_ang=topk_slab_half_width_ang,
                coord_tolerance=args.coord_tolerance,
                project_2d_lattice=not args.disable_2d_lattice_projection,
                min_c_length=args.min_c_length,
                max_c_length=args.max_c_length,
                gamma_min_deg=args.gamma_min_deg,
                gamma_max_deg=args.gamma_max_deg,
                max_unique_elements=args.max_unique_elements,
                forbid_atomic_numbers=args.forbid_atomic_numbers,
                coord_oor_penalty=args.coord_oor_penalty,
                slab_penalty=args.slab_penalty,
                min_valid_element_prob=args.min_valid_element_prob,
            )
        else:
            structures = convert_to_structure(
                lattice_vectors=lattice_vectors,
                atom_features=atom_features,
                max_length=args.max_length,
                filter_rare_gases=args.filter_rare_gases,
                allowed_atomic_numbers=allowed_atomic_numbers,
                coord_tolerance=args.coord_tolerance,
                slab_center=args.slab_center,
                slab_half_width=args.slab_half_width,
                element_sigma=args.element_sigma,
                min_valid_element_prob=args.min_valid_element_prob,
                project_2d_lattice=not args.disable_2d_lattice_projection,
                min_c_length=args.min_c_length,
                max_c_length=args.max_c_length,
                gamma_min_deg=args.gamma_min_deg,
                gamma_max_deg=args.gamma_max_deg,
                max_unique_elements=args.max_unique_elements,
                forbid_atomic_numbers=args.forbid_atomic_numbers,
            )

        all_structures.extend(structures)

    print(f"Successfully generated {len(all_structures)} crystal structures")

    analyze_generated_structures(all_structures)
    save_structures(all_structures, args.output_dir)

    return all_structures


# ============================================================================
# Multi-GPU Parallel Generation
# ============================================================================

def run_parallel_generation(args):
    """
    Run parallel generation on multiple GPUs using multiprocessing.

    Spawns separate processes for each GPU, dividing the total samples
    evenly (with remainder assigned to the last GPU).

    Args:
        args: Generation arguments namespace
    """
    available_gpus = get_available_gpus()

    if not available_gpus:
        print("No available GPUs detected, using CPU generation")
        device = torch.device("cpu")
        args.device = "cpu"
        return generate_crystals_single_device(args)

    print(f"Detected {len(available_gpus)} available GPUs: {available_gpus}")

    os.makedirs(args.output_dir, exist_ok=True)

    total_samples = args.num_samples
    samples_per_gpu = total_samples // len(available_gpus)
    remainder = total_samples % len(available_gpus)

    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    processes = []
    start_idx = 0

    for i, gpu_id in enumerate(available_gpus):
        # Distribute remainder to last GPU
        if i == len(available_gpus) - 1:
            gpu_samples = samples_per_gpu + remainder
        else:
            gpu_samples = samples_per_gpu

        p = mp.Process(
            target=generate_crystals_on_gpu,
            args=(gpu_id, args, gpu_samples, start_idx)
        )

        processes.append(p)
        start_idx += gpu_samples

    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"All GPU work completed, generated structures saved to {args.output_dir}")
