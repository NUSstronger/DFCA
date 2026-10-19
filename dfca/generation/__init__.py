"""
DFCA Generation Module.

This module provides crystal generation functionality using the CrystalDiT model
with physical constraints for 2D slab crystal generation.

Submodules:
    - decoder: Element decoding from model outputs
    - structure: Structure construction and I/O
    - generator: Main generation orchestration
"""

from dfca.generation.decoder import (
    map_to_element,
    load_allowed_atomic_numbers,
    RARE_GAS_ELEMENTS,
)

from dfca.generation.structure import (
    project_lattice_to_2d,
    convert_to_structure,
    save_structures,
    analyze_generated_structures,
    periodic_distance,
    vector_angle_degrees,
)

from dfca.generation.generator import (
    load_model,
    build_diffusion_from_checkpoint,
    prepare_generation_args,
    generate_crystals_single_device,
    generate_crystals_on_gpu,
    run_parallel_generation,
    get_available_gpus,
)

__all__ = [
    # decoder
    "map_to_element",
    "load_allowed_atomic_numbers",
    "RARE_GAS_ELEMENTS",
    # structure
    "project_lattice_to_2d",
    "convert_to_structure",
    "save_structures",
    "analyze_generated_structures",
    "periodic_distance",
    "vector_angle_degrees",
    # generator
    "load_model",
    "build_diffusion_from_checkpoint",
    "prepare_generation_args",
    "generate_crystals_single_device",
    "generate_crystals_on_gpu",
    "run_parallel_generation",
    "get_available_gpus",
]
