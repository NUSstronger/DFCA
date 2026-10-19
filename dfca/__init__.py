"""
DFCA: Diffusion Framework for Crystal Anisotropy

A training framework for 2D slab crystal structure generation using discrete diffusion models.
"""

__version__ = "0.1.0"

# Configuration loading
from dfca.config import (
    load_config,
    load_config_with_cli,
    save_config,
    ConfigNamespace,
    TrainingConfig,
    GenerationConfig,
)

__all__ = [
    'load_config',
    'load_config_with_cli',
    'save_config',
    'ConfigNamespace',
    'TrainingConfig',
    'GenerationConfig',
]
