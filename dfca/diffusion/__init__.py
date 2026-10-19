"""
DFCA Diffusion Module.

Provides Gaussian diffusion process with XY/Z anisotropic treatment for crystal structures.

Submodules:
    - gaussian: CrystalGaussianDiffusion implementation
"""

from dfca.diffusion.gaussian import (
    CrystalGaussianDiffusion,
    ATOM_XY_FEATURE_WEIGHTS,
    ATOM_Z_LOSS_WEIGHT,
    ATOM_TOTAL_LOSS_COMBINE_WEIGHT,
)

__all__ = [
    'CrystalGaussianDiffusion',
    'ATOM_XY_FEATURE_WEIGHTS',
    'ATOM_Z_LOSS_WEIGHT',
    'ATOM_TOTAL_LOSS_COMBINE_WEIGHT',
]
