"""
DFCA Validation Electronic Module.

Functions for electronic structure calculations including:
- Band gap probing with fixed-density method
- Spacegroup analysis
- Adaptive electronic parameters based on band gap
- k-point mesh generation
"""

from dfca.validation.electronic.gap import probe_electronic_gap
from dfca.validation.electronic.symmetry import get_spacegroup_info
from dfca.validation.electronic.params import (
    adaptive_electronic_params,
    adaptive_kpoints,
    adaptive_kpoints_for_supercell,
)

__all__ = [
    'probe_electronic_gap',
    'get_spacegroup_info',
    'adaptive_electronic_params',
    'adaptive_kpoints',
    'adaptive_kpoints_for_supercell',
]
