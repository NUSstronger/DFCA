"""
DFCA Validation Magnetic Module.

Functions for magnetic ground state comparison and magnetic moment handling.
"""

from dfca.validation.magnetic.comparison import compare_magnetic_ground_states
from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms

__all__ = [
    'compare_magnetic_ground_states',
    'classify_magnetic',
    'build_initial_magmoms',
]
