"""
DFCA Validation Utilities.

Helper functions for the GPAW validation pipeline.
"""

from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms
from dfca.validation.utils.kpoints import adaptive_kpoints
from dfca.validation.utils.io import scan_output_directory

# GPAW utilities (migrated from original gpaw_validation.py)
from dfca.validation.utils.gpaw_utils import (
    PW,
    FermiDirac,
    _check_dependencies,
    _timer,
    _find_structure_file,
    compute_dft_energy_per_atom,
    generate_band_qpoints,
    get_2d_high_symmetry_path,
    print_elastic_analysis,
    parameters_to_fc,
    spectrum_from_fc,
)

__all__ = [
    'classify_magnetic',
    'build_initial_magmoms',
    'adaptive_kpoints',
    'scan_output_directory',
    # GPAW utilities
    'PW',
    'FermiDirac',
    '_check_dependencies',
    '_timer',
    '_find_structure_file',
    'compute_dft_energy_per_atom',
    'generate_band_qpoints',
    'get_2d_high_symmetry_path',
    'print_elastic_analysis',
    'parameters_to_fc',
    'spectrum_from_fc',
]
