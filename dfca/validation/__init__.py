"""
DFCA Validation Module.

First-principles DFT validation pipeline using GPAW for 2D materials.

Screening Stages:
    M1: Structure relaxation (FIRE/BFGS with 2D constraints)
    M1+: Electronic gap probe (adaptive parameters)
    M2: Elastic constants (Born stability)
    M3: Phonon dispersion (phonopy + GPAW, ZA/Huang rescue)
    M4: Convex hull (Materials Project API)

Features:
    - Adaptive electronic parameters based on band gap
    - Branch-resolved phonon stability criteria
    - Fine-grained checkpoint with artifact recovery
    - PBS walltime recovery support

Module Structure:
    validation/
    ├── __init__.py           # Main exports
    ├── config.py             # GPAWConfig dataclass
    ├── record.py             # ValidationRecord dataclass
    ├── constants.py          # Physical constants
    ├── mpi.py                # MPI utilities
    ├── pipeline.py           # GPAWValidationPipeline
    ├── calculators.py        # GPAW calculator builder
    ├── metadata.py           # Screen metadata utilities
    ├── io.py                 # Structure I/O utilities
    ├── electronic/           # Electronic structure
    │   ├── gap.py           # Band gap probing
    │   ├── params.py        # Adaptive parameters
    │   └── symmetry.py      # Spacegroup analysis
    ├── magnetic/             # Magnetic calculations
    │   └── comparison.py    # FM/NM ground state
    ├── mechanical/           # Elastic properties
    │   └── elastic.py       # 2D elastic constants
    ├── relaxation/           # Structure relaxation
    │   └── relax_2d.py     # 2D relaxation
    ├── phonon/              # Phonon calculations
    │   ├── branches.py     # Branch identification
    │   ├── dispersion.py   # Phonon dispersion
    │   ├── stability.py    # Stability analysis
    │   └── plotting.py     # Phonon plotting
    ├── thermodynamics/       # Thermodynamic stability
    │   └── ehull.py        # Convex hull
    └── utils/               # Utility functions
        ├── magnetic.py      # Magnetic classification
        ├── kpoints.py      # k-point generation
        └── io.py           # I/O helpers
"""

# Core configuration and records
from dfca.validation.config import GPAWConfig
from dfca.validation.record import ValidationRecord
from dfca.validation.constants import MAG_3D, MAG_4D5D, MAG_RE, MAGMOM_REF
from dfca.validation.calculators import GPAW_DEFAULTS

# Pipeline
from dfca.validation.pipeline import GPAWValidationPipeline

# MPI utilities
from dfca.validation.mpi import (
    MPI_RANK,
    MPI_SIZE,
    mpi_broadcast_object,
    mpi_broadcast_numpy,
    mpi_world_sum,
    mpi_trace,
)

# Electronic structure
from dfca.validation.electronic import (
    probe_electronic_gap,
    get_spacegroup_info,
    adaptive_electronic_params,
    adaptive_kpoints,
    adaptive_kpoints_for_supercell,
)

# Magnetic calculations
from dfca.validation.magnetic import (
    compare_magnetic_ground_states,
    classify_magnetic,
    build_initial_magmoms,
)

# Elastic constants
from dfca.validation.mechanical import compute_elastic_matrix_2d_gpaw

# Structure relaxation
from dfca.validation.relaxation import relax_structure_2d

# Phonon calculations
from dfca.validation.phonon import (
    compute_phonon_dispersion_gpaw,
    get_adaptive_supercell,
    identify_phonon_branches,
    detect_za_rotational_rescue_candidate,
    _try_huang_za_rescue,
    _evaluate_phonon_stability_spectrum,
)

# Thermodynamic stability
from dfca.validation.thermodynamics import (
    compute_ehull_stability,
    compute_ehull_stability_v2,
)

# Utilities
from dfca.validation.utils import (
    classify_magnetic,
    build_initial_magmoms,
    adaptive_kpoints,
)

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

# Metadata utilities
from dfca.validation.metadata import (
    load_screen_report_index,
    get_screen_metadata,
)

# I/O utilities
from dfca.validation.io import (
    read_structure,
    scan_output_directory,
)

__all__ = [
    # Core
    'GPAWConfig',
    'ValidationRecord',
    'GPAW_DEFAULTS',
    'MAG_3D',
    'MAG_4D5D',
    'MAG_RE',
    'MAGMOM_REF',
    # Pipeline
    'GPAWValidationPipeline',
    # MPI
    'MPI_RANK',
    'MPI_SIZE',
    'mpi_broadcast_object',
    'mpi_broadcast_numpy',
    'mpi_world_sum',
    'mpi_trace',
    # Electronic
    'probe_electronic_gap',
    'get_spacegroup_info',
    'adaptive_electronic_params',
    'adaptive_kpoints',
    'adaptive_kpoints_for_supercell',
    # Magnetic
    'compare_magnetic_ground_states',
    'classify_magnetic',
    'build_initial_magmoms',
    # Mechanical
    'compute_elastic_matrix_2d_gpaw',
    # Relaxation
    'relax_structure_2d',
    # Phonon
    'compute_phonon_dispersion_gpaw',
    'get_adaptive_supercell',
    'identify_phonon_branches',
    'detect_za_rotational_rescue_candidate',
    '_try_huang_za_rescue',
    '_evaluate_phonon_stability_spectrum',
    # Thermodynamic
    'compute_ehull_stability',
    'compute_ehull_stability_v2',
    # Metadata
    'load_screen_report_index',
    'get_screen_metadata',
    # I/O
    'read_structure',
    'scan_output_directory',
    # GPAW utilities (migrated from original gpaw_validation.py)
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
