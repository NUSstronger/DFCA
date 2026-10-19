"""
Screening Configuration dataclass.

All screening thresholds and parameters are defined here.
CRITICAL: These values are calibrated to C2DB standards and MUST NOT be modified.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# Default MAGMOM values per element (for VASP INCAR export)
# These are used when magnetic classification is nonmag
DEFAULT_MAGMOM: Dict[str, float] = {
    "H": 0.0, "He": 0.0, "Li": 0.0, "Be": 0.0, "B": 0.0, "C": 0.0, "N": 0.0,
    "O": 0.0, "F": 0.0, "Ne": 0.0, "Na": 0.0, "Mg": 0.0, "Al": 0.0, "Si": 0.0,
    "P": 0.0, "S": 0.0, "Cl": 0.0, "Ar": 0.0, "K": 0.0, "Ca": 0.0, "Sc": 2.0,
    "Ti": 2.0, "V": 2.0, "Cr": 2.0, "Mn": 2.0, "Fe": 2.5, "Co": 2.0,
    "Ni": 2.0, "Cu": 0.5, "Zn": 0.5, "Ga": 0.0, "Ge": 0.0, "As": 0.0,
    "Se": 0.0, "Br": 0.0, "Kr": 0.0, "Rb": 0.0, "Sr": 0.0, "Y": 2.0,
    "Zr": 2.0, "Nb": 2.0, "Mo": 2.0, "Tc": 2.0, "Ru": 2.0, "Rh": 2.0,
    "Pd": 0.5, "Ag": 0.5, "Cd": 0.5, "In": 0.0, "Sn": 0.0, "Sb": 0.0,
    "Te": 0.0, "I": 0.0, "Xe": 0.0, "Cs": 0.0, "Ba": 0.0, "La": 3.0,
    "Ce": 3.0, "Pr": 3.0, "Nd": 3.0, "Pm": 3.0, "Sm": 3.0, "Eu": 3.0,
    "Gd": 3.0, "Tb": 3.0, "Dy": 3.0, "Ho": 3.0, "Er": 3.0, "Tm": 3.0,
    "Yb": 3.0, "Lu": 3.0, "Hf": 2.0, "Ta": 2.0, "W": 2.0, "Re": 2.0,
    "Os": 2.0, "Ir": 2.0, "Pt": 0.5, "Au": 0.5, "Hg": 0.5, "Tl": 0.0,
    "Pb": 0.0, "Bi": 0.0, "Po": 0.0, "At": 0.0, "Rn": 0.0, "Fr": 0.0,
    "Ra": 0.0, "Ac": 3.0, "Th": 3.0, "Pa": 3.0, "U": 3.0, "Np": 3.0,
    "Pu": 3.0, "Am": 3.0,
}


@dataclass
class ScreeningConfig:
    """
    Configuration for 2D materials screening pipeline.

    All threshold values are calibrated to C2DB standards.
    DO NOT modify default values without scientific justification.

    Attributes:
        input_dir: Directory containing input .cif files
        output_dir: Directory for screening outputs
        train_csv: Path to training set CSV for novelty filtering
        device: Computing device for ML model ('cpu' or 'cuda')
        mace_model: MACE model size ('small', 'medium', 'large')
        vacuum_min_angstrom: Minimum vacuum gap (Å)
        slab_thickness_max: Maximum slab thickness (Å)
        min_bond_distance: Minimum allowed bond distance (Å)
        lattice_ab_min: Minimum lattice a/b parameter (Å)
        lattice_ab_max: Maximum lattice a/b parameter (Å)
        ltol: Lattice tolerance for StructureMatcher
        stol: Structure tolerance for StructureMatcher
        angle_tol: Angle tolerance for StructureMatcher (degrees)
        symprec: Symmetry detection precision (Å)
        relax_fmax: Maximum force for ML relaxation (eV/Å)
        relax_steps: Maximum steps for ML relaxation
        relax_steps_retry: Steps for retry relaxation
        fix_z_lattice: Fix c-axis during relaxation
        elastic_delta: Strain amplitude for elastic matrix calculation
        skip_phonon: Skip phonon stability check
        phonon_supercell: Supercell dimensions for phonon calculation
        phonon_displacement: Atomic displacement for phonons (Å)
        phonon_mesh: q-point mesh for phonon dispersion
        imaginary_freq_tol: Tolerance for imaginary frequencies (THz)
        ehull_threshold_ml: E_hull threshold for thermodynamic stability (eV/atom)
        enable_hard_ehull_gate: Enable strict E_hull gate (default: False, C2DB standard)
        mp_api_key: Materials Project API key for convex hull
        enable_md: Enable MD screening (default: False, optional per C2DB)
        md_temperature: MD temperature (K)
        md_dt_fs: MD timestep (fs)
        md_s7a_steps: Steps for S7a MD (3ps)
        md_s7b_steps: Steps for S7b MD (10ps)
        reject_p1: Reject Phase 1 candidates
        verbose: Enable verbose logging
        export_mag_buckets: Export VASP jobs by magnetic classification
        reference_energies: Precomputed element reference energies (eV/atom)
        save_interval: Checkpoint save interval (structures)
        vasp_vacuum_angstrom: Vacuum gap for VASP POSCAR export (Å)
        pareto_plot: Generate Pareto front plots
        pareto_plot_top_n: Number of top clusters to plot
        rerun_phase2: Reuse Phase 1 data, only rerun Phase 2
        force_rerun_all: Force rerun all records in Phase 2
    """

    # I/O
    input_dir: str = "./generated_crystals"
    output_dir: str = "./dft_ready"
    train_csv: str = ""

    # Device & ML
    device: str = "cpu"
    mace_model: str = "medium"

    # S1 Geometry thresholds
    vacuum_min_angstrom: float = 5.0  # Minimum vacuum gap
    slab_thickness_max: float = 15.0  # Maximum slab thickness
    min_bond_distance: float = 0.4  # Minimum bond distance
    lattice_ab_min: float = 1.0  # Min lattice a/b
    lattice_ab_max: float = 50.0  # Max lattice a/b

    # S2 Symmetry thresholds
    ltol: float = 0.30  # Lattice tolerance
    stol: float = 0.30  # Structure tolerance
    angle_tol: float = 10.0  # Angle tolerance (degrees)
    symprec: float = 0.1  # Symmetry precision

    # S3.5 Relaxation
    relax_fmax: float = 0.05  # Maximum force (eV/Å)
    relax_steps: int = 400  # Maximum steps
    relax_steps_retry: int = 800  # Retry steps
    fix_z_lattice: bool = True  # Fix c-axis

    # S4 Elastic Matrix
    elastic_delta: float = 0.005  # Strain amplitude

    # S5 Phonon
    skip_phonon: bool = False  # Skip phonon check
    phonon_supercell: List[int] = field(default_factory=lambda: [3, 3, 1])
    phonon_displacement: float = 0.03  # Displacement (Å)
    phonon_mesh: List[int] = field(default_factory=lambda: [20, 20, 1])
    imaginary_freq_tol: float = -0.02  # THz, C2DB Sec. 2.7

    # S5' Convex Hull
    ehull_threshold_ml: float = 0.2  # eV/atom
    enable_hard_ehull_gate: bool = False  # C2DB default: False
    mp_api_key: str = ""

    # S7 MD
    enable_md: bool = False  # Optional per C2DB
    md_temperature: float = 300.0  # K
    md_dt_fs: float = 1.0  # fs
    md_s7a_steps: int = 3000  # 3ps
    md_s7b_steps: int = 10000  # 10ps

    # Pipeline
    reject_p1: bool = False
    verbose: bool = True
    export_mag_buckets: bool = True
    reference_energies: Dict[str, float] = field(default_factory=dict)
    save_interval: int = 10
    vasp_vacuum_angstrom: float = 20.0  # VASP POSCAR vacuum

    # Pareto visualization
    pareto_plot: bool = False
    pareto_plot_top_n: int = 10

    # V3.2 Phase 2
    rerun_phase2: bool = False
    force_rerun_all: bool = False


__all__ = ['ScreeningConfig', 'DEFAULT_MAGMOM']
