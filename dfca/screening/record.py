"""
Screening Record dataclass.

Tracks all screening stage outcomes for a single structure.
v3.2 C2DB Standard - complete field inventory.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ScreeningRecord:
    """
    Complete screening record for a single structure.

    Tracks outcomes of all 13 screening stages (S0-S7b, S5').

    Attributes:
        # Identity
        name: Structure name (CIF filename stem)
        formula: Reduced formula
        n_atoms: Number of atoms

        # S0 Composition
        s0_pass: Composition check passed
        s0_reason: Composition check reason

        # S1 Geometry
        s1_pass: Geometry check passed
        s1_reason: Geometry check reason
        s1_vacuum_angstrom: Vacuum gap (Å)
        s1_thickness_angstrom: Slab thickness (Å)
        s1_min_bond: Minimum bond distance (Å)
        s1_gamma: Gamma coordinate
        s1_aspect_ratio: Aspect ratio

        # S2 Symmetry
        s2_pass: Symmetry check passed
        s2_reason: Symmetry check reason
        s2_space_group: Space group number
        s2_layer_group: Layer group number

        # S3 Deduplication
        s3_is_duplicate: Is duplicate of another structure
        s3_duplicate_of: Name of duplicate structure
        s3_pareto_dominated: Is Pareto-dominated
        s3_pareto_dominated_by: Name of dominating structure

        # S-dim Connectivity
        sdim_pass: 2D connectivity check passed
        sdim_dimensionality: Dimensionality value
        sdim_reason: Connectivity check reason

        # S3.5 ML Relaxation
        s35_pass: ML relaxation passed
        s35_reason: Relaxation reason
        s35_ml_energy_eV: Total ML energy (eV)
        s35_ml_energy_per_atom: ML energy per atom (eV/atom)
        s35_relax_converged: Relaxation converged
        s35_convergence_status: "converged" | "not_converged" | "uncertain"
        s35_post_geom_ok: Post-relaxation geometry valid

        # S3T/S6 Thermodynamics
        s3t_pass: Thermodynamic coarse screening passed
        s3t_reason: Thermodynamic reason
        s3t_formation_energy_per_atom: Formation energy (eV/atom)
        s3t_e_above_hull_elem: Energy above hull (eV/atom)
        s6_formation_energy: Total formation energy (eV)
        s6_formation_energy_per_atom: Formation energy per atom (eV/atom)
        s6_pass: S6 passed (same as S3T)

        # S4 Elastic Matrix
        s4_pass: Elastic matrix check passed
        s4_reason: Elastic reason
        s4_c11: Elastic constant C11 (N/m)
        s4_c12: Elastic constant C12 (N/m)
        s4_c22: Elastic constant C22 (N/m)
        s4_c66: Elastic constant C66 (N/m)
        s4_c16: Elastic constant C16 (N/m)
        s4_c26: Elastic constant C26 (N/m)
        s4_min_eigenvalue: Minimum eigenvalue of Born matrix
        s4_stress_residual: Stress residual
        s4_energy_traj: Energy trajectory

        # S5 Phonon
        s5_pass: Phonon stability passed (None=skipped)
        s5_reason: Phonon reason
        s5_gamma_min_THz: Minimum gamma frequency (THz)
        s5_min_freq_THz: Minimum frequency (THz)
        s5_imaginary_count: Number of imaginary frequencies

        # S5' Convex Hull (MP API)
        s5p_pass: Convex hull check passed
        s5p_reason: Convex hull reason
        s5p_ehull_eV: Energy above hull (eV/atom)
        s5p_competing_phases_found: Number of competing phases
        s5p_missing_from_MP: Elements missing from MP
        s5p_stability_label: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"

        # S7a MD Fast
        s7a_pass: S7a MD passed
        s7a_reason: S7a reason
        s7a_struct_integrity_ok: Structure integrity maintained
        s7a_max_z_drift: Maximum z-coordinate drift (Å)
        s7a_fragment_count: Number of fragments

        # S7b MD Full
        s7b_pass: S7b MD passed
        s7b_reason: S7b reason
        s7b_struct_integrity_ok: Structure integrity maintained
        s7b_energy_trend_ok: Energy trend stable
        s7b_temperature_ok: Temperature stable
        s7b_t_max: Maximum temperature (K)
        s7b_energy_curve_2ps: Last 2000 energy values
        s7b_atom_snapshot: Atom configuration snapshot

        # Final
        final_pass: Passed all screening stages
        uncertain_final: Final result uncertain (not converged)
        failure_type: Type of failure
        raw_error: Raw error message
        retry_performed: Whether retry was performed
        elapsed_s: Total elapsed time (seconds)

        # Post-processing
        mag_type: Magnetic classification
        select_score: Distribution-based selection score
        dft_poscar_path: Path to exported POSCAR
    """

    # Identity
    name: str = ""
    formula: str = ""
    n_atoms: int = 0

    # S0 Composition
    s0_pass: bool = False
    s0_reason: str = ""

    # S1 Geometry
    s1_pass: bool = False
    s1_reason: str = ""
    s1_vacuum_angstrom: float = 0.0
    s1_thickness_angstrom: float = 0.0
    s1_min_bond: float = 0.0
    s1_gamma: float = 0.0
    s1_aspect_ratio: float = 0.0

    # S2 Symmetry
    s2_pass: bool = False
    s2_reason: str = ""
    s2_space_group: str = "?"
    s2_layer_group: str = "?"

    # S3 Deduplication
    s3_is_duplicate: bool = False
    s3_duplicate_of: str = ""
    s3_pareto_dominated: bool = False
    s3_pareto_dominated_by: str = ""

    # S-dim Connectivity
    sdim_pass: bool = False
    sdim_dimensionality: float = -1.0
    sdim_reason: str = ""

    # S3.5 ML Relaxation
    s35_pass: bool = False
    s35_reason: str = ""
    s35_ml_energy_eV: float = 0.0
    s35_ml_energy_per_atom: float = 0.0
    s35_relax_converged: bool = False
    s35_convergence_status: str = ""  # "converged" | "not_converged" | "uncertain"
    s35_post_geom_ok: bool = False

    # S3T/S6 Thermodynamics
    s3t_pass: bool = False
    s3t_reason: str = ""
    s3t_formation_energy_per_atom: float = 0.0
    s3t_e_above_hull_elem: float = 0.0
    s6_formation_energy: float = 0.0
    s6_formation_energy_per_atom: float = 0.0
    s6_pass: bool = False

    # S4 Elastic Matrix
    s4_pass: bool = False
    s4_reason: str = ""
    s4_c11: float = 0.0
    s4_c12: float = 0.0
    s4_c22: float = 0.0
    s4_c66: float = 0.0
    s4_c16: float = 0.0
    s4_c26: float = 0.0
    s4_min_eigenvalue: float = 0.0
    s4_stress_residual: float = 0.0
    s4_energy_traj: List = field(default_factory=list)

    # S5 Phonon
    s5_pass: Optional[bool] = None  # None = skipped
    s5_reason: str = ""
    s5_gamma_min_THz: float = 0.0
    s5_min_freq_THz: float = 0.0
    s5_imaginary_count: int = 0

    # S5' Convex Hull
    s5p_pass: bool = False
    s5p_reason: str = ""
    s5p_ehull_eV: float = 0.0
    s5p_competing_phases_found: int = 0
    s5p_missing_from_MP: List = field(default_factory=list)
    s5p_stability_label: str = ""

    # S7a MD Fast
    s7a_pass: bool = False
    s7a_reason: str = ""
    s7a_struct_integrity_ok: bool = False
    s7a_max_z_drift: float = 0.0
    s7a_fragment_count: int = 0

    # S7b MD Full
    s7b_pass: bool = False
    s7b_reason: str = ""
    s7b_struct_integrity_ok: bool = False
    s7b_energy_trend_ok: bool = False
    s7b_temperature_ok: bool = False
    s7b_t_max: float = 0.0
    s7b_energy_curve_2ps: List = field(default_factory=list)
    s7b_atom_snapshot: str = ""

    # Final
    final_pass: bool = False
    uncertain_final: bool = False
    failure_type: str = ""
    raw_error: str = ""
    retry_performed: bool = False
    elapsed_s: float = 0.0

    # Post-processing
    mag_type: str = ""
    select_score: float = 0.0
    dft_poscar_path: str = ""


__all__ = ['ScreeningRecord']
