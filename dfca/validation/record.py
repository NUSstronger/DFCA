"""
Validation Record for DFT Validation Pipeline.

Dataclass containing all results for a single structure's DFT validation.
Fields must not be renamed or removed to maintain checkpoint compatibility.

References:
    - C2DB validation criteria
    - GPAW validation pipeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ValidationRecord:
    """
    Single structure validation record.

    Contains all results from M1-M4 validation stages plus metadata.
    Field names are locked for checkpoint compatibility.

    Categories:
        - Identification: name, formula, n_atoms
        - Magnetic: mag_type, is_magnetic, moment config
        - M1: Relaxation results and FM/NM comparison
        - GAP: Electronic gap probe results
        - M2: Elastic constants
        - M3: Phonon dispersion
        - M4: Convex hull stability
        - Summary: all_dft_pass, error handling
    """

    # ── Identification ──────────────────────────────────────────────────────
    name: str = ""
    formula: str = ""
    mag_type: str = ""
    n_atoms: int = 0

    # ── Magnetic State ──────────────────────────────────────────────────────
    is_magnetic: bool = False
    magnetic_moment: float = 0.0
    magmom_config: List[float] = field(default_factory=list)

    # ── M1: Relaxation Results ──────────────────────────────────────────────
    m1_relax_converged: bool = False
    m1_fmax_reached: float = 0.0
    m1_steps: int = 0
    m1_energy_eV: float = 0.0
    m1_energy_per_atom: float = 0.0
    m1_time_s: float = 0.0
    m1_pass: bool = False
    m1_reason: str = ""

    # Space group
    m1_spacegroup_before: str = ""      # Pre-relaxation space group
    m1_spacegroup_after: str = ""       # Post-relaxation space group

    # Ground state
    m1_ground_state: str = "not_evaluated"  # FM | NM | not_evaluated
    m1_fm_nm_gap_eV: float = 0.0       # FM vs NM energy difference

    # Zero-step verification
    m1_zero_step_flag: bool = False    # True = initial structure met convergence
    m1_zero_step_verified: bool = False  # True = secondary verification passed
    m1_zero_step_energy_delta: float = 0.0  # eV/atom, perturbation energy delta
    m1_zero_step_verify_passed: bool = False  # True = delta within threshold

    # NM relaxation failure detail
    m1_nm_relax_failure_detail: str = ""

    # Checkpoint resume flag
    m1_resumed_from_partial: bool = False

    # ── Electronic Gap Probe Results ────────────────────────────────────────
    probed_gap_eV: float = -1.0        # HOMO-LUMO gap (eV)
    probed_dos_at_ef: float = -1.0     # DOS at Fermi level (states/eV/atom)
    probed_probe_success: bool = False

    # ── Adaptive Parameters Usage Record ────────────────────────────────────
    used_kpoint_multiplier: float = 1.0
    used_fermi_smearing: float = 0.05  # eV
    used_supercell_min_length: float = 10.0  # Å

    # ── Screen -> GPAW Metadata ─────────────────────────────────────────────
    # Diagnostic/provenance fields. MUST NOT alter pass/fail until validated.
    screen_report_found: bool = False
    screen_name: str = ""
    screen_formula: str = ""
    screen_n_atoms: int = 0
    screen_mag_type: str = ""

    screen_s4_min_eigenvalue: float = 0.0
    screen_s4_stress_residual: float = 0.0
    screen_s4_pass: bool = False

    screen_s5_gamma_min_THz: float = 0.0
    screen_s5_min_freq_THz: float = 0.0
    screen_s5_imaginary_count: int = 0
    screen_s5_pass: bool = False
    screen_s5_reason: str = ""

    screen_s7a_pass: bool = False
    screen_s7b_pass: bool = False

    screen_s5p_ehull_eV: float = 0.0
    screen_s5p_pass: bool = False
    screen_s5p_stability_label: str = ""

    screen_select_score: float = 0.0
    screen_final_pass: bool = False

    # ── M2: Elastic Constants ────────────────────────────────────────────────
    m2_c11: float = 0.0                # N/m
    m2_c12: float = 0.0                # N/m
    m2_c22: float = 0.0                # N/m
    m2_c66: float = 0.0                # N/m
    m2_stress_residual: float = 0.0    # eV/Å, baseline stress norm
    m2_noise_floor: float = 0.0        # N/m, intrinsic noise scale
    m2_elastic_pass: bool = False
    m2_reason: str = ""

    # ── M3: Phonon Dispersion ───────────────────────────────────────────────
    m3_gamma_min_THz: float = 0.0
    m3_min_freq_THz: float = 0.0
    m3_imaginary_count: int = 0
    m3_phonon_pass: bool = False
    m3_reason: str = ""
    m3_phonon_pdf_path: str = ""

    # M3 execution state: False = not finished, True = finished (pass or fail)
    m3_completed: bool = False

    # Dynamic noise tolerance
    m3_noise_floor_THz: float = 0.0
    m3_gamma_tol_dynamic: float = 0.0  # THz
    m3_boundary_tol_dynamic: float = 0.0  # THz

    # Branch-specific results
    m3_za_branch_idx: int = -1
    m3_ta_branch_idx: int = -1
    m3_la_branch_idx: int = -1
    m3_za_gamma_freq: float = 0.0      # THz
    m3_ta_gamma_freq: float = 0.0
    m3_la_gamma_freq: float = 0.0
    m3_optical_gamma_min: float = 0.0
    m3_za_tol_dynamic: float = 0.0
    m3_optical_tol_dynamic: float = 0.0
    m3_za_ok: bool = True
    m3_ta_ok: bool = True
    m3_la_ok: bool = True
    m3_optical_ok: bool = True

    # Noise reduction record
    m3_used_displacement: float = 0.0  # Å
    m3_used_supercell: str = ""        # [a,b,c]
    m3_is_small_lattice: bool = False
    m3_noise_estimation_method: str = ""

    # ── M4: Convex Hull ─────────────────────────────────────────────────────
    m4_ehull_eV: float = 0.0
    m4_convex_hull_pass: bool = False
    m4_reason: str = ""
    m4_used_local_ref: bool = False    # True = scheme B (GPAW reference)

    # ── Summary ─────────────────────────────────────────────────────────────
    all_dft_pass: bool = False
    dft_poscar_path: str = ""

    # Error handling
    error_type: str = ""   # scf_converge, relaxation, elastic, phonon, ehull, unknown
    raw_error: str = ""
    retry_count: int = 0
    checkpoint_data: Dict = field(default_factory=dict)


__all__ = ['ValidationRecord']
