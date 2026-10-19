"""
GPAW Configuration for DFT Validation.

Dataclass containing all parameters for the GPAW DFT validation pipeline.
Values are immutable scientific constants per C2DB standards.

References:
    - C2DB DFT settings
    - GPAW documentation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class GPAWConfig:
    """
    GPAW DFT validation pipeline configuration.

    All values are scientific constants following C2DB standards.
    DO NOT modify default values during refactoring.

    Categories:
        - Electronic: encut, xc, fermi_smearing, k-points
        - Adaptive: gap interpolation parameters
        - Relaxation: M1 optimization parameters
        - Elastic: M2 strain calculation parameters
        - Phonon: M3 dispersion parameters
        - Convex Hull: M4 thermodynamic parameters
    """

    # ── Input/Output ────────────────────────────────────────────────────────
    input_dir: str = ""  # Required: path to screening output (set via CLI or environment)
    output_dir: str = "./gpaw_validation"

    # ── GPAW Electronic Parameters ──────────────────────────────────────────
    encut: int = 550                    # eV, plane-wave cutoff
    xc: str = "PBE"                     # Exchange-correlation functional
    fermi_smearing: float = 0.05        # eV, semiconductor default
    dk_per_ang: float = 0.03            # Å⁻¹, max k-point spacing
    spinpol: Optional[bool] = None      # None = auto-detect
    maxiter: int = 500                  # Max SCF iterations

    # ── Adaptive Electronic Parameters (Gap Interpolation) ──────────────────
    # These control how electronic parameters adapt based on the DFT-probed band gap.
    # The old discrete classification logic has been removed.
    gap_interp_scale: float = 0.5       # eV, gap interpolation decay scale
    dos_interp_scale: float = 1.0       # states/eV/atom, DOS interpolation scale
    k_max_boost: float = 2.0            # Max k-density multiplier for metals
    fermi_smearing_metal: float = 0.02  # eV, metal limit smearing
    fermi_smearing_semiconductor: float = 0.05  # eV, semiconductor limit
    supercell_min_length_base: float = 10.0  # Å, base supercell minimum
    supercell_min_length_boost: float = 5.0   # Å, metal limit extra length

    # ── M1: Relaxation Parameters ───────────────────────────────────────────
    relax_fmax: float = 0.01            # eV/Å, force convergence
    relax_steps: int = 200              # Max relaxation steps
    optimizer: str = "FIRE"             # FIRE or BFGS
    fix_cell: str = "ab"                # "ab" = fix c-axis
    symmetry_tol: float = 0.01          # Å, spglib symmetry tolerance
    always_spinpol: bool = False        # Always spin-polarized relaxation

    # FM/NM ground state comparison
    fm_nm_degenerate_tol_eV: float = 1e-3  # eV/atom, FM/NM degeneracy threshold
    fm_nm_compare_before_m2: bool = True   # Compare before M2, may replace structure

    # Zero-step convergence verification
    zero_step_verify: bool = True       # Verify zero-step convergence
    zero_step_displacement: float = 0.05   # Å, displacement amplitude
    zero_step_verify_threshold: float = 0.05  # eV/atom, energy delta threshold

    # ── M2: Elastic Constants Parameters ────────────────────────────────────
    elastic_delta: float = 0.005        # Strain amplitude (±0.5%)
    elastic_fmax: float = 0.01          # eV/Å, force convergence
    elastic_relax_steps: int = 100      # Internal relaxation steps
    born_noise_sigma: float = 3.0       # Born criteria noise confidence
    symmetry_noise_sigma: float = 3.0   # Hexagonal symmetry noise confidence

    # ── M3: Phonon Parameters ───────────────────────────────────────────────
    phonon_supercell: List[int] = field(default_factory=lambda: [3, 3, 1])
    phonon_displacement: float = 0.01   # Å, atomic displacement

    # Displacement convergence testing (default: fixed 0.01 Å)
    phonon_displacement_convergence_test: bool = False
    phonon_displacement_candidates: List[float] = field(
        default_factory=lambda: [0.005, 0.0075, 0.01, 0.015])
    phonon_displacement_tolerance: float = 0.02  # THz, convergence tolerance

    # Dynamic noise tolerance
    phonon_noise_sigma: float = 3.0     # Γ-point tolerance sigma
    phonon_noise_floor_max_THz: float = 1.0  # THz, noise estimation ceiling
    phonon_boundary_tighten_factor: float = 5.0  # BZ boundary tightening factor

    # Static fallback values
    phonon_gamma_tol: float = -0.1      # THz, Γ-point fallback
    phonon_boundary_tol: float = -0.1   # THz, BZ boundary fallback
    phonon_npoints_per_segment: int = 51  # q-points per segment

    # Branch-specific tolerances
    phonon_za_tol: float = -0.2         # THz, ZA branch Γ-point tolerance
    phonon_za_sigma_factor: float = 5.0 # ZA noise amplification factor
    phonon_optical_tol: float = -0.05   # THz, optical branch tolerance

    # Small lattice material handling
    small_lattice_elements: List[str] = field(
        default_factory=lambda: ['C', 'B', 'N', 'O'])
    small_lattice_min_length_multiplier: float = 2.0
    small_lattice_min_length: float = 15.0  # Å, hard minimum
    small_lattice_max_supercell: List[int] = field(
        default_factory=lambda: [7, 7, 1])

    # ── M4: Convex Hull Parameters ──────────────────────────────────────────
    ehull_threshold: float = 0.2        # eV/atom, stability threshold
    ehull_ref_pool_size: int = 3        # MP candidates per formula
    use_local_ref_ehull: bool = True    # Scheme B: local GPAW reference

    # ── Parallel & Retry ────────────────────────────────────────────────────
    parallel: Dict[str, Any] = field(default_factory=lambda: {
        'auto_cores': True,
        'max_cores': 32,
        'mpi': True,
    })
    scf_retry_max: int = 2
    scf_retry_perturbs: List[float] = field(default_factory=lambda: [0.05, 0.1])

    # ── Gap Probe Control ───────────────────────────────────────────────────
    skip_gap_probe: bool = False        # Skip gap probe, use metal defaults

    # ── Output Control ──────────────────────────────────────────────────────
    save_poscar: bool = True
    save_phonon_pdf: bool = True
    verbose: bool = True

    # ── 2D Constraint Mask ──────────────────────────────────────────────────
    # mask = [True, True, False, False, False, True]
    # [xx, yy, zz, xy, yz, zx]
    # a, b allowed; c, angles fixed; xy shear allowed
    relax_2d_mask: List[bool] = field(
        default_factory=lambda: [True, True, False, False, False, True]
    )


__all__ = ['GPAWConfig']
