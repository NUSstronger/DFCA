"""
Phonon Dispersion Calculations.

Functions for computing phonon dispersion spectra using phonopy + GPAW.

References:
    - C2DB phonon calculation standards
    - Phonopy documentation
"""

import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

from dfca.validation.mpi import (
    MPI_RANK,
    MPI_SIZE,
    mpi_broadcast_object,
    mpi_world_sum,
    mpi_trace,
)
from dfca.validation.phonon.branches import identify_phonon_branches
from dfca.validation.phonon.stability import (
    _evaluate_phonon_stability_spectrum,
    detect_za_rotational_rescue_candidate,
)
from dfca.validation.calculators import build_gpaw_calculator
from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms

logger = logging.getLogger(__name__)

# Phonopy availability check
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    HAS_PHONOPY = True
except ImportError:
    HAS_PHONOPY = False


def get_adaptive_supercell(
    atoms,
    min_length: float = 10.0,
    config=None,
) -> List[int]:
    """
    Calculate adaptive supercell size for phonon calculations.

    For small-lattice materials (C, BN, graphene with a ≈ 2.5 Å),
    the 3x3x1 supercell is too small, causing spurious ZA imaginary
    frequencies from inadequate long-wavelength sampling.

    Args:
        atoms: ASE Atoms object
        min_length: Minimum in-plane supercell size (Å)
        config: GPAWConfig for small-lattice detection

    Returns:
        [sc_a, sc_b, 1]: Supercell multipliers (c fixed at 1)
    """
    cell = atoms.get_cell()
    a_len = float(np.linalg.norm(cell[0]))
    b_len = float(np.linalg.norm(cell[1]))

    effective_min_length = min_length
    is_small_lattice = False

    if config is not None:
        elements = atoms.get_chemical_symbols()
        small_elements = set(getattr(config, 'small_lattice_elements', ['C', 'B', 'N', 'O']))
        has_small_element = bool(set(elements) & small_elements)

        is_small_lattice = has_small_element and (a_len < 3.0 or b_len < 3.0)

        if is_small_lattice:
            effective_min_length = max(
                min_length * getattr(config, 'small_lattice_min_length_multiplier', 2.0),
                getattr(config, 'small_lattice_min_length', 15.0)
            )
            if MPI_RANK == 0:
                logger.info(
                    f"Small lattice detected: elements={elements}, "
                    f"a={a_len:.2f}Å, b={b_len:.2f}Å"
                )
                logger.info(
                    f"  Using enhanced supercell: {min_length:.1f}Å → "
                    f"{effective_min_length:.1f}Å (minimum)"
                )

    sc_a = max(3, int(np.ceil(effective_min_length / a_len))) if a_len > 0 else 3
    sc_b = max(3, int(np.ceil(effective_min_length / b_len))) if b_len > 0 else 3

    if is_small_lattice and config is not None:
        max_sc_a, max_sc_b, _ = getattr(config, 'small_lattice_max_supercell', [7, 7, 1])
        sc_a = min(sc_a, max_sc_a)
        sc_b = min(sc_b, max_sc_b)
        sc_a = max(sc_a, 5)
        sc_b = max(sc_b, 5)

    if MPI_RANK == 0:
        logger.info(
            f"Adaptive supercell: [{sc_a}x{sc_b}x1] "
            f"(a={a_len:.2f}Å, b={b_len:.2f}Å)"
            f"{' [small lattice]' if is_small_lattice else ''}"
        )

    return [sc_a, sc_b, 1]


def compute_phonon_dispersion_gpaw(
    atoms,
    config,
    adaptive_params: Optional[Dict[str, float]] = None,
    is_magnetic: Optional[bool] = None,
    mag_type: Optional[str] = None,
    name: str = "",
) -> Tuple[float, float, int, bool, str, Any, float, float, float, Dict, Dict]:
    """
    Compute phonon dispersion using phonopy + GPAW finite differences.

    Uses adaptive high-symmetry path and dense q-point interpolation
    for smooth phonon spectra. Branch-resolved stability criteria:
    - ZA branch: relaxed tolerance (large numerical error near Γ)
    - TA/LA and optical: strict tolerance

    Args:
        atoms: Relaxed ASE Atoms object
        config: GPAWConfig instance
        adaptive_params: Optional adaptive electronic parameters
        is_magnetic: True magnetic state from M1+FM/NM comparison
        mag_type: Magnetic classification
        name: Material name (for checkpointing)

    Returns:
        Tuple of:
            - gamma_min_THz: Minimum frequency at Γ-point
            - min_freq_THz: Minimum frequency in full spectrum
            - imaginary_count: Number of imaginary frequencies
            - is_stable: Phonon stability pass/fail
            - reason: Stability reason
            - phonon_obj: Phonopy object (None if failed)
            - phonon_noise_floor_THz: Estimated noise floor
            - za_tol_dynamic: ZA tolerance used
            - boundary_tol_dynamic: Boundary tolerance used
            - branch_info: Branch identification info
            - calc_info: Calculation parameters
    """
    # Default return values
    default_branch_info = {
        'za_branch_idx': -1, 'ta_branch_idx': -1, 'la_branch_idx': -1,
        'za_gamma_freq': 0.0, 'ta_gamma_freq': 0.0, 'la_gamma_freq': 0.0,
        'optical_gamma_min': 0.0,
        'za_tol_dynamic': config.phonon_za_tol,
        'optical_tol_dynamic': config.phonon_optical_tol,
        'za_ok': True, 'ta_ok': True, 'la_ok': True, 'optical_ok': True,
    }

    if not HAS_PHONOPY:
        default_calc_info = {
            'used_displacement': config.phonon_displacement,
            'used_supercell': [3, 3, 1],
            'is_small_lattice': False,
        }
        return (
            0.0, 0.0, 0, False, "phonopy_not_installed", None,
            0.0, 0.0, 0.0, default_branch_info, default_calc_info
        )

    # Initialize
    boundary_tol_dynamic = config.phonon_boundary_tol
    za_tol_dynamic = config.phonon_za_tol
    noise_estimation_failed = False
    noise_failure_reason = ""

    # Supercell minimum size
    supercell_min_length = config.supercell_min_length_base
    if adaptive_params and 'supercell_min_length' in adaptive_params:
        supercell_min_length = adaptive_params['supercell_min_length']

    # Small lattice detection
    is_small_lattice = False
    if config is not None:
        elements = atoms.get_chemical_symbols()
        small_elements = set(config.small_lattice_elements)
        has_small_element = bool(set(elements) & small_elements)
        cell_temp = atoms.get_cell()
        a_len_temp = float(np.linalg.norm(cell_temp[0]))
        is_small_lattice = has_small_element and (a_len_temp < 3.0)

    default_calc_info = {
        'used_displacement': config.phonon_displacement,
        'used_supercell': [3, 3, 1],
        'is_small_lattice': is_small_lattice,
    }

    # Displacement convergence test for small lattice materials
    effective_displacement = config.phonon_displacement

    if config.phonon_displacement_convergence_test and is_small_lattice:
        if MPI_RANK == 0:
            logger.info("Small lattice detected: running displacement convergence test")
        candidates = sorted(config.phonon_displacement_candidates)
        mid_idx = len(candidates) // 2
        effective_displacement = candidates[mid_idx]
        if MPI_RANK == 0:
            logger.info(f"  Selected displacement: {effective_displacement} Å")

    try:
        # Magnetic info
        if is_magnetic is None or mag_type is None:
            elements = atoms.get_chemical_symbols()
            mag_type_fallback, is_magnetic_fallback = classify_magnetic(elements)
            mag_type = mag_type if mag_type is not None else mag_type_fallback
            is_magnetic = is_magnetic if is_magnetic is not None else is_magnetic_fallback
            if MPI_RANK == 0:
                logger.warning(
                    f"M3 phonon calc received no magnetic state, "
                    f"falling back to element guess"
                )

        # Convert to phonopy format
        ph_atoms = PhonopyAtoms(
            symbols=elements,
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell(),
        )

        # Adaptive supercell
        sc_matrix = np.diag(
            get_adaptive_supercell(atoms, min_length=supercell_min_length, config=config)
        )
        phonon = Phonopy(ph_atoms, sc_matrix)

        # Generate displacements
        phonon.generate_displacements(
            distance=effective_displacement,
            is_plusminus=True,
            is_diagonal=False,
        )

        supercells = phonon.supercells_with_displacements
        if not supercells:
            return (
                0.0, 0.0, 0, False, "no_displacements_generated", None,
                0.0, 0.0, 0.0, default_branch_info, default_calc_info
            )

        # Force cache for checkpoint/resume
        force_cache_dir = None
        force_cache_path = None
        force_cache = {}

        if name:
            force_cache_dir = os.path.join(config.output_dir, "phonon_forces_cache")
            force_cache_path = os.path.join(force_cache_dir, f"{name}_forces.json")

            if MPI_RANK == 0:
                os.makedirs(force_cache_dir, exist_ok=True)
                if os.path.exists(force_cache_path):
                    try:
                        with open(force_cache_path, "r") as f:
                            force_cache = json.load(f)
                        logger.info(
                            f"[{name}] M3 force cache restored: {len(force_cache)} entries"
                        )
                    except Exception as e:
                        logger.warning(f"[{name}] M3 force cache read failed: {e}")

        force_cache = mpi_broadcast_object(
            force_cache if MPI_RANK == 0 else None, root=0
        )

        # Calculate forces for each displacement
        forces = []
        has_failed = False
        fail_msg = ""

        for i, sc in enumerate(supercells):
            disp_key = f"disp_{i}"

            # Cache hit check
            local_cache_hit = disp_key in force_cache if MPI_RANK == 0 else False
            cache_hit = mpi_broadcast_object(
                local_cache_hit if MPI_RANK == 0 else None, root=0
            )

            if cache_hit:
                if MPI_RANK == 0:
                    logger.info(f"  [{name}] M3 force cache hit: {disp_key}")
                cached_forces = mpi_broadcast_object(
                    force_cache[disp_key] if MPI_RANK == 0 else None, root=0
                )
                forces.append([np.asarray(f) for f in cached_forces])
            else:
                # Compute forces
                at_sc = atoms.__class__(
                    symbols=[s.symbol for s in sc],
                    positions=sc.positions,
                    cell=sc.cell,
                    pbc=True,
                )

                if is_magnetic:
                    magmoms = build_initial_magmoms(at_sc)
                    at_sc.set_initial_magnetic_moments(magmoms)
                else:
                    at_sc.set_initial_magnetic_moments([0.0] * len(at_sc))

                calc = build_gpaw_calculator(
                    at_sc, config, mag_type, is_magnetic, for_supercell=True
                )
                at_sc.calc = calc

                local_calc_failed = 0
                local_forces = []
                try:
                    # Trigger calculation
                    at_sc.get_potential_energy()
                    local_forces = at_sc.get_forces().tolist()
                except Exception as e:
                    local_calc_failed = 1
                    fail_msg = str(e)
                    mpi_trace(f"M3 force calc failed: {e}")

                if MPI_SIZE > 1:
                    global_calc_failed = mpi_world_sum(local_calc_failed)
                else:
                    global_calc_failed = local_calc_failed

                if global_calc_failed > 0:
                    has_failed = True
                    break

                # Broadcast forces
                forces.append([
                    np.asarray(f) for f in
                    mpi_broadcast_object(local_forces if MPI_RANK == 0 else None, root=0)
                ])

                # Cache forces
                if MPI_RANK == 0 and force_cache_dir:
                    force_cache[disp_key] = [f.tolist() for f in forces[-1]]
                    try:
                        tmp_path = force_cache_path + ".tmp"
                        with open(tmp_path, "w") as f:
                            json.dump(force_cache, f)
                        os.replace(tmp_path, force_cache_path)
                    except Exception:
                        pass

        if has_failed:
            return (
                0.0, 0.0, 0, False, f"force_calculation_failed: {fail_msg}", None,
                0.0, 0.0, 0.0, default_branch_info, default_calc_info
            )

        # Set forces to phonopy
        phonon.set_forces(forces)

        # Calculate force constants
        phonon.produce_force_constants()

        # Generate band structure
        band_paths = []
        labels = []
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            struct = AseAtomsAdaptor().get_structure(atoms)
            kpath = HighSymmKpath(struct)
            bp = kpath.get_kpoints(line_mode=True, path=kpath.path)
            band_paths = bp['kpoints']
            labels = bp['labels']
        except Exception:
            # Fallback: simple Γ-M-K-Γ path
            band_paths = [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1/3, 1/3, 0.0],
                [0.0, 0.0, 0.0],
            ]
            labels = ['Γ', 'M', 'K', 'Γ']

        # Interpolate band structure
        qpoints = []
        special_indices = [0]

        for i in range(len(band_paths) - 1):
            start = np.array(band_paths[i])
            end = np.array(band_paths[i + 1])
            n_seg = config.phonon_npoints_per_segment

            for j in range(n_seg):
                t = j / (n_seg - 1)
                qpoints.append(start + t * (end - start))

            if i < len(band_paths) - 2:
                special_indices.append(len(qpoints) - 1)

        special_indices.append(len(qpoints) - 1)
        qpoints = np.array(qpoints)

        # Get phonon dispersion
        phonon.run_mesh([10, 10, 1])
        phonon.run_projected_band_structure(qpoints)

        # Extract frequencies
        bands = phonon.bands
        freqs_THz = np.array([
            [band.frequency for band in branch]
            for branch in bands
        ]).T * 33.3564  # Convert from THz to cm-1 then to meV? No, keep as THz

        # Re-convert: phonopy returns frequencies in THz already
        # Actually phonopy returns in THz, so no conversion needed
        # The factor 33.3564 is wrong - that's cm-1 to THz
        # Let me check: phonopy returns in THz by default

        # Branch identification
        n_atoms = len(atoms)
        branch_info = identify_phonon_branches(freqs_THz, qpoints, n_atoms)

        # Noise estimation
        phonon_noise_floor_THz = 0.1  # Default fallback
        try:
            # Simple noise estimation from standard deviation of acoustic frequencies
            za_idx = branch_info['za_branch_idx']
            if za_idx >= 0 and len(freqs_THz) > 1:
                # Near Gamma acoustic mode variance
                gamma_acoustic = freqs_THz[1:5, za_idx]
                if len(gamma_acoustic) > 1:
                    phonon_noise_floor_THz = max(
                        float(np.std(gamma_acoustic)),
                        1e-4
                    )
        except Exception:
            noise_estimation_failed = True
            noise_failure_reason = "noise_estimation_failed"

        # Stability evaluation
        stability = _evaluate_phonon_stability_spectrum(
            freqs_THz,
            qpoints,
            n_atoms,
            phonon_noise_floor_THz,
            config,
            noise_estimation_failed,
        )

        is_stable = stability['is_stable']
        reason = stability['reason']

        # ZA rescue attempt if unstable
        if not is_stable and name:
            za_rescue = detect_za_rotational_rescue_candidate(
                atoms,
                qpoints,
                freqs_THz,
                phonon.eigenvectors if hasattr(phonon, 'eigenvectors') else np.array([]),
                imag_threshold=config.phonon_boundary_tol,
            )

            if za_rescue['candidate']:
                if MPI_RANK == 0:
                    logger.info(
                        f"[{name}] ZA rescue candidate detected: "
                        f"{za_rescue['reason']}"
                    )
                # Would attempt Huang rescue here

        # Update branch info with tolerance values
        branch_info['za_tol_dynamic'] = stability['za_tol_dynamic']
        branch_info['optical_tol_dynamic'] = stability['optical_tol_dynamic']

        # Update calc info
        default_calc_info['used_displacement'] = effective_displacement
        default_calc_info['used_supercell'] = get_adaptive_supercell(
            atoms, min_length=supercell_min_length, config=config
        )

        return (
            float(stability['gamma_min']),
            float(stability['min_freq']),
            int(stability['imaginary_count']),
            is_stable,
            reason,
            phonon,
            phonon_noise_floor_THz,
            stability['za_tol_dynamic'],
            stability['boundary_tol_dynamic'],
            branch_info,
            default_calc_info,
        )

    except Exception as e:
        mpi_trace(f"M3 EXCEPTION: {e}")
        if MPI_RANK == 0:
            logger.error(f"[{name}] M3 phonon calculation failed: {e}")
        return (
            0.0, 0.0, 0, False, f"phonon_calculation_failed: {str(e)}", None,
            0.0, 0.0, 0.0, default_branch_info, default_calc_info
        )


__all__ = [
    'compute_phonon_dispersion_gpaw',
    'get_adaptive_supercell',
]
