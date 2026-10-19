"""
2D Elastic Constant Calculations.

Functions for computing elastic constants using finite difference method with GPAW.

References:
    - C2DB elastic constant standards
    - Born stability criteria for 2D materials
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from dfca.validation.mpi import (
    MPI_RANK,
    MPI_SIZE,
    mpi_broadcast_object,
    mpi_world_sum,
    mpi_trace,
)
from dfca.validation.calculators import build_gpaw_calculator
from dfca.validation.electronic.symmetry import get_spacegroup_info
from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms

logger = logging.getLogger(__name__)

# Conversion factor: 1 eV/A³ = 160.21766208 GPa
GPA_PER_EV_A3 = 160.21766208


def compute_elastic_matrix_2d_gpaw(
    atoms,
    config,
    is_magnetic: Optional[bool] = None,
    mag_type: Optional[str] = None,
    name: str = "",
) -> Tuple[Dict[str, float], bool, str, float, float]:
    """
    Compute 2D elastic constants using GPAW finite difference method.

    Uses 4 strain amplitudes (±0.5%, ±1.0%) to extract stress responses
    via finite differences.

    Args:
        atoms: Relaxed ASE Atoms object
        config: GPAWConfig instance
        is_magnetic: True magnetic state from M1+FM/NM comparison
        mag_type: Magnetic classification
        name: Material name (for checkpointing)

    Returns:
        Tuple of:
            - c2d_dict: {"C11":, "C12":, "C22":, "C66":, "C16":, "C26":} in N/m
            - is_stable: Born stability criterion pass/fail
            - reason: Stability reason string
            - stress_residual: Reference state stress residual (eV/Å³)
            - noise_floor_N_per_m: Intrinsic noise floor (N/m)
    """
    # Get c-axis length
    c_axis_length = float(atoms.cell[2, 2])
    if c_axis_length <= 0:
        return {}, False, f"invalid_c_axis({c_axis_length})", 0.0, 0.0

    # Get magnetic info: prefer passed value, fallback to element-based guess
    if is_magnetic is None or mag_type is None:
        elements = atoms.get_chemical_symbols()
        mag_type_fallback, is_magnetic_fallback = classify_magnetic(elements)
        mag_type = mag_type if mag_type is not None else mag_type_fallback
        is_magnetic = is_magnetic if is_magnetic is not None else is_magnetic_fallback
        if MPI_RANK == 0:
            logger.warning(
                f"M2 elastic calc received no magnetic state, "
                f"falling back to element guess: mag_type={mag_type}, "
                f"is_magnetic={is_magnetic}"
            )

    # Initial magnetic moments
    if is_magnetic:
        magmoms = build_initial_magmoms(atoms)
        atoms.set_initial_magnetic_moments(magmoms)
    else:
        atoms.set_initial_magnetic_moments([0.0] * len(atoms))

    # Helper functions
    def voigt_to_strain(voigt: np.ndarray) -> np.ndarray:
        """Convert Voigt strain [e1..e6] to 3x3 symmetric strain matrix."""
        e = np.zeros((3, 3))
        e[0, 0] = voigt[0]
        e[1, 1] = voigt[1]
        e[2, 2] = voigt[2]
        e[1, 2] = e[2, 1] = voigt[3] / 2.0
        e[0, 2] = e[2, 0] = voigt[4] / 2.0
        e[0, 1] = e[1, 0] = voigt[5] / 2.0
        return e

    def apply_strain(cell: np.ndarray, voigt: np.ndarray) -> np.ndarray:
        """Apply Voigt strain to cell.

        ASE's cell array stores lattice vectors as ROWS.
        Strain: C' = C @ (I + eps) where eps is symmetric.
        """
        return cell @ (np.eye(3) + voigt_to_strain(voigt))

    def fit_slope(strains: List[float], stresses: List[float]) -> float:
        """1st-order polyfit: slope = d(stress)/d(strain)"""
        if len(strains) < 2:
            return 0.0
        coeffs = np.polyfit(strains, stresses, deg=1)
        return float(coeffs[0])

    # Reference state symmetry
    sg_before_sym, sg_num_before = get_spacegroup_info(
        atoms, symprec=config.symmetry_tol
    )

    # Reference state: fixed cell, relax atoms
    atoms_ref = atoms.copy()
    atoms_ref.set_constraint(None)  # Clear FixSymmetry from M1

    if is_magnetic:
        magmoms = build_initial_magmoms(atoms_ref)
        atoms_ref.set_initial_magnetic_moments(magmoms)
    else:
        atoms_ref.set_initial_magnetic_moments([0.0] * len(atoms_ref))

    calc_ref = build_gpaw_calculator(atoms_ref, config, mag_type, is_magnetic)
    atoms_ref.calc = calc_ref

    # Reference relaxation
    local_ref_failed = 0
    local_ref_error = ""
    try:
        from ase.optimize import FIRE
        FIRE(atoms_ref, logfile=None).run(
            fmax=config.elastic_fmax,
            steps=config.elastic_relax_steps
        )
    except Exception as e:
        local_ref_failed = 1
        local_ref_error = str(e)

    if MPI_SIZE > 1:
        global_ref_failed = mpi_world_sum(local_ref_failed)
    else:
        global_ref_failed = local_ref_failed

    if global_ref_failed > 0:
        mpi_trace("M2 reference relaxation failed")
        return {}, False, "reference_relax_failed: " + local_ref_error, 0.0, 0.0

    # Check convergence
    fmax_ref = float(np.linalg.norm(atoms_ref.get_forces(), axis=1).max())
    if fmax_ref > config.elastic_fmax:
        if MPI_RANK == 0:
            logger.warning(
                f"Reference elastic relaxation not fully converged: "
                f"fmax={fmax_ref:.4f} eV/Å > {config.elastic_fmax}"
            )

    stress_ref = np.asarray(atoms_ref.get_stress(), dtype=float).copy()
    stress_residual = float(np.linalg.norm(stress_ref))

    # Detach GPAW after reference relaxation
    try:
        atoms_ref.calc = None
    except Exception:
        pass

    # Strain calculation
    strain_amps = [
        -config.elastic_delta, -config.elastic_delta/2,
        config.elastic_delta/2, config.elastic_delta
    ]

    def strain_and_relax(comp_idx: int, delta_val: float) -> np.ndarray:
        """Compute stress for one strain component."""
        voigt = np.zeros(6)
        voigt[comp_idx] = delta_val
        deformed = apply_strain(atoms_ref.cell[:], voigt)

        at = atoms_ref.copy()
        at.set_cell(deformed, scale_atoms=True)

        # Magnetic moments
        if is_magnetic:
            magmoms = build_initial_magmoms(at)
            at.set_initial_magnetic_moments(magmoms)
        else:
            at.set_initial_magnetic_moments([0.0] * len(at))

        calc = build_gpaw_calculator(at, config, mag_type, is_magnetic)
        at.calc = calc

        # Fixed cell, relax atoms only
        local_strain_failed = 0
        local_strain_error = ""
        try:
            from ase.optimize import FIRE
            FIRE(at, logfile=None).run(
                fmax=config.elastic_fmax,
                steps=config.elastic_relax_steps
            )
        except Exception as e:
            local_strain_failed = 1
            local_strain_error = str(e)

        if MPI_SIZE > 1:
            global_strain_failed = mpi_world_sum(local_strain_failed)
        else:
            global_strain_failed = local_strain_failed

        if global_strain_failed > 0:
            mpi_trace("M2 strain calc failed")
            raise RuntimeError("strain_calc_failed: " + local_strain_error)

        stress = np.asarray(at.get_stress(), dtype=float).copy()

        try:
            at.calc = None
        except Exception:
            pass

        return stress

    # Components for 2D elastic constants
    components = [("xx", 0), ("yy", 1), ("xy", 5)]
    c_mat = np.zeros((6, 6))
    all_stresses = {}

    # M2 strain-level persistent checkpoint
    m2_cache_dir = os.path.join(config.output_dir, "m2_strain_cache")
    m2_cache_path = os.path.join(m2_cache_dir, f"{name}_stresses.json")

    m2_fingerprint = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.asarray(atoms.cell.array, dtype=float).round(10).tolist(),
        "positions": np.asarray(atoms.positions, dtype=float).round(10).tolist(),
        "elastic_delta": float(config.elastic_delta),
        "elastic_fmax": float(config.elastic_fmax),
        "elastic_relax_steps": int(config.elastic_relax_steps),
        "is_magnetic": bool(is_magnetic),
        "mag_type": str(mag_type),
    }

    m2_cache = {}

    if MPI_RANK == 0:
        try:
            os.makedirs(m2_cache_dir, exist_ok=True)

            if os.path.exists(m2_cache_path):
                with open(m2_cache_path, "r") as f:
                    loaded = json.load(f)

                if (
                    isinstance(loaded, dict)
                    and loaded.get("fingerprint") == m2_fingerprint
                    and isinstance(loaded.get("stresses"), dict)
                ):
                    m2_cache = loaded["stresses"]
                    logger.info(
                        f"[{name}] M2 strain cache loaded: "
                        f"{len(m2_cache)}/12 completed"
                    )
                else:
                    logger.warning(
                        f"[{name}] M2 strain cache fingerprint mismatch; "
                        "old cache ignored"
                    )
        except Exception as cache_err:
            logger.warning(f"[{name}] M2 strain cache load failed: {cache_err}")
            m2_cache = {}

    m2_cache = mpi_broadcast_object(
        m2_cache if MPI_RANK == 0 else None, root=0
    )

    for comp_name, idx in components:
        strains_list = []
        stresses_dict = {i: [] for i in range(6)}

        for delta_val in strain_amps:
            cache_key = f"{comp_name}_{delta_val:+.8f}"

            cached_sig = m2_cache.get(cache_key)

            if isinstance(cached_sig, list) and len(cached_sig) == 6:
                sig = np.asarray(cached_sig, dtype=float).copy()
                if MPI_RANK == 0:
                    logger.info(f"  [{name}] M2 strain cache HIT: {cache_key}")
            else:
                sig = np.asarray(
                    strain_and_relax(idx, delta_val),
                    dtype=float,
                ).copy()

                m2_cache[cache_key] = sig.tolist()

                if MPI_RANK == 0:
                    try:
                        tmp_path = m2_cache_path + ".tmp"
                        with open(tmp_path, "w") as f:
                            json.dump(
                                {"fingerprint": m2_fingerprint, "stresses": m2_cache},
                                f, indent=2
                            )
                        os.replace(tmp_path, m2_cache_path)
                        logger.info(
                            f"  [{name}] M2 strain cached: {cache_key} "
                            f"({len(m2_cache)}/12)"
                        )
                    except Exception as cache_err:
                        logger.warning(
                            f"[{name}] M2 strain cache write failed: {cache_err}"
                        )

            strains_list.append(delta_val)
            for i in range(6):
                stresses_dict[i].append(float(sig[i]))

        for i in range(6):
            c_mat[i, idx] = fit_slope(strains_list, stresses_dict[i])

        all_stresses[idx] = stresses_dict

    # Nonlinear response check
    d = abs(strain_amps[1])
    idx_d_plus, idx_d_minus = 2, 1
    idx_2d_plus, idx_2d_minus = 3, 0

    max_rel_diff = 0.0
    for comp_name, idx in components:
        stresses = all_stresses[idx]
        slope_d = (stresses[idx][idx_d_plus] - stresses[idx][idx_d_minus]) / (2 * d)
        slope_2d = (stresses[idx][idx_2d_plus] - stresses[idx][idx_2d_minus]) / (2 * 2 * d)

        if abs(slope_d) > 1e-10:
            rel_diff = abs(slope_2d - slope_d) / abs(slope_d)
            max_rel_diff = max(max_rel_diff, rel_diff)

    if max_rel_diff > 0.20:
        if MPI_RANK == 0:
            logger.warning(
                f"Large nonlinear elastic response (rel_diff={max_rel_diff:.1%}), "
                "continuing with first-order fit"
            )

    # Convert to 2D elastic constants (N/m)
    factor = c_axis_length * 0.1

    c11 = c_mat[0, 0] * GPA_PER_EV_A3 * factor
    c22 = c_mat[1, 1] * GPA_PER_EV_A3 * factor
    c12 = (c_mat[0, 1] + c_mat[1, 0]) / 2.0 * GPA_PER_EV_A3 * factor
    c66 = c_mat[5, 5] * GPA_PER_EV_A3 * factor
    c16 = c_mat[5, 0] * GPA_PER_EV_A3 * factor
    c26 = c_mat[5, 1] * GPA_PER_EV_A3 * factor

    # Noise floor from reference state stress residual
    noise_floor_N_per_m = stress_residual * GPA_PER_EV_A3 * factor

    c2d = {
        "C11": float(c11),
        "C12": float(c12),
        "C22": float(c22),
        "C66": float(c66),
        "C16": float(c16),
        "C26": float(c26),
    }

    # Hexagonal symmetry check
    cell = atoms.get_cell()
    a_vec, b_vec = np.array(cell[0]), np.array(cell[1])
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    cos_gamma = float(np.dot(a_vec, b_vec) / (a_len * b_len)) if a_len > 0 and b_len > 0 else 0.0
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma_deg = float(np.degrees(np.arccos(cos_gamma)))

    is_hexagonal = (
        abs(a_len - b_len) / max(a_len, b_len) < 0.05
        and (abs(gamma_deg - 120.0) < 10.0 or abs(gamma_deg - 60.0) < 10.0)
    )

    symmetry_warning = ""
    if is_hexagonal:
        c66_theory = (c11 - c12) / 2.0
        if abs(c66_theory) > 1e-10:
            deviation_N_per_m = abs(c66 - c66_theory)
            if deviation_N_per_m > config.symmetry_noise_sigma * noise_floor_N_per_m:
                symmetry_warning = (
                    f"hexagonal_symmetry_violation("
                    f"dev={deviation_N_per_m:.2f}N/m, "
                    f"noise_floor={noise_floor_N_per_m:.2f}N/m)"
                )
                if MPI_RANK == 0:
                    logger.warning(
                        f"Hexagonal symmetry deviation: C66={c66:.2f} vs "
                        f"theory={c66_theory:.2f} N/m "
                        f"(deviation {deviation_N_per_m:.2f}N/m > "
                        f"{config.symmetry_noise_sigma}σ="
                        f"{config.symmetry_noise_sigma*noise_floor_N_per_m:.2f}N/m)"
                    )

    # Shear coupling check
    if abs(c16) > 1.0 or abs(c26) > 1.0:
        if MPI_RANK == 0:
            logger.warning(
                f"Significant shear coupling: C16={c16:.2f} N/m, "
                f"C26={c26:.2f} N/m"
            )

    # Born stability criterion (3x3 eigenvalue decomposition)
    elastic_3x3 = np.array([
        [c2d["C11"], c2d["C12"], c2d["C16"]],
        [c2d["C12"], c2d["C22"], c2d["C26"]],
        [c2d["C16"], c2d["C26"], c2d["C66"]],
    ])
    elastic_3x3 = (elastic_3x3 + elastic_3x3.T) / 2.0

    eigenvalues = np.linalg.eigvalsh(elastic_3x3)
    min_eigenvalue = float(np.min(eigenvalues))

    tolerance = config.born_noise_sigma * max(noise_floor_N_per_m, 1e-6)
    born_ok = bool(min_eigenvalue > -tolerance)

    if born_ok:
        reason = "ok"
        if min_eigenvalue < noise_floor_N_per_m:
            reason += f"; min_eig={min_eigenvalue:.2f}N/m"
        if symmetry_warning:
            reason += f"; {symmetry_warning}"
    else:
        parts = []
        if min_eigenvalue <= -tolerance:
            parts.append(
                f"min_eig={min_eigenvalue:.2f}N/m<=-{tolerance:.2f}N/m"
            )
        reason = "born_unstable(" + "|".join(parts) + ")"
        if symmetry_warning:
            reason += f"; {symmetry_warning}"

    return c2d, born_ok, reason, stress_residual, noise_floor_N_per_m


__all__ = [
    'compute_elastic_matrix_2d_gpaw',
]
