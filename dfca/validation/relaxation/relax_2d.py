"""
2D Structure Relaxation.

Functions for high-precision 2D structure relaxation using GPAW.
Fixed c-axis, symmetry constraints, and checkpoint/resume support.

References:
    - C2DB relaxation standards
    - ASE optimization documentation
"""

import logging
import os
import shutil
import time
from typing import Dict, Tuple, Any, Optional

import numpy as np

from dfca.validation.mpi import (
    MPI_RANK,
    MPI_SIZE,
    _gpaw_world,
    mpi_broadcast_object,
    mpi_world_sum,
    mpi_trace,
)
from dfca.validation.calculators import build_gpaw_calculator
from dfca.validation.electronic.symmetry import get_spacegroup_info
from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms

logger = logging.getLogger(__name__)

# FixSymmetry availability
try:
    try:
        from ase.constraints import FixSymmetry
    except ImportError:
        from ase.spacegroup.symmetrize import FixSymmetry
    HAS_FIX_SYMMETRY = True
except ImportError:
    HAS_FIX_SYMMETRY = False


def relax_structure_2d(
    atoms,
    config,
    max_retries: int = 2,
    name: str = "",
) -> Tuple[Any, Dict]:
    """
    High-precision 2D structure relaxation with GPAW.

    Fixed c-axis and vacuum layer, optimizes in-plane lattice constants a, b
    and atomic coordinates. Optional symmetry constraints prevent high-symmetry
    materials from drifting from theoretical symmetric positions.

    Features:
        - 2D constraints (fix c-axis)
        - Optional symmetry constraints (FixSymmetry)
        - Checkpoint/resume for interrupted relaxations
        - Zero-step convergence verification
        - MPI-safe with broadcast synchronization

    Args:
        atoms: Initial ASE Atoms object
        config: GPAWConfig instance
        max_retries: Max SCF non-convergence retries
        name: Material name (for checkpointing and logging)

    Returns:
        Tuple of (relaxed_atoms, result_dict)
    """
    result = {
        'converged': False,
        'fmax_reached': 0.0,
        'steps': 0,
        'energy_eV': 0.0,
        'energy_per_atom': 0.0,
        'time_s': 0.0,
        'pass': False,
        'reason': '',
        'error': '',
        'spacegroup_before': '',
        'spacegroup_after': '',
        'symmetry_constrained': False,
        'resumed_from_partial': False,
    }

    # Checkpoint/resume variables
    traj_dir = None
    traj_file = None
    resume_from_partial = False
    start_atoms = atoms
    partial_n_frames = 0

    if name and MPI_RANK == 0:
        traj_dir = os.path.join(config.output_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = os.path.join(traj_dir, f"{name}_relax.traj")

        # Check for interrupted trajectory
        if os.path.exists(traj_file) and os.path.getsize(traj_file) > 0:
            traj_read_ok = False
            try:
                from ase.io.trajectory import Trajectory
                with Trajectory(traj_file, 'r') as old_traj:
                    n_frames = len(old_traj)
                    if n_frames > 0:
                        last_frame = old_traj[-1]
                        if (last_frame.get_chemical_formula() == atoms.get_chemical_formula()
                                and len(last_frame) == len(atoms)):
                            start_atoms = last_frame.copy()
                            resume_from_partial = True
                            partial_n_frames = n_frames
                            traj_read_ok = True
                            logger.info(
                                f"[{name}] Detected incomplete trajectory "
                                f"({n_frames} frames), resuming from frame {n_frames}"
                            )
                        else:
                            logger.warning(
                                f"[{name}] Trajectory file mismatch, restarting"
                            )
                            try:
                                shutil.move(traj_file, traj_file + ".bak")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"[{name}] Failed to read old trajectory: {e}")
                try:
                    shutil.move(traj_file, traj_file + ".corrupted_bak")
                except Exception:
                    pass

    # MPI broadcast for resume state
    try:
        _resume_data = None
        if MPI_RANK == 0:
            if name:
                traj_dir_local = os.path.join(config.output_dir, "trajectories")
                os.makedirs(traj_dir_local, exist_ok=True)
                restart_struct_file_local = os.path.join(
                    traj_dir_local, f"{name}_restart.traj"
                )
                traj_file_local = os.path.join(traj_dir_local, f"{name}_relax.traj")

                # 1. Prefer restart.traj
                resume_atoms = None
                partial_n_frames_local = 0
                if (restart_struct_file_local
                        and os.path.exists(restart_struct_file_local)
                        and os.path.getsize(restart_struct_file_local) > 0):
                    try:
                        from ase.io.trajectory import Trajectory
                        with Trajectory(restart_struct_file_local, 'r') as tj:
                            if len(tj) > 0:
                                last_frame = tj[-1]
                                if (last_frame.get_chemical_formula()
                                        == atoms.get_chemical_formula()
                                        and len(last_frame) == len(atoms)):
                                    resume_atoms = last_frame.copy()
                                    logger.info(
                                        f"[{name}] Detected restart structure file"
                                    )
                    except Exception as e:
                        logger.warning(f"[{name}] Failed to read restart file: {e}")

                # 2. Fallback to legacy .traj
                if (resume_atoms is None and traj_file_local
                        and os.path.exists(traj_file_local)
                        and os.path.getsize(traj_file_local) > 0):
                    try:
                        from ase.io.trajectory import Trajectory
                        with Trajectory(traj_file_local, 'r') as old_traj:
                            n_frames = len(old_traj)
                            if n_frames > 0:
                                last_frame = old_traj[-1]
                                if (last_frame.get_chemical_formula()
                                        == atoms.get_chemical_formula()
                                        and len(last_frame) == len(atoms)):
                                    resume_atoms = last_frame.copy()
                                    partial_n_frames_local = n_frames
                                    logger.info(
                                        f"[{name}] No restart file, using legacy .traj"
                                    )
                    except Exception as e:
                        logger.warning(f"[{name}] Failed to read .traj: {e}")

                # Construct resume_data
                if resume_atoms is not None:
                    _resume_data = {
                        'resume': True,
                        'positions': resume_atoms.positions.tolist(),
                        'cell': resume_atoms.cell.tolist(),
                        'n_frames': partial_n_frames_local,
                        'source': ('restart_file'
                                   if os.path.exists(restart_struct_file_local)
                                   else 'legacy_traj')
                    }
                else:
                    _resume_data = {'resume': False}
            else:
                _resume_data = {'resume': False}

        # Broadcast to all ranks
        resume_data = mpi_broadcast_object(_resume_data, root=0)
        mpi_trace(
            "M1 after resume sync resume="
            + str(resume_data.get('resume', False))
        )

        # Apply resume state
        if resume_data.get('resume', False):
            resume_from_partial = True
            atoms.positions = np.asarray(resume_data['positions'], dtype=float)
            atoms.cell = np.asarray(resume_data['cell'], dtype=float)
            partial_n_frames = resume_data.get('n_frames', 0)
            if MPI_RANK == 0:
                logger.info(
                    f"[{name}] MPI broadcast complete: all ranks synchronized "
                    f"to checkpoint (resuming from frame {partial_n_frames})"
                )
        else:
            resume_from_partial = False

    except Exception as bc_err:
        if MPI_RANK == 0:
            logger.error(f"[{name}] Resume MPI broadcast failed (fatal): {bc_err}")
            logger.error("Atom coordinates/cell mismatch causes MPI deadlock, exiting")
        _gpaw_world.barrier()
        import sys
        sys.exit(1)

    # Magnetic classification
    elements = atoms.get_chemical_symbols()
    screen_mag_type = atoms.info.get('screen_mag_type')

    if screen_mag_type in ('nonmag', 'mag_3d', 'mag_4d5d', 'mag_re'):
        mag_type = screen_mag_type
        is_magnetic = (screen_mag_type != 'nonmag')
    else:
        mag_type, is_magnetic = classify_magnetic(elements)

    # Pre-relaxation logging
    if MPI_RANK == 0:
        if hasattr(atoms, 'cell'):
            a_ang = float(np.linalg.norm(atoms.cell[0]))
            b_ang = float(np.linalg.norm(atoms.cell[1]))
            dk = config.dk_per_ang
            b1_len = 2 * np.pi / a_ang if a_ang > 0 else 1.0
            b2_len = 2 * np.pi / b_ang if b_ang > 0 else 1.0
            kx = max(2, int(np.ceil(b1_len / dk)))
            ky = max(2, int(np.ceil(b2_len / dk)))
            kpts_repr = f"[{kx}, {ky}, 1]"
        else:
            kpts_repr = "unknown"
        recovery_mode = "resuming from partial" if resume_from_partial else "starting fresh"
        logger.info(
            f"[{name}] M1 relaxation start: mode={recovery_mode} "
            f"| kpts={kpts_repr} | mag={mag_type} "
            f"| optimizer={config.optimizer}"
        )

    # Record pre-relaxation spacegroup
    sg_before_sym, sg_num_before = get_spacegroup_info(
        atoms, symprec=config.symmetry_tol
    )
    result['spacegroup_before'] = (
        f"{sg_before_sym}({sg_num_before})"
        if sg_num_before > 0 else sg_before_sym
    )

    # Set magnetic moments
    result['mag_type'] = mag_type
    result['is_magnetic'] = is_magnetic

    if config.always_spinpol:
        magmoms = build_initial_magmoms(atoms)
        for i, el in enumerate(elements):
            if el not in {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                          'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd',
                          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                          'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                          'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}:
                magmoms[i] = 0.1
        atoms.set_initial_magnetic_moments(magmoms)
        is_magnetic = True
    elif is_magnetic:
        magmoms = build_initial_magmoms(atoms)
        atoms.set_initial_magnetic_moments(magmoms)

    result['magmom_config'] = (
        list(atoms.get_initial_magnetic_moments())
        if hasattr(atoms, 'get_initial_magnetic_moments') else []
    )

    # Build calculator
    calc = build_gpaw_calculator(atoms, config, mag_type, is_magnetic)
    atoms.calc = calc

    # Choose optimizer
    optimizer_class = (
        __import__('ase.optimize', fromlist=['FIRE', 'BFGS']).FIRE
        if config.optimizer == "FIRE" else
        __import__('ase.optimize', fromlist=['FIRE', 'BFGS']).BFGS
    )

    # 2D constraint mask: [xx, yy, zz, xy, yz, zx]
    # a, b allowed; c, angles fixed; xy shear allowed
    if config.fix_cell == "ab":
        mask = [True, True, False, False, False, True]
    elif config.fix_cell == "full":
        mask = [False, False, False, False, False, False]
    else:
        mask = [True, True, False, False, False, True]

    # Apply symmetry constraint if available
    if HAS_FIX_SYMMETRY:
        local_sym_failed = 0
        try:
            sym_constraint = FixSymmetry(atoms, symprec=config.symmetry_tol)
            atoms.set_constraint(sym_constraint)
        except Exception as e:
            local_sym_failed = 1
            mpi_trace(f"FixSymmetry EXCEPTION: {e}")
            if MPI_RANK == 0:
                logger.warning(f"Failed to apply symmetry constraint: {e}")

        if MPI_SIZE > 1:
            global_sym_failed = mpi_world_sum(local_sym_failed)
        else:
            global_sym_failed = local_sym_failed

        if global_sym_failed > 0:
            atoms.set_constraint(None)
            result['symmetry_constrained'] = False
            if MPI_RANK == 0:
                logger.warning(
                    "Symmetry constraint failed, continuing without"
                )
        else:
            result['symmetry_constrained'] = True
            if MPI_RANK == 0:
                logger.info(
                    f"Applied symmetry constraint (symprec={config.symmetry_tol})"
                )

    # Create CellFilter
    from ase.constraints import FrechetCellFilter
    cell_filter = FrechetCellFilter(atoms, mask=mask)

    # Callbacks
    def _log_m1_step():
        """FIRE step log callback - only reads cached results, no GPAW collective."""
        if MPI_RANK != 0:
            return
        try:
            results = getattr(atoms.calc, "results", {})
            energy_cached = results.get("energy", results.get("free_energy", None))
            if energy_cached is not None:
                logger.info(f"[{name}] M1 FIRE step cached_energy={energy_cached}")
        except Exception as e:
            logger.warning(f"[{name}] M1 step log failed: {e}")

    def _save_m1_restart_step():
        """Save restart structure each step - only reads cached results."""
        if MPI_RANK != 0 or not name:
            return
        try:
            restart_struct_file = os.path.join(
                config.output_dir, "trajectories", name + "_restart.traj"
            )
            results = getattr(atoms.calc, "results", {})
            energy_cached = results.get("energy", results.get("free_energy", None))
            forces_cached = results.get("forces", None)
            if energy_cached is None or forces_cached is None:
                return
            from ase.calculators.singlepoint import SinglePointCalculator
            from ase.io.trajectory import Trajectory
            atoms_snap = atoms.copy()
            atoms_snap.set_constraint(None)
            atoms_snap.calc = SinglePointCalculator(
                atoms_snap,
                energy=float(energy_cached),
                forces=np.asarray(forces_cached).copy()
            )
            tmp_path = restart_struct_file + ".tmp"
            with Trajectory(tmp_path, "w") as traj:
                traj.write(atoms_snap)
            os.replace(tmp_path, restart_struct_file)
        except Exception as e:
            logger.warning(f"[{name}] restart save failed: {e}")

    # Relaxation loop
    t0 = time.time()
    retry_count = 0

    while retry_count <= max_retries:
        local_failed = 0
        local_error = ""
        local_fmax = 0.0
        local_energy = 0.0
        local_n_steps = 0

        mpi_trace("M1 before optimizer construction")
        opt = optimizer_class(cell_filter, logfile=None, trajectory=None)
        mpi_trace("M1 after optimizer construction")

        opt.attach(_log_m1_step, interval=1)
        if os.environ.get("DFCA_DISABLE_RESTART_SAVE") != "1":
            opt.attach(_save_m1_restart_step, interval=1)
        else:
            mpi_trace("[MPI_DIAG] restart callback DISABLED")

        try:
            mpi_trace("M1 before opt.run")
            opt.run(fmax=config.relax_fmax, steps=config.relax_steps)
            mpi_trace("M1 after opt.run")

            local_fmax = float(
                np.linalg.norm(atoms.get_forces(), axis=1).max()
            )
            local_energy = atoms.get_potential_energy()
            local_n_steps = opt.nsteps

        except Exception as e:
            local_failed = 1
            local_error = str(e)
            mpi_trace("M1 retry=" + str(retry_count) + " EXCEPTION: " + local_error)

        if MPI_SIZE > 1:
            global_failed = mpi_world_sum(local_failed)
        else:
            global_failed = local_failed

        if global_failed > 0:
            retry_count += 1
            if retry_count > max_retries:
                mpi_trace("M1 EXIT retry exhausted")
                return atoms, {
                    'pass': False,
                    'error': f'relaxation_failed_after_{max_retries}_retries: '
                             + local_error,
                    'time_s': time.time() - t0
                }
            if MPI_RANK == 0:
                logger.warning(
                    f"Relaxation not converged, retrying ({retry_count}/"
                    f"{max_retries}): {local_error}"
                )
            try:
                calc = build_gpaw_calculator(atoms, config, mag_type, is_magnetic)
                atoms.calc = calc
            except Exception as e2:
                mpi_trace("M1 EXIT calc rebuild failed")
                return atoms, {
                    'pass': False,
                    'error': 'calculator_rebuild_failed: ' + str(e2),
                    'time_s': time.time() - t0
                }
        else:
            # Success - broadcast results
            energy = mpi_broadcast_object(
                local_energy if MPI_RANK == 0 else None, root=0
            )
            fmax = mpi_broadcast_object(
                local_fmax if MPI_RANK == 0 else None, root=0
            )
            n_steps = mpi_broadcast_object(
                local_n_steps if MPI_RANK == 0 else None, root=0
            )

            # Zero-step verification
            zero_step_verified = False
            zero_step_energy_delta = 0.0
            zero_step_verify_passed = False

            if n_steps == 0 and config.zero_step_verify:
                if MPI_RANK == 0:
                    logger.info("Zero-step convergence detected, running verification...")

                local_zero_failed = 0
                local_zero_verified = False
                local_zero_passed = False
                local_zero_delta = 0.0

                try:
                    atoms_perturbed = atoms.copy()
                    disp = config.zero_step_displacement
                    np.random.seed(42)
                    for _i in range(len(atoms_perturbed)):
                        rng_i = np.random.RandomState(42 + _i)
                        dx = rng_i.uniform(-disp, disp)
                        rng_i = np.random.RandomState(42 + _i + 1000)
                        dy = rng_i.uniform(-disp, disp)
                        cell = atoms_perturbed.get_cell()
                        atoms_perturbed.positions[_i] += dx * cell[0] + dy * cell[1]

                    calc_v = build_gpaw_calculator(
                        atoms_perturbed, config, mag_type, is_magnetic
                    )
                    atoms_perturbed.calc = calc_v
                    cell_filter_v = FrechetCellFilter(atoms_perturbed, mask=mask)
                    opt_v = optimizer_class(cell_filter_v, logfile=None)
                    opt_v.run(fmax=config.relax_fmax, steps=config.relax_steps)
                    energy_perturbed = atoms_perturbed.get_potential_energy()
                    local_zero_delta = abs(
                        energy_perturbed / len(atoms_perturbed)
                        - energy / len(atoms)
                    )

                    if local_zero_delta <= config.zero_step_verify_threshold:
                        local_zero_verified = True
                        local_zero_passed = True
                        if MPI_RANK == 0:
                            logger.info(
                                f"Zero-step verification passed: ΔE="
                                f"{local_zero_delta:.6f} <= "
                                f"{config.zero_step_verify_threshold} eV/atom"
                            )
                    else:
                        local_zero_passed = False
                        if MPI_RANK == 0:
                            logger.warning(
                                f"Zero-step verification failed: ΔE="
                                f"{local_zero_delta:.6f} > "
                                f"{config.zero_step_verify_threshold} eV/atom"
                            )

                except Exception as e:
                    local_zero_failed = 1
                    if MPI_RANK == 0:
                        logger.warning(f"Zero-step verification failed: {e}")

                if MPI_SIZE > 1:
                    global_zero_failed = mpi_world_sum(local_zero_failed)
                else:
                    global_zero_failed = local_zero_failed

                if global_zero_failed > 0:
                    zero_step_verified = False
                    zero_step_verify_passed = False
                    zero_step_energy_delta = 0.0
                else:
                    zero_step_verified = mpi_broadcast_object(
                        local_zero_verified if MPI_RANK == 0 else None, root=0
                    )
                    zero_step_verify_passed = mpi_broadcast_object(
                        local_zero_passed if MPI_RANK == 0 else None, root=0
                    )
                    zero_step_energy_delta = mpi_broadcast_object(
                        local_zero_delta if MPI_RANK == 0 else None, root=0
                    )

                try:
                    atoms_perturbed.calc = None
                except Exception:
                    pass

            result.update({
                'converged': True,
                'fmax_reached': float(fmax),
                'steps': n_steps,
                'energy_eV': float(energy),
                'energy_per_atom': float(energy / len(atoms)),
                'time_s': time.time() - t0,
                'pass': True,
                'reason': 'ok' if fmax < config.relax_fmax else (
                    f'fmax={fmax}>{config.relax_fmax}'
                ),
                'retry_count': retry_count,
                'resumed_from_partial': resume_from_partial,
                'zero_step_verification_triggered': (n_steps == 0),
                'zero_step_verified': zero_step_verified,
                'zero_step_energy_delta': zero_step_energy_delta,
                'zero_step_verify_passed': zero_step_verify_passed,
            })
            break

    # Record post-relaxation spacegroup
    sg_after_sym, sg_num_after = get_spacegroup_info(
        atoms, symprec=config.symmetry_tol
    )
    result['spacegroup_after'] = (
        f"{sg_after_sym}({sg_num_after})"
        if sg_num_after > 0 else sg_after_sym
    )

    if sg_num_before > 0 and sg_num_after > 0 and sg_num_before != sg_num_after:
        result['reason'] += f"; spacegroup_changed:{sg_num_before}->{sg_num_after}"

    # Record final magnetic moment
    if hasattr(atoms, 'get_magnetic_moments'):
        try:
            moments = atoms.get_magnetic_moments()
            result['final_magnetic_moment'] = float(sum(moments))
        except Exception:
            pass

    # Detach GPAW calculator
    try:
        atoms.calc = None
    except Exception:
        pass

    return atoms, result


__all__ = [
    'relax_structure_2d',
]
