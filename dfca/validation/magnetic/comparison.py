"""
Magnetic Ground State Comparison.

Functions for comparing ferromagnetic (FM) and non-magnetic (NM) ground states.

References:
    - C2DB magnetic ground state determination
    - DFT spin-polarization studies
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

from dfca.validation.mpi import (
    MPI_RANK,
    MPI_SIZE,
    mpi_broadcast_object,
    mpi_broadcast_numpy,
    mpi_world_sum,
)
from dfca.validation.calculators import build_gpaw_calculator
from dfca.validation.electronic.symmetry import get_spacegroup_info
from dfca.validation.utils.magnetic import build_initial_magmoms

logger = logging.getLogger(__name__)


def compare_magnetic_ground_states(
    atoms_fm,
    config,
    force_nm_relax: bool = False,
) -> Dict[str, Any]:
    """
    Compare FM and NM ground state energies for a relaxed structure.

    All GPAW calculations must be executed by all MPI ranks simultaneously.
    Results are synchronized via MPI broadcast.

    Args:
        atoms_fm: Relaxed ASE Atoms object (from FM relaxation)
        config: GPAWConfig instance
        force_nm_relax: Force NM relaxation regardless of energy comparison

    Returns:
        Dictionary with:
            - fm_energy_per_atom: FM energy (eV/atom)
            - nm_energy_per_atom: NM single-point energy (eV/atom)
            - ground_state: 'FM' | 'NM' | 'degenerate'
            - fm_nm_gap_eV: Energy difference (eV/atom)
            - should_use_nm: Whether to use NM structure for further calculations
            - atoms_nm: NM-relaxed structure (if applicable)
            - nm_final_energy: NM relaxation final energy
            - nm_final_magmom: NM total magnetic moment
    """
    # FM energy - all ranks must execute GPAW calculation
    local_fm_energy = atoms_fm.get_potential_energy() / len(atoms_fm)
    fm_energy = mpi_broadcast_object(
        local_fm_energy if MPI_RANK == 0 else None,
        root=0
    )

    # NM single-point energy - all ranks must execute
    atoms_nm_sp = atoms_fm.copy()
    atoms_nm_sp.set_initial_magnetic_moments([0.0] * len(atoms_nm_sp))
    calc_nm_sp = build_gpaw_calculator(
        atoms_nm_sp, config,
        mag_type='nonmag',
        is_magnetic=False,
    )
    atoms_nm_sp.calc = calc_nm_sp
    local_nm_sp_energy = atoms_nm_sp.get_potential_energy() / len(atoms_nm_sp)
    nm_sp_energy = mpi_broadcast_object(
        local_nm_sp_energy if MPI_RANK == 0 else None,
        root=0
    )

    # Energy difference
    gap = nm_sp_energy - fm_energy
    degenerate_tol = config.fm_nm_degenerate_tol_eV
    if abs(gap) <= degenerate_tol:
        ground_state = 'degenerate'
        should_use_nm = True
    elif gap > degenerate_tol:
        ground_state = 'NM'
        should_use_nm = True
    else:
        ground_state = 'FM'
        should_use_nm = False

    result = {
        'fm_energy_per_atom': float(fm_energy),
        'nm_energy_per_atom': float(nm_sp_energy),
        'ground_state': ground_state,
        'fm_nm_gap_eV': float(gap),
        'should_use_nm': should_use_nm,
        'atoms_nm': None,
        'nm_final_energy': None,
        'nm_final_magmom': 0.0,
    }

    # NM relaxation
    if (should_use_nm and force_nm_relax) or force_nm_relax:
        if MPI_RANK == 0:
            logger.info(f"Executing NM relaxation (ground state: {ground_state})")

        local_nm_failed = 0
        local_nm_energy = 0.0
        local_nm_magmom = 0.0
        local_nm_sg = ""
        local_nm_steps = 0
        local_nm_atoms = None

        try:
            atoms_nm = atoms_fm.copy()
            atoms_nm.set_initial_magnetic_moments([0.0] * len(atoms_nm))
            calc_nm = build_gpaw_calculator(
                atoms_nm, config,
                mag_type='nonmag',
                is_magnetic=False,
            )
            atoms_nm.calc = calc_nm

            from ase.optimize import FIRE
            from ase.constraints import FrechetCellFilter
            mask = [True, True, False, False, False, True]
            cell_filter = FrechetCellFilter(atoms_nm, mask=mask)
            opt = FIRE(cell_filter, logfile=None)
            opt.run(fmax=config.relax_fmax, steps=config.relax_steps)

            local_nm_energy = atoms_nm.get_potential_energy() / len(atoms_nm)

            nm_sg_sym, nm_sg_num = get_spacegroup_info(
                atoms_nm, symprec=config.symmetry_tol
            )
            local_nm_sg = f"{nm_sg_sym}({nm_sg_num})" if nm_sg_num > 0 else nm_sg_sym

            nm_moments = 0.0
            if hasattr(atoms_nm, 'get_magnetic_moments'):
                try:
                    nm_moments = float(sum(atoms_nm.get_magnetic_moments()))
                except Exception:
                    pass
            local_nm_magmom = nm_moments
            local_nm_steps = opt.nsteps
            local_nm_atoms = atoms_nm

            if MPI_RANK == 0:
                logger.info(f"NM relaxation complete: E_NM={local_nm_energy:.6f} eV/atom")

        except Exception as e:
            local_nm_failed = 1
            from dfca.validation.mpi import mpi_trace
            mpi_trace("NM relaxation EXCEPTION: " + str(e))
            if MPI_RANK == 0:
                logger.warning(f"NM relaxation failed: {e}, falling back to FM result")

        # MPI synchronization
        if MPI_SIZE > 1:
            global_nm_failed = mpi_world_sum(local_nm_failed)
            local_nm_energy = mpi_broadcast_object(
                local_nm_energy if MPI_RANK == 0 else None, root=0
            )
            local_nm_magmom = mpi_broadcast_object(
                local_nm_magmom if MPI_RANK == 0 else None, root=0
            )
            local_nm_sg = mpi_broadcast_object(
                local_nm_sg if MPI_RANK == 0 else None, root=0
            )
            local_nm_steps = mpi_broadcast_object(
                local_nm_steps if MPI_RANK == 0 else None, root=0
            )
        else:
            global_nm_failed = local_nm_failed

        if global_nm_failed > 0:
            result['should_use_nm'] = False
            result['nm_relax_failure_detail'] = "nm_relax_failed"
        else:
            result['should_use_nm'] = True

            # Synchronize NM geometry
            if MPI_SIZE > 1:
                if MPI_RANK == 0:
                    if local_nm_atoms is None:
                        raise RuntimeError(
                            "NM relaxation succeeded but rank0 local_nm_atoms is None"
                        )
                    local_nm_positions = np.asarray(
                        local_nm_atoms.positions, dtype=float
                    )
                    local_nm_cell = np.asarray(
                        local_nm_atoms.cell, dtype=float
                    )
                else:
                    local_nm_positions = None
                    local_nm_cell = None

                nm_positions = mpi_broadcast_numpy(local_nm_positions, root=0)
                nm_cell = mpi_broadcast_numpy(local_nm_cell, root=0)
            else:
                nm_positions = np.asarray(local_nm_atoms.positions, dtype=float)
                nm_cell = np.asarray(local_nm_atoms.cell, dtype=float)

            atoms_nm_synced = atoms_fm.copy()
            atoms_nm_synced.positions = nm_positions
            atoms_nm_synced.cell = nm_cell
            atoms_nm_synced.set_initial_magnetic_moments(
                [0.0] * len(atoms_nm_synced)
            )

            result['atoms_nm'] = atoms_nm_synced
            result['nm_final_energy'] = float(local_nm_energy)
            result['nm_final_magmom'] = float(local_nm_magmom)
            result['nm_spacegroup_after'] = local_nm_sg
            result['nm_magmom_config'] = [0.0] * len(atoms_fm)
            result['nm_steps'] = local_nm_steps

    return result


__all__ = [
    'compare_magnetic_ground_states',
]
