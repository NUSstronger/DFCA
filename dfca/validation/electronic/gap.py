"""
Electronic Gap Probing.

Functions for probing electronic band gaps using GPAW fixed-density calculations.

References:
    - GPAW band structure documentation
    - C2DB electronic structure standards
"""

import logging
from typing import Dict, Any

import numpy as np

from dfca.validation.mpi import MPI_RANK, MPI_SIZE, mpi_trace
from dfca.validation.electronic.symmetry import HAS_SPGLIB

logger = logging.getLogger(__name__)


def probe_electronic_gap(
    atoms_relaxed,
    config,
    mag_type: str,
    is_magnetic: bool,
) -> Dict[str, Any]:
    """
    Probe electronic band gap using fixed-density band calculation.

    Performs:
    1. Self-consistent SCF to obtain density and Fermi level
    2. Fixed-density band calculation along 2D high-symmetry path
    3. Gap determination based on electron count and band occupation

    All GPAW collective calculations must be executed by all MPI ranks simultaneously.

    Args:
        atoms_relaxed: Relaxed ASE Atoms object
        config: GPAWConfig instance
        mag_type: Magnetic classification
        is_magnetic: Whether structure is magnetic

    Returns:
        Dictionary with:
            - gap_eV: Band gap (eV)
            - dos_at_ef: DOS at Fermi level (states/eV/atom)
            - probe_success: Whether calculation succeeded
    """
    local_probe_result = None

    try:
        atoms = atoms_relaxed.copy()

        # ------------------------------------------------------------
        # 1. Self-consistent ground state
        # ------------------------------------------------------------
        from dfca.validation.calculators import build_gpaw_calculator
        calc = build_gpaw_calculator(
            atoms,
            config,
            mag_type,
            is_magnetic,
            for_supercell=False,
        )
        atoms.calc = calc
        atoms.get_potential_energy()

        efermi = float(calc.get_fermi_level())
        n_spins = int(calc.get_number_of_spins())
        n_electrons = float(calc.get_number_of_electrons())

        # ------------------------------------------------------------
        # 2. SCF mesh DOS estimation
        # ------------------------------------------------------------
        energy_window = 0.05
        n_atoms = len(atoms)
        window_count = 0
        mesh_eigs = []

        try:
            ibz_kpts = calc.get_ibz_k_points()

            if ibz_kpts is not None:
                for spin in range(n_spins):
                    for k in range(len(ibz_kpts)):
                        try:
                            eig = np.asarray(
                                calc.get_eigenvalues(kpt=k, spin=spin),
                                dtype=float,
                            )
                            mesh_eigs.append(eig)
                            window_count += int(
                                np.sum(np.abs(eig - efermi) <= energy_window)
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        if n_atoms > 0 and energy_window > 0 and len(mesh_eigs) > 0:
            dos_at_ef = (
                window_count
                / (2.0 * energy_window * n_atoms * max(n_spins, 1))
            )
        else:
            dos_at_ef = 999.0

        # ------------------------------------------------------------
        # 3. Fixed-density calculation along high-symmetry path
        # ------------------------------------------------------------
        bandpath = atoms.cell.bandpath(npoints=121)

        band_calc = calc.fixed_density(
            kpts=bandpath,
            symmetry='off',
            txt=None,
        )

        atoms.calc = band_calc
        atoms.get_potential_energy()

        nk = len(bandpath.kpts)
        spin_bands = []

        for spin in range(n_spins):
            rows = []
            for k in range(nk):
                rows.append(
                    np.asarray(
                        band_calc.get_eigenvalues(kpt=k, spin=spin),
                        dtype=float,
                    )
                )
            spin_bands.append(np.asarray(rows))

        # ------------------------------------------------------------
        # 4. Gap determination
        # ------------------------------------------------------------
        gap_candidates = []

        if n_spins == 1:
            # Non-spin-polarized: each band holds 2 electrons
            n_occ_float = n_electrons / 2.0
            n_occ = int(round(n_occ_float))

            if (
                abs(n_occ_float - n_occ) < 1e-6
                and n_occ >= 1
                and spin_bands[0].shape[1] > n_occ
            ):
                eigs = spin_bands[0]

                # Full path indirect gap
                vbm = float(np.max(eigs[:, n_occ - 1]))
                cbm = float(np.min(eigs[:, n_occ]))
                indirect_gap = cbm - vbm

                # Minimum direct gap along path
                direct_gaps = eigs[:, n_occ] - eigs[:, n_occ - 1]
                min_direct_gap = float(np.min(direct_gaps))

                gap_eV = max(
                    0.0,
                    min(indirect_gap, min_direct_gap),
                )

                gap_candidates.append(gap_eV)
            else:
                # Non-integer filling: treat as metal
                gap_candidates.append(0.0)

        else:
            # Spin-polarized: conservative metal detection
            metallic = False
            spin_gaps = []

            for eigs in spin_bands:
                if np.any(np.abs(eigs - efermi) <= 0.05):
                    metallic = True
                    break

                below = eigs[eigs < efermi]
                above = eigs[eigs > efermi]

                if len(below) == 0 or len(above) == 0:
                    metallic = True
                    break

                spin_gaps.append(
                    float(np.min(above) - np.max(below))
                )

            if metallic or not spin_gaps:
                gap_candidates.append(0.0)
            else:
                gap_candidates.append(max(0.0, min(spin_gaps)))

        gap_eV = (
            float(min(gap_candidates))
            if gap_candidates
            else 0.0
        )

        # Numerical zero gap: below 1 meV treated as zero
        if gap_eV < 1.0e-3:
            gap_eV = 0.0

        if MPI_RANK == 0:
            logger.info(
                f"Band gap probe: gap={gap_eV:.6f} eV, "
                f"DOS(EF)={dos_at_ef:.4f} states/eV/atom, "
                f"ne={n_electrons:.3f}, nspin={n_spins}"
            )

        local_probe_result = {
            'gap_eV': float(gap_eV),
            'dos_at_ef': float(dos_at_ef),
            'probe_success': True,
        }

    except Exception as e:
        if MPI_RANK == 0:
            logger.warning(
                f"Band gap probe failed: {e}, using conservative (metal) settings"
            )

        local_probe_result = {
            'gap_eV': 0.0,
            'dos_at_ef': 999.0,
            'probe_success': False,
        }

    # GAP calculation above is already executed collectively by all ranks.
    # Every rank therefore owns the same logical probe result locally.
    probe_result = local_probe_result

    mpi_trace(
        f"[GAP_LOCAL_RESULT] "
        f"result_type={type(probe_result).__name__} "
        f"result_value={repr(probe_result)}"
    )

    if not isinstance(probe_result, dict):
        raise RuntimeError(
            f"GAP local result invalid: "
            f"type={type(probe_result).__name__}, "
            f"value={repr(probe_result)}"
        )

    required = {
        'gap_eV',
        'dos_at_ef',
        'probe_success',
    }

    missing = required.difference(probe_result)

    if missing:
        raise RuntimeError(
            "GAP local result missing keys: "
            + repr(sorted(missing))
        )

    return probe_result


__all__ = [
    'probe_electronic_gap',
]
