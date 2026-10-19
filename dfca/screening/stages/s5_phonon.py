"""
Screening Stage S5: Phonon Stability.

Features:
    - Phonon dispersion calculation using phonopy
    - Adaptive supercell sizing
    - C2DB Sec. 2.7 compliance (global min_freq >= -0.02 THz)

References:
    - C2DB Sec. 2.7: Dynamical stability
"""

import logging
import numpy as np
from typing import Tuple, Dict, List
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

# Optional phonopy import
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    HAS_PHONOPY = True
except ImportError:
    HAS_PHONOPY = False

# Optional ASE import
try:
    from ase.atoms import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Atoms = None


def _pmg_to_phonopy_atoms(struct: Structure) -> PhonopyAtoms:
    """Convert pymatgen Structure to phonopy Atoms."""
    return PhonopyAtoms(
        symbols=[str(s.specie) for s in struct],
        scaled_positions=[s.frac_coords for s in struct],
        cell=struct.lattice.matrix,
    )


def _phonopy_to_ase(ph_atoms: PhonopyAtoms) -> Atoms:
    """Convert phonopy Atoms to ASE Atoms."""
    return Atoms(
        symbols=ph_atoms.symbols,
        scaled_positions=ph_atoms.scaled_positions,
        cell=ph_atoms.cell,
        pbc=[True, True, False],  # 2D: no pbc in z
    )


def _adaptive_supercell(n_atoms: int, base: List[int], max_atoms: int = 200) -> List[int]:
    """
    Adaptively adjust supercell size based on atom count.

    Ensures sufficient supercell for phonon calculation while
    staying within max_atoms limit.

    Args:
        n_atoms: Number of atoms in primitive cell
        base: Base supercell dimensions [nx, ny, nz]
        max_atoms: Maximum atoms in supercell

    Returns:
        Adjusted supercell dimensions
    """
    sc = list(base)

    # Adjust based on atom count
    if n_atoms <= 2:
        sc[0], sc[1] = max(sc[0], 5), max(sc[1], 5)
    elif n_atoms <= 4:
        sc[0], sc[1] = max(sc[0], 4), max(sc[1], 4)
    elif n_atoms <= 8:
        sc[0], sc[1] = max(sc[0], 3), max(sc[1], 3)

    # Reduce if too large
    while n_atoms * sc[0] * sc[1] > max_atoms and (sc[0] > 1 or sc[1] > 1):
        if sc[0] > sc[1]:
            sc[0] = max(1, sc[0] - 1)
        else:
            sc[1] = max(1, sc[1] - 1)

    sc[2] = 1  # No extension in z
    return sc


def check_phonon_stability(
    struct: Structure,
    backend,
    cfg
) -> Tuple[bool, Dict]:
    """
    S5: Phonon dynamical stability check.

    C2DB Sec. 2.7 standard:
        "Dynamical stability requires that the dynamical matrix has no imaginary
         eigenvalues, i.e. all frequencies are real and positive."

    Simplified implementation (aligned with C2DB standard):
        - Global minimum frequency tolerance: cfg.imaginary_freq_tol (default -0.02 THz)
        - Allows small numerical noise tolerance

    Bug #15 fix: Explicit skip via cfg.skip_phonon flag.

    Args:
        struct: pymatgen Structure
        backend: MLBackend instance
        cfg: ScreeningConfig

    Returns:
        (passed, info_dict)
    """
    # Bug #15 fix: explicit skip
    if cfg.skip_phonon:
        return True, {
            "min_freq_THz": 0.0,
            "gamma_min_THz": 0.0,
            "imaginary_count": 0,
            "supercell": [],
            "reason": "phonon_skipped",
        }

    if not (HAS_PHONOPY and HAS_ASE):
        return False, {
            "reason": "phonopy/ase_not_installed",
            "min_freq_THz": 0.0,
        }

    try:
        n_atoms = len(struct)

        # Adaptive supercell
        sc_dims = _adaptive_supercell(n_atoms, cfg.phonon_supercell)
        sc_mat = np.diag(sc_dims)

        # Convert to phonopy format
        ph_atoms = _pmg_to_phonopy_atoms(struct)
        phonon = Phonopy(ph_atoms, sc_mat)
        phonon.generate_displacements(distance=cfg.phonon_displacement)

        # Get displaced supercells
        supercells = phonon.supercells_with_displacements
        calc = backend.get_calculator()

        # Compute forces for each displaced supercell
        forces = []
        for sc in supercells:
            atoms = _phonopy_to_ase(sc)
            atoms.calc = calc
            forces.append(atoms.get_forces())

        phonon.forces = forces
        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()

        # Adaptive mesh for frequency evaluation
        min_ab = min(struct.lattice.a, struct.lattice.b)
        ref_lattice = 3.0
        mesh_scale = max(0.5, min(1.5, ref_lattice / min_ab))
        adaptive_mesh = [
            max(8, int(cfg.phonon_mesh[0] * mesh_scale)),
            max(8, int(cfg.phonon_mesh[1] * mesh_scale)),
            1
        ]

        # Compute phonon dispersion
        phonon.run_mesh(adaptive_mesh, with_eigenvectors=False, is_mesh_symmetry=True)

        # Extract frequencies
        if hasattr(phonon, "mesh") and phonon.mesh is not None:
            freqs = np.array(phonon.mesh.frequencies)
        else:
            mesh_dict = phonon.get_mesh_dict()
            freqs = np.array(mesh_dict["frequencies"])

        # C2DB Sec. 2.7: Global minimum frequency criterion
        min_freq_THz = float(freqs.min())
        tolerance_THz = cfg.imaginary_freq_tol  # Default: -0.02 THz
        imaginary_count = int((freqs < tolerance_THz).sum())

        # Stability check: min_freq >= tolerance
        is_stable = bool(min_freq_THz >= tolerance_THz)

        if is_stable:
            reason = "ok"
        else:
            reason = f"phonon_unstable(min_freq={min_freq_THz:.3f}THz<{tolerance_THz}THz)"

        return is_stable, {
            "min_freq_THz": round(min_freq_THz, 4),
            "gamma_min_THz": round(min_freq_THz, 4),  # Simplified: use global min
            "imaginary_count": imaginary_count,
            "supercell": sc_dims,
            "reason": reason,
        }

    except Exception as e:
        logger.warning(f"Phonon check failed for {struct.composition.reduced_formula}: {e}")
        return False, {
            "reason": f"phonon_check_failed:{e}",
            "min_freq_THz": 0.0,
            "gamma_min_THz": 0.0,
            "imaginary_count": 0,
        }


__all__ = [
    'check_phonon_stability',
    '_adaptive_supercell',
    '_pmg_to_phonopy_atoms',
    '_phonopy_to_ase',
]
