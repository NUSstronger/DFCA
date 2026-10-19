"""
K-point Sampling Utilities.

Functions for adaptive k-point mesh calculation based on cell size.
Follows C2DB standards: Δk ≤ 0.03 Å⁻¹.

References:
    - C2DB k-point convergence criteria
    - Monkhorst-Pack grid generation
"""

from typing import List, Tuple

import numpy as np


def adaptive_kpoints(
    atoms,
    dk: float = 0.03,
    max_nk: int = 10,
) -> Tuple[List[int], List[int]]:
    """
    Calculate adaptive k-point mesh for 2D materials.

    Uses reciprocal space lattice vectors to determine k-point density.
    Ensures Δk ≤ dk Å⁻¹ for convergence.

    Args:
        atoms: ASE Atoms object
        dk: Maximum k-point spacing (Å⁻¹)
        max_nk: Maximum k-points per direction (prevents mesh explosion)

    Returns:
        Tuple of (kpts, koffset)
            - kpts: [nk1, nk2, 1] k-point grid
            - koffset: [0, 0, 0] (Gamma-centered)

    Note:
        c-direction is always 1 (vacuum layer)
    """
    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    # c = np.linalg.norm(cell[2])  # Not used for k-points

    # Calculate reciprocal lattice vector lengths
    b1_length = 2 * np.pi / a if a > 0 else 1.0
    b2_length = 2 * np.pi / b if b > 0 else 1.0

    # k-points = reciprocal length / spacing
    # Minimum 2 for good Gamma-point sampling
    # Maximum max_nk to prevent mesh explosion
    nk1 = min(max(2, int(np.ceil(b1_length / dk))), max_nk)
    nk2 = min(max(2, int(np.ceil(b2_length / dk))), max_nk)

    # c-direction fixed at 1 (vacuum)
    return [nk1, nk2, 1], [0, 0, 0]  # Gamma-centered


def kpoints_to_string(kpts: List[int], koffset: List[int] = None) -> str:
    """
    Format k-points as a string for GPAW.

    Args:
        kpts: K-point grid [nk1, nk2, nk3]
        koffset: K-point offset (default [0, 0, 0])

    Returns:
        String representation
    """
    if koffset is None:
        koffset = [0, 0, 0]
    return f"{kpts[0]} {kpts[1]} {kpts[2]} 0 0 0"


def estimate_kpoint_convergence(
    atoms,
    target_dk: float = 0.03,
    min_kpts: int = 2,
) -> dict:
    """
    Estimate k-point convergence parameters for a structure.

    Args:
        atoms: ASE Atoms object
        target_dk: Target k-point spacing (Å⁻¹)
        min_kpts: Minimum k-points per direction

    Returns:
        Dictionary with convergence info
    """
    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])

    b1_length = 2 * np.pi / a if a > 0 else 1.0
    b2_length = 2 * np.pi / b if b > 0 else 1.0

    nk1 = int(np.ceil(b1_length / target_dk))
    nk2 = int(np.ceil(b2_length / target_dk))

    actual_dk1 = b1_length / max(nk1, 1)
    actual_dk2 = b2_length / max(nk2, 1)

    return {
        'nk1': max(nk1, min_kpts),
        'nk2': max(nk2, min_kpts),
        'target_dk': target_dk,
        'actual_dk1': actual_dk1,
        'actual_dk2': actual_dk2,
        'converged': actual_dk1 <= target_dk and actual_dk2 <= target_dk,
    }


__all__ = [
    'adaptive_kpoints',
    'kpoints_to_string',
    'estimate_kpoint_convergence',
]
