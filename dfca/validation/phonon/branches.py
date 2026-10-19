"""
Phonon Branch Identification.

Functions for identifying acoustic and optical branches in phonon spectra.

References:
    - C2DB phonon analysis standards
    - 2D materials phonon physics
"""

from typing import Dict, Any

import numpy as np


def identify_phonon_branches(
    freqs_THz: np.ndarray,
    qpoints: np.ndarray,
    n_atoms: int,
) -> Dict[str, Any]:
    """
    Identify acoustic and optical branches in phonon spectrum.

    In 2D materials there are 3N phonon branches where 3 are acoustic
    and 3N-3 are optical. The acoustic branches are:
    - ZA (flexural): out-of-plane, ω(k) ~ k²
    - TA (transverse): in-plane transverse, ω(k) ~ k
    - LA (longitudinal): in-plane longitudinal, ω(k) ~ k

    Args:
        freqs_THz: Frequency array (n_qpoints, n_branches) in THz
        qpoints: q-point array (n_qpoints, 3)
        n_atoms: Number of atoms in unit cell

    Returns:
        Dictionary with:
            - za_branch_idx: ZA branch index (-1 if not found)
            - ta_branch_idx: TA branch index
            - la_branch_idx: LA branch index
            - acoustic_indices: List of acoustic branch indices
            - optical_branch_indices: List of optical branch indices
            - is_identified: Whether identification succeeded
    """
    n_branches = freqs_THz.shape[1]
    n_acoustic = 3

    # Γ-point frequencies (average of first few points to reduce noise)
    n_gamma_points = min(5, len(freqs_THz))
    gamma_avg_freqs = np.mean(freqs_THz[:n_gamma_points], axis=0)

    # Sort by Γ-point frequency to identify acoustic branches
    sorted_indices = np.argsort(gamma_avg_freqs)

    # First 3 lowest-frequency branches are acoustic
    acoustic_indices = sorted_indices[:n_acoustic].tolist()
    optical_indices = sorted_indices[n_acoustic:].tolist()

    # Identify ZA/TA/LA by frequency vs q dependence
    # ZA: smallest slope (quadratic behavior)
    # TA/LA: larger slopes (linear behavior)

    acoustic_slopes = []
    for idx in acoustic_indices:
        # Frequencies near Γ-point (skip Γ itself)
        q_norms = np.linalg.norm(qpoints[1:10], axis=1)
        freqs_near_gamma = freqs_THz[1:10, idx]

        if len(freqs_near_gamma) > 1 and np.max(q_norms) > 1e-6:
            valid_mask = freqs_near_gamma > 0
            if np.sum(valid_mask) >= 2:
                slope = np.polyfit(
                    q_norms[valid_mask],
                    freqs_near_gamma[valid_mask],
                    1
                )[0]
            else:
                slope = 0.0
        else:
            slope = 0.0
        acoustic_slopes.append((idx, slope, gamma_avg_freqs[idx]))

    # Sort by slope: smallest = ZA
    acoustic_slopes.sort(key=lambda x: (x[1], x[2]))

    if len(acoustic_slopes) >= 3:
        za_idx = acoustic_slopes[0][0]
        ta_idx = acoustic_slopes[1][0]
        la_idx = acoustic_slopes[2][0]
    else:
        # Fallback: use default order
        za_idx = acoustic_indices[0]
        ta_idx = acoustic_indices[1] if len(acoustic_indices) > 1 else -1
        la_idx = acoustic_indices[2] if len(acoustic_indices) > 2 else -1

    return {
        'za_branch_idx': int(za_idx),
        'ta_branch_idx': int(ta_idx) if ta_idx >= 0 else -1,
        'la_branch_idx': int(la_idx) if la_idx >= 0 else -1,
        'acoustic_indices': acoustic_indices,
        'optical_branch_indices': optical_indices,
        'is_identified': True,
    }


def _get_2d_layer_normal(atoms) -> np.ndarray:
    """
    Return unit normal of the first two lattice vectors.

    Args:
        atoms: ASE Atoms object

    Returns:
        Unit normal vector (3D)
    """
    cell = np.asarray(atoms.cell.array, dtype=float)

    normal = np.cross(cell[0], cell[1])
    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        raise ValueError("Degenerate in-plane lattice vectors")

    return normal / norm


def _phonon_mode_out_of_plane_fraction(
    eigenvector: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Return fraction of phonon eigenvector polarized normal to layer.

    Args:
        eigenvector: Phonon eigenvector (3*n_atoms)
        normal: Unit layer normal vector

    Returns:
        Fraction of out-of-plane polarization
    """
    vec = np.asarray(eigenvector).reshape((-1, 3))

    projection = vec @ normal

    out_power = np.sum(np.abs(projection) ** 2)
    total_power = np.sum(np.abs(vec) ** 2)

    if total_power < 1e-20:
        return 0.0

    return float(out_power / total_power)


def _physical_q_norms_2d(atoms, qpoints: np.ndarray) -> np.ndarray:
    """
    Convert reduced q coordinates to reciprocal-space |q| in Å⁻¹.

    Args:
        atoms: ASE Atoms object
        qpoints: q-point array in reduced coordinates

    Returns:
        Array of q-vector magnitudes
    """
    cell = np.asarray(atoms.cell.array, dtype=float)

    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T

    qcart = np.asarray(qpoints, dtype=float) @ reciprocal

    return np.linalg.norm(qcart, axis=1)


__all__ = [
    'identify_phonon_branches',
    '_get_2d_layer_normal',
    '_phonon_mode_out_of_plane_fraction',
    '_physical_q_norms_2d',
]
