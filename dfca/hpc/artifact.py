"""
Artifact Recovery for PBS Walltime Recovery.

Handles recovery of M1 relaxed POSCAR files that were written but
not checkpointed due to PBS walltime expiration.

Features:
    - Search for relaxed POSCAR files
    - PBS walltime artifact detection
    - Magnetic type path resolution
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# Magnetic type directories
MAGNETIC_TYPES = ['nonmag', 'mag_3d', 'mag_4d5d', 'mag_re']


def find_relaxed_poscar(
    name: str,
    output_dir: str,
    mag_type: Optional[str] = None
) -> Optional[str]:
    """
    Find a relaxed POSCAR file for a structure.

    Searches in the dft_relaxed directory hierarchy for a matching
    POSCAR file. Used for artifact recovery when PBS walltime expires
    after M1 completes but before checkpoint is saved.

    Args:
        name: Structure name (e.g., 'MoS2_001')
        output_dir: Output directory containing dft_relaxed/
        mag_type: Specific magnetic type to search (optional)

    Returns:
        Path to POSCAR file, or None if not found

    Example:
        >>> find_relaxed_poscar('MoS2_001', './gpaw_validation', 'mag_4d5d')
        './gpaw_validation/dft_relaxed/mag_4d5d/POSCAR_MoS2_001'
    """
    # Search specific magnetic type first
    if mag_type:
        search_types = [mag_type]
    else:
        search_types = MAGNETIC_TYPES

    for mtype in search_types:
        relaxed_dir = os.path.join(output_dir, 'dft_relaxed', mtype)
        poscar_path = os.path.join(relaxed_dir, f'POSCAR_{name}')

        if os.path.exists(poscar_path):
            logger.info(f"Found relaxed POSCAR: {poscar_path}")
            return poscar_path

        # Also check NM variant
        nm_path = os.path.join(relaxed_dir, f'POSCAR_{name}_nm')
        if os.path.exists(nm_path):
            logger.info(f"Found relaxed POSCAR (NM): {nm_path}")
            return nm_path

    # Try without _nm suffix if not found
    if not mag_type:
        for mtype in MAGNETIC_TYPES:
            relaxed_dir = os.path.join(output_dir, 'dft_relaxed', mtype)
            for fname in os.listdir(relaxed_dir):
                if fname.startswith(f'POSCAR_{name}') or fname.startswith(f'POSCAR_{name}_'):
                    poscar_path = os.path.join(relaxed_dir, fname)
                    logger.info(f"Found relaxed POSCAR (pattern match): {poscar_path}")
                    return poscar_path

    return None


def find_all_relaxed_poscars(output_dir: str) -> List[tuple]:
    """
    Find all relaxed POSCAR files in the output directory.

    Args:
        output_dir: Output directory containing dft_relaxed/

    Returns:
        List of tuples (name, mag_type, path)
    """
    results = []
    relaxed_dir = os.path.join(output_dir, 'dft_relaxed')

    if not os.path.exists(relaxed_dir):
        return results

    for mtype in MAGNETIC_TYPES:
        mtype_dir = os.path.join(relaxed_dir, mtype)
        if not os.path.exists(mtype_dir):
            continue

        for fname in os.listdir(mtype_dir):
            if fname.startswith('POSCAR_'):
                name = fname.replace('POSCAR_', '')
                path = os.path.join(mtype_dir, fname)
                results.append((name, mtype, path))

    return results


def is_valid_relaxed_poscar(path: str) -> bool:
    """
    Check if a POSCAR file is valid (exists and is readable).

    Args:
        path: Path to POSCAR file

    Returns:
        True if file exists and is readable
    """
    return os.path.exists(path) and os.path.isfile(path) and os.access(path, os.R_OK)


__all__ = [
    'find_relaxed_poscar',
    'find_all_relaxed_poscars',
    'is_valid_relaxed_poscar',
    'MAGNETIC_TYPES',
]
