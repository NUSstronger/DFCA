"""
Structure I/O Utilities.

Functions for reading and scanning structures from directories.

References:
    - ASE I/O documentation
"""

import logging
import os
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# ASE availability check
try:
    from ase.io import read, write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


def read_structure(poscar_path: str) -> Tuple[Any, Any]:
    """
    Read a structure from POSCAR/VASP file.

    Args:
        poscar_path: Path to POSCAR file

    Returns:
        Tuple of (ase_atoms, pymatgen_structure)
    """
    if not HAS_ASE:
        raise ImportError("ASE is required for structure I/O")

    atoms = read(poscar_path)

    # Also create pymatgen structure for convex hull calculations
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        struct = AseAtomsAdaptor().get_structure(atoms)
    except Exception:
        struct = None
        logger.warning(f"Failed to create pymatgen structure from {poscar_path}")

    return atoms, struct


def scan_output_directory(input_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scan input directory for candidate structures.

    Looks for subdirectories containing POSCAR files with naming pattern:
    - {mag_type}/POSCAR_{name}

    Args:
        input_dir: Directory to scan

    Returns:
        List of (mag_type, name, poscar_path) tuples
    """
    structures = []

    if not os.path.isdir(input_dir):
        logger.warning(f"Input directory not found: {input_dir}")
        return structures

    # Scan for subdirectories containing POSCAR files
    for root, dirs, files in os.walk(input_dir):
        # Sort for deterministic ordering
        dirs.sort()
        files.sort()

        for fname in files:
            if not fname.startswith("POSCAR_"):
                continue

            name = fname.replace("POSCAR_", "")

            # Determine mag_type from parent directory
            rel_path = os.path.relpath(root, input_dir)
            if rel_path == ".":
                mag_type = "unknown"
            else:
                # Use parent directory name as mag_type
                mag_type = os.path.basename(root)

            poscar_path = os.path.join(root, fname)
            structures.append((mag_type, name, poscar_path))

            if MPI_RANK == 0:
                logger.debug(f"Found structure: {mag_type}/{name}")

    return structures


# Import MPI rank for logging control
try:
    from dfca.validation.mpi import MPI_RANK
except ImportError:
    MPI_RANK = 0


__all__ = [
    'read_structure',
    'scan_output_directory',
]
