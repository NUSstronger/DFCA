"""
I/O Utilities for GPAW Validation Pipeline.

Functions for scanning directories and reading/writing structure files.
"""

import logging
import os
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def scan_output_directory(input_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scan a screening output directory for structures ready for DFT validation.

    Searches for POSCAR files in magnetic type subdirectories.

    Args:
        input_dir: Path to screening output directory

    Returns:
        List of tuples: (mag_type, name, poscar_path)
    """
    structures = []

    if not os.path.isdir(input_dir):
        logger.warning(f"Input directory not found: {input_dir}")
        return structures

    # Magnetic type directories
    magnetic_dirs = ['nonmag', 'mag_3d', 'mag_4d5d', 'mag_re']

    for mag_dir in magnetic_dirs:
        mag_path = os.path.join(input_dir, mag_dir)

        if not os.path.isdir(mag_path):
            continue

        for fname in os.listdir(mag_path):
            if not fname.startswith('POSCAR_'):
                continue

            # Extract structure name
            name = fname.replace('POSCAR_', '')
            poscar_path = os.path.join(mag_path, fname)

            if os.path.isfile(poscar_path):
                structures.append((mag_dir, name, poscar_path))

    logger.info(f"Found {len(structures)} structures in {input_dir}")
    return structures


def read_structure(path: str):
    """
    Read a structure from a POSCAR file.

    Args:
        path: Path to POSCAR file

    Returns:
        Tuple of (atoms, structure) where structure is a pymatgen Structure
    """
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = read(path)

    # Convert to pymatgen Structure
    structure = AseAtomsAdaptor.get_structure(atoms)

    return atoms, structure


def write_structure(atoms, path: str, direct: bool = True) -> None:
    """
    Write a structure to a POSCAR file.

    Args:
        atoms: ASE Atoms object
        path: Output path
        direct: Use direct (fractional) coordinates
    """
    from ase.io import write
    write(path, atoms, direct=direct, format='vasp')


def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(output_dir: str) -> dict:
    """
    Get standard output file paths for the validation pipeline.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of path names
    """
    return {
        'checkpoint': os.path.join(output_dir, 'checkpoint.json'),
        'csv': os.path.join(output_dir, 'validation_results.csv'),
        'json': os.path.join(output_dir, 'validation_summary.json'),
        'dft_relaxed': os.path.join(output_dir, 'dft_relaxed'),
        'ref_cache': os.path.join(output_dir, 'ref_energies_cache.json'),
        'phonon_dir': os.path.join(output_dir, 'phonon_pdfs'),
    }


__all__ = [
    'scan_output_directory',
    'read_structure',
    'write_structure',
    'ensure_directory',
    'get_output_paths',
]
