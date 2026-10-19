"""
Electronic Structure Utilities.

Functions for spacegroup analysis and electronic structure calculations.

References:
    - spglib documentation
    - ASE spacegroup analysis
"""

from typing import Tuple, Any

import numpy as np

# spglib availability check
try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False


def get_spacegroup_info(atoms, symprec: float = 0.01) -> Tuple[str, int]:
    """
    Get spacegroup information for an ASE Atoms object.

    Args:
        atoms: ASE Atoms object
        symprec: spglib symmetry tolerance

    Returns:
        Tuple of (spacegroup_symbol, spacegroup_number)
        Returns ("Unknown", 0) on failure
    """
    if not HAS_SPGLIB:
        return "Unknown", 0

    try:
        cell = (
            atoms.cell[:].tolist(),
            atoms.get_scaled_positions().tolist(),
            atoms.numbers.tolist()
        )
        spacegroup = spglib.get_spacegroup(cell, symprec=symprec)
        if spacegroup:
            # spglib returns format: "Pm-3m (221)" or "1"
            parts = spacegroup.split()
            if len(parts) >= 2:
                symbol = parts[0]
                number_str = parts[1].strip("()")
                try:
                    number = int(number_str)
                    return symbol, number
                except ValueError:
                    return symbol, 0
            elif parts:
                try:
                    return "", int(parts[0])
                except ValueError:
                    return "Unknown", 0
    except Exception:
        pass
    return "Unknown", 0


__all__ = [
    'HAS_SPGLIB',
    'get_spacegroup_info',
]
