"""
Magnetic Classification Utilities.

Functions for classifying structures by magnetic type and building
initial magnetic moment configurations.

References:
    - C2DB magnetic classification scheme
    - Element-specific magnetic moments
"""

from typing import List, Tuple

# 3d transition metals (Sc-Zn)
MAG_3D = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
}

# 4d/5d transition metals
MAG_4D5D = {
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}

# Lanthanides (rare earth)
MAG_RE = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
}

# Default magnetic moments (μB per atom)
# Based on C2DB standards and experimental/DFT data
MAGMOM_REF = {
    # Non-magnetic elements
    "H": 0.0, "He": 0.0, "Li": 0.0, "Be": 0.0, "B": 0.0, "C": 0.0, "N": 0.0,
    "O": 0.0, "F": 0.0, "Ne": 0.0, "Na": 0.0, "Mg": 0.0, "Al": 0.0, "Si": 0.0,
    "P": 0.0, "S": 0.0, "Cl": 0.0, "Ar": 0.0, "K": 0.0, "Ca": 0.0,
    "Ga": 0.0, "Ge": 0.0, "As": 0.0, "Se": 0.0, "Br": 0.0, "Kr": 0.0,
    "Rb": 0.0, "Sr": 0.0, "In": 0.0, "Sn": 0.0, "Sb": 0.0,
    "Te": 0.0, "I": 0.0, "Xe": 0.0, "Cs": 0.0, "Ba": 0.0,
    "Tl": 0.0, "Pb": 0.0, "Bi": 0.0, "Po": 0.0, "At": 0.0, "Rn": 0.0,
    "Fr": 0.0, "Ra": 0.0,
    # 3d transition metals
    "Sc": 2.0, "Ti": 2.0, "V": 2.0, "Cr": 2.0, "Mn": 2.0,
    "Fe": 2.5, "Co": 2.0, "Ni": 2.0, "Cu": 0.5, "Zn": 0.5,
    # 4d/5d transition metals
    "Y": 2.0, "Zr": 2.0, "Nb": 2.0, "Mo": 2.0, "Tc": 2.0,
    "Ru": 2.0, "Rh": 2.0, "Pd": 0.5, "Ag": 0.5, "Cd": 0.5,
    "Hf": 2.0, "Ta": 2.0, "W": 2.0, "Re": 2.0, "Os": 2.0,
    "Ir": 2.0, "Pt": 0.5, "Au": 0.5, "Hg": 0.5,
    # Lanthanides
    "La": 3.0, "Ce": 3.0, "Pr": 3.0, "Nd": 3.0, "Pm": 3.0,
    "Sm": 3.0, "Eu": 3.0, "Gd": 3.0, "Tb": 3.0, "Dy": 3.0,
    "Ho": 3.0, "Er": 3.0, "Tm": 3.0, "Yb": 3.0, "Lu": 3.0,
    # Actinides (forbidden in screening, but include for completeness)
    "Ac": 3.0, "Th": 3.0, "Pa": 3.0, "U": 3.0, "Np": 3.0,
    "Pu": 3.0, "Am": 3.0,
}


def classify_magnetic(elements: List[str]) -> Tuple[str, bool]:
    """
    Classify structure by magnetic type based on element composition.

    Args:
        elements: List of element symbols

    Returns:
        Tuple of (mag_type, is_magnetic)
            - mag_type: 'nonmag' | 'mag_3d' | 'mag_4d5d' | 'mag_re'
            - is_magnetic: True if any magnetic element present
    """
    el_set = set(elements)

    if el_set & MAG_RE:
        return 'mag_re', True
    if el_set & MAG_3D:
        return 'mag_3d', True
    if el_set & MAG_4D5D:
        return 'mag_4d5d', True

    return 'nonmag', False


def build_initial_magmoms(
    atoms,
    default_magmom: float = 0.6
) -> List[float]:
    """
    Build initial magnetic moment configuration for a structure.

    Uses element-specific default values for magnetic elements.
    Non-magnetic elements get a small default value.

    Args:
        atoms: ASE Atoms object
        default_magmom: Default magnetic moment for non-magnetic elements (μB)

    Returns:
        List of magnetic moments (one per atom)
    """
    magmoms = []
    for el in atoms.get_chemical_symbols():
        if el in MAGMOM_REF:
            magmoms.append(MAGMOM_REF[el])
        else:
            magmoms.append(default_magmom)
    return magmoms


def is_magnetic_element(element: str) -> bool:
    """
    Check if an element is magnetic.

    Args:
        element: Element symbol

    Returns:
        True if element is magnetic
    """
    return (
        element in MAG_3D or
        element in MAG_4D5D or
        element in MAG_RE
    )


__all__ = [
    'MAG_3D',
    'MAG_4D5D',
    'MAG_RE',
    'MAGMOM_REF',
    'classify_magnetic',
    'build_initial_magmoms',
    'is_magnetic_element',
]
