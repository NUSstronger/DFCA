"""
VASP Export Utilities.

Features:
    - POSCAR generation with vacuum padding and centering
    - INCAR generation by magnetic classification
    - KPOINTS generation

References:
    - C2DB VASP settings
"""

import numpy as np
from typing import List, Dict, Tuple
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar


def pad_vacuum(struct: Structure, target_vacuum: float = 20.0) -> Structure:
    """
    Center slab and adjust vacuum thickness.

    Finds the material layer by maximum gap, shifts atoms to center,
    and adjusts c-axis to achieve target vacuum.

    Args:
        struct: pymatgen Structure
        target_vacuum: Target vacuum gap (Å)

    Returns:
        Centered structure with adjusted vacuum
    """
    from dfca.screening.stages.s1_geometry import _vacuum_by_max_gap

    c_old = struct.lattice.c
    cart_z = np.array([s.coords[2] for s in struct])

    _, thickness = _vacuum_by_max_gap(cart_z, c_old)
    c_new = thickness + target_vacuum

    # Find max gap position
    z_sorted = np.sort(cart_z % c_old)
    all_gaps = np.diff(z_sorted)
    wrap_gap = (z_sorted[0] + c_old) - z_sorted[-1]
    gaps_all = np.append(all_gaps, wrap_gap)
    max_gap_idx = int(np.argmax(gaps_all))

    if max_gap_idx < len(all_gaps):
        layer_top = z_sorted[max_gap_idx]
        layer_bottom = z_sorted[max_gap_idx + 1] if max_gap_idx + 1 < len(all_gaps) else z_sorted[0]
    else:
        layer_top = z_sorted[-1]
        layer_bottom = z_sorted[0] + c_old

    # Center the layer
    layer_center = (layer_top + layer_bottom) / 2 % c_old
    shift = c_new / 2 - layer_center
    new_cart_z = (cart_z + shift) % c_new

    # Create new lattice
    new_matrix = struct.lattice.matrix.copy()
    new_matrix[2, 2] = c_new
    new_lattice = Lattice(new_matrix)

    # Convert back to fractional coordinates
    new_frac_coords = []
    for i, site in enumerate(struct):
        fc = site.frac_coords.copy()
        fc[2] = new_cart_z[i] / c_new
        new_frac_coords.append(fc)

    return Structure(
        new_lattice,
        [str(s.specie) for s in struct],
        new_frac_coords,
        coords_are_cartesian=False
    )


def export_poscar(
    struct: Structure,
    name: str,
    output_dir: str,
    cfg
) -> str:
    """
    Export DFT-ready POSCAR (standardized + centered + 20Å vacuum).

    Args:
        struct: pymatgen Structure
        name: Structure name
        output_dir: Output directory
        cfg: ScreeningConfig

    Returns:
        Path to exported POSCAR
    """
    import os

    padded = pad_vacuum(struct, cfg.vasp_vacuum_angstrom)

    # Optional spglib refinement
    try:
        import spglib
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        sga = SpacegroupAnalyzer(padded, symprec=cfg.symprec)
        refined = sga.get_refined_structure()
        refined_lattice = refined.lattice.matrix.copy()
        refined_lattice[2, 2] = padded.lattice.c
        refined = Structure(
            Lattice(refined_lattice),
            refined.species,
            refined.frac_coords,
            coords_are_cartesian=False
        )
        padded = refined
    except Exception:
        pass

    out_path = os.path.join(output_dir, f"POSCAR_{name}")
    Poscar(padded).write_file(out_path)
    return out_path


# ────────────────────────────────────────────────────────────────
# Magnetic Classification
# ────────────────────────────────────────────────────────────────

# 3d transition metals (Sc-Zn)
MAG_3D = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
}

# 4d/5d transition metals
MAG_4D5D = {
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}

# Lanthanides
MAG_RE = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
}

# Default MAGMOM values
MAGMOM_REF: Dict[str, float] = {
    "H": 0.0, "He": 0.0, "Li": 0.0, "Be": 0.0, "B": 0.0, "C": 0.0, "N": 0.0,
    "O": 0.0, "F": 0.0, "Ne": 0.0, "Na": 0.0, "Mg": 0.0, "Al": 0.0, "Si": 0.0,
    "P": 0.0, "S": 0.0, "Cl": 0.0, "Ar": 0.0, "K": 0.0, "Ca": 0.0, "Sc": 2.0,
    "Ti": 2.0, "V": 2.0, "Cr": 2.0, "Mn": 2.0, "Fe": 2.5, "Co": 2.0,
    "Ni": 2.0, "Cu": 0.5, "Zn": 0.5, "Ga": 0.0, "Ge": 0.0, "As": 0.0,
    "Se": 0.0, "Br": 0.0, "Kr": 0.0, "Rb": 0.0, "Sr": 0.0, "Y": 2.0,
    "Zr": 2.0, "Nb": 2.0, "Mo": 2.0, "Tc": 2.0, "Ru": 2.0, "Rh": 2.0,
    "Pd": 0.5, "Ag": 0.5, "Cd": 0.5, "In": 0.0, "Sn": 0.0, "Sb": 0.0,
    "Te": 0.0, "I": 0.0, "Xe": 0.0, "Cs": 0.0, "Ba": 0.0, "La": 3.0,
    "Ce": 3.0, "Pr": 3.0, "Nd": 3.0, "Pm": 3.0, "Sm": 3.0, "Eu": 3.0,
    "Gd": 3.0, "Tb": 3.0, "Dy": 3.0, "Ho": 3.0, "Er": 3.0, "Tm": 3.0,
    "Yb": 3.0, "Lu": 3.0, "Hf": 2.0, "Ta": 2.0, "W": 2.0, "Re": 2.0,
    "Os": 2.0, "Ir": 2.0, "Pt": 0.5, "Au": 0.5, "Hg": 0.5, "Tl": 0.0,
    "Pb": 0.0, "Bi": 0.0, "Po": 0.0, "At": 0.0, "Rn": 0.0, "Fr": 0.0,
    "Ra": 0.0, "Ac": 3.0, "Th": 3.0, "Pa": 3.0, "U": 3.0, "Np": 3.0,
    "Pu": 3.0, "Am": 3.0,
}


def classify(elements: List[str]) -> str:
    """
    Classify structure by magnetic type.

    Args:
        elements: List of element symbols

    Returns:
        Magnetic type: 'nonmag' | 'mag_3d' | 'mag_4d5d' | 'mag_re'
    """
    has_3d = any(el in MAG_3D for el in elements)
    has_4d5d = any(el in MAG_4D5D for el in elements)
    has_re = any(el in MAG_RE for el in elements)

    if has_re:
        return "mag_re"
    elif has_3d:
        return "mag_3d"
    elif has_4d5d:
        return "mag_4d5d"
    else:
        return "nonmag"


def make_incar(mag_type: str, elements: List[str], counts: List[int]) -> str:
    """
    Generate VASP INCAR file.

    Args:
        mag_type: Magnetic type ('nonmag', 'mag_3d', 'mag_4d5d', 'mag_re')
        elements: Element symbols
        counts: Atom counts per element

    Returns:
        INCAR file content
    """
    lines = [
        "SYSTEM = 2D Materials DFT Validation",
        "ENCUT  = 520",
        "EDIFF  = 1E-5",
        "EDIFFG = -0.02",
        "NSW    = 500",
        "ISIF   = 3",
        "ISMEAR = 0",
        "SIGMA  = 0.05",
        "LCHARG = .FALSE.",
        "LWAVE  = .FALSE.",
        "LASPH  = .TRUE.",
    ]

    # Add MAGMOM for magnetic systems
    if mag_type != "nonmag":
        lines.append("ISPIN  = 2")
        magmom = []
        for el, count in zip(elements, counts):
            mom = MAGMOM_REF.get(el, 0.0)
            magmom.append(f"{count}*{mom}")
        lines.append(f"MAGMOM = {' '.join(magmom)}")
    else:
        lines.append("ISPIN  = 1")
        lines.append("MAGMOM = 0.0")

    return "\n".join(lines)


def make_kpoints() -> str:
    """
    Generate VASP KPOINTS file.

    Returns:
        KPOINTS file content (gamma-centered Monkhorst-Pack)
    """
    return """Automatic mesh
0
Gamma
6 6 1
0 0 0
"""


__all__ = [
    'pad_vacuum',
    'export_poscar',
    'classify',
    'make_incar',
    'make_kpoints',
    'MAG_3D',
    'MAG_4D5D',
    'MAG_RE',
    'MAGMOM_REF',
]
