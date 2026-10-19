"""
Screening Stage S1: Geometry Check.

Checks:
    - Minimum vacuum gap (≥5.0 Å)
    - Maximum slab thickness (≤15.0 Å)
    - Minimum bond distance (≥0.4 Å)
    - Lattice parameter bounds

References:
    - C2DB Sec. 2.1: Vacuum gap and thickness criteria
    - C2DB Sec. 2.4: Structural validity
"""

import numpy as np
from typing import Tuple, Dict
from pymatgen.core import Structure


def _vacuum_by_max_gap(cart_z: np.ndarray, c: float) -> Tuple[float, float]:
    """
    Calculate vacuum gap using maximum gap method.

    Finds the largest continuous gap in the z-direction,
    treating periodic boundaries (z=0 and z=c are connected).

    Args:
        cart_z: Array of Cartesian z-coordinates
        c: Lattice c-axis length

    Returns:
        (vacuum_gap, slab_thickness)
    """
    # Sort z-coordinates
    z_sorted = np.sort(cart_z % c)
    n = len(z_sorted)

    if n == 0:
        return c, 0.0

    # Compute gaps between consecutive atoms
    gaps = np.diff(z_sorted)
    # Wrap-around gap: from last atom to first (crossing c boundary)
    wrap_gap = (z_sorted[0] + c) - z_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)

    # Maximum gap is the vacuum region
    max_gap = np.max(all_gaps)
    slab_thickness = c - max_gap

    return float(max_gap), float(slab_thickness)


def occupied_arc(cart_z: np.ndarray, c: float) -> float:
    """
    Calculate the occupied arc length in fractional coordinates.

    Args:
        cart_z: Array of Cartesian z-coordinates
        c: Lattice c-axis length

    Returns:
        Occupied arc length (fraction of c)
    """
    if len(cart_z) == 0:
        return 0.0

    z_sorted = np.sort(cart_z % c)
    gaps = np.diff(z_sorted)
    wrap_gap = (z_sorted[0] + c) - z_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)

    # Occupied = c - vacuum (max gap)
    max_gap = np.max(all_gaps)
    occupied = c - max_gap

    return float(occupied)


def min_pairwise_distance(struct: Structure) -> float:
    """
    Calculate minimum pairwise distance between atoms.

    Uses direct method for small structures, periodic neighbor
    search for larger structures.

    Args:
        struct: pymatgen Structure

    Returns:
        Minimum pairwise distance (Å)
    """
    n = len(struct)
    if n < 2:
        return float('inf')

    # Get fractional coordinates and lattice
    fcoords = struct.frac_coords
    lattice = struct.lattice

    min_dist_sq = float('inf')

    # Direct O(n^2) for small structures
    if n <= 50:
        for i in range(n):
            for j in range(i + 1, n):
                # Check all periodic images (3x3x3 grid)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            offset = np.array([dx, dy, dz])
                            dist_vec = fcoords[j] - fcoords[i] + offset
                            cart_dist = lattice.get_cartesian_coords(dist_vec)
                            dist_sq = np.dot(cart_dist, cart_dist)
                            min_dist_sq = min(min_dist_sq, dist_sq)
    else:
        # Use pymatgen's neighbor list for larger structures
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            vnn = VoronoiNN()
            min_dist = float('inf')
            for site in struct:
                neighbors = vnn.get_nn(struct, struct.index(site))
                for nbr in neighbors:
                    dist = site.distance(nbr)
                    min_dist = min(min_dist, dist)
            return float(min_dist)
        except Exception:
            # Fallback to direct method
            for i in range(n):
                for j in range(i + 1, n):
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                offset = np.array([dx, dy, dz])
                                dist_vec = fcoords[j] - fcoords[i] + offset
                                cart_dist = lattice.get_cartesian_coords(dist_vec)
                                dist_sq = np.dot(cart_dist, cart_dist)
                                min_dist_sq = min(min_dist_sq, dist_sq)

    return float(np.sqrt(min_dist_sq))


def check_geometry_2d(struct: Structure, cfg) -> Tuple[bool, Dict]:
    """
    S1: Geometry and non-physical severe distortion check.

    Checks:
        1. Vacuum gap >= vacuum_min_angstrom (default 5.0 Å)
        2. Slab thickness <= slab_thickness_max (default 15.0 Å)
        3. Min bond distance >= min_bond_distance (default 0.4 Å)
        4. Lattice a, b within bounds [lattice_ab_min, lattice_ab_max]

    Args:
        struct: pymatgen Structure
        cfg: ScreeningConfig

    Returns:
        (passed, info_dict)
    """
    cart_z = np.array([s.coords[2] for s in struct])
    c = struct.lattice.c

    # 1. Vacuum gap check
    vacuum, thickness = _vacuum_by_max_gap(cart_z, c)
    if vacuum < cfg.vacuum_min_angstrom:
        return False, {
            "reason": f"insufficient_vacuum:{vacuum:.2f}A<{cfg.vacuum_min_angstrom}A",
            "vacuum_angstrom": vacuum,
            "thickness_angstrom": thickness,
        }

    # 2. Slab thickness check
    if thickness > cfg.slab_thickness_max:
        return False, {
            "reason": f"slab_too_thick:{thickness:.2f}A>{cfg.slab_thickness_max}A",
            "vacuum_angstrom": vacuum,
            "thickness_angstrom": thickness,
        }

    # 3. Bond distance check
    min_bond = min_pairwise_distance(struct)
    if min_bond < cfg.min_bond_distance:
        return False, {
            "reason": f"unphysically_close_atoms:{min_bond:.3f}A<{cfg.min_bond_distance}A",
            "vacuum_angstrom": vacuum,
            "thickness_angstrom": thickness,
            "min_bond_angstrom": min_bond,
        }

    # 4. Lattice parameter bounds
    a, b = struct.lattice.a, struct.lattice.b
    if a < cfg.lattice_ab_min or a > cfg.lattice_ab_max:
        return False, {
            "reason": f"invalid_lattice_a:{a:.2f}A",
            "vacuum_angstrom": vacuum,
            "thickness_angstrom": thickness,
        }
    if b < cfg.lattice_ab_min or b > cfg.lattice_ab_max:
        return False, {
            "reason": f"invalid_lattice_b:{b:.2f}A",
            "vacuum_angstrom": vacuum,
            "thickness_angstrom": thickness,
        }

    # 5. Additional geometry metrics for record
    gamma = struct.lattice.gamma
    aspect_ratio = max(a, b) / min(a, b) if min(a, b) > 0 else 0.0

    return True, {
        "reason": "ok",
        "vacuum_angstrom": vacuum,
        "thickness_angstrom": thickness,
        "min_bond_angstrom": min_bond,
        "gamma": gamma,
        "aspect_ratio": aspect_ratio,
    }


def geometry_metrics(struct: Structure) -> 'GeometryMetrics':
    """
    Compute full geometry metrics for a structure.

    Args:
        struct: pymatgen Structure

    Returns:
        GeometryMetrics namedtuple with computed values
    """
    from dataclasses import dataclass

    cart_z = np.array([s.coords[2] for s in struct])
    c = struct.lattice.c

    vacuum, thickness = _vacuum_by_max_gap(cart_z, c)
    min_bond = min_pairwise_distance(struct)

    # Density: atoms per unit volume (for 2D: area * thickness)
    n_atoms = len(struct)
    area = struct.lattice.abc[0] * struct.lattice.abc[1] * np.sin(
        np.radians(struct.lattice.gamma)
    )
    density = n_atoms / (area * thickness) if thickness > 0 else 0.0

    @dataclass
    class GeometryMetrics:
        slab_thickness_ang: float
        density: float
        atom_count: int
        reduced_formula: str
        vacuum_angstrom: float = 0.0
        min_bond_angstrom: float = 0.0

    return GeometryMetrics(
        slab_thickness_ang=thickness,
        density=density,
        atom_count=n_atoms,
        reduced_formula=struct.composition.reduced_formula,
        vacuum_angstrom=vacuum,
        min_bond_angstrom=min_bond,
    )


__all__ = [
    'check_geometry_2d',
    'geometry_metrics',
    'min_pairwise_distance',
    '_vacuum_by_max_gap',
    'occupied_arc',
]
