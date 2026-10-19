"""
DFCA Crystal Representation Module.

This module defines the core scientific representation for converting between
pymatgen crystal structures and neural network input tensors.

CRITICAL SCIENTIFIC CONSTANTS:
    - MAX_LATTICE_LENGTH: 46.7425 (lattice normalization constant)
    - Element mapping: atomic number → (period, group) in periodic table
    - Coordinate convention: fractional [0, 1], z-centered for 2D slabs

DO NOT MODIFY these constants without version bump and backward compatibility check.
"""

import numpy as np
import torch


# ============================================================================
# LATTICE NORMALIZATION CONSTANT
# ============================================================================

# Lattice normalization constant — used in both preprocessing and training.
# All lattice vectors are divided by this value to normalize to approximately [-1, 1].
MAX_LATTICE_LENGTH = 46.7425


# ============================================================================
# ELEMENT ENCODING — PERIODIC TABLE POSITION MAPPING
# ============================================================================

# Periodic table mapping: atomic number to (period, group) position.
# This is the PRIMARY element encoding for DFCA.
# Period (row): 1-7, Group (column): 1-18.
# Lanthanides (Z=58-71) and Actinides (Z=90-103) are merged into group 3.
ATOMIC_NUMBER_TO_POSITION = {
    # Period 1
    1: (1, 1),    # H
    2: (1, 18),   # He

    # Period 2
    3: (2, 1),    # Li
    4: (2, 2),    # Be
    5: (2, 13),   # B
    6: (2, 14),   # C
    7: (2, 15),   # N
    8: (2, 16),   # O
    9: (2, 17),   # F
    10: (2, 18),  # Ne

    # Period 3
    11: (3, 1),   # Na
    12: (3, 2),   # Mg
    13: (3, 13),  # Al
    14: (3, 14),  # Si
    15: (3, 15),  # P
    16: (3, 16),  # S
    17: (3, 17),  # Cl
    18: (3, 18),  # Ar

    # Period 4
    19: (4, 1),   # K
    20: (4, 2),   # Ca
    21: (4, 3),   # Sc
    22: (4, 4),   # Ti
    23: (4, 5),   # V
    24: (4, 6),   # Cr
    25: (4, 7),   # Mn
    26: (4, 8),   # Fe
    27: (4, 9),   # Co
    28: (4, 10),  # Ni
    29: (4, 11),  # Cu
    30: (4, 12),  # Zn
    31: (4, 13),  # Ga
    32: (4, 14),  # Ge
    33: (4, 15),  # As
    34: (4, 16),  # Se
    35: (4, 17),  # Br
    36: (4, 18),  # Kr

    # Period 5
    37: (5, 1),   # Rb
    38: (5, 2),   # Sr
    39: (5, 3),   # Y
    40: (5, 4),   # Zr
    41: (5, 5),   # Nb
    42: (5, 6),   # Mo
    43: (5, 7),   # Tc
    44: (5, 8),   # Ru
    45: (5, 9),   # Rh
    46: (5, 10),  # Pd
    47: (5, 11),  # Ag
    48: (5, 12),  # Cd
    49: (5, 13),  # In
    50: (5, 14),  # Sn
    51: (5, 15),  # Sb
    52: (5, 16),  # Te
    53: (5, 17),  # I
    54: (5, 18),  # Xe

    # Period 6 (lanthanides merged into group 3)
    55: (6, 1),   # Cs
    56: (6, 2),   # Ba
    57: (6, 3),   # La
    58: (6, 3),   # Ce
    59: (6, 3),   # Pr
    60: (6, 3),   # Nd
    61: (6, 3),   # Pm
    62: (6, 3),   # Sm
    63: (6, 3),   # Eu
    64: (6, 3),   # Gd
    65: (6, 3),   # Tb
    66: (6, 3),   # Dy
    67: (6, 3),   # Ho
    68: (6, 3),   # Er
    69: (6, 3),   # Tm
    70: (6, 3),   # Yb
    71: (6, 3),   # Lu
    72: (6, 4),   # Hf
    73: (6, 5),   # Ta
    74: (6, 6),   # W
    75: (6, 7),   # Re
    76: (6, 8),   # Os
    77: (6, 9),   # Ir
    78: (6, 10),  # Pt
    79: (6, 11),  # Au
    80: (6, 12),  # Hg
    81: (6, 13),  # Tl
    82: (6, 14),  # Pb
    83: (6, 15),  # Bi
    84: (6, 16),  # Po
    85: (6, 17),  # At
    86: (6, 18),  # Rn

    # Period 7 (actinides merged into group 3)
    87: (7, 1),   # Fr
    88: (7, 2),   # Ra
    89: (7, 3),   # Ac
    90: (7, 3),   # Th
    91: (7, 3),   # Pa
    92: (7, 3),   # U
    93: (7, 3),   # Np
    94: (7, 3),   # Pu
    95: (7, 3),   # Am
    96: (7, 3),   # Cm
    97: (7, 3),   # Bk
    98: (7, 3),   # Cf
    99: (7, 3),   # Es
    100: (7, 3),  # Fm
    101: (7, 3),  # Md
    102: (7, 3),  # No
    103: (7, 3),  # Lr
    104: (7, 4),   # Rf
    105: (7, 5),   # Db
    106: (7, 6),   # Sg
    107: (7, 7),   # Bh
    108: (7, 8),   # Hs
    109: (7, 9),   # Mt
    110: (7, 10),  # Ds
    111: (7, 11),  # Rg
    112: (7, 12),  # Cn
    113: (7, 13),  # Nh
    114: (7, 14),  # Fl
    115: (7, 15),  # Mc
    116: (7, 16),  # Lv
    117: (7, 17),  # Ts
    118: (7, 18),  # Og

    # Invalid atom placeholder
    0: (0, 0),
}

# Reverse mapping: (period, group) → atomic number
POSITION_TO_ATOMIC_NUMBER = {position: number for number, position in ATOMIC_NUMBER_TO_POSITION.items()}


# ============================================================================
# ATOMIC MASSES
# ============================================================================

# Atomic masses (amu) for all Z 1-94, indexed by Z.
# Source: standard IUPAC atomic weights (averaged over isotopes).
ATOMIC_NUMBER_TO_MASS = {
    1: 1.008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180,
    11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45, 18: 39.95,
    19: 39.098, 20: 40.078, 21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996, 25: 54.938, 26: 55.845, 27: 58.933,
    28: 58.693, 29: 63.546, 30: 65.38,
    31: 69.723, 32: 72.630, 33: 74.922, 34: 78.971, 35: 79.904, 36: 83.798,
    37: 85.468, 38: 87.62, 39: 88.906, 40: 91.224, 41: 92.906, 42: 95.95, 43: 98.0, 44: 101.07, 45: 102.91, 46: 106.42,
    47: 107.87, 48: 112.41, 49: 114.82, 50: 118.71,
    51: 121.76, 52: 127.60, 53: 126.90, 54: 131.29,
    55: 132.91, 56: 137.33, 57: 138.91, 58: 140.12, 59: 140.91, 60: 144.24, 61: 145.0, 62: 150.36, 63: 151.96,
    64: 157.25, 65: 158.93, 66: 162.50, 67: 164.93, 68: 167.26, 69: 168.93, 70: 173.05,
    71: 174.97, 72: 178.49, 73: 180.95, 74: 183.84, 75: 186.21, 76: 190.23, 77: 192.22, 78: 195.08, 79: 196.97,
    80: 200.59, 81: 204.38, 82: 207.2, 83: 208.98, 84: 209.0, 85: 210.0, 86: 222.0,
    87: 223.0, 88: 226.0, 89: 227.0, 90: 232.04, 91: 231.04, 92: 238.03, 93: 237.0, 94: 244.0, 95: 243.0, 96: 247.0,
    97: 247.0, 98: 251.0, 99: 252.0, 100: 257.0, 101: 258.0, 102: 259.0, 103: 262.0, 104: 267.0, 105: 268.0, 106: 269.0,
    107: 270.0, 108: 277.0, 109: 278.0, 110: 281.0, 111: 282.0, 112: 285.0, 113: 286.0, 114: 289.0, 115: 290.0,
    116: 293.0, 117: 294.0, 118: 294.0
}


# ============================================================================
# NORMALIZATION / DENORMALIZATION FUNCTIONS
# ============================================================================

def normalize_row(period):
    """
    Normalize periodic table period number to [-1, 1] range.

    Formula: (period / 7.0) * 2 - 1
    Range: [1, 7] → [-1, 1]

    Args:
        period: Integer period number (1-7)

    Returns:
        float: Normalized value in [-1, 1]
    """
    return (period / 7.0) * 2 - 1


def normalize_column(column):
    """
    Normalize periodic table group number to [-1, 1] range.

    Formula: (group / 18.0) * 2 - 1
    Range: [1, 18] → [-1, 1]

    Args:
        column: Integer group number (1-18)

    Returns:
        float: Normalized value in [-1, 1]
    """
    return (column / 18.0) * 2 - 1


def denormalize_row(norm_row):
    """
    Convert normalized row coordinate back to period number.

    Inverse of normalize_row: [−1, 1] → [1, 7], then round.

    Args:
        norm_row: Normalized row value in [-1, 1]

    Returns:
        int: Period number (rounded)
    """
    row = ((norm_row + 1) / 2) * 7
    return round(row)


def denormalize_column(norm_column):
    """
    Convert normalized column coordinate back to group number.

    Inverse of normalize_column: [−1, 1] → [1, 18], then round.

    Args:
        norm_column: Normalized column value in [-1, 1]

    Returns:
        int: Group number (rounded)
    """
    column = ((norm_column + 1) / 2) * 18
    return round(column)


# ============================================================================
# DEDUPLICATED ELEMENT LOOKUP TABLES
# ============================================================================

def _build_unique_element_rowcol_and_masses():
    """
    Build deduplicated element lookup tables.

    After lanthanide/actinide group-3 merging:
    - 94 atomic numbers → 72 unique (period, group) positions
    - Deduplication preserves the first-seen Z for each (row, col) coordinate

    Returns:
        tuple: (VALID_ELEMENT_ROWCOL, VALID_ELEMENT_MASSES)
            - VALID_ELEMENT_ROWCOL: torch.Tensor [E, 2]
            - VALID_ELEMENT_MASSES: torch.Tensor [E]
    """
    seen = {}
    result_rowcol = []
    result_masses = []
    for atomic_number, (elem_row, elem_column) in ATOMIC_NUMBER_TO_POSITION.items():
        if atomic_number == 0:
            continue
        norm_row = normalize_row(elem_row)
        norm_col = normalize_column(elem_column)
        key = (round(norm_row, 6), round(norm_col, 6))
        if key not in seen:
            seen[key] = True
            result_rowcol.append([norm_row, norm_col])
            result_masses.append(ATOMIC_NUMBER_TO_MASS.get(atomic_number, 1.0))
    return torch.tensor(result_rowcol, dtype=torch.float32), torch.tensor(result_masses, dtype=torch.float32)


# Global lookup tables (computed once at import time)
VALID_ELEMENT_ROWCOL, VALID_ELEMENT_MASSES = _build_unique_element_rowcol_and_masses()


# ============================================================================
# ELEMENT PROPERTIES
# ============================================================================

def get_element_properties():
    """
    Returns a dict keyed by atomic number with element properties.

    Properties include: symbol, electronegativity, atomic_radius, atomic_mass,
    group (column), period (row).

    Returns:
        dict: {atomic_number: {property_name: value}}
    """
    from pymatgen.core.periodic_table import Element as PymatgenElement

    properties = {}
    for atomic_number, (row, column) in ATOMIC_NUMBER_TO_POSITION.items():
        if atomic_number == 0:
            continue
        try:
            el = PymatgenElement.from_Z(atomic_number)
        except Exception:
            continue
        try:
            electroneg = float(el.X) if el.X is not None else None
        except Exception:
            electroneg = None
        try:
            radius = float(el.atomic_radius) if el.atomic_radius is not None else None
        except Exception:
            radius = None
        try:
            mass = float(el.atomic_mass) if el.atomic_mass is not None else None
        except Exception:
            mass = None
        properties[atomic_number] = {
            'symbol': el.symbol,
            'electronegativity': electroneg,
            'atomic_radius': radius,
            'atomic_mass': mass,
            'group': column,
            'period': row,
        }
    return properties


# ============================================================================
# SOFT DECODING: AVERAGE ATOMIC MASS
# ============================================================================

def decode_avg_atomic_mass(atom_features, temperature=0.0788):
    """
    Compute average atomic mass per sample via soft expectation over element distribution.

    Uses softmax-based continuous probability over periodic-table coordinates to avoid
    discrete argmax (which risks out-of-bounds on device-side asserts).
    All operations are fully differentiable matrix computations with no indexing.

    Args:
        atom_features: torch.Tensor of shape [B, N, 5]
            Features: [norm_row, norm_col, x, y, z]
        temperature: float, softmax temperature (default 0.0788)

    Returns:
        torch.Tensor of shape [B]: Average atomic mass per sample
    """
    device = atom_features.device
    dtype = atom_features.dtype

    # Valid-atom mask and extract row/column coordinates.
    valid_mask = (atom_features[:, :, 0] != -1).float()  # [B, N]
    pred_rowcol = atom_features[:, :, :2]  # [B, N, 2]

    valid_element_rowcol = VALID_ELEMENT_ROWCOL.to(device=device, dtype=dtype)  # [E, 2]
    element_masses = VALID_ELEMENT_MASSES.to(device=device, dtype=dtype)  # [E]

    # [B, N, E] Squared distance from each atom to every valid element in row-column space.
    dist_sq = (pred_rowcol.unsqueeze(2) - valid_element_rowcol.view(1, 1, -1, 2)).pow(2).sum(dim=-1)

    # Lower-bound temperature to prevent softmax overflow.
    temperature = max(float(temperature), 0.05)

    # Softmax turns distances into a valid probability distribution.
    probs = torch.softmax(-dist_sq / temperature, dim=-1)  # [B, N, E]

    # [B, N] Expected mass per atom from weighted sum over element masses.
    expected_atom_mass = (probs * element_masses.view(1, 1, -1)).sum(dim=-1)

    # [B] Total mass divided by atom count (skip division by zero).
    total_mass = (expected_atom_mass * valid_mask).sum(dim=1)
    num_atoms = valid_mask.sum(dim=1).clamp(min=1.0)

    return total_mass / num_atoms


# ============================================================================
# 2D CRYSTAL: PERIODIC Z-CENTERING
# ============================================================================

def periodic_span_center(values):
    """
    Find the center of the minimal arc containing all fractional coordinates.

    This is more stable than a plain mean when a 2D slab crosses z=0/1 boundary.
    Handles periodic boundary conditions correctly.

    Args:
        values: list of fractional coordinates [0, 1)

    Returns:
        float: Center of the minimal arc containing all values
    """
    if not values:
        return 0.5

    points = sorted(float(v) % 1.0 for v in values)
    if len(points) == 1:
        return points[0]

    extended = points + [points[0] + 1.0]
    gaps = [extended[i + 1] - extended[i] for i in range(len(points))]
    max_gap_idx = int(np.argmax(gaps))
    span_start = extended[max_gap_idx + 1] % 1.0
    span = 1.0 - gaps[max_gap_idx]
    return (span_start + span / 2.0) % 1.0


# ============================================================================
# ENCODING UTILITIES (for preprocessing)
# ============================================================================

def encode_element(atomic_number):
    """
    Encode atomic number to [norm_row, norm_col] coordinates.

    Args:
        atomic_number: int, atomic number (Z)

    Returns:
        tuple: (norm_row, norm_col) in [-1, 1] range
    """
    if atomic_number == 0:
        return (-1, -1)
    period, group = ATOMIC_NUMBER_TO_POSITION[atomic_number]
    return (normalize_row(period), normalize_column(group))


def encode_lattice(lattice_matrix, max_length=MAX_LATTICE_LENGTH):
    """
    Normalize lattice matrix.

    Args:
        lattice_matrix: numpy array [3, 3] in Ångström
        max_length: Normalization constant (default MAX_LATTICE_LENGTH)

    Returns:
        numpy array [3, 3], normalized to [-1, 1] approximately
    """
    return lattice_matrix / max_length


def decode_lattice(normalized_lattice, max_length=MAX_LATTICE_LENGTH):
    """
    Denormalize lattice matrix.

    Args:
        normalized_lattice: numpy array [3, 3], normalized
        max_length: Denormalization constant (default MAX_LATTICE_LENGTH)

    Returns:
        numpy array [3, 3] in Ångström
    """
    return normalized_lattice * max_length
