"""
Structure Construction for Crystal Generation.

This module handles:
    - 2D lattice projection (projecting 3D lattice to slab-friendly form)
    - Converting model output to pymatgen Structure objects
    - Saving structures to CIF files
    - Analyzing generated structure statistics

IMPORTANT: This code is migrated directly from the original generate_crystals.py.
Scientific behavior must be preserved exactly.
"""

import os
import math
from collections import defaultdict

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element


# Rare gas elements list - must match the original exactly
RARE_GAS_ELEMENTS = [2, 10, 18, 36, 54, 86]


# ============================================================================
# Geometry Utilities
# ============================================================================

def periodic_distance(value, center=0.5):
    """Shortest wrapped distance on [0, 1)."""
    diff = abs((value % 1.0) - center)
    return min(diff, 1.0 - diff)


def vector_angle_degrees(vec_a, vec_b):
    """Return the angle between two vectors in degrees."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 90.0
    cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return np.degrees(np.arccos(cosine))


def project_lattice_to_2d(lattice_matrix, min_c_length=18.0, max_c_length=35.0,
                           gamma_min_deg=60.0, gamma_max_deg=120.0):
    """
    Project a 3D lattice matrix to a 2D slab-friendly form.

    Preserves a and b lengths in XY plane, constrains gamma angle,
    and ensures minimum/maximum c-length for proper vacuum layer.

    This function handles the anisotropic treatment of the lattice:
    - XY plane: arbitrary lattice vectors with angle constraint
    - Z axis: c-length for vacuum layer (min 18Å, max 35Å by default)

    Args:
        lattice_matrix: 3x3 lattice matrix (numpy array)
        min_c_length: Minimum c-axis length in Angstrom (default 18.0)
        max_c_length: Maximum c-axis length in Angstrom (default 35.0)
        gamma_min_deg: Minimum gamma angle in degrees (default 60.0)
        gamma_max_deg: Maximum gamma angle in degrees (default 120.0)

    Returns:
        Projected 3x3 lattice matrix with 2D slab geometry
    """
    a_vec = lattice_matrix[0].astype(np.float64).copy()
    b_vec = lattice_matrix[1].astype(np.float64).copy()
    c_vec = lattice_matrix[2].astype(np.float64).copy()

    # Extract XY components for 2D in-plane lattice
    a_xy = a_vec[:2]
    b_xy = b_vec[:2]

    # Handle degenerate cases where XY projection is zero
    if np.linalg.norm(a_xy) < 1e-6:
        a_xy = np.array([max(np.linalg.norm(a_vec), 1.0), 0.0], dtype=np.float64)

    if np.linalg.norm(b_xy) < 1e-6:
        b_xy = np.array([0.0, max(np.linalg.norm(b_vec), 1.0)], dtype=np.float64)

    # Compute in-plane lattice parameters
    a_len = max(np.linalg.norm(a_xy), 1.0)
    b_len = max(np.linalg.norm(b_xy), 1.0)
    gamma_deg = vector_angle_degrees(a_xy, b_xy)
    gamma_deg = float(np.clip(gamma_deg, gamma_min_deg, gamma_max_deg))
    gamma_rad = math.radians(gamma_deg)

    # Compute c-length with floor and ceiling constraints
    raw_c_len = max(np.linalg.norm(c_vec), abs(c_vec[2]))
    if min_c_length is not None:
        c_len = max(raw_c_len, min_c_length)
    else:
        c_len = max(raw_c_len, 1e-3)
    # Cap c_len to avoid abnormally large vacuum layers.
    if max_c_length is not None:
        c_len = min(c_len, max_c_length)

    # Construct orthogonal 2D-projected lattice matrix
    projected = np.array(
        [
            [a_len, 0.0, 0.0],
            [b_len * math.cos(gamma_rad), b_len * math.sin(gamma_rad), 0.0],
            [0.0, 0.0, c_len],
        ],
        dtype=np.float64,
    )
    return projected


# ============================================================================
# Structure Conversion
# ============================================================================

def convert_to_structure(
        lattice_vectors,
        atom_features,
        max_length=46.7425,
        filter_rare_gases=True,
        allowed_atomic_numbers=None,
        coord_tolerance=0.05,
        slab_center=0.5,
        slab_half_width=0.15,
        element_sigma=0.05,
        min_valid_element_prob=1e-3,
        project_2d_lattice=True,
        min_c_length=None,
        max_c_length=None,
        gamma_min_deg=60.0,
        gamma_max_deg=120.0,
        max_unique_elements=4,
        forbid_atomic_numbers=None,
):
    """
    Convert model output (lattice vectors + atom features) to pymatgen Structure objects.

    This function implements the complete decoding pipeline:
    1. Project lattice to 2D slab form (if enabled)
    2. Filter atoms by position (slab region) and element validity
    3. Map row/column coordinates to element atomic numbers
    4. Limit element diversity per structure
    5. Create Structure objects with fractional coordinates

    Args:
        lattice_vectors: Model output lattice [B, 3, 3] normalized to max_length
        atom_features: Model output atoms [B, max_atoms, 5] with (row, col, x, y, z)
        max_length: Denormalization factor (default 46.7425)
        filter_rare_gases: Exclude noble gas elements (default True)
        allowed_atomic_numbers: Restrict to elements in this set (default None = all)
        coord_tolerance: Allow small coordinate overflow (default 0.05)
        slab_center: Expected fractional z center of 2D slab (default 0.5)
        slab_half_width: Half-width of slab region in fractional coords (default 0.15)
        element_sigma: Gaussian sigma for element decoding (default 0.05)
        min_valid_element_prob: Minimum element confidence threshold (default 1e-3)
        project_2d_lattice: Apply 2D lattice projection (default True)
        min_c_length: Minimum c-axis length (default None = 18.0)
        max_c_length: Maximum c-axis length (default None = 35.0)
        gamma_min_deg: Minimum gamma angle for projection (default 60.0)
        gamma_max_deg: Maximum gamma angle for projection (default 120.0)
        max_unique_elements: Max element types per structure (default 4)
        forbid_atomic_numbers: Exclude these atomic numbers (default None)

    Returns:
        List of pymatgen Structure objects
    """
    from dfca.generation.decoder import map_to_element

    batch_size = lattice_vectors.shape[0]
    structures = []
    forbidden = set(forbid_atomic_numbers or [])

    # Apply defaults for optional parameters
    effective_min_c_length = min_c_length if min_c_length is not None else 18.0
    effective_max_c_length = max_c_length if max_c_length is not None else 35.0

    for i in range(batch_size):
        # Denormalize and optionally project lattice
        lattice_matrix = lattice_vectors[i].cpu().numpy() * max_length
        actual_c_len = None
        if project_2d_lattice:
            lattice_matrix = project_lattice_to_2d(
                lattice_matrix,
                min_c_length=effective_min_c_length,
                max_c_length=effective_max_c_length,
                gamma_min_deg=gamma_min_deg,
                gamma_max_deg=gamma_max_deg,
            )
            actual_c_len = lattice_matrix[2, 2]

        atoms = atom_features[i].cpu().numpy()

        atom_candidates = []

        # Convert slab_half_width from fractional to Ångstrom using actual c_len.
        # If c_len not available (projection disabled), fall back to min_c_length (fixed 18 Å).
        if actual_c_len is not None:
            slab_half_width_ang = slab_half_width * actual_c_len
        else:
            slab_half_width_ang = slab_half_width * effective_min_c_length

        for atom in atoms:
            # Skip invalid atom slots (row <= -0.5 indicates padding)
            if atom[0] <= -0.5:
                continue
            coords = atom[2:5].copy()

            # Filter out-of-range coordinates
            if np.any(coords < -coord_tolerance) or np.any(coords > 1.0 + coord_tolerance):
                continue
            coords = np.clip(coords, 0.0, 1.0)

            # Slab filtering in Å: convert fractional z to Å, then compare.
            if actual_c_len is not None:
                z_ang = coords[2] * actual_c_len
                slab_dist_ang = min(z_ang, actual_c_len - z_ang)
                if slab_dist_ang > slab_half_width_ang:
                    continue
            else:
                # Fallback to fractional distance if c_len unavailable.
                if periodic_distance(coords[2], slab_center) > slab_half_width:
                    continue

            # Decode element from row/column coordinates
            atom_z, atom_prob = map_to_element(
                atom[0],
                atom[1],
                sigma=element_sigma,
                filter_rare_gases=filter_rare_gases,
                max_atomic_number=94,
                allowed_atomic_numbers=allowed_atomic_numbers,
                min_valid_prob=min_valid_element_prob,
                return_confidence=True,
            )

            if atom_z > 0 and atom_z not in forbidden:
                atom_candidates.append((atom_z, float(atom_prob), coords))

        # Limit element diversity per structure
        if max_unique_elements is not None and max_unique_elements > 0 and atom_candidates:
            element_scores = defaultdict(float)
            for atom_z, atom_prob, _ in atom_candidates:
                element_scores[atom_z] += atom_prob
            kept_elements = {
                atom_z
                for atom_z, _ in sorted(
                    element_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:max_unique_elements]
            }
            atom_candidates = [item for item in atom_candidates if item[0] in kept_elements]

        atomic_numbers = [atom_z for atom_z, _, _ in atom_candidates]
        frac_coords = [coords for _, _, coords in atom_candidates]

        if not atomic_numbers:
            continue

        elements = [Element.from_Z(z) for z in atomic_numbers]

        try:
            structure = Structure(
                lattice=lattice_matrix,
                species=elements,
                coords=frac_coords,
                coords_are_cartesian=False
            )
            structures.append(structure)
        except Exception as e:
            print(f"Error creating structure {i}: {e}")

    return structures


# ============================================================================
# Structure I/O and Analysis
# ============================================================================

def save_structures(structures, output_dir, prefix="gen", start_idx=0):
    """
    Save structures to CIF files.

    Args:
        structures: List of pymatgen Structure objects
        output_dir: Output directory path
        prefix: Filename prefix (default "gen")
        start_idx: Starting index for numbering (default 0)

    Returns:
        Number of structures saved
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for i, structure in enumerate(structures):
        cif_path = os.path.join(output_dir, f"{prefix}_{start_idx + i + 1}.cif")
        structure.to(filename=cif_path)
        saved_count += 1

    return saved_count


def analyze_generated_structures(structures):
    """
    Analyze and print statistics of generated structures.

    Args:
        structures: List of pymatgen Structure objects

    Returns:
        Dictionary with element statistics or None if empty
    """
    if not structures:
        return None

    element_counter = {}
    for struct in structures:
        for site in struct:
            element = site.specie.symbol
            z = site.specie.Z
            if z not in element_counter:
                element_counter[z] = 0
            element_counter[z] += 1

    total_atoms = sum(element_counter.values())
    element_percentages = {z: count / total_atoms * 100 for z, count in element_counter.items()}
    sorted_elements = sorted(element_percentages.items(), key=lambda x: x[1], reverse=True)

    print(f"\nGenerated structure analysis - Total atoms: {total_atoms}")
    print("Element\tAtomic#\tCount\tPercentage(%)")
    print("-" * 40)
    for z, percentage in sorted_elements[:20]:
        element_symbol = Element.from_Z(z).symbol
        count = element_counter[z]
        print(f"{element_symbol}\t{z}\t{count}\t{percentage:.2f}")

    rare_gases_present = [z for z in element_counter if z in RARE_GAS_ELEMENTS]
    if rare_gases_present:
        print("\nWarning: Generated structures contain rare gas elements:")
        for z in rare_gases_present:
            element_symbol = Element.from_Z(z).symbol
            print(f"{element_symbol}(Z={z}): {element_counter[z]} atoms, {element_percentages[z]:.2f}%")

    return {
        "total_atoms": total_atoms,
        "element_counts": element_counter,
        "element_percentages": element_percentages
    }
