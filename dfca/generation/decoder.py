"""
Element Decoding for Crystal Generation.

This module handles:
    - Mapping model row/column outputs to element atomic numbers
    - Element filtering based on allowed set and rarity
    - Confidence scoring for element predictions

IMPORTANT: This code is migrated directly from the original generate_crystals.py.
Scientific behavior (element mapping, probability calculations) must be preserved exactly.
"""

import numpy as np
from scipy.stats import norm

# Authoritative imports (Stage 9 tested)
from dfca.data.representation import ATOMIC_NUMBER_TO_POSITION, normalize_row, normalize_column


# Rare gas elements list - must match the original exactly
RARE_GAS_ELEMENTS = [2, 10, 18, 36, 54, 86]

# Step sizes for probability integration (periodic table grid)
_ROW_STEP = 1.0 / 7.0
_COLUMN_STEP = 1.0 / 18.0


def map_to_element(row_value, column_value, sigma=0.01, filter_rare_gases=True, max_atomic_number=94,
                   allowed_atomic_numbers=None, min_valid_prob=1e-4, invalid_margin=0.15, return_confidence=False):
    """
    Map row/column coordinates to element atomic number using Gaussian probability.

    This function implements the periodic-table-based element decoding:
    1. Compute probability of row/column values under Gaussian distribution
    2. Compare valid element probability vs invalid slot probability
    3. Return the most likely element or 0 (invalid) if confidence too low

    The decoding uses a soft probability approach rather than hard classification:
    - Each element occupies a cell in the (period, group) periodic table grid
    - The probability of predicting element Z is the joint probability of:
      P(row | period) * P(col | group)
    - Invalid slots (padding) are centered at normalized (0, 0) with a small margin

    Args:
        row_value: Normalized row coordinate (from model output, typically in [-1, 1])
        column_value: Normalized column coordinate (from model output, typically in [-1, 1])
        sigma: Gaussian standard deviation for probability calculation (default 0.01)
        filter_rare_gases: Exclude noble gas elements from predictions (default True)
        max_atomic_number: Maximum atomic number to consider (default 94)
        allowed_atomic_numbers: Restrict predictions to this set (default None = all)
        min_valid_prob: Minimum element probability threshold (default 1e-4)
        invalid_margin: Invalid slot threshold (default 0.15)
        return_confidence: Return (element, confidence) tuple instead of just element

    Returns:
        Atomic number (int) or (atomic_number, confidence) tuple if return_confidence=True
        Returns 0 if the slot is invalid or confidence is below threshold
    """
    # Invalid slot detection: row <= -0.5 indicates padding
    if row_value <= -0.5:
        return (0, invalid_margin) if return_confidence else 0

    # Get normalized coordinates for invalid slot (centered at 0, 0)
    invalid_row = normalize_row(0)
    invalid_column = normalize_column(0)

    # Compute probability mass for invalid slot region
    invalid_row_prob = norm.cdf(invalid_row + _ROW_STEP, loc=row_value, scale=sigma) - \
                      norm.cdf(invalid_row - _ROW_STEP, loc=row_value, scale=sigma)
    invalid_column_prob = norm.cdf(invalid_column + _COLUMN_STEP, loc=column_value, scale=sigma) - \
                         norm.cdf(invalid_column - _COLUMN_STEP, loc=column_value, scale=sigma)
    invalid_prob = invalid_row_prob * invalid_column_prob

    # Compute probability for each valid element
    element_probs = []
    for z, (elem_row, elem_column) in ATOMIC_NUMBER_TO_POSITION.items():
        # Skip atomic number 0 (invalid placeholder)
        if z == 0:
            continue
        # Filter rare gases if requested
        if filter_rare_gases and z in RARE_GAS_ELEMENTS:
            continue
        # Filter by maximum atomic number
        if z > max_atomic_number:
            continue
        # Filter by allowed set
        if allowed_atomic_numbers is not None and z not in allowed_atomic_numbers:
            continue

        # Get normalized coordinates for this element
        norm_row = normalize_row(elem_row)
        norm_column = normalize_column(elem_column)

        # Compute probability bounds (cell boundaries in normalized space)
        row_upper = norm_row + _ROW_STEP if norm_row < 1 else np.inf
        row_lower = norm_row - _ROW_STEP if norm_row > -1 else -np.inf
        column_upper = norm_column + _COLUMN_STEP if norm_column < 1 else np.inf
        column_lower = norm_column - _COLUMN_STEP if norm_column > -1 else -np.inf

        # Compute probability mass under Gaussian
        row_prob = norm.cdf(row_upper, loc=row_value, scale=sigma) - \
                   norm.cdf(row_lower, loc=row_value, scale=sigma)
        column_prob = norm.cdf(column_upper, loc=column_value, scale=sigma) - \
                      norm.cdf(column_lower, loc=column_value, scale=sigma)

        # Joint probability (assuming independence)
        joint_prob = row_prob * column_prob
        element_probs.append((z, joint_prob))

    if not element_probs:
        return (0, 0.0) if return_confidence else 0

    # Select element with highest probability
    best_z, best_prob = max(element_probs, key=lambda x: x[1])

    # Compare with invalid probability to decide validity
    if best_prob < min_valid_prob or invalid_prob >= best_prob:
        return (0, best_prob) if return_confidence else 0

    return (best_z, best_prob) if return_confidence else best_z


def load_allowed_atomic_numbers(data_dir):
    """
    Load allowed atomic numbers from dataset CSV files.

    Scans train.csv, val.csv, and test.csv to collect all element types
    present in the training distribution. This restricts generation to
    elements observed during training.

    Args:
        data_dir: Path to dataset directory containing CSV files

    Returns:
        Sorted list of allowed atomic numbers, or None if no data found
    """
    import os
    import csv
    from pymatgen.core.structure import Structure

    allowed_atomic_numbers = set()
    if not data_dir or not os.path.isdir(data_dir):
        return None

    for split_name in ("train.csv", "val.csv", "test.csv"):
        csv_path = os.path.join(data_dir, split_name)
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cif_string = row.get("cif")
                if not cif_string:
                    continue
                try:
                    structure = Structure.from_str(cif_string, fmt="cif")
                except Exception:
                    continue
                for site in structure.sites:
                    z = site.specie.Z
                    if z not in RARE_GAS_ELEMENTS:
                        allowed_atomic_numbers.add(z)

    return sorted(allowed_atomic_numbers) if allowed_atomic_numbers else None
