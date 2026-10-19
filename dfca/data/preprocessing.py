"""
Dataset Preprocessing Module.

Handles the conversion of raw CSV datasets (CIF strings) into preprocessed
pickle files containing DFCA representation tensors.

This module is responsible for:
    - Reading CSV files with CIF strings
    - Parsing crystal structures
    - Converting to DFCA representation
    - Caching preprocessed data

The preprocessing uses dfca.data.representation for all representation logic.
"""

import os
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from dfca.data.representation import (
    MAX_LATTICE_LENGTH,
    ATOMIC_NUMBER_TO_POSITION,
    normalize_row,
    normalize_column,
    periodic_span_center,
)


def preprocess_dataset(csv_file, output_file, max_atoms=20, max_length=MAX_LATTICE_LENGTH):
    """
    Preprocess dataset by parsing CIF strings into lattice vectors and atom features.

    Args:
        csv_file: Path to dataset CSV file with 'cif' column
        output_file: Path to preprocessed data output file (.pkl)
        max_atoms: Maximum number of atoms per crystal (default 20)
        max_length: Maximum lattice vector length for normalization (default MAX_LATTICE_LENGTH)

    Returns:
        int: Number of successfully preprocessed samples

    Note:
        Failed samples are skipped silently. Only valid samples are written.
    """
    from pymatgen.core.structure import Structure

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Skip if already preprocessed
    if os.path.exists(output_file) and not output_file.endswith('.tmp'):
        print(f"Preprocessed file already exists: {output_file}, skipping preprocessing")
        return 0

    data = pd.read_csv(csv_file)
    print(f"Starting dataset preprocessing: {csv_file}, total samples: {len(data)}")

    processed_data = []
    success_count = 0
    error_count = 0

    temp_file = output_file + '.tmp'

    try:
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing progress"):
            try:
                cif_string = row['cif']
                structure = Structure.from_str(cif_string, fmt="cif")

                # Get lattice vectors and normalize
                lattice_vectors = structure.lattice.matrix
                normalized_lattice = lattice_vectors / max_length

                # Get atom information
                atomic_numbers = []
                atom_coords = []

                for site in structure.sites:
                    if len(atomic_numbers) >= max_atoms:
                        break

                    atomic_numbers.append(site.specie.Z)
                    atom_coords.append(list(site.frac_coords))

                # Center z coordinates for 2D materials.
                # Use a periodic center instead of a plain mean so slabs that cross
                # z=0/1 are still aligned to the middle of the vacuum.
                if len(atom_coords) > 0:
                    valid_z = [c[2] for c in atom_coords]
                    z_center = periodic_span_center(valid_z)
                    z_shift = 0.5 - z_center
                    for c in atom_coords:
                        c[2] = (c[2] + z_shift) % 1.0  # Keep in [0, 1) range

                # Pad with invalid atoms if necessary
                while len(atomic_numbers) < max_atoms:
                    atomic_numbers.append(0)
                    atom_coords.append([-1, -1, -1])

                # Convert atomic numbers to periodic table positions and normalize
                atom_features = []
                for z, coords in zip(atomic_numbers, atom_coords):
                    if z > 0:  # Valid atom
                        elem_row, elem_column = ATOMIC_NUMBER_TO_POSITION[z]
                        norm_row = normalize_row(elem_row)
                        norm_column = normalize_column(elem_column)
                        atom_features.append([norm_row, norm_column, coords[0], coords[1], coords[2]])
                    else:  # Invalid atom
                        atom_features.append([-1, -1, -1, -1, -1])

                lattice_vectors_np = normalized_lattice.astype(np.float32)
                atom_features_np = np.array(atom_features, dtype=np.float32)

                processed_data.append({
                    'lattice_vectors': lattice_vectors_np,
                    'atom_features': atom_features_np,
                    'index': idx
                })

                success_count += 1

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                error_count += 1
                # Skip failed samples — do not write zero-filled placeholders.

        # Save preprocessed data to temporary file
        with open(temp_file, 'wb') as f:
            pickle.dump(processed_data, f)

        # Atomic rename operation
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)

        print(f"Preprocessing completed: wrote {success_count} valid samples, skipped {error_count} failed samples.")
        print(f"Preprocessed data saved to: {output_file}")

    except Exception as e:
        print(f"Serious error during preprocessing: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

    return success_count
