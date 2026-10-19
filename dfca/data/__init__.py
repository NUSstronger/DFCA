"""
DFCA Data Module.

Provides crystal representation, preprocessing, and dataset interfaces.

Submodules:
    - representation: Core tensor representation for crystals
    - preprocessing: CSV to pickle conversion
    - dataset: PyTorch Dataset interface
"""

from dfca.data.representation import (
    # Constants
    MAX_LATTICE_LENGTH,
    ATOMIC_NUMBER_TO_POSITION,
    POSITION_TO_ATOMIC_NUMBER,
    ATOMIC_NUMBER_TO_MASS,
    # Lookups
    VALID_ELEMENT_ROWCOL,
    VALID_ELEMENT_MASSES,
    # Functions
    normalize_row,
    normalize_column,
    denormalize_row,
    denormalize_column,
    periodic_span_center,
    get_element_properties,
    decode_avg_atomic_mass,
    encode_element,
    encode_lattice,
    decode_lattice,
)

from dfca.data.preprocessing import preprocess_dataset

from dfca.data.dataset import CrystalDataset

__all__ = [
    # Constants
    "MAX_LATTICE_LENGTH",
    "ATOMIC_NUMBER_TO_POSITION",
    "POSITION_TO_ATOMIC_NUMBER",
    "ATOMIC_NUMBER_TO_MASS",
    "VALID_ELEMENT_ROWCOL",
    "VALID_ELEMENT_MASSES",
    # Functions
    "normalize_row",
    "normalize_column",
    "denormalize_row",
    "denormalize_column",
    "periodic_span_center",
    "get_element_properties",
    "decode_avg_atomic_mass",
    "encode_element",
    "encode_lattice",
    "decode_lattice",
    # Preprocessing
    "preprocess_dataset",
    # Dataset
    "CrystalDataset",
]
