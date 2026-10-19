"""
Crystal Dataset Module.

PyTorch Dataset interface for loading preprocessed crystal data.

This module provides the standard PyTorch Dataset interface used by training.
It loads preprocessed pickle files and returns tensor dictionaries.
"""

import os
import time
import pickle

import torch
from torch.utils.data import Dataset

from dfca.data.representation import MAX_LATTICE_LENGTH
from dfca.data.preprocessing import preprocess_dataset


class CrystalDataset(Dataset):
    """
    Crystal structure dataset for PyTorch.

    Loads preprocessed crystal data (lattice vectors + atom features) and provides
    standard PyTorch Dataset interface.

    Representation schema:
        - lattice_vectors: [3, 3] float32, normalized lattice matrix
        - atom_features: [max_atoms, 5] float32
            - [0] = normalized row (element period)
            - [1] = normalized column (element group)
            - [2] = fractional x
            - [3] = fractional y
            - [4] = fractional z (centered for 2D)

    Args:
        csv_file: Path to dataset CSV file with 'cif' column
        max_atoms: Maximum number of atoms per crystal (default 20)
        max_length: Maximum lattice vector length for normalization (default MAX_LATTICE_LENGTH)
        transform: Optional transform to apply to samples
        preprocessed_file: Path to preprocessed file, auto-generated if None
        force_preprocess: Force repreprocessing even if preprocessed file exists
    """

    def __init__(self, csv_file, max_atoms=20, max_length=MAX_LATTICE_LENGTH,
                 transform=None, preprocessed_file=None, force_preprocess=False):
        self.csv_file = csv_file
        self.max_atoms = max_atoms
        self.max_length = max_length
        self.transform = transform

        # Auto-generate preprocessed file path if not provided
        if preprocessed_file is None:
            preprocessed_dir = os.path.join(os.path.dirname(csv_file), "preprocessed")
            os.makedirs(preprocessed_dir, exist_ok=True)

            csv_basename = os.path.basename(csv_file).split('.')[0]
            self.preprocessed_file = os.path.join(
                preprocessed_dir,
                f"{csv_basename}_preprocessed_ma{max_atoms}.pkl"
            )
        else:
            self.preprocessed_file = preprocessed_file

        # Check if preprocessing is needed
        if force_preprocess or not os.path.exists(self.preprocessed_file):
            preprocess_dataset(csv_file, self.preprocessed_file, max_atoms, max_length)

        # Load preprocessed data with retry mechanism
        max_retries = 5
        retry_count = 0
        retry_delay = 1

        while retry_count < max_retries:
            try:
                with open(self.preprocessed_file, 'rb') as f:
                    self.data = pickle.load(f)
                break
            except (EOFError, pickle.UnpicklingError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Failed to load dataset, retrying ({retry_count}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(
                        f"Failed to load preprocessed data {self.preprocessed_file} after multiple attempts: {e}"
                    )

        print(f"Loaded dataset: {csv_file}, samples: {len(self.data)}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index (int or torch.Tensor)

        Returns:
            dict: {
                'lattice_vectors': torch.Tensor [3, 3],
                'atom_features': torch.Tensor [max_atoms, 5]
            }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data = self.data[idx]

        lattice_vectors = sample_data['lattice_vectors']
        atom_features = sample_data['atom_features']

        # Convert numpy arrays to torch tensors
        lattice_vectors_tensor = torch.from_numpy(lattice_vectors)
        atom_features_tensor = torch.from_numpy(atom_features)

        sample = {
            'lattice_vectors': lattice_vectors_tensor,
            'atom_features': atom_features_tensor,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
