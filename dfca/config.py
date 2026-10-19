"""
Configuration Loading Module for DFCA.

Provides YAML configuration loading with support for:
    - Loading configurations from YAML files
    - Merging CLI arguments with YAML configs
    - Validating configuration parameters
    - Converting between dataclass and dict formats

Usage:
    from dfca.config import load_config, ConfigNamespace

    # Load from YAML file
    cfg = load_config('configs/train.yaml')

    # Load with overrides
    cfg = load_config('configs/train.yaml',
                      overrides={'batch_size': 32, 'epochs': 50})

    # Convert to namespace for attribute access
    ns = ConfigNamespace.from_dict(cfg)
    print(ns.model.hidden_size)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of values to override in the loaded config

    Returns:
        Dictionary containing merged configuration

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> cfg = load_config('configs/train.yaml')
        >>> cfg = load_config('configs/train.yaml', overrides={'epochs': 100})
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def load_config_with_cli(
    config_path: Union[str, Path],
    cli_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load YAML config and merge with CLI arguments.

    CLI arguments take precedence over YAML values.

    Args:
        config_path: Path to YAML configuration file
        cli_args: Dictionary of CLI argument values

    Returns:
        Merged configuration dictionary
    """
    # Load YAML config
    yaml_config = load_config(config_path)

    # CLI arguments override YAML
    return _deep_merge(yaml_config, cli_args)


def save_config(
    config: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path for output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Override values take precedence. Lists are replaced, not merged.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class ConfigNamespace:
    """
    Namespace class for accessing config values as attributes.

    Provides dot-notation access to nested configuration values.

    Example:
        >>> cfg = load_config('configs/train.yaml')
        >>> ns = ConfigNamespace.from_dict(cfg)
        >>> print(ns.model.hidden_size)
        512
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigNamespace':
        """Create namespace from dictionary."""
        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert namespace back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                items.append(f"{key}=<ConfigNamespace>")
            else:
                items.append(f"{key}={value!r}")
        return f"ConfigNamespace({', '.join(items)})"


class TrainingConfig:
    """
    Training configuration dataclass with type hints.

    This class provides a structured interface for training parameters.
    Use load_config() to populate from YAML.
    """

    def __init__(self, **kwargs):
        # Random seed
        self.seed: int = kwargs.get('seed', 42)

        # Data
        self.data_dir: str = kwargs.get('data_dir', './Data')
        self.max_atoms: int = kwargs.get('max_atoms', 20)
        self.max_length: float = kwargs.get('max_length', 46.7425)
        self.batch_size: int = kwargs.get('batch_size', 16)
        self.num_workers: int = kwargs.get('num_workers', 4)

        # Model
        self.hidden_size: int = kwargs.get('hidden_size', 512)
        self.depth: int = kwargs.get('depth', 18)
        self.num_heads: int = kwargs.get('num_heads', 8)
        self.mlp_ratio: float = kwargs.get('mlp_ratio', 4.0)

        # Diffusion
        self.diffusion_steps: int = kwargs.get('diffusion_steps', 1000)
        self.z_noise_scale: float = kwargs.get('z_noise_scale', 0.113001)
        self.lattice_c_noise_scale: float = kwargs.get('lattice_c_noise_scale', 0.340659)

        # Optimizer
        self.learning_rate: float = kwargs.get('learning_rate', 0.000203886)
        self.weight_decay: float = kwargs.get('weight_decay', 0.0468414)
        self.clip_grad: float = kwargs.get('clip_grad', 2.803345)

        # Constraints
        self.use_physical_constraints: bool = kwargs.get('use_physical_constraints', True)
        self.lattice_loss_z_weight: float = kwargs.get('lattice_loss_z_weight', 0.706323)
        self.decay_mid: float = kwargs.get('decay_mid', 0.224396)
        self.decay_late: float = kwargs.get('decay_late', 0.122513)

        # Training
        self.epochs: int = kwargs.get('epochs', 100)
        self.output_dir: str = kwargs.get('output_dir', './output')

        # Checkpointing
        self.ema_decay: float = kwargs.get('ema_decay', 0.9999)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        cfg = load_config(path)
        # Flatten nested config for dataclass
        flat = _flatten_dict(cfg)
        return cls(**flat)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()


class GenerationConfig:
    """
    Generation configuration dataclass with type hints.
    """

    def __init__(self, **kwargs):
        self.checkpoint: str = kwargs.get('checkpoint', '')
        self.data_dir: str = kwargs.get('data_dir', './Data')
        self.max_atoms: int = kwargs.get('max_atoms', 20)
        self.max_length: float = kwargs.get('max_length', 46.7425)
        self.hidden_size: int = kwargs.get('hidden_size', 512)
        self.depth: int = kwargs.get('depth', 18)
        self.num_heads: int = kwargs.get('num_heads', 8)
        self.diffusion_steps: int = kwargs.get('diffusion_steps', 1000)
        self.num_samples: int = kwargs.get('num_samples', 1000)
        self.batch_size: int = kwargs.get('batch_size', 32)
        self.output_dir: str = kwargs.get('output_dir', './generated_crystals')
        self.device: str = kwargs.get('device', 'cuda')
        self.coord_tolerance: float = kwargs.get('coord_tolerance', 0.05)
        self.slab_center: float = kwargs.get('slab_center', 0.5)
        self.slab_half_width: float = kwargs.get('slab_half_width', 0.15)
        self.element_sigma: float = kwargs.get('element_sigma', 0.05)
        self.min_valid_element_prob: float = kwargs.get('min_valid_element_prob', 0.001)
        self.min_c_length: float = kwargs.get('min_c_length', 18.0)
        self.max_c_length: float = kwargs.get('max_c_length', 35.0)
        self.gamma_min_deg: float = kwargs.get('gamma_min_deg', 60.0)
        self.gamma_max_deg: float = kwargs.get('gamma_max_deg', 120.0)
        self.topk_seed: int = kwargs.get('topk_seed', 20260504)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'GenerationConfig':
        """Load configuration from YAML file."""
        cfg = load_config(path)
        flat = _flatten_dict(cfg)
        return cls(**flat)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten nested dictionary with underscore-separated keys.

    Example:
        {'model': {'hidden_size': 512}} -> {'model_hidden_size': 512}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


__all__ = [
    'load_config',
    'load_config_with_cli',
    'save_config',
    'ConfigNamespace',
    'TrainingConfig',
    'GenerationConfig',
]
