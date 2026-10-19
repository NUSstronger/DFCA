"""
Checkpoint Management for GPAW Validation Pipeline.

Handles loading and saving of ValidationRecord checkpoints with numpy serialization.

Features:
    - JSON checkpoint format v2.0
    - Automatic numpy type conversion
    - Version tracking
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Checkpoint version
CHECKPOINT_VERSION = '2.0'


def _serialize_value(v: Any) -> Any:
    """
    Serialize a value for JSON, handling numpy types.

    Args:
        v: Value to serialize

    Returns:
        JSON-serializable value
    """
    if isinstance(v, (np.integer,)):
        return int(v)
    elif isinstance(v, (np.floating,)):
        return float(v)
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, list) and v:
        # Handle list of numpy scalars
        if isinstance(v[0], (np.integer, np.floating)):
            return [x.item() if hasattr(x, 'item') else x for x in v]
    return v


def serialize_records(records: List[dataclass]) -> List[Dict]:
    """
    Serialize a list of dataclass records to checkpoint format.

    Args:
        records: List of ValidationRecord objects

    Returns:
        List of dictionaries suitable for JSON serialization
    """
    from dataclasses import asdict

    serialized = []
    for rec in records:
        rec_dict = asdict(rec)
        rec_serialized = {}
        for k, v in rec_dict.items():
            rec_serialized[k] = _serialize_value(v)
        serialized.append(rec_serialized)
    return serialized


def load_checkpoint(checkpoint_path: str) -> tuple:
    """
    Load a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file

    Returns:
        Tuple of (records, version, timestamp)
    """
    if not os.path.exists(checkpoint_path):
        return [], CHECKPOINT_VERSION, None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        version = data.get('version', '1.0')
        timestamp = data.get('timestamp')
        records = data.get('records', [])

        logger.info(f"Loaded checkpoint: {len(records)} records (v{version})")
        return records, version, timestamp

    except json.JSONDecodeError as e:
        logger.warning(f"Checkpoint parse error: {e}, starting fresh")
        return [], CHECKPOINT_VERSION, None
    except Exception as e:
        logger.warning(f"Checkpoint load error: {e}, starting fresh")
        return [], CHECKPOINT_VERSION, None


def save_checkpoint(
    checkpoint_path: str,
    records: List[dataclass],
    version: str = CHECKPOINT_VERSION
) -> bool:
    """
    Save a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file
        records: List of ValidationRecord objects
        version: Checkpoint version string

    Returns:
        True if saved successfully

    Note:
        This function should only be called from rank 0 to avoid
        file corruption from concurrent writes.
    """
    try:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        serialized = serialize_records(records)

        ckpt = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'records': serialized,
        }

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved checkpoint: {len(records)} records")
        return True

    except Exception as e:
        logger.warning(f"Checkpoint save failed: {e}")
        return False


def checkpoint_exists(checkpoint_path: str) -> bool:
    """
    Check if a checkpoint file exists.

    Args:
        checkpoint_path: Path to checkpoint JSON file

    Returns:
        True if checkpoint exists
    """
    return os.path.exists(checkpoint_path)


def get_checkpoint_version(checkpoint_path: str) -> str:
    """
    Get the version of a checkpoint file without loading full data.

    Args:
        checkpoint_path: Path to checkpoint JSON file

    Returns:
        Version string, or 'unknown' if not found
    """
    if not os.path.exists(checkpoint_path):
        return 'unknown'

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('version', 'unknown')
    except Exception:
        return 'unknown'


__all__ = [
    'CHECKPOINT_VERSION',
    'serialize_records',
    'load_checkpoint',
    'save_checkpoint',
    'checkpoint_exists',
    'get_checkpoint_version',
]
