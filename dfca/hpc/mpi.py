"""
MPI Utilities for GPAW Validation Pipeline.

Provides collective communication primitives with sequence tracking for debugging.

Features:
    - mpi_broadcast_object: Broadcast Python objects with tracking
    - mpi_trace: Log MPI flow with sequence numbers
    - world: GPAW world communicator reference

Note:
    These functions are critical for checkpoint compatibility.
    Broadcast order and sequence tracking must be preserved during refactoring.
"""

import logging
from typing import Any, Optional

# GPAW MPI imports
try:
    from gpaw.mpi import world
    HAS_GPAW_MPI = True
except ImportError:
    HAS_GPAW_MPI = False
    world = None

logger = logging.getLogger(__name__)

# Global sequence counter for MPI object broadcasts
# CRITICAL: This tracks the order of mpi_broadcast_object calls
_MPI_OBJECT_BCAST_SEQ = 0

# Export rank info
MPI_RANK = world.rank if HAS_GPAW_MPI and world else 0
MPI_SIZE = world.size if HAS_GPAW_MPI and world else 1


def mpi_trace(msg: str) -> None:
    """
    Log MPI flow with sequence numbers.

    Used for debugging MPI issues and verifying broadcast order.
    Only rank 0 logs to avoid duplicate messages.

    Args:
        msg: Log message
    """
    if MPI_RANK == 0:
        logger.debug(msg)


def mpi_broadcast_object(obj: Any, root: int = 0) -> Any:
    """
    Broadcast a Python object from root to all ranks.

    This function wraps the GPAW world.broadcast with:
    1. Sequence tracking for debugging
    2. Automatic logging of broadcast order
    3. Type preservation

    Args:
        obj: Object to broadcast (only value at root matters)
        root: Root rank (default 0)

    Returns:
        Broadcast object on all ranks

    Raises:
        RuntimeError: If GPAW MPI is not available

    Warning:
        DO NOT change the call order of this function.
        Sequence tracking depends on consistent ordering.
    """
    global _MPI_OBJECT_BCAST_SEQ

    if not HAS_GPAW_MPI:
        return obj

    _MPI_OBJECT_BCAST_SEQ += 1
    mpi_trace(f"BCAST_SEQ={_MPI_OBJECT_BCAST_SEQ}")

    try:
        result = world.broadcast(obj, root=root)
        mpi_trace(f"BCAST_SEQ={_MPI_OBJECT_BCAST_SEQ} complete")
        return result
    except Exception as e:
        mpi_trace(f"[MPI_ERROR] BCAST_SEQ={_MPI_OBJECT_BCAST_SEQ} failed: {e}")
        raise


def mpi_barrier() -> None:
    """
    Synchronize all ranks.

    Use this to ensure all ranks reach a certain point before proceeding.
    """
    if HAS_GPAW_MPI:
        world.barrier()


def reset_mpi_sequence() -> None:
    """
    Reset the MPI broadcast sequence counter.

    WARNING: This should only be used for testing.
    In production, sequence tracking must be consistent.
    """
    global _MPI_OBJECT_BCAST_SEQ
    _MPI_OBJECT_BCAST_SEQ = 0


def get_mpi_sequence() -> int:
    """
    Get the current MPI broadcast sequence number.

    Returns:
        Current sequence value
    """
    return _MPI_OBJECT_BCAST_SEQ


__all__ = [
    'mpi_broadcast_object',
    'mpi_trace',
    'mpi_barrier',
    'reset_mpi_sequence',
    'get_mpi_sequence',
    'MPI_RANK',
    'MPI_SIZE',
    'world',
    'HAS_GPAW_MPI',
]
