"""
MPI Utilities for GPAW Validation.

MPI-safe utilities for distributed GPAW calculations.
All GPAW collective operations must be executed by all MPI ranks simultaneously.

References:
    - GPAW MPI documentation
    - C2DB parallelization scheme
"""

from __future__ import annotations

import sys
import logging
from typing import Any, Optional

import numpy as np

# MPI initialization
try:
    from gpaw.mpi import world as _gpaw_world
    from gpaw.mpi import broadcast as _gpaw_broadcast
    MPI_RANK = _gpaw_world.rank
    MPI_SIZE = _gpaw_world.size
except ImportError:
    MPI_RANK = 0
    MPI_SIZE = 1
    _gpaw_broadcast = None
    _gpaw_world = None

logger = logging.getLogger("dfca.validation.mpi")

# Sequence counter for broadcast diagnostics
_MPI_OBJECT_BCAST_SEQ = 0


def mpi_trace(msg: str) -> None:
    """
    Unified MPI tracing function for diagnostics.

    All ranks print to stderr to avoid confusion with stdout.
    Prints rank information for debugging MPI divergence issues.
    """
    sys.stderr.write(
        f"[MPI_TRACE rank={MPI_RANK}/{MPI_SIZE}] {msg}\n"
    )
    sys.stderr.flush()


def mpi_broadcast_object(obj: Any, root: int = 0) -> Any:
    """
    Broadcast one Python object collectively.

    Every rank must call this function in the same sequence to avoid
    deadlock. This is a critical requirement for GPAW collective operations.

    Args:
        obj: Object to broadcast from root rank. None for non-root ranks.
        root: Root rank number (default 0)

    Returns:
        The same object on all ranks
    """
    global _MPI_OBJECT_BCAST_SEQ

    seq = _MPI_OBJECT_BCAST_SEQ
    _MPI_OBJECT_BCAST_SEQ += 1

    mpi_trace(
        f"[OBJ_BCAST_ENTER seq={seq}] "
        f"root={root} local_type={type(obj).__name__}"
    )

    if MPI_SIZE <= 1:
        mpi_trace(
            f"[OBJ_BCAST_EXIT seq={seq}] "
            f"result_type={type(obj).__name__}"
        )
        return obj

    if _gpaw_broadcast is None:
        raise RuntimeError(
            "mpi_broadcast_object: GPAW broadcast unavailable"
        )

    try:
        result = _gpaw_broadcast(obj, root)

        mpi_trace(
            f"[OBJ_BCAST_EXIT seq={seq}] "
            f"result_type={type(result).__name__}"
        )

        return result

    except Exception as e:
        mpi_trace(
            f"[OBJ_BCAST_FATAL seq={seq}] "
            f"{type(e).__name__}: {e}"
        )
        raise


def mpi_broadcast_numpy(arr: Optional[np.ndarray], root: int = 0) -> np.ndarray:
    """
    Broadcast a numpy array to all ranks.

    Steps:
    1. Broadcast shape (tuple of int)
    2. Broadcast dtype string
    3. All ranks create empty buffer
    4. Broadcast actual data buffer (numpy uint8)

    Args:
        arr: numpy array on root rank, None on other ranks
        root: Root rank number (default 0)

    Returns:
        Identical numpy array on all ranks
    """
    if MPI_SIZE <= 1:
        return arr

    # Step 1: Prepare shape and dtype on root, keep None on others
    if MPI_RANK == root:
        if arr is None:
            raise RuntimeError("mpi_broadcast_numpy: root array is None")
        arr = np.ascontiguousarray(arr)
        shape = tuple(arr.shape)
        dtype_str = arr.dtype.str
    else:
        shape = None
        dtype_str = None

    # Broadcast shape
    shape = mpi_broadcast_object(shape, root)
    dtype_str = mpi_broadcast_object(dtype_str, root)

    # Step 2: Calculate buffer size
    dtype = np.dtype(dtype_str)
    nbytes = int(np.prod(shape)) * dtype.itemsize

    # Step 3: All ranks create buffer
    if MPI_RANK == root:
        buf = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    else:
        buf = np.empty(nbytes, dtype=np.uint8)

    # Step 4: Broadcast buffer
    if _gpaw_world is not None:
        _gpaw_world.broadcast(buf, root)

    # Reconstruct array
    result = buf.view(dtype).reshape(shape).copy()
    return result


def mpi_world_sum(local_value: int, root: int = 0) -> int:
    """
    Perform MPI all-reduce sum operation.

    Used for synchronizing failure flags across all ranks.

    Args:
        local_value: Local value to sum
        root: Root rank (for logging only)

    Returns:
        Sum across all ranks
    """
    if MPI_SIZE <= 1:
        return local_value

    if _gpaw_world is not None:
        return _gpaw_world.sum(local_value)
    return local_value


def mpi_barrier() -> None:
    """Synchronize all MPI ranks at a barrier."""
    if _gpaw_world is not None:
        _gpaw_world.barrier()


__all__ = [
    'MPI_RANK',
    'MPI_SIZE',
    'mpi_trace',
    'mpi_broadcast_object',
    'mpi_broadcast_numpy',
    'mpi_world_sum',
    'mpi_barrier',
]
