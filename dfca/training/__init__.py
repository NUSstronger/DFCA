"""
DFCA Training Module

Provides training infrastructure for CrystalDiT with physical constraints.
"""

from dfca.training.constraints import (
    compute_physical_constraints,
    compute_weighted_constraint_loss,
    get_decay_factor,
    build_weighted_terms,
    CONSTRAINT_KEYS,
    DECAY_BOUNDARY_MID,
    DECAY_BOUNDARY_LATE,
)

from dfca.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

from dfca.training.trainer import (
    train,
    train_epoch,
    evaluate_epoch,
    set_seed,
    seed_worker,
    setup_logging,
    setup,
    cleanup,
    reduce_mean,
)

__all__ = [
    # Constraints
    "compute_physical_constraints",
    "compute_weighted_constraint_loss",
    "get_decay_factor",
    "build_weighted_terms",
    "CONSTRAINT_KEYS",
    "DECAY_BOUNDARY_MID",
    "DECAY_BOUNDARY_LATE",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    # Trainer
    "train",
    "train_epoch",
    "evaluate_epoch",
    "set_seed",
    "seed_worker",
    "setup_logging",
    "setup",
    "cleanup",
    "reduce_mean",
]
