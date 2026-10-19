"""
Checkpoint Management for CrystalDiT Training.

This module handles saving and loading training checkpoints with:
    - Model weights
    - Optimizer state
    - Scheduler state
    - RNG state for reproducibility
    - Epoch and global step tracking

IMPORTANT: This code is migrated directly from the original train.py.
Checkpoint schema keys are preserved exactly to maintain backward compatibility.
"""

import os
import logging
import random

import numpy as np
import torch
import torch.distributed as dist


# ============================================================================
# Atomic Save Utility
# ============================================================================

def _atomic_torch_save(obj, path):
    """
    Atomic checkpoint save: write to temp file, then atomically rename.
    Prevents corrupted checkpoints from partial writes during interruption.
    """
    tmp_path = path + '.tmp'
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {path}: {type(e).__name__}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logging.warning(
                    f"Failed to remove partial checkpoint file {tmp_path}: "
                    f"{type(cleanup_err).__name__}: {cleanup_err}"
                )
        return False


# ============================================================================
# Checkpoint Save / Load
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, rank, is_best=False, is_periodic=False):
    """Save model checkpoint (weights-only for periodic, full state for latest/best)."""
    if rank != 0:
        return
    checkpoint = {'model': model.module.state_dict()}
    if not is_periodic:
        checkpoint.update({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'global_step': global_step,
            'args': args,
            'random_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        })
    if not is_periodic:
        latest_path = os.path.join(args.output_dir, 'latest_checkpoint.pt')
        if _atomic_torch_save(checkpoint, latest_path):
            logging.info(f"Saved latest checkpoint: {latest_path}")
    if is_periodic:
        epoch_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        if _atomic_torch_save(checkpoint, epoch_path):
            logging.info(f"Saved periodic checkpoint: {epoch_path} (weights only)")
    if is_best:
        best_path = os.path.join(args.output_dir, 'best_model.pt')
        if _atomic_torch_save(checkpoint, best_path):
            logging.info(f"Saved best model: {best_path}")


def load_checkpoint(model, optimizer, scheduler, args, rank):
    """Load checkpoint and restore training state."""
    if not os.path.exists(args.resume):
        if rank == 0:
            logging.info(f"Checkpoint not found: {args.resume}, starting from scratch.")
        return 0, 0
    if rank == 0:
        logging.info(f"Loading checkpoint: {args.resume}")
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(args.resume, map_location=map_location, weights_only=False)
    model.module.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0:
                logging.info("Successfully loaded optimizer state")
        except (RuntimeError, ValueError, KeyError) as e:
            if rank == 0:
                logging.warning(f"Optimizer state load failed ({type(e).__name__}: {e}); using new optimizer state")
    if 'scheduler' in checkpoint and scheduler is not None and checkpoint.get('scheduler') is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                logging.info("Successfully loaded scheduler state")
        except (RuntimeError, ValueError, KeyError) as e:
            if rank == 0:
                logging.warning(f"Scheduler state load failed ({type(e).__name__}: {e}); using new scheduler state")
    if 'random_state' in checkpoint:
        random_state = checkpoint['random_state']
        if random_state.get('python') is not None:
            random.setstate(random_state['python'])
        if random_state.get('numpy') is not None:
            np.random.set_state(random_state['numpy'])
        if random_state.get('torch') is not None:
            torch.set_rng_state(random_state['torch'])
        if random_state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_state['cuda'])
        if rank == 0:
            logging.info("Successfully restored random state")
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    if rank == 0:
        if epoch > 0:
            logging.info(f"Resuming training from epoch {epoch} and step {global_step}")
        else:
            logging.info(f"Loaded model weights, but starting from epoch 0 (simplified checkpoint)")
    return epoch, global_step
