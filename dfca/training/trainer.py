"""
Training Loop and Distributed Training Utilities for CrystalDiT.

This module contains:
    - Main distributed training function (train)
    - Epoch-level training and evaluation loops
    - Gradient checking and clipping utilities
    - Distributed synchronization primitives
    - Logging and metrics utilities
    - LR scheduler restart epoch calculation

IMPORTANT: This code is migrated directly from the original train.py.
The execution order and logic flow are preserved exactly.
"""

import os
import time
import math
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm

# Import from project modules (authoritative paths - Stage 9 tested)
from dfca.data.dataset import CrystalDataset
from dfca.data.preprocessing import preprocess_dataset
from dfca.models.dit import CrystalDiT, count_parameters
from dfca.diffusion.gaussian import CrystalGaussianDiffusion

# Import from sibling modules
from dfca.training.constraints import (
    compute_physical_constraints,
    compute_weighted_constraint_loss,
    get_decay_factor,
    build_weighted_terms,
    _empty_constraint_stats,
    _average_constraint_stats,
    CONSTRAINT_KEYS,
    DECAY_BOUNDARY_MID,
    DECAY_BOUNDARY_LATE,
)
from dfca.training.checkpoint import save_checkpoint, load_checkpoint

# ============================================================================
# Scheduler Configuration Constants
# ============================================================================

SCHEDULER_WARMUP_EPOCHS = 5
SCHEDULER_T0_EPOCHS = 25
SCHEDULER_T_MULT = 2


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def reduce_mean(value, device, world_size):
    """Average a scalar across all distributed workers."""
    value_tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        value_tensor /= world_size
    return value_tensor.item()


def _sync_bad_flag(local_bad, device):
    """Sync bad-flag across workers (for detecting NaN/Inf gradients)."""
    bad_tensor = torch.tensor([1.0 if local_bad else 0.0], device=device)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(bad_tensor, op=dist.ReduceOp.MAX)
    return bad_tensor.item() > 0.0


def _find_nonfinite_gradient(model):
    """Check if any model parameter has non-finite gradients."""
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            return True, name
    return False, None


def _optimizer_step_with_grad_guard(model, optimizer, scheduler, args, rank, device, epoch, global_step):
    """Perform optimizer step with gradient sanity checks and clipping."""
    local_grad_bad, bad_grad_name = _find_nonfinite_gradient(model)
    if _sync_bad_flag(local_grad_bad, device):
        if rank == 0:
            logging.warning(
                f"Epoch {epoch} | Step {global_step} | Non-finite gradient detected "
                f"(local_bad_param={bad_grad_name}); dropping accumulated gradients."
            )
        optimizer.zero_grad(set_to_none=True)
        return False

    if args.clip_grad:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        local_clip_bad = not torch.isfinite(grad_norm).item()
        if _sync_bad_flag(local_clip_bad, device):
            if rank == 0:
                grad_norm_str = f"{grad_norm.item():.6f}" if torch.isfinite(grad_norm) else "nan"
                logging.warning(
                    f"Epoch {epoch} | Step {global_step} | Non-finite gradient norm after clipping "
                    f"(grad_norm={grad_norm_str}); dropping accumulated gradients."
                )
            optimizer.zero_grad(set_to_none=True)
            return False

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()
    return True


# ============================================================================
# Random Seed Utilities
# ============================================================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, rank=0):
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(actual_seed)


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logging(rank, args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'train.log')
    if rank == 0:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        root_logger.setLevel(logging.INFO)
        root_logger.info(f"Logging to file: {log_file}")
    else:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(stream_handler)
        root_logger.setLevel(logging.INFO)
    return logging.getLogger()


# ============================================================================
# Distributed Training Setup
# ============================================================================

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29527'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# ============================================================================
# Training Step Logging Utilities
# ============================================================================

def _log_step(rank, iterator, logged_loss, logged_constraint_terms, args, global_step, epoch, lr, use_constraints):
    """Update tqdm progress bar and log step metrics."""
    if isinstance(iterator, tqdm):
        postfix = {
            'total': f"{logged_loss['total']:.4f}",
            'base': f"{logged_loss['base']:.4f}",
            'constraint': f"{logged_loss['constraint']:.4f}",
            'invalid': f"{logged_constraint_terms['invalid_slot_loss']:.4f}",
            'count': f"{logged_constraint_terms['soft_atom_count_loss']:.4f}",
            'RowCol': f"{logged_constraint_terms['rowcol_validity_loss']:.4f}",
            'SlabC': f"{logged_constraint_terms['slab_center_loss']:.4f}",
            'SlabS': f"{logged_constraint_terms['slab_spread_loss']:.4f}",
            'LatLen': f"{logged_constraint_terms['lattice_len_loss']:.4f}",
            'LatAng': f"{logged_constraint_terms['lattice_angle_loss']:.4f}",
            'ElemDist': f"{logged_constraint_terms['elem_count_dist_loss']:.4f}",
            'Density': f"{logged_constraint_terms['density_loss']:.4f}",
            'CLenCeil': f"{logged_constraint_terms['c_length_ceiling_loss']:.4f}",
            'lr': f"{lr:.6f}"
        }
        iterator.set_postfix(postfix)
    if args.use_physical_constraints:
        logging.info(
            f"Epoch {epoch} | Step {global_step} | "
            f"Total: {logged_loss['total']:.6f} | "
            f"Base: {logged_loss['base']:.6f} | "
            f"Constraint: {logged_loss['constraint']:.6f} | "
            f"Invalid: {logged_constraint_terms['invalid_slot_loss']:.6f} | "
            f"Count: {logged_constraint_terms['soft_atom_count_loss']:.6f} | "
            f"RowCol: {logged_constraint_terms['rowcol_validity_loss']:.6f} | "
            f"SlabC: {logged_constraint_terms['slab_center_loss']:.6f} | "
            f"SlabS: {logged_constraint_terms['slab_spread_loss']:.6f} | "
            f"LatLen: {logged_constraint_terms['lattice_len_loss']:.6f} | "
            f"LatAng: {logged_constraint_terms['lattice_angle_loss']:.6f} | "
            f"ElemDist: {logged_constraint_terms['elem_count_dist_loss']:.6f} | "
            f"Density: {logged_constraint_terms['density_loss']:.6f} | "
            f"CLenCeil: {logged_constraint_terms['c_length_ceiling_loss']:.6f} | "
            f"LR: {lr:.6f}"
        )


def _accumulate_batch_metrics(
        accumulated, batch_loss_dict, loss_dict, constraint_losses, constraint_terms
):
    """Accumulate batch metrics into running totals."""
    accumulated['loss'] += batch_loss_dict['loss']
    accumulated['base_loss'] += batch_loss_dict['base_loss']
    accumulated['constraint_loss'] += batch_loss_dict['constraint_loss']
    accumulated['lat_xy'] += loss_dict.get('lattice_loss_xy', torch.tensor(0.0)).item()
    accumulated['lat_z'] += loss_dict.get('lattice_loss_z', torch.tensor(0.0)).item()
    accumulated['atm_xy'] += loss_dict.get('atom_loss_xy', torch.tensor(0.0)).item()
    accumulated['atm_z'] += loss_dict.get('atom_loss_z', torch.tensor(0.0)).item()
    accumulated['batches'] += 1
    if constraint_losses is not None:
        for key in CONSTRAINT_KEYS:
            accumulated['constraint_terms'][key] += constraint_terms[key]


def _compute_logged_metrics(accumulated, accumulation_steps, is_tail=False):
    """
    Compute averaged logged metrics from accumulated values.

    Tail windows (< accumulation_steps batches) use accumulated_loss / divisor (no scaling)
    because they represent a partial accumulation window. Regular windows scale by
    accumulation_steps to normalize to per-batch units.
    """
    divisor = max(accumulated['batches'], 1)
    total_scale = 1.0 if is_tail else accumulation_steps
    return {
        'total': accumulated['loss'] * total_scale / divisor,
        'base': accumulated['base_loss'] / divisor,
        'lat_xy': accumulated['lat_xy'] / divisor,
        'lat_z': accumulated['lat_z'] / divisor,
        'atm_xy': accumulated['atm_xy'] / divisor,
        'atm_z': accumulated['atm_z'] / divisor,
        'constraint': accumulated['constraint_loss'] / divisor,
        'constraint_terms': {key: accumulated['constraint_terms'][key] / divisor for key in CONSTRAINT_KEYS},
    }


def _flush_accumulator(accumulated):
    """Reset accumulator to zero."""
    accumulated['loss'] = 0.0
    accumulated['base_loss'] = 0.0
    accumulated['constraint_loss'] = 0.0
    accumulated['lat_xy'] = 0.0
    accumulated['lat_z'] = 0.0
    accumulated['atm_xy'] = 0.0
    accumulated['atm_z'] = 0.0
    accumulated['constraint_terms'] = _empty_constraint_stats()
    accumulated['batches'] = 0


# ============================================================================
# Training and Evaluation Loops
# ============================================================================

def train_epoch(model, diffusion, dataloader, optimizer, scheduler, device, epoch, global_step, args, rank):
    model.train()
    epoch_total_loss = 0.0
    epoch_base_loss = 0.0
    epoch_constraint_loss = 0.0
    epoch_constraint_terms = _empty_constraint_stats()
    epoch_applied_steps = 0
    epoch_start_time = time.time()
    accumulation_steps = args.accumulation_steps
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    epoch_progress = float(epoch) / float(args.epochs)
    epoch_decay_f = get_decay_factor(epoch_progress, args)

    iterator = dataloader
    if rank == 0:
        iterator = tqdm(dataloader, desc=f"Epoch {epoch}")

    accumulated = {
        'loss': 0.0, 'base_loss': 0.0, 'constraint_loss': 0.0,
        'lat_xy': 0.0, 'lat_z': 0.0, 'atm_xy': 0.0, 'atm_z': 0.0,
        'constraint_terms': _empty_constraint_stats(), 'batches': 0,
    }

    for step, batch in enumerate(iterator):
        lattice_vectors = batch["lattice_vectors"].to(device)
        atom_features = batch["atom_features"].to(device)

        loss_dict = diffusion.training_losses(model, lattice_vectors, atom_features)
        base_loss = loss_dict["loss"]

        constraint_losses = None
        weighted_constraint_loss = torch.zeros((), device=device)

        if args.use_physical_constraints:
            constraint_losses = compute_physical_constraints(
                pred_lattice_x0=loss_dict['pred_lattice_x0'],
                pred_atom_x0=loss_dict['pred_atom_x0'],
                gt_lattice_vectors=lattice_vectors,
                gt_atom_features=atom_features,
                timesteps=loss_dict['timesteps'],
                diffusion_steps=args.diffusion_steps,
                slab_center=args.slab_center,
                min_angle_deg=args.min_angle_deg,
                max_angle_deg=args.max_angle_deg,
                min_angle_ab_deg=args.min_angle_ab_deg,
                max_angle_ab_deg=args.max_angle_ab_deg,
                rowcol_temperature=args.rowcol_temperature,
                rowcol_active_threshold=args.rowcol_active_threshold,
                lattice_quad_power=args.lattice_quad_power,
            )
            _, weighted_constraint_loss = compute_weighted_constraint_loss(constraint_losses, args, progress=epoch_progress)

        batch_total_loss = base_loss + weighted_constraint_loss

        local_bad = torch.tensor([0.0 if torch.isfinite(batch_total_loss).item() else 1.0], device=device)
        if world_size > 1:
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        if local_bad.item() > 0.0:
            if rank == 0:
                bad_terms = []
                if constraint_losses is not None:
                    for key, value in constraint_losses.items():
                        if not torch.isfinite(value):
                            bad_terms.append(key)
                base_str = f"{base_loss.item():.6f}" if torch.isfinite(base_loss) else "nan"
                cons_str = f"{weighted_constraint_loss.item():.6f}" if torch.isfinite(weighted_constraint_loss) else "nan"
                logging.warning(
                    f"Epoch {epoch} | Step {global_step} | Non-finite loss detected "
                    f"(base={base_str}, constraint={cons_str}, bad_terms={bad_terms}); "
                    f"skipping optimizer update for current window."
                )
            optimizer.zero_grad(set_to_none=True)
            _flush_accumulator(accumulated)
            continue

        loss = batch_total_loss / accumulation_steps
        loss.backward()

        constraint_terms = {key: (constraint_losses[key].item() if constraint_losses is not None else 0.0)
                            for key in CONSTRAINT_KEYS}
        _accumulate_batch_metrics(
            accumulated,
            {'loss': loss.item(), 'base_loss': base_loss.item(), 'constraint_loss': weighted_constraint_loss.item()},
            loss_dict, constraint_losses, constraint_terms
        )

        if (step + 1) % accumulation_steps == 0:
            logged_metrics = _compute_logged_metrics(accumulated, accumulation_steps)

            step_applied = _optimizer_step_with_grad_guard(
                model, optimizer, scheduler, args, rank, device, epoch, global_step
            )

            if step_applied:
                epoch_total_loss += accumulated['loss'] * accumulation_steps
                epoch_base_loss += accumulated['base_loss']
                epoch_constraint_loss += accumulated['constraint_loss']
                for key in CONSTRAINT_KEYS:
                    epoch_constraint_terms[key] += accumulated['constraint_terms'][key]
                epoch_applied_steps += accumulated['batches']

            _flush_accumulator(accumulated)

            if step_applied:
                global_step += 1

            if step_applied and rank == 0 and global_step % args.log_every == 0:
                _log_step(
                    rank, iterator, logged_metrics, logged_metrics['constraint_terms'],
                    args, global_step, epoch, optimizer.param_groups[0]['lr'], args.use_physical_constraints
                )

    if accumulated['batches'] > 0:
        logged_metrics = _compute_logged_metrics(accumulated, accumulation_steps, is_tail=True)

        step_applied = _optimizer_step_with_grad_guard(
            model, optimizer, scheduler, args, rank, device, epoch, global_step
        )

        if step_applied:
            epoch_total_loss += accumulated['loss'] * accumulation_steps
            epoch_base_loss += accumulated['base_loss']
            epoch_constraint_loss += accumulated['constraint_loss']
            for key in CONSTRAINT_KEYS:
                epoch_constraint_terms[key] += accumulated['constraint_terms'][key]
            epoch_applied_steps += accumulated['batches']
            global_step += 1

        if step_applied and rank == 0 and global_step % args.log_every == 0:
            _log_step(
                rank, iterator, logged_metrics, logged_metrics['constraint_terms'],
                args, global_step, epoch, optimizer.param_groups[0]['lr'], args.use_physical_constraints
            )

    n_steps = max(epoch_applied_steps, 1)
    avg_total_loss = reduce_mean(epoch_total_loss / n_steps, device, world_size)
    avg_base_loss = reduce_mean(epoch_base_loss / n_steps, device, world_size)
    avg_constraint_loss = reduce_mean(epoch_constraint_loss / n_steps, device, world_size)
    avg_constraint_terms = _average_constraint_stats(epoch_constraint_terms, n_steps, device, world_size)

    if rank == 0:
        epoch_time = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch} completed | Time: {epoch_time:.2f}s | "
            f"Average total loss: {avg_total_loss:.6f} | "
            f"Average base loss (for comparison with val): {avg_base_loss:.6f}"
        )
        if args.use_physical_constraints:
            weighted_terms = build_weighted_terms(avg_constraint_terms, args, decay_f=epoch_decay_f)
            sum_weighted = sum(weighted_terms.values())
            logging.info(
                f"Epoch {epoch} constraints | Total: {avg_constraint_loss:.6f} (sum={sum_weighted:.6f}, decay_f={epoch_decay_f:.2f}) | "
                + " | ".join([
                    f"Invalid: {weighted_terms['invalid_slot_loss']:.6f}",
                    f"Count: {weighted_terms['soft_atom_count_loss']:.6f}",
                    f"RowCol: {weighted_terms['rowcol_validity_loss']:.6f}",
                    f"SlabCenter: {weighted_terms['slab_center_loss']:.6f}",
                    f"SlabSpread: {weighted_terms['slab_spread_loss']:.6f}",
                    f"LatLen: {weighted_terms['lattice_len_loss']:.6f}",
                    f"LatAngle: {weighted_terms['lattice_angle_loss']:.6f}",
                    f"ElemDist: {weighted_terms['elem_count_dist_loss']:.6f}",
                    f"Density: {weighted_terms['density_loss']:.6f}",
                    f"CLenFlr: {weighted_terms['c_length_floor_loss']:.6f}",
                    f"CLenCeil: {weighted_terms['c_length_ceiling_loss']:.6f}",
                ])
            )

    return global_step, {
        'total_loss': avg_total_loss,
        'base_loss': avg_base_loss,
        'constraint_loss': avg_constraint_loss,
        'constraint_terms': avg_constraint_terms,
    }


def evaluate_epoch(model, diffusion, dataloader, device, epoch, rank, args):
    model.eval()
    total_base_loss = 0.0
    total_total_loss = 0.0
    total_constraint_loss = 0.0
    total_lat_xy = 0.0
    total_lat_z = 0.0
    total_atm_xy = 0.0
    total_atm_z = 0.0
    total_constraint_terms = _empty_constraint_stats()
    batch_count = 0
    iterator = dataloader

    if rank == 0:
        iterator = tqdm(dataloader, desc=f"Val {epoch}", leave=False)

    with torch.no_grad():
        for batch in iterator:
            lattice_vectors = batch["lattice_vectors"].to(device)
            atom_features = batch["atom_features"].to(device)
            loss_dict = diffusion.training_losses(model, lattice_vectors, atom_features)
            base_loss = loss_dict["loss"]
            weighted_constraint_loss = torch.zeros((), device=device)
            constraint_losses = None

            if args.use_physical_constraints:
                constraint_losses = compute_physical_constraints(
                    pred_lattice_x0=loss_dict['pred_lattice_x0'],
                    pred_atom_x0=loss_dict['pred_atom_x0'],
                    gt_lattice_vectors=lattice_vectors,
                    gt_atom_features=atom_features,
                    timesteps=loss_dict['timesteps'],
                    diffusion_steps=args.diffusion_steps,
                    slab_center=args.slab_center,
                    min_angle_deg=args.min_angle_deg,
                    max_angle_deg=args.max_angle_deg,
                    min_angle_ab_deg=args.min_angle_ab_deg,
                    max_angle_ab_deg=args.max_angle_ab_deg,
                    rowcol_temperature=args.rowcol_temperature,
                    rowcol_active_threshold=args.rowcol_active_threshold,
                    lattice_quad_power=args.lattice_quad_power,
                )
                _, weighted_constraint_loss = compute_weighted_constraint_loss(constraint_losses, args, progress=None)

            batch_total_loss = base_loss + weighted_constraint_loss
            if not torch.isfinite(batch_total_loss):
                if rank == 0:
                    logging.warning(f"Epoch {epoch} validation | Non-finite loss detected; skipping validation batch.")
                continue

            if constraint_losses is not None:
                for key in CONSTRAINT_KEYS:
                    total_constraint_terms[key] += constraint_losses[key].item()

            total_base_loss += base_loss.item()
            total_total_loss += batch_total_loss.item()
            total_constraint_loss += weighted_constraint_loss.item()
            total_lat_xy += loss_dict.get('lattice_loss_xy', torch.tensor(0.0)).item()
            total_lat_z += loss_dict.get('lattice_loss_z', torch.tensor(0.0)).item()
            total_atm_xy += loss_dict.get('atom_loss_xy', torch.tensor(0.0)).item()
            total_atm_z += loss_dict.get('atom_loss_z', torch.tensor(0.0)).item()
            batch_count += 1

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    loss_sums = torch.tensor([
        total_base_loss, total_total_loss, total_constraint_loss, float(batch_count),
        total_lat_xy, total_lat_z, total_atm_xy, total_atm_z
    ], device=device, dtype=torch.float32)
    constraint_sums = torch.tensor([total_constraint_terms[key] for key in CONSTRAINT_KEYS],
                                   device=device, dtype=torch.float32)

    if world_size > 1:
        dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(constraint_sums, op=dist.ReduceOp.SUM)

    global_batch_count = int(loss_sums[3].item())
    if global_batch_count > 0:
        avg_base_loss = (loss_sums[0] / global_batch_count).item()
        avg_total_loss = (loss_sums[1] / global_batch_count).item()
        avg_constraint_loss = (loss_sums[2] / global_batch_count).item()
        avg_lat_xy = (loss_sums[4] / global_batch_count).item()
        avg_lat_z = (loss_sums[5] / global_batch_count).item()
        avg_atm_xy = (loss_sums[6] / global_batch_count).item()
        avg_atm_z = (loss_sums[7] / global_batch_count).item()
        avg_constraint_terms = {key: (constraint_sums[i] / global_batch_count).item() for i, key in
                                enumerate(CONSTRAINT_KEYS)}
    else:
        avg_base_loss = float('inf')
        avg_total_loss = float('inf')
        avg_constraint_loss = float('inf')
        avg_lat_xy = float('inf')
        avg_lat_z = float('inf')
        avg_atm_xy = float('inf')
        avg_atm_z = float('inf')
        avg_constraint_terms = {key: 0.0 for key in CONSTRAINT_KEYS}

    if rank == 0:
        if args.use_physical_constraints:
            # Training optimizes L_total = L_base + L_physical where L_physical uses epoch-decayed weights.
            # Model selection is intentionally based on validation base diffusion loss alone,
            # keeping results comparable with the unconstrained baseline.
            # Validation constraint loss here uses undecayed weights (no progress argument passed)
            # for diagnostic reporting only; it does NOT participate in model selection.
            weighted_terms = build_weighted_terms(avg_constraint_terms, args)
            sum_weighted = sum(weighted_terms.values())
            logging.info(
                f"Epoch {epoch} validation completed | Average total loss: {avg_total_loss:.6f} | Average base loss: {avg_base_loss:.6f}")
            logging.info(
                f"Epoch {epoch} validation constraints | Total: {avg_constraint_loss:.6f} (sum={sum_weighted:.6f}) | "
                + " | ".join([
                    f"Invalid: {weighted_terms['invalid_slot_loss']:.6f}",
                    f"Count: {weighted_terms['soft_atom_count_loss']:.6f}",
                    f"RowCol: {weighted_terms['rowcol_validity_loss']:.6f}",
                    f"SlabCenter: {weighted_terms['slab_center_loss']:.6f}",
                    f"SlabSpread: {weighted_terms['slab_spread_loss']:.6f}",
                    f"LatLen: {weighted_terms['lattice_len_loss']:.6f}",
                    f"LatAngle: {weighted_terms['lattice_angle_loss']:.6f}",
                    f"ElemDist: {weighted_terms['elem_count_dist_loss']:.6f}",
                    f"Density: {weighted_terms['density_loss']:.6f}",
                    f"CLenFlr: {weighted_terms['c_length_floor_loss']:.6f}",
                    f"CLenCeil: {weighted_terms['c_length_ceiling_loss']:.6f}",
                ])
            )
        else:
            logging.info(
                f"Epoch {epoch} validation completed | Avg loss: {avg_base_loss:.6f} | "
                f"LatXY: {avg_lat_xy:.6f} | LatZ: {avg_lat_z:.6f} | "
                f"AtmXY: {avg_atm_xy:.6f} | AtmZ: {avg_atm_z:.6f}"
            )

    return {
        'total_loss': avg_total_loss,
        'base_loss': avg_base_loss,
        'constraint_loss': avg_constraint_loss,
        'constraint_terms': avg_constraint_terms,
    }


# ============================================================================
# Scheduler Utilities
# ============================================================================

def _check_decay_transition(epoch, args):
    """Check if training has crossed a decay boundary; returns (prev_decay_f, curr_decay_f)."""
    prev_progress = float(epoch - 1) / float(args.epochs) if epoch > 0 else 0.0
    curr_progress = float(epoch) / float(args.epochs)
    prev_decay_f = get_decay_factor(prev_progress, args)
    curr_decay_f = get_decay_factor(curr_progress, args)
    return prev_decay_f, curr_decay_f, prev_progress, curr_progress


def _get_lr_restart_epochs(args):
    """Compute LR warm-restart epoch indices derived from scheduler config constants."""
    restart_epochs = []
    cursor = SCHEDULER_WARMUP_EPOCHS
    period = SCHEDULER_T0_EPOCHS
    while cursor <= args.epochs:
        cursor += period
        if cursor <= args.epochs:
            restart_epochs.append(cursor)
        period *= SCHEDULER_T_MULT
    return restart_epochs


# ============================================================================
# Main Training Function
# ============================================================================

def train(rank, world_size, args):
    setup(rank, world_size)
    set_seed(args.seed, rank)
    device = torch.device(f"cuda:{rank}")

    setup_logging(rank, args)

    import json
    if rank == 0:
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info(f"Training parameters: {args}")
        logging.info(f"World size: {world_size}, Device: {device}")
        logging.info(f"Random seed: {args.seed}")

    if args.preprocessed_dir is None:
        preprocessed_dir = os.path.join(args.data_dir, "preprocessed")
    else:
        preprocessed_dir = args.preprocessed_dir
    val_csv = os.path.join(args.data_dir, "val.csv")
    use_validation = os.path.exists(val_csv)

    if rank == 0:
        os.makedirs(preprocessed_dir, exist_ok=True)
        train_preprocessed = os.path.join(preprocessed_dir, f"train_preprocessed_ma{args.max_atoms}.pkl")
        if args.force_preprocess or not os.path.exists(train_preprocessed):
            preprocess_dataset(os.path.join(args.data_dir, "train.csv"), train_preprocessed, args.max_atoms)
        if use_validation:
            val_preprocessed = os.path.join(preprocessed_dir, f"val_preprocessed_ma{args.max_atoms}.pkl")
            if args.force_preprocess or not os.path.exists(val_preprocessed):
                preprocess_dataset(val_csv, val_preprocessed, args.max_atoms)

    dist.barrier()

    train_preprocessed = os.path.join(preprocessed_dir, f"train_preprocessed_ma{args.max_atoms}.pkl")
    val_preprocessed = os.path.join(preprocessed_dir, f"val_preprocessed_ma{args.max_atoms}.pkl")

    train_dataset = CrystalDataset(os.path.join(args.data_dir, "train.csv"), max_atoms=args.max_atoms,
                                   preprocessed_file=train_preprocessed, force_preprocess=False)
    val_dataset = None
    if use_validation:
        val_dataset = CrystalDataset(val_csv, max_atoms=args.max_atoms, preprocessed_file=val_preprocessed,
                                     force_preprocess=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed + rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                              generator=g)
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=seed_worker, generator=g)

    model = CrystalDiT(max_atoms=args.max_atoms, hidden_size=args.hidden_size, depth=args.depth,
                       num_heads=args.num_heads, mlp_ratio=4.0)

    if rank == 0:
        logging.info(f"Total model parameters: {count_parameters(model):,}")

    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    diffusion = CrystalGaussianDiffusion(
        timesteps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        z_noise_scale=args.z_noise_scale,
        lattice_c_noise_scale=args.lattice_c_noise_scale,
        lattice_loss_z_weight=args.lattice_loss_z_weight
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)

    # Warmup (LinearLR) + CosineAnnealingWarmRestarts, concatenated via SequentialLR
    warmup_iters = SCHEDULER_WARMUP_EPOCHS * updates_per_epoch
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.03,
        end_factor=1.0,
        total_iters=warmup_iters
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=SCHEDULER_T0_EPOCHS * updates_per_epoch,
        T_mult=SCHEDULER_T_MULT,
        eta_min=args.min_learning_rate,
        last_epoch=-1
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters]
    )
    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, args, rank)

    best_metric = float('inf')
    no_improve_count = 0
    lr_restart_epochs = _get_lr_restart_epochs(args)

    for epoch in range(start_epoch, args.epochs):
        try:
            train_sampler.set_epoch(epoch)
            global_step, train_metrics = train_epoch(model, diffusion, train_loader, optimizer, scheduler, device, epoch,
                                                     global_step, args, rank)

            val_metrics = None
            if val_loader is not None:
                val_metrics = evaluate_epoch(model, diffusion, val_loader, device, epoch, rank, args)

            current_metric = val_metrics['base_loss'] if val_metrics is not None else train_metrics['base_loss']
            # Training optimizes L_total = L_base + L_physical (epoch-decayed constraint weights).
            # Model selection uses validation base diffusion loss only, for baseline comparability.
            is_best = current_metric < best_metric
            if is_best:
                best_metric = current_metric
                no_improve_count = 0
                if rank == 0:
                    logging.info(f"Found new best model (base_loss): {current_metric:.6f}")
            else:
                no_improve_count += 1

            # Reset early-stop counter on decay boundary transitions (constraint loss discontinuity)
            prev_decay_f, curr_decay_f, prev_prog, curr_prog = _check_decay_transition(epoch, args)
            if prev_decay_f != curr_decay_f:
                no_improve_count = 0
                if rank == 0:
                    logging.info(
                        f"Decay transition at epoch {epoch} (progress {prev_prog:.3f}->{curr_prog:.3f}), "
                        f"decay_f {prev_decay_f:.4f}->{curr_decay_f:.4f}, resetting no_improve_count to 0."
                    )

            # Reset early-stop counter on LR warm-restart boundaries (LR spike causes base_loss transient)
            if epoch in lr_restart_epochs:
                no_improve_count = 0
                if rank == 0:
                    logging.info(
                        f"LR warm-restart at epoch {epoch}, resetting no_improve_count to 0 "
                        f"(restart schedule: {lr_restart_epochs})."
                    )

            if epoch >= 200 and no_improve_count >= args.early_stopping_patience:
                if rank == 0:
                    logging.info(f"No improvement for {no_improve_count} epochs, stopping early at epoch {epoch}.")
                break

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, args, rank, is_best=False,
                                is_periodic=True)
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, args, rank, is_best=is_best,
                            is_periodic=False)

        except (KeyboardInterrupt, SystemExit):
            if rank == 0:
                logging.info(f"Training interrupted by user at epoch {epoch}, step {global_step}. Saving checkpoint before exit...")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, rank,
                                is_best=False, is_periodic=False)
            raise
        except Exception as e:
            # Non-recoverable training error (CUDA OOM, corrupted data batch, NCCL deadlock, etc.).
            # Log with full traceback, attempt emergency checkpoint, then propagate the exception.
            if rank == 0:
                logging.error(
                    f"[TRAINING ERROR] Epoch {epoch} | Step {global_step} | "
                    f"{type(e).__name__}: {e}", exc_info=True
                )
                logging.error("Attempting emergency checkpoint before aborting...")
                try:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, rank,
                                    is_best=False, is_periodic=False)
                    logging.info("Emergency checkpoint saved successfully.")
                except Exception as save_err:
                    logging.error(f"Emergency checkpoint save also failed: {type(save_err).__name__}: {save_err}")
            # Synchronize all ranks before re-raising to avoid DDP inconsistency
            try:
                cleanup()
            except Exception:
                pass  # Cleanup may fail if process group is already broken
            raise

    cleanup()
