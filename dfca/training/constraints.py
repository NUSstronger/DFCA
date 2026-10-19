"""
Physical Constraint Loss Functions for CrystalDiT Training.

This module contains all physical constraint loss computations that enforce structural
validity on the generated 2D slab crystal structures:
    - Atom count / element distribution alignment
    - Row/column validity (valid periodic table positions)
    - Slab z-center and spread (2D slab geometry)
    - Lattice length and angle constraints
    - Mass density conservation
    - Vacuum layer bounds on c-axis

IMPORTANT: This code is migrated directly from the original train.py.
No changes have been made to the mathematical formulas, loss weights, or computation logic.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Import from authoritative representation module (Stage 9 tested)
from dfca.data.representation import (
    VALID_ELEMENT_ROWCOL,
    decode_avg_atomic_mass,
    MAX_LATTICE_LENGTH,
    normalize_column,
    normalize_row,
)

# ============================================================================
# Physical Constraint Constants
# ============================================================================

MIN_VACUUM_THICKNESS = 15.0   # Minimum vacuum layer thickness (Å) to avoid periodic interactions

# Invalid row/column position for padding
INVALID_ROWCOL = torch.tensor([normalize_row(0), normalize_column(0)], dtype=torch.float32)
INVALID_ATOM_FEATURE = -1.0

# Decay stage boundaries (global constants for all decay calculations and early-stop resets)
DECAY_BOUNDARY_MID = 0.30   # progress <= 0.30 -> decay_f = 1.0
DECAY_BOUNDARY_LATE = 0.70  # progress <= 0.70 -> decay_f = decay_mid

# Constraint keys for loss aggregation and logging
CONSTRAINT_KEYS = (
    'invalid_slot_loss',
    'soft_atom_count_loss',
    'rowcol_validity_loss',
    'slab_center_loss',
    'slab_spread_loss',
    'lattice_len_loss',
    'lattice_angle_loss',
    'elem_count_dist_loss',
    'density_loss',
    'c_length_floor_loss',
    'c_length_ceiling_loss',
)


# ============================================================================
# Physical Constraint Loss Computation
# ============================================================================

def compute_physical_constraints(
        pred_lattice_x0,
        pred_atom_x0,
        gt_lattice_vectors,
        gt_atom_features,
        timesteps,
        diffusion_steps=1000,
        slab_center=0.5,
        min_angle_deg=60.0,
        max_angle_deg=125.0,
        min_angle_ab_deg=80.0,
        max_angle_ab_deg=100.0,
        rowcol_temperature=0.0604,
        rowcol_active_threshold=0.403809,
        lattice_quad_power=4,
):
    """
    Physical constraints applied to model-predicted x0 (gradient flows to model).

    Uses quadratic time-weighting: high-timestep predictions are less reliable,
    so constraint contribution decays toward zero. Clamps extreme predicted values
    before constraint computation to prevent gradient explosion.
    """
    # Quadratic decay: high-t predictions are less reliable
    t_frac = (1.0 - timesteps.float() / diffusion_steps).clamp(min=0.0)
    t_weight = (t_frac.pow(lattice_quad_power)).clamp(min=0.03)  # [B]

    # Clip predictions to prevent gradient explosion from extreme denoised values
    atom_clip = 4.0
    lattice_clip = 8.0
    pred_atom_safe = pred_atom_x0.clamp(-atom_clip, atom_clip)
    pred_lattice_safe = pred_lattice_x0.clamp(-lattice_clip, lattice_clip)
    gt_lattice_safe = gt_lattice_vectors.clamp(-lattice_clip, lattice_clip)

    slab_center_tensor = torch.tensor(slab_center, device=pred_atom_safe.device, dtype=torch.float32)

    valid_mask = (gt_atom_features[:, :, 0] != -1).float()  # [B, N]
    invalid_mask = 1.0 - valid_mask
    num_valid = valid_mask.sum(dim=1).clamp(min=1.0)  # [B]
    num_invalid = invalid_mask.sum(dim=1).clamp(min=1.0)  # [B]

    # ---- Soft atom count: relative error normalized to atom count ----
    pred_rowcol = pred_atom_safe[:, :, :2]
    soft_valid_prob, nearest_valid_dist_sq = _compute_soft_valid_prob(pred_rowcol, rowcol_temperature)
    pred_atom_count = soft_valid_prob.sum(dim=1)  # [B]
    gt_atom_count = valid_mask.sum(dim=1)         # [B]
    atom_count_error = ((pred_atom_count - gt_atom_count).abs() / gt_atom_count.clamp(min=1.0)).clamp(0.0, 2.0)
    soft_atom_count_loss = (F.smooth_l1_loss(
        atom_count_error, torch.zeros_like(atom_count_error), beta=0.1, reduction='none'
    ) * t_weight).mean()

    # ---- Row/column validity: L1 distance to nearest valid element ----
    rowcol_active_mask = (t_frac > rowcol_active_threshold).float()
    nearest_valid_dist_l1 = torch.sqrt(nearest_valid_dist_sq + 1e-8)
    per_sample_loss = ((nearest_valid_dist_l1 * valid_mask).sum(dim=1) / num_valid) * t_weight * rowcol_active_mask
    active_count = rowcol_active_mask.sum().clamp(min=1.0)
    rowcol_validity_loss = per_sample_loss.sum() / active_count

    # ---- Invalid slot: Huber loss pushing invalid positions to -1.0 ----
    invalid_target = torch.full_like(pred_atom_safe, INVALID_ATOM_FEATURE)
    invalid_slot_error = F.smooth_l1_loss(
        pred_atom_safe, invalid_target, beta=1.0, reduction='none'
    ).mean(dim=-1)  # [B, N]
    invalid_slot_loss = (((invalid_slot_error * invalid_mask).sum(dim=1) / num_invalid) * t_weight).mean()

    # ---- Slab center: z-direction center alignment ----
    z_pred = pred_atom_safe[:, :, 4]  # [B, N]
    z_weighted_sum = (z_pred * soft_valid_prob).sum(dim=1)
    z_valid_weight_sum = soft_valid_prob.sum(dim=1).clamp(min=1e-6)
    z_mean = z_weighted_sum / z_valid_weight_sum
    slab_center_loss = (F.smooth_l1_loss(
        z_mean, slab_center_tensor.expand_as(z_mean), beta=0.05, reduction='none'
    ) * t_weight).mean()

    # ---- Lattice length: relative error normalized per axis ----
    # a, b axes weighted higher (structurally meaningful); c axis lower (vacuum parameter)
    LATTICE_AXIS_WEIGHTS = [1.0, 1.0, 0.5]  # [a, b, c]
    axis_weight = torch.tensor(LATTICE_AXIS_WEIGHTS, device=pred_lattice_safe.device, dtype=pred_lattice_safe.dtype)
    pred_lens = _safe_vector_norm(pred_lattice_safe * MAX_LATTICE_LENGTH, dim=-1)  # [B, 3] in Å
    gt_lens = _safe_vector_norm(gt_lattice_safe * MAX_LATTICE_LENGTH, dim=-1)  # [B, 3] in Å
    lattice_len_error = ((pred_lens - gt_lens).abs() / gt_lens.clamp(min=1e-6)).clamp(0.0, 2.0)
    lattice_len_loss = ((F.smooth_l1_loss(
        lattice_len_error, torch.zeros_like(lattice_len_error), beta=0.1, reduction='none'
    ) * axis_weight).mean(dim=1) * t_weight).mean()

    # ---- Slab spread: z-direction physical thickness constraint ----
    # Physical RMS spread: fractional RMS * c_length in Å
    z_dist_sq = (z_pred - slab_center_tensor).pow(2)
    z_rms_fraction = ((z_dist_sq * soft_valid_prob).sum(dim=1) / z_valid_weight_sum).sqrt()  # [B]
    c_length = pred_lens[:, 2]  # [B] c-axis length in Å
    z_rms_physical = z_rms_fraction * c_length.detach()  # [B] RMS spread in Å
    SLAB_SPREAD_PHYSICAL = 3.0  # Å, fixed physical threshold
    above_spread = torch.relu(z_rms_physical - SLAB_SPREAD_PHYSICAL)  # [B]
    slab_spread_loss = (F.smooth_l1_loss(
        above_spread, torch.zeros_like(above_spread),
        beta=0.1, reduction='none'
    ) * t_weight).mean()

    # ---- Lattice angle constraints ----
    angle_loss = _compute_angle_loss(
        pred_lattice_safe, min_angle_deg, max_angle_deg, t_weight,
        min_angle_ab_deg=min_angle_ab_deg, max_angle_ab_deg=max_angle_ab_deg
    )

    # ---- Element count distribution: scalar regression ----
    pred_elem_count = _compute_unique_elements(pred_atom_safe, is_gt=False, temperature=rowcol_temperature)
    elem_count_dist_loss = _compute_element_count_distribution_loss(
        pred_elem_count, gt_atom_features, t_weight
    )

    # ---- Mass density alignment (g/cm³) ----
    AMU_TO_G_PER_CM3 = 1.66054

    gt_volume = _compute_lattice_volume(gt_lattice_safe)
    gt_volume_ang3 = gt_volume * (MAX_LATTICE_LENGTH ** 3)  # [B]
    gt_avg_mass = decode_avg_atomic_mass(gt_atom_features, temperature=rowcol_temperature)  # [B]
    gt_density = gt_atom_count * gt_avg_mass / gt_volume_ang3 * AMU_TO_G_PER_CM3  # [B]

    pred_lattice_ang = pred_lattice_safe * MAX_LATTICE_LENGTH  # [B, 3, 3] in Å
    pred_volume = _compute_lattice_volume(pred_lattice_ang)  # [B]
    pred_avg_mass = decode_avg_atomic_mass(pred_atom_safe, temperature=rowcol_temperature)  # [B]
    pred_density = pred_atom_count * pred_avg_mass / pred_volume.clamp(min=1e-6) * AMU_TO_G_PER_CM3  # [B]

    density_rel_error = ((pred_density - gt_density) / gt_density.clamp(min=1e-6)).clamp(-2.0, 2.0)
    density_loss = (F.smooth_l1_loss(
        density_rel_error, torch.zeros_like(density_rel_error), beta=0.1, reduction='none'
    ) * t_weight).mean()

    # ---- c-axis floor: minimum vacuum thickness constraint ----
    # Required c_length scales with predicted slab thickness (z_rms_physical.detach() avoids circular coupling)
    required_c = MIN_VACUUM_THICKNESS + 2 * z_rms_physical  # [B]
    c_too_short = torch.relu(required_c - c_length)  # [B]
    c_length_floor_loss = (F.smooth_l1_loss(
        c_too_short, torch.zeros_like(c_too_short),
        beta=0.1, reduction='none'
    ) * t_weight).mean()

    # ---- c-axis ceiling: maximum vacuum thickness constraint ----
    VACUUM_CEILING_MARGIN = 14.0  # Å
    c_too_long = torch.relu(c_length - (required_c + VACUUM_CEILING_MARGIN))  # [B]
    c_length_ceiling_loss = (F.smooth_l1_loss(
        c_too_long, torch.zeros_like(c_too_long),
        beta=0.1, reduction='none'
    ) * t_weight).mean()

    return {
        'invalid_slot_loss': invalid_slot_loss,
        'soft_atom_count_loss': soft_atom_count_loss,
        'rowcol_validity_loss': rowcol_validity_loss,
        'slab_center_loss': slab_center_loss,
        'slab_spread_loss': slab_spread_loss,
        'lattice_len_loss': lattice_len_loss,
        'lattice_angle_loss': angle_loss,
        'elem_count_dist_loss': elem_count_dist_loss,
        'density_loss': density_loss,
        'c_length_floor_loss': c_length_floor_loss,
        'c_length_ceiling_loss': c_length_ceiling_loss,
    }


def _compute_soft_valid_prob(pred_rowcol, rowcol_temperature):
    """Compute soft probability that each position is a valid atom (vs. invalid padding)."""
    valid_element_rowcol = VALID_ELEMENT_ROWCOL.to(device=pred_rowcol.device, dtype=pred_rowcol.dtype)
    invalid_rowcol = INVALID_ROWCOL.to(device=pred_rowcol.device, dtype=pred_rowcol.dtype)

    valid_dist_sq = (pred_rowcol.unsqueeze(2) - valid_element_rowcol.view(1, 1, -1, 2)).pow(2).sum(dim=-1)
    nearest_valid_dist_sq = valid_dist_sq.min(dim=-1).values

    invalid_dist_sq = (pred_rowcol - invalid_rowcol.view(1, 1, 2)).pow(2).sum(dim=-1)

    rowcol_temperature = max(float(rowcol_temperature), 0.05)
    valid_logit = -nearest_valid_dist_sq / rowcol_temperature
    invalid_logit = -invalid_dist_sq / rowcol_temperature

    max_logit = torch.maximum(valid_logit, invalid_logit).detach()
    valid_score = torch.exp(valid_logit - max_logit)
    invalid_score = torch.exp(invalid_logit - max_logit)
    soft_valid_prob = valid_score / (valid_score + invalid_score + 1e-8)

    return soft_valid_prob, nearest_valid_dist_sq


def _compute_element_count_distribution_loss(pred_elem_counts, gt_atom_features, t_weight):
    """
    ElemDist does NOT participate in epoch decay (kept constant throughout training).
    """
    gt_elem_counts = _compute_unique_elements(gt_atom_features, is_gt=True).detach()  # [B]
    rel_error = ((pred_elem_counts - gt_elem_counts).abs() / gt_elem_counts.clamp(min=1.0)).clamp(0.0, 2.0)
    mse_loss = rel_error.pow(2).clamp(0.0, 4.0)
    return (mse_loss * t_weight).mean()


def _compute_unique_elements(atom_features, is_gt=False, temperature=0.03):
    """Return count of unique elements per sample (scalar per batch element)."""
    valid_mask = (atom_features[:, :, 0] != -1).float()  # [B, N]
    rowcol = atom_features[:, :, :2]  # [B, N, 2]

    if is_gt:
        # Hard round + bitcount for exact discrete counting
        row_idx = (atom_features[:, :, 0].round().clamp(-0.5, 27.5) + 0.5).long()
        col_idx = (atom_features[:, :, 1].round().clamp(-0.5, 27.5) + 0.5).long()
        elem_idx = row_idx * 28 + col_idx
        B, N = atom_features.shape[:2]
        elem_idx_masked = elem_idx.masked_fill((valid_mask < 0.5).bool(), 0)
        bitpacked = torch.zeros(B, 784, device=atom_features.device, dtype=torch.int64)
        bitpacked.scatter_add_(1, elem_idx_masked, valid_mask.long())
        unique_counts = (bitpacked > 0).long().sum(dim=1).float()
    else:
        # Pred: max-based counting avoids sum-based overestimation
        elem_temperature = max(float(temperature), 0.03)
        valid_element_rowcol = VALID_ELEMENT_ROWCOL.to(device=atom_features.device, dtype=atom_features.dtype)
        n_elems = valid_element_rowcol.shape[0]
        dist_sq = (rowcol.unsqueeze(2) - valid_element_rowcol.view(1, 1, n_elems, 2)).pow(2).sum(dim=-1)
        neg_dist = -dist_sq / max(elem_temperature, 1e-4)
        atom_elem_probs = torch.softmax(neg_dist, dim=-1)  # [B, N, n_elems]
        atom_elem_probs = atom_elem_probs * valid_mask.unsqueeze(-1)
        atom_elem_max = atom_elem_probs.max(dim=1).values  # [B, n_elems]
        elem_present_prob = torch.sigmoid((atom_elem_max - 0.5) / 0.2)
        unique_counts = elem_present_prob.sum(dim=-1)

    return unique_counts


def _safe_vector_norm(x, dim=-1, eps=1e-12):
    """Stable vector norm that avoids NaN for zero vectors."""
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim), min=eps))


def _compute_lattice_volume(lattice_vectors):
    """Compute lattice volume as |a · (b × c)|."""
    a = lattice_vectors[:, 0, :]
    b = lattice_vectors[:, 1, :]
    c = lattice_vectors[:, 2, :]
    b_cross_c = torch.cross(b, c, dim=-1)
    volume = (a * b_cross_c).sum(dim=-1).abs()
    return volume.clamp(min=1e-6)


def _compute_angle_loss(lattice_vectors, min_angle_deg, max_angle_deg, t_weight,
                         min_angle_ab_deg=80.0, max_angle_ab_deg=100.0):
    """
    Angle constraints: γ (a,b plane) and α/β (a-c, b-c) separately.
    For 2D slab representation, c must be perpendicular to ab plane (α=β≈90°).
    γ is kept loose as it depends on lattice type.
    """
    a = lattice_vectors[:, 0, :]
    b = lattice_vectors[:, 1, :]
    c = lattice_vectors[:, 2, :]

    def _angle_deg(v1, v2):
        dot = (v1 * v2).sum(dim=-1)
        n1 = _safe_vector_norm(v1, dim=-1)
        n2 = _safe_vector_norm(v2, dim=-1)
        cos = (dot / (n1 * n2)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(cos) * (180.0 / np.pi)

    alpha = _angle_deg(b, c)  # [B]
    beta = _angle_deg(a, c)   # [B]
    gamma = _angle_deg(a, b)  # [B]

    def _angle_violation(angle, lo, hi):
        below_v = torch.relu(lo - angle) / float(lo)
        above_v = torch.relu(angle - hi) / (180.0 - float(hi))
        below_v = below_v.clamp(0.0, 1.0)
        above_v = above_v.clamp(0.0, 1.0)
        return F.smooth_l1_loss(below_v, torch.zeros_like(below_v), beta=0.1, reduction='none') + \
               F.smooth_l1_loss(above_v, torch.zeros_like(above_v), beta=0.1, reduction='none')

    gamma_loss = (_angle_violation(gamma, min_angle_deg, max_angle_deg) * t_weight).mean()
    ab_violation = _angle_violation(torch.stack([alpha, beta], dim=-1), min_angle_ab_deg, max_angle_ab_deg).mean(dim=-1)
    ab_angle_loss = (ab_violation * t_weight).mean()

    return gamma_loss + ab_angle_loss


# ============================================================================
# Constraint Loss Aggregation
# ============================================================================

def compute_weighted_constraint_loss(constraint_losses, args, progress=None):
    """
    Aggregate constraint losses with per-weight scaling and epoch-based decay.

    ElemDist is kept at fixed weight (does not decay with epoch) to ensure
    gradient signal remains sufficient in late training stages.
    """
    raw_loss_decayed = (
            args.invalid_slot_weight * constraint_losses['invalid_slot_loss'] +
            args.soft_atom_count_weight * constraint_losses['soft_atom_count_loss'] +
            args.rowcol_validity_weight * constraint_losses['rowcol_validity_loss'] +
            args.slab_center_weight * constraint_losses['slab_center_loss'] +
            args.slab_spread_weight * constraint_losses['slab_spread_loss'] +
            args.lattice_len_weight * constraint_losses['lattice_len_loss'] +
            args.lattice_angle_weight * constraint_losses['lattice_angle_loss'] +
            args.density_weight * constraint_losses['density_loss'] +
            args.c_length_floor_weight * constraint_losses['c_length_floor_loss'] +
            args.c_length_ceiling_weight * constraint_losses['c_length_ceiling_loss']
    )
    raw_loss_fixed = args.elem_count_dist_weight * constraint_losses['elem_count_dist_loss']

    if progress is None:
        decay_f = 1.0
    elif progress <= DECAY_BOUNDARY_MID:
        decay_f = 1.0
    elif progress <= DECAY_BOUNDARY_LATE:
        decay_f = args.decay_mid
    else:
        decay_f = args.decay_late

    total_loss = decay_f * raw_loss_decayed + raw_loss_fixed
    return decay_f, total_loss


def get_decay_factor(progress, args):
    """Get epoch decay factor based on training progress."""
    if progress <= DECAY_BOUNDARY_MID:
        return 1.0
    elif progress <= DECAY_BOUNDARY_LATE:
        return args.decay_mid
    else:
        return args.decay_late


def build_weighted_terms(avg_constraint_terms, args, decay_f=None):
    """Build weighted constraint terms dict from raw averages and weights."""
    weighted_terms = {
        'invalid_slot_loss': args.invalid_slot_weight * avg_constraint_terms['invalid_slot_loss'],
        'soft_atom_count_loss': args.soft_atom_count_weight * avg_constraint_terms['soft_atom_count_loss'],
        'rowcol_validity_loss': args.rowcol_validity_weight * avg_constraint_terms['rowcol_validity_loss'],
        'slab_center_loss': args.slab_center_weight * avg_constraint_terms['slab_center_loss'],
        'slab_spread_loss': args.slab_spread_weight * avg_constraint_terms['slab_spread_loss'],
        'lattice_len_loss': args.lattice_len_weight * avg_constraint_terms['lattice_len_loss'],
        'lattice_angle_loss': args.lattice_angle_weight * avg_constraint_terms['lattice_angle_loss'],
        'elem_count_dist_loss': args.elem_count_dist_weight * avg_constraint_terms['elem_count_dist_loss'],
        'density_loss': args.density_weight * avg_constraint_terms['density_loss'],
        'c_length_floor_loss': args.c_length_floor_weight * avg_constraint_terms['c_length_floor_loss'],
        'c_length_ceiling_loss': args.c_length_ceiling_weight * avg_constraint_terms['c_length_ceiling_loss'],
    }
    if decay_f is not None:
        # All terms except ElemDist participate in decay
        decay_keys = [k for k in weighted_terms if k != 'elem_count_dist_loss']
        for k in decay_keys:
            weighted_terms[k] *= decay_f
    return weighted_terms


def _empty_constraint_stats():
    return {key: 0.0 for key in CONSTRAINT_KEYS}


def _average_constraint_stats(constraint_sums, divisor, device, world_size):
    import torch.distributed as dist
    return {
        key: _reduce_mean(constraint_sums[key] / max(divisor, 1), device, world_size)
        for key in CONSTRAINT_KEYS
    }


def _reduce_mean(value, device, world_size):
    """Average a scalar across all distributed workers."""
    import torch.distributed as dist
    value_tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        value_tensor /= world_size
    return value_tensor.item()
