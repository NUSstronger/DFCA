"""
Top-k atom selector for crystal decoding.

Samples a target atom count N from the training-set atom-count empirical distribution,
then ranks all max_atoms slots by occupancy confidence and retains the top-N.
"""

import math
import os
from collections import defaultdict

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from scipy.stats import norm

# Authoritative imports (Stage 9 tested)
from dfca.data.representation import (
    ATOMIC_NUMBER_TO_POSITION,
    normalize_row,
    normalize_column,
)

RARE_GAS_ELEMENTS = {2, 10, 18, 36, 54, 86}


# Training-set atom-count prior


def _atom_count_from_features(atom_features_np):
    # Valid atoms in training data have x in [0, 1]; padding = -1.
    return int(np.sum(atom_features_np[:, 2] > -0.5))


def build_atom_count_prior(data_dir, max_atoms=20, max_length=46.7425,
                           split="train", smooth_eps=1e-3):
    """
    Returns (pmf, raw_counts):
        pmf: probability vector of shape [max_atoms+1], pmf[k] = P(N=k); pmf[0] = 0
        raw_counts: atom-count array of shape [N_samples] (for diagnostics)
    smooth_eps prevents any N from becoming unreachable if absent from the training set.
    """
    from dfca.data.dataset import CrystalDataset

    csv_path = os.path.join(data_dir, f"{split}.csv")
    ds = CrystalDataset(csv_file=csv_path, max_atoms=max_atoms, max_length=max_length)

    counts = np.array(
        [_atom_count_from_features(entry["atom_features"]) for entry in ds.data],
        dtype=np.int64,
    )

    pmf = np.zeros(max_atoms + 1, dtype=np.float64)
    for c in counts:
        if 1 <= c <= max_atoms:
            pmf[c] += 1.0

    pmf[1:] += smooth_eps  # Light smoothing: ensures every N seen in training is reachable.
    pmf /= pmf.sum()
    return pmf, counts


def sample_target_counts(pmf, n, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return rng.choice(np.arange(len(pmf)), size=n, p=pmf).astype(np.int64)


def summarize_pmf(pmf, raw_counts):
    """Diagnostic summary: mean, median, and percentiles of the atom-count distribution."""
    nz = np.where(pmf > 0)[0]
    return {
        "n_samples": int(raw_counts.size),
        "mean": float(np.mean(raw_counts)),
        "median": float(np.median(raw_counts)),
        "p10": float(np.percentile(raw_counts, 10)),
        "p90": float(np.percentile(raw_counts, 90)),
        "min": int(raw_counts.min()) if raw_counts.size else 0,
        "max": int(raw_counts.max()) if raw_counts.size else 0,
        "support": [int(nz.min()), int(nz.max())] if nz.size else [0, 0],
    }


# Per-slot element scoring (keeps full information for ranking)

_ROW_STEP = 1.0 / 7.0
_COLUMN_STEP = 1.0 / 18.0
_INVALID_NORM_ROW = normalize_row(0)
_INVALID_NORM_COLUMN = normalize_column(0)


def _build_element_arrays(filter_rare_gases=True, max_atomic_number=94,
                          allowed_atomic_numbers=None):
    """Return (z_arr, norm_row_arr, norm_col_arr) for vectorized element scoring."""
    z_list, nr_list, nc_list = [], [], []
    for z, (elem_row, elem_column) in ATOMIC_NUMBER_TO_POSITION.items():
        if z == 0:
            continue
        if filter_rare_gases and z in RARE_GAS_ELEMENTS:
            continue
        if z > max_atomic_number:
            continue
        if allowed_atomic_numbers is not None and z not in allowed_atomic_numbers:
            continue
        z_list.append(z)
        nr_list.append(normalize_row(elem_row))
        nc_list.append(normalize_column(elem_column))
    return (
        np.array(z_list, dtype=np.int64),
        np.array(nr_list, dtype=np.float64),
        np.array(nc_list, dtype=np.float64),
    )


def _decode_slots_vec(row_values, col_values, z_arr, nr_arr, nc_arr, sigma=0.05):
    """
    Vectorized decoding: score all slots in a batch simultaneously.
    Args:
        row_values, col_values: shape [N_slots]
        z_arr, nr_arr, nc_arr: shape [N_elements]
    Returns:
        best_z: shape [N_slots], int
        best_prob: shape [N_slots], float
        invalid_prob: shape [N_slots], float
    """
    rv = np.asarray(row_values, dtype=np.float64)
    cv = np.asarray(col_values, dtype=np.float64)
    n_slots = rv.size
    if n_slots == 0:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))

    inv_row_p = norm.cdf(_INVALID_NORM_ROW + _ROW_STEP, loc=rv, scale=sigma) - \
                norm.cdf(_INVALID_NORM_ROW - _ROW_STEP, loc=rv, scale=sigma)
    inv_col_p = norm.cdf(_INVALID_NORM_COLUMN + _COLUMN_STEP, loc=cv, scale=sigma) - \
                norm.cdf(_INVALID_NORM_COLUMN - _COLUMN_STEP, loc=cv, scale=sigma)
    invalid_prob = inv_row_p * inv_col_p  # [N_slots]

    # Broadcast: [N_elements, N_slots]
    rp = norm.cdf(nr_arr[:, None] + _ROW_STEP, loc=rv[None, :], scale=sigma) - \
         norm.cdf(nr_arr[:, None] - _ROW_STEP, loc=rv[None, :], scale=sigma)
    cp = norm.cdf(nc_arr[:, None] + _COLUMN_STEP, loc=cv[None, :], scale=sigma) - \
         norm.cdf(nc_arr[:, None] - _COLUMN_STEP, loc=cv[None, :], scale=sigma)
    elem_probs = rp * cp  # [N_elements, N_slots]

    best_idx = np.argmax(elem_probs, axis=0)  # [N_slots]
    best_prob = elem_probs[best_idx, np.arange(n_slots)]
    best_z = z_arr[best_idx]
    return best_z, best_prob, invalid_prob


# 2D lattice projection (mirrors generate_crystals.project_lattice_to_2d;
# duplicated here to avoid circular imports)

def _vector_angle_degrees(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 90.0
    cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _project_lattice_to_2d(lattice_matrix, min_c_length=18.0, max_c_length=35.0,
                           gamma_min_deg=60.0, gamma_max_deg=120.0):
    a_vec = lattice_matrix[0].astype(np.float64).copy()
    b_vec = lattice_matrix[1].astype(np.float64).copy()
    c_vec = lattice_matrix[2].astype(np.float64).copy()

    a_xy = a_vec[:2]
    b_xy = b_vec[:2]
    if np.linalg.norm(a_xy) < 1e-6:
        a_xy = np.array([max(np.linalg.norm(a_vec), 1.0), 0.0], dtype=np.float64)
    if np.linalg.norm(b_xy) < 1e-6:
        b_xy = np.array([0.0, max(np.linalg.norm(b_vec), 1.0)], dtype=np.float64)

    a_len = max(np.linalg.norm(a_xy), 1.0)
    b_len = max(np.linalg.norm(b_xy), 1.0)
    gamma_deg = _vector_angle_degrees(a_xy, b_xy)
    gamma_deg = float(np.clip(gamma_deg, gamma_min_deg, gamma_max_deg))
    gamma_rad = math.radians(gamma_deg)

    raw_c_len = max(np.linalg.norm(c_vec), abs(c_vec[2]))
    if min_c_length is not None:
        c_len = max(raw_c_len, min_c_length)
    else:
        c_len = max(raw_c_len, 1e-3)
    # Cap c_len to avoid abnormally large vacuum layers.
    if max_c_length is not None:
        c_len = min(c_len, max_c_length)

    return np.array(
        [
            [a_len, 0.0, 0.0],
            [b_len * math.cos(gamma_rad), b_len * math.sin(gamma_rad), 0.0],
            [0.0, 0.0, c_len],
        ],
        dtype=np.float64,
    )


def _periodic_distance(value, center=0.5):
    diff = abs((value % 1.0) - center)
    return min(diff, 1.0 - diff)


# Top-k decoding: retain top-N slots per sample and assemble into Structure

def convert_to_structure_topk(
    lattice_vectors,
    atom_features,
    target_counts,
    max_length=46.7425,
    filter_rare_gases=True,
    allowed_atomic_numbers=None,
    element_sigma=0.01,
    slab_center=0.5,
    slab_half_width_ang=3.5,
    coord_tolerance=0.05,
    project_2d_lattice=True,
    min_c_length=18.0,
    max_c_length=35.0,
    gamma_min_deg=60.0,
    gamma_max_deg=120.0,
    max_unique_elements=4,
    forbid_atomic_numbers=None,
    coord_oor_penalty=100.0,
    slab_penalty=10.0,
    min_valid_element_prob=None,
):
    """
    Per sample, rank all max_atoms slots by occupancy confidence and retain the top target_counts[i].

    score_per_slot = (best_prob - invalid_prob)
                     - coord_oor_penalty * I(coord out of [-tol, 1+tol])
                     - slab_penalty * relu(|z - slab_center| - slab_half_width_ang)

    Args:
        min_valid_element_prob: If not None, minimum (best_prob - invalid_prob) threshold.
            Slots below this confidence are excluded from top-k selection. Default: None (disabled).
    """
    forbidden_arr = np.array(list(forbid_atomic_numbers or []), dtype=np.int64)
    z_arr, nr_arr, nc_arr = _build_element_arrays(
        filter_rare_gases=filter_rare_gases,
        max_atomic_number=94,
        allowed_atomic_numbers=allowed_atomic_numbers,
    )

    batch_size = lattice_vectors.shape[0]
    target_counts = np.asarray(target_counts).astype(np.int64)
    assert target_counts.shape[0] == batch_size, \
        f"target_counts size {target_counts.shape[0]} != batch_size {batch_size}"

    # Pre-project all lattices and extract c_len (needed for slab half-width unit conversion).
    lattice_np = lattice_vectors.detach().cpu().numpy() if hasattr(lattice_vectors, "detach") \
        else lattice_vectors.cpu().numpy()
    c_lens = np.array([
        _project_lattice_to_2d(lattice_np[i], min_c_length=min_c_length,
                               max_c_length=max_c_length,
                               gamma_min_deg=gamma_min_deg, gamma_max_deg=gamma_max_deg)[2, 2]
        for i in range(batch_size)
    ])

    # Batch pre-decoding: element scores for all slots across all samples in one pass (main cost).
    atom_features_np = atom_features.detach().cpu().numpy() if hasattr(atom_features, "detach") \
        else atom_features.cpu().numpy()
    n_slots_per_sample = atom_features_np.shape[1]
    flat_rows = atom_features_np[:, :, 0].reshape(-1)
    flat_cols = atom_features_np[:, :, 1].reshape(-1)
    flat_coords = atom_features_np[:, :, 2:5].reshape(-1, 3).astype(np.float64)

    flat_best_z, flat_best_prob, flat_invalid_prob = _decode_slots_vec(
        flat_rows, flat_cols, z_arr, nr_arr, nc_arr, sigma=element_sigma
    )

    # Mask forbidden elements: mark best_z that are in the forbidden set as 0.
    if forbidden_arr.size > 0:
        forbidden_mask = np.isin(flat_best_z, forbidden_arr)
        flat_best_z = np.where(forbidden_mask, 0, flat_best_z)

    # Geometric penalties: out-of-range coordinates + slab boundary violation.
    # slab_half_width_ang is in Ångström; must convert to fractional coords.
    oor_mask = (flat_coords < -coord_tolerance).any(axis=1) | \
               (flat_coords > 1.0 + coord_tolerance).any(axis=1)
    z_mod = flat_coords[:, 2] % 1.0
    slab_dist = np.minimum(np.abs(z_mod - slab_center),
                           1.0 - np.abs(z_mod - slab_center))
    # Each sample converts Ångström half-width to fractional using its own c_len,
    # ensuring the criterion is identical to SLAB_SPREAD_PHYSICAL=3.5 during training.
    flat_c_lens = np.repeat(c_lens, n_slots_per_sample)
    flat_slab_half_width = slab_half_width_ang / np.maximum(flat_c_lens, 1e-6)
    slab_excess = np.maximum(0.0, slab_dist - flat_slab_half_width)

    # Pure element decoding confidence (best_prob - invalid_prob), separate from geometric penalties.
    # Used only for min_valid_element_prob filtering, NOT for ranking.
    flat_confidence = flat_best_prob - flat_invalid_prob

    flat_scores = flat_confidence \
                  - coord_oor_penalty * oor_mask.astype(np.float64) \
                  - slab_penalty * slab_excess
    flat_scores = np.where(flat_best_z > 0, flat_scores, -np.inf)

    flat_clipped_coords = np.clip(flat_coords, 0.0, 1.0)

    # Reshape back to [batch, n_slots, ...].
    sample_scores = flat_scores.reshape(batch_size, n_slots_per_sample)
    sample_confidence = flat_confidence.reshape(batch_size, n_slots_per_sample)
    sample_z = flat_best_z.reshape(batch_size, n_slots_per_sample)
    sample_coords = flat_clipped_coords.reshape(batch_size, n_slots_per_sample, 3)

    structures = []

    for i in range(batch_size):
        target_n = int(max(1, target_counts[i]))

        lattice_matrix = lattice_vectors[i].cpu().numpy() * max_length
        if project_2d_lattice:
            lattice_matrix = _project_lattice_to_2d(
                lattice_matrix,
                min_c_length=min_c_length,
                max_c_length=max_c_length,
                gamma_min_deg=gamma_min_deg,
                gamma_max_deg=gamma_max_deg,
            )

        scores = sample_scores[i]
        confidence = sample_confidence[i]
        decoded_z = sample_z[i]
        decoded_coords = sample_coords[i]

        # Retain top-N (only from slots decoded as non-zero elements).
        valid_mask = decoded_z > 0
        # Apply minimum element confidence filter if enabled.
        # Uses pure element decoding confidence (best_prob - invalid_prob), NOT the geometric-penalized scores.
        if min_valid_element_prob is not None:
            valid_mask = valid_mask & (confidence >= min_valid_element_prob)
        if not np.any(valid_mask):
            print(f"Warning: Sample {i} has zero decodable slots, skipping")
            continue

        order = np.argsort(-scores)
        selected = []
        for idx in order:
            if not valid_mask[idx]:
                continue
            selected.append(int(idx))
            if len(selected) >= target_n:
                break

        if not selected:
            print(f"Warning: Sample {i} produced no atoms after top-k, skipping")
            continue

        atomic_numbers = [int(decoded_z[k]) for k in selected]
        frac_coords = [decoded_coords[k] for k in selected]
        sel_scores = [scores[k] for k in selected]

        # Limit element diversity: keep top-K element types by cumulative score.
        if max_unique_elements is not None and max_unique_elements > 0:
            element_score = defaultdict(float)
            for z, sc in zip(atomic_numbers, sel_scores):
                element_score[z] += sc
            kept = {
                z for z, _ in sorted(
                    element_score.items(), key=lambda x: x[1], reverse=True
                )[:max_unique_elements]
            }
            kept_pairs = [(z, c) for z, c in zip(atomic_numbers, frac_coords) if z in kept]
            if not kept_pairs:
                continue
            atomic_numbers = [p[0] for p in kept_pairs]
            frac_coords = [p[1] for p in kept_pairs]

        try:
            elements = [Element.from_Z(z) for z in atomic_numbers]
            structure = Structure(
                lattice=lattice_matrix,
                species=elements,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
            structures.append(structure)
        except Exception as e:
            print(f"Error creating structure {i}: {e}")

    return structures
