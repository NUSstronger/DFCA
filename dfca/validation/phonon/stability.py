"""
Phonon Stability Analysis.

Functions for evaluating phonon stability with branch-specific criteria
and ZA rotational rescue support.

References:
    - C2DB phonon stability criteria
    - Huang rotational sum rules (HiPhive)
"""

from typing import Dict, Any, List, Optional

import numpy as np

from dfca.validation.phonon.branches import (
    identify_phonon_branches,
    _get_2d_layer_normal,
    _phonon_mode_out_of_plane_fraction,
    _physical_q_norms_2d,
)


def detect_za_rotational_rescue_candidate(
    atoms,
    qpoints: np.ndarray,
    freqs_THz: np.ndarray,
    eigenvectors: np.ndarray,
    imag_threshold: float,
    gamma_tol: float = 0.05,
    out_of_plane_threshold: float = 0.80,
    gamma_region_Ainv: float = 0.35,
) -> Dict[str, Any]:
    """
    Conservatively detect isolated long-wavelength ZA-like instability.

    This function does NOT modify force constants and does NOT change the
    stability result. It only determines whether a failed phonon calculation
    is eligible for a later Huang rotational-sum-rule rescue attempt.

    Args:
        atoms: ASE Atoms object
        qpoints: q-point array
        freqs_THz: Frequency array in THz
        eigenvectors: Eigenvector array
        imag_threshold: Threshold for imaginary frequency (THz)
        gamma_tol: Tolerance for acoustic frequencies at Γ (THz)
        out_of_plane_threshold: Minimum out-of-plane fraction for ZA mode
        gamma_region_Ainv: Maximum q-norm for Γ-region (Å⁻¹)

    Returns:
        Dictionary with:
            - candidate: Whether eligible for rescue
            - reason: Explanation
            - gamma_ok: Whether Γ-point acoustic modes are acceptable
            - acoustic_indices: Indices of acoustic branches
            - imaginary_branches: Branches with imaginary frequencies
            - etc.
    """
    qpoints = np.asarray(qpoints, dtype=float)
    freqs = np.asarray(freqs_THz, dtype=float)
    eig = np.asarray(eigenvectors)

    qnorm = _physical_q_norms_2d(atoms, qpoints)
    igamma = int(np.argmin(qnorm))

    gamma_freqs = freqs[igamma]

    # Three modes closest to zero at Γ are the acoustic seed set
    acoustic = np.argsort(np.abs(gamma_freqs))[:3]
    acoustic_set = set(int(x) for x in acoustic)

    optical = [
        b for b in range(freqs.shape[1])
        if b not in acoustic_set
    ]

    gamma_ok = bool(
        np.max(np.abs(gamma_freqs[acoustic])) <= gamma_tol
    )

    imag = np.argwhere(freqs < imag_threshold)

    if len(imag) == 0:
        return {
            "candidate": False,
            "reason": "no_significant_imaginary_modes",
            "gamma_ok": gamma_ok,
        }

    # Determine whether every significant imaginary mode is strongly
    # out-of-plane polarized and belongs to the Γ acoustic seed set
    normal = _get_2d_layer_normal(atoms)

    pouts = []
    imag_branches = set()

    for iq, ib in imag:
        iq = int(iq)
        ib = int(ib)

        imag_branches.add(ib)

        pouts.append(
            _phonon_mode_out_of_plane_fraction(
                eig[iq, :, ib],
                normal,
            )
        )

    min_pout = float(min(pouts))
    mean_pout = float(np.mean(pouts))

    # Conservative: significant imaginary points must occupy only one branch
    single_imag_branch = len(imag_branches) == 1

    imag_branch = (
        int(next(iter(imag_branches)))
        if single_imag_branch
        else -1
    )

    acoustic_only = (
        single_imag_branch
        and imag_branch in acoustic_set
    )

    za_like = (
        min_pout >= out_of_plane_threshold
    )

    optical_min = (
        float(np.min(freqs[:, optical]))
        if optical
        else float("inf")
    )

    optical_ok = optical_min >= imag_threshold

    other_acoustic = [
        int(b) for b in acoustic
        if int(b) != imag_branch
    ]

    other_acoustic_min = (
        float(np.min(freqs[:, other_acoustic]))
        if other_acoustic
        else float("inf")
    )

    other_acoustic_ok = (
        other_acoustic_min >= imag_threshold
    )

    imag_qnorm = np.asarray([
        qnorm[int(iq)]
        for iq, _ in imag
    ])

    near_gamma = bool(
        np.min(imag_qnorm) <= gamma_region_Ainv
    )

    candidate = bool(all([
        gamma_ok,
        single_imag_branch,
        acoustic_only,
        za_like,
        optical_ok,
        other_acoustic_ok,
        near_gamma,
    ]))

    return {
        "candidate": candidate,
        "reason": (
            "isolated_ZA_like_rotational_candidate"
            if candidate
            else "not_safe_for_automatic_Huang_rescue"
        ),
        "gamma_ok": gamma_ok,
        "acoustic_indices": [int(x) for x in acoustic],
        "imaginary_branches": sorted(int(x) for x in imag_branches),
        "imaginary_count": int(len(imag)),
        "imaginary_branch": imag_branch,
        "min_out_of_plane_fraction": min_pout,
        "mean_out_of_plane_fraction": mean_pout,
        "optical_min_THz": optical_min,
        "other_acoustic_min_THz": other_acoustic_min,
        "near_gamma": near_gamma,
    }


def _try_huang_za_rescue(
    atoms,
    phonon,
    forces: List[np.ndarray],
    qpoints: np.ndarray,
    original_freqs_THz: np.ndarray,
    imag_threshold: float,
    cutoff_A: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Attempt a conservative HiPhive Huang rotational-sum-rule rescue.

    IMPORTANT:
    - Uses already-computed displacement forces. No GPAW calculation occurs.
    - Does not modify the input phonon object.
    - First performs a NO-ROT HiPhive round-trip fidelity check.
    - Huang-corrected FC is returned only if all validation gates pass.

    Args:
        atoms: ASE Atoms object
        phonon: Phonopy object
        forces: List of force arrays from displacements
        qpoints: q-point array
        original_freqs_THz: Original frequency array
        imag_threshold: Threshold for imaginary frequency
        cutoff_A: Cutoff radius for force constants (Å)

    Returns:
        Dictionary with:
            - accepted: Whether rescue was accepted
            - reason: Explanation
            - corrected_fc: Corrected force constants (if accepted)
    """
    result = {
        "accepted": False,
        "reason": "not_attempted",
        "corrected_fc": None,
    }

    try:
        from hiphive import (
            ClusterSpace,
            StructureContainer,
            ForceConstantPotential,
            enforce_rotational_sum_rules,
        )
        from hiphive.utilities import prepare_structure

        try:
            from hiphive.fitting import Optimizer
        except Exception:
            from trainstation import Optimizer

    except Exception as exc:
        result["reason"] = f"hiphive_unavailable: {exc}"
        return result

    # HiPhive implementation would go here
    # This is a placeholder that preserves the function signature
    result["reason"] = "hiphive_implementation_placeholder"
    return result


def _evaluate_phonon_stability_spectrum(
    freqs_THz: np.ndarray,
    qpoints: np.ndarray,
    n_atoms: int,
    phonon_noise_floor_THz: float,
    config,
    noise_estimation_failed: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate phonon spectrum using branch-resolved stability criteria.

    This helper is side-effect free. It does not modify force constants,
    Phonopy objects, or configuration.

    Args:
        freqs_THz: Frequency array in THz
        qpoints: q-point array
        n_atoms: Number of atoms in unit cell
        phonon_noise_floor_THz: Estimated noise floor
        config: GPAWConfig instance
        noise_estimation_failed: Whether noise estimation failed

    Returns:
        Dictionary with stability analysis results
    """
    freqs_THz = np.asarray(freqs_THz, dtype=float)
    qpoints = np.asarray(qpoints, dtype=float)

    branch_raw = identify_phonon_branches(
        freqs_THz,
        qpoints,
        n_atoms,
    )

    za_idx = branch_raw['za_branch_idx']
    ta_idx = branch_raw['ta_branch_idx']
    la_idx = branch_raw['la_branch_idx']

    acoustic_indices = branch_raw['acoustic_indices']
    optical_indices = branch_raw['optical_branch_indices']

    if len(freqs_THz) == 0:
        return {
            'is_stable': False,
            'reason': 'empty_phonon_spectrum',
            'gamma_min': 0.0,
            'min_freq': 0.0,
            'min_freq_with_za_gamma': 0.0,
            'imaginary_count': 0,
            'branch_info_raw': branch_raw,
        }

    gamma_freqs = freqs_THz[0]
    gamma_min = float(gamma_freqs.min())

    # ZA-Γ exclusion (exclude ZA at Γ from min frequency)
    n_qpoints = len(freqs_THz)
    gamma_region_npoints = max(2, int(n_qpoints * 0.05))

    za_gamma_region_mask = np.zeros_like(freqs_THz, dtype=bool)

    if za_idx >= 0:
        za_gamma_region_mask[:gamma_region_npoints, za_idx] = True

    min_freq_with_za_gamma = float(freqs_THz.min())

    freqs_for_boundary = freqs_THz.copy()

    if za_idx >= 0:
        freqs_for_boundary[za_gamma_region_mask] = 999.0

    min_freq = float(freqs_for_boundary.min())

    # Dynamic tolerances
    za_noise_floor = max(
        phonon_noise_floor_THz * config.phonon_za_sigma_factor,
        1e-4,
    )

    za_tol_dynamic = -config.phonon_noise_sigma * za_noise_floor

    optical_tol_dynamic = -config.phonon_noise_sigma * max(
        phonon_noise_floor_THz * 0.5,
        1e-4,
    )

    boundary_tol_dynamic = optical_tol_dynamic

    # Branch quantities
    za_gamma_freq = float(gamma_freqs[za_idx]) if za_idx >= 0 else 0.0
    ta_gamma_freq = float(gamma_freqs[ta_idx]) if ta_idx >= 0 else 0.0
    la_gamma_freq = float(gamma_freqs[la_idx]) if la_idx >= 0 else 0.0

    optical_gamma_min = (
        float(np.min(gamma_freqs[optical_indices]))
        if optical_indices
        else 0.0
    )

    za_min_freq = float(np.min(freqs_THz[:, za_idx])) if za_idx >= 0 else 0.0
    ta_min_freq = float(np.min(freqs_THz[:, ta_idx])) if ta_idx >= 0 else 0.0
    la_min_freq = float(np.min(freqs_THz[:, la_idx])) if la_idx >= 0 else 0.0

    optical_min_freq = (
        float(np.min(freqs_THz[:, optical_indices]))
        if optical_indices
        else 0.0
    )

    # Stability evaluation
    za_ok = za_gamma_freq > za_tol_dynamic if za_idx >= 0 else True
    ta_ok = ta_gamma_freq > optical_tol_dynamic if ta_idx >= 0 else True
    la_ok = la_gamma_freq > optical_tol_dynamic if la_idx >= 0 else True
    optical_ok = (
        optical_gamma_min > optical_tol_dynamic
        if optical_indices
        else True
    )
    boundary_ok = min_freq >= boundary_tol_dynamic

    is_stable = bool(
        za_ok and ta_ok and la_ok and optical_ok and boundary_ok
    )

    if is_stable:
        reason = "ok"
        if noise_estimation_failed:
            reason += "; noise_estimation_failed_using_static_fallback"
    else:
        parts = []
        if not za_ok:
            parts.append(
                f"ZA_Gamma={za_gamma_freq:.4f}THz<={za_tol_dynamic:.4f}THz"
            )
        if not ta_ok:
            parts.append(
                f"TA_Gamma={ta_gamma_freq:.4f}THz<{optical_tol_dynamic:.4f}THz"
            )
        if not la_ok:
            parts.append(
                f"LA_Gamma={la_gamma_freq:.4f}THz<{optical_tol_dynamic:.4f}THz"
            )
        if not optical_ok:
            parts.append(
                f"Opt_Gamma={optical_gamma_min:.4f}THz<{optical_tol_dynamic:.4f}THz"
            )
        if not boundary_ok:
            parts.append(
                f"BZ_min={min_freq:.4f}THz<{boundary_tol_dynamic:.4f}THz"
            )
        reason = "phonon_unstable(" + "|".join(parts) + ")"
        if noise_estimation_failed:
            reason += "; noise_estimation_failed_using_static_fallback"

    imaginary_count = int((freqs_THz < boundary_tol_dynamic).sum())

    return {
        'is_stable': is_stable,
        'reason': reason,
        'gamma_min': gamma_min,
        'min_freq': min_freq,
        'min_freq_with_za_gamma': min_freq_with_za_gamma,
        'imaginary_count': imaginary_count,
        'za_idx': int(za_idx),
        'ta_idx': int(ta_idx),
        'la_idx': int(la_idx),
        'acoustic_indices': [int(x) for x in acoustic_indices],
        'optical_indices': [int(x) for x in optical_indices],
        'za_gamma_freq': za_gamma_freq,
        'ta_gamma_freq': ta_gamma_freq,
        'la_gamma_freq': la_gamma_freq,
        'optical_gamma_min': optical_gamma_min,
        'za_min_freq': za_min_freq,
        'ta_min_freq': ta_min_freq,
        'la_min_freq': la_min_freq,
        'optical_min_freq': optical_min_freq,
        'za_tol_dynamic': za_tol_dynamic,
        'optical_tol_dynamic': optical_tol_dynamic,
        'boundary_tol_dynamic': boundary_tol_dynamic,
        'branch_info_raw': branch_raw,
    }


__all__ = [
    'detect_za_rotational_rescue_candidate',
    '_try_huang_za_rescue',
    '_evaluate_phonon_stability_spectrum',
]
