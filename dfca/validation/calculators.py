"""
GPAW Calculator Construction.

Functions for building GPAW calculators with adaptive parameters.
Follows C2DB standards for 2D materials DFT.

References:
    - C2DB GPAW settings
    - GPAW documentation
"""

import logging
from typing import Optional, Dict, List

import numpy as np

from dfca.validation.utils.kpoints import adaptive_kpoints

logger = logging.getLogger(__name__)

# GPAW defaults (C2DB standards)
GPAW_DEFAULTS = {
    'xc': 'PBE',
    'convergence': {
        'energy': 5e-4,  # eV, C2DB standard
        'density': 1e-4,
        'bands': 'occupied',
    },
    'maxiter': 500,
}


def build_gpaw_calculator(
    atoms,
    config,
    mag_type: str = 'nonmag',
    is_magnetic: bool = False,
    for_supercell: bool = False,
    adaptive_params: Optional[Dict[str, float]] = None,
):
    """
    Build a GPAW calculator with adaptive electronic parameters.

    Args:
        atoms: ASE Atoms object
        config: GPAWConfig instance
        mag_type: Magnetic classification
        is_magnetic: Whether structure is magnetic
        for_supercell: Use supercell k-point density
        adaptive_params: Optional adaptive electronic parameters:
            - kpoint_multiplier: k-point density multiplier
            - fermi_smearing: Fermi smearing width (eV)

    Returns:
        Configured GPAW calculator
    """
    # Lazy import GPAW to allow testing without it
    try:
        from gpaw import GPAW, PW, FermiDirac
    except ImportError:
        # Return None for testing
        return None

    # k-point mesh
    if for_supercell:
        dk = config.dk_per_ang
        if adaptive_params and 'kpoint_multiplier' in adaptive_params:
            dk = dk / adaptive_params['kpoint_multiplier']
        kpts, koffset = _adaptive_kpoints_supercell(atoms, dk=dk)
    else:
        dk = config.dk_per_ang
        if adaptive_params and 'kpoint_multiplier' in adaptive_params:
            dk = dk / adaptive_params['kpoint_multiplier']
        kpts, koffset = adaptive_kpoints(atoms, dk=dk)

    # Spin polarization
    spinpol = is_magnetic if config.spinpol is None else config.spinpol

    # Fermi smearing
    fermi_smearing = config.fermi_smearing
    if adaptive_params and 'fermi_smearing' in adaptive_params:
        fermi_smearing = adaptive_params['fermi_smearing']

    # Debug logging
    k_mult = adaptive_params.get('kpoint_multiplier', 1.0) if adaptive_params else 1.0
    cell = atoms.get_cell()
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    delta_k = 2 * np.pi / (kpts[0] * a_len) if kpts[0] > 0 and a_len > 0 else 0
    logger.debug(
        f"[kpts] a={a_len:.3f}Å b={b_len:.3f}Å | "
        f"dk_input={config.dk_per_ang:.4f}Å⁻¹ k_mult={k_mult:.2f} | "
        f"dk_eff={dk:.4f}Å⁻¹ Δk≈{delta_k:.4f}Å⁻¹ | "
        f"kpts={kpts} mode={'supercell' if for_supercell else 'unit'}"
    )

    # GPAW requires zero magnetic moments for non-spin-polarized calculations
    if not spinpol:
        atoms.set_initial_magnetic_moments([0.0] * len(atoms))
    elif is_magnetic:
        from dfca.validation.utils.magnetic import build_initial_magmoms
        magmoms = build_initial_magmoms(atoms)
        atoms.set_initial_magnetic_moments(magmoms)

    # Build calculator
    calc_params = {
        'mode': PW(config.encut),
        'xc': config.xc,
        'occupations': FermiDirac(fermi_smearing),
        'kpts': kpts,
        'spinpol': spinpol,
        'convergence': GPAW_DEFAULTS['convergence'],
        'maxiter': config.maxiter,
    }

    return GPAW(**calc_params)


def _adaptive_kpoints_supercell(atoms, dk: float = 0.03) -> tuple:
    """
    Calculate k-points for a supercell.

    Uses the supercell lattice constants directly.

    Args:
        atoms: ASE Atoms object
        dk: Maximum k-point spacing (Å⁻¹)

    Returns:
        Tuple of (kpts, koffset)
    """
    return adaptive_kpoints(atoms, dk=dk, max_nk=6)


def adaptive_electronic_params(
    gap_eV: float,
    dos_at_ef: float,
    config,
) -> Dict[str, float]:
    """
    Calculate adaptive electronic parameters based on band gap.

    Uses continuous interpolation based on gap and DOS at Fermi level.
    This replaces the old discrete metal/semiconductor classification.

    Args:
        gap_eV: DFT-probed band gap (eV)
        dos_at_ef: DOS at Fermi level (states/eV/atom)
        config: GPAWConfig instance

    Returns:
        Dictionary of adaptive parameters:
            - kpoint_multiplier: k-point density multiplier
            - fermi_smearing: Fermi smearing width (eV)
            - supercell_min_length: Minimum supercell length (Å)
    """
    # Interpolate between semiconductor and metal limits
    gap_scale = config.gap_interp_scale
    dos_scale = config.dos_interp_scale

    # Semiconductor limit (large gap)
    semi_k_mult = 1.0
    semi_smearing = config.fermi_smearing_semiconductor
    semi_sc_min = config.supercell_min_length_base

    # Metal limit (zero gap, large DOS)
    metal_k_mult = 1.0 + config.k_max_boost
    metal_smearing = config.fermi_smearing_metal
    metal_sc_min = config.supercell_min_length_base + config.supercell_min_length_boost

    # Dirac/semimetal handling (zero gap, small DOS)
    # These need extra k-point density like metals
    dirac_k_mult = metal_k_mult * 1.2
    dirac_smearing = metal_smearing
    dirac_sc_min = metal_sc_min

    # Continuous interpolation
    # Gap-based interpolation (how "semiconductor-like")
    gap_factor = 1.0 - np.exp(-max(gap_eV, 0) / gap_scale)

    # DOS-based interpolation (how "metal-like")
    # Large DOS at Ef indicates metallic character
    max_dos = 10.0  # Normalization factor
    dos_factor = np.clip(dos_at_ef / max_dos, 0, 1)
    dos_decay = np.exp(-dos_factor * dos_scale)

    # Dirac/semimetal: gap ≈ 0 but DOS small
    is_dirac = (gap_eV < 0.1 and dos_at_ef < 0.5)
    is_metal = (dos_at_ef >= 0.5)

    if is_dirac:
        # Dirac semimetals need extra k-density
        k_mult = dirac_k_mult
        smearing = dirac_smearing
        sc_min = dirac_sc_min
    elif is_metal:
        # True metals
        k_mult = metal_k_mult
        smearing = metal_smearing
        sc_min = metal_sc_min
    else:
        # Continuous interpolation for semiconductors/near-insulators
        # Weight by semiconductor character
        k_mult = metal_k_mult - (metal_k_mult - semi_k_mult) * gap_factor
        smearing = metal_smearing - (metal_smearing - semi_smearing) * gap_factor
        sc_min = metal_sc_min - (metal_sc_min - semi_sc_min) * gap_factor

    return {
        'kpoint_multiplier': float(k_mult),
        'fermi_smearing': float(smearing),
        'supercell_min_length': float(sc_min),
    }


__all__ = [
    'GPAW_DEFAULTS',
    'build_gpaw_calculator',
    'adaptive_electronic_params',
]
