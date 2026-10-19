"""
Screening Stage S3.5: ML Relaxation.

Features:
    - 2D-aware ML relaxation with fixed c-axis
    - FIRE optimizer with retry logic
    - LBFGS fallback for difficult structures
    - ASE/pymatgen format conversion

References:
    - C2DB Sec. 2.4: Structural optimization
"""

import logging
from typing import Tuple, Optional
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

# Optional ASE imports
try:
    from ase.atoms import Atoms
    from ase.optimize import FIRE, LBFGS
    from ase.filters import FrechetCellFilter
    from ase.io import write as ase_write
    from ase.calculators.calculator import Calculator
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Atoms = None
    FIRE = None
    LBFGS = None
    FrechetCellFilter = None


def _pmg_to_ase(struct: Structure) -> Atoms:
    """Convert pymatgen Structure to ASE Atoms."""
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor().get_atoms(struct)


def _ase_to_pmg(atoms: Atoms) -> Structure:
    """Convert ASE Atoms to pymatgen Structure."""
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor().get_structure(atoms)


def relax_2d(
    struct: Structure,
    backend,  # MLBackend
    cfg,
    steps_override: Optional[int] = None
) -> Tuple[bool, Structure, float, str]:
    """
    S3.5: 2D-aware ML relaxation.

    Uses FIRE optimizer with optional c-axis fixing.
    Returns convergence status based on optimizer result.

    Args:
        struct: pymatgen Structure to relax
        backend: MLBackend instance
        cfg: ScreeningConfig
        steps_override: Override max steps (for retry)

    Returns:
        (converged, relaxed_struct, energy, reason)
    """
    if not HAS_ASE:
        return False, struct, 0.0, "ase_not_installed"

    steps = steps_override if steps_override is not None else cfg.relax_steps

    try:
        atoms = _pmg_to_ase(struct)
        atoms.calc = backend.get_calculator()

        # 2D constraint: fix c-axis and angles, allow in-plane relaxation
        if cfg.fix_z_lattice:
            # [xx, yy, zz, yz, xz, xy] - fix xx, yy, yz, xz, xy, free zz
            cell_filter = FrechetCellFilter(atoms, mask=[True, True, False, False, False, True])
        else:
            cell_filter = FrechetCellFilter(atoms)

        # FIRE optimization
        opt = FIRE(cell_filter, logfile=None)
        converged = opt.run(fmax=cfg.relax_fmax, steps=steps)
        energy = float(atoms.get_potential_energy())

        relaxed = _ase_to_pmg(atoms)
        return bool(converged), relaxed, energy, "ok"

    except Exception as e:
        return False, struct, 0.0, f"relaxation_error:{e}"


def relax_2d_with_retry(
    struct: Structure,
    backend,  # MLBackend
    cfg
) -> Tuple[bool, Structure, float, str, str]:
    """
    S3.5: ML relaxation with retry logic.

    Strategy:
        1. FIRE + relax_steps
        2. If not converged → FIRE + relax_steps_retry (double steps)
        3. If still not converged → LBFGS + relax_steps_retry
        4. If still not converged → mark as uncertain

    Args:
        struct: pymatgen Structure
        backend: MLBackend
        cfg: ScreeningConfig

    Returns:
        (converged, relaxed_struct, energy, reason, convergence_status)
            - convergence_status: "converged" | "not_converged" | "uncertain"
    """
    # First attempt: FIRE + relax_steps
    converged, rel_struct, ml_energy, reason = relax_2d(
        struct, backend, cfg, steps_override=None
    )
    if converged:
        return converged, rel_struct, ml_energy, reason, "converged"

    # Second attempt: FIRE + relax_steps_retry
    logger.warning(f"FIRE 未收敛，重试（步数翻倍）: {struct.composition.reduced_formula}")
    converged, rel_struct, ml_energy, reason = relax_2d(
        struct, backend, cfg, steps_override=cfg.relax_steps_retry
    )
    if converged:
        return converged, rel_struct, ml_energy, f"{reason}; retry_steps", "converged"

    # Third attempt: LBFGS
    logger.warning(f"FIRE 仍未收敛，换用 LBFGS 优化器: {struct.composition.reduced_formula}")

    # Initialize defaults
    energy = ml_energy
    rel_struct = struct

    try:
        atoms = _pmg_to_ase(struct)
        atoms.calc = backend.get_calculator()

        if cfg.fix_z_lattice:
            cell_filter = FrechetCellFilter(atoms, mask=[True, True, False, False, False, True])
        else:
            cell_filter = FrechetCellFilter(atoms)

        opt = LBFGS(cell_filter, logfile=None)
        converged = opt.run(fmax=cfg.relax_fmax, steps=cfg.relax_steps_retry)
        energy = float(atoms.get_potential_energy())
        rel_struct = _ase_to_pmg(atoms)

        # LBFGS still not converged → uncertain
        if not converged:
            raise RuntimeError("LBFGS did not converge within step limit")

        return True, rel_struct, energy, "ok; optimizer_switched_to_LBFGS", "converged"

    except Exception as e:
        # Two retries failed → mark as uncertain
        logger.warning(f"LBFGS 仍未收敛，标记为 uncertain: {struct.composition.reduced_formula} - {e}")
        return False, rel_struct, energy, f"not_converged_after_retry:{e}", "uncertain"


__all__ = [
    'relax_2d',
    'relax_2d_with_retry',
    '_pmg_to_ase',
    '_ase_to_pmg',
]
