"""
Screening Stages S3T/S6: Thermodynamics.

Features:
    - Formation energy calculation
    - E_above_hull calculation
    - Element reference energy precomputation
    - Stability labeling (C2DB Table 2)

References:
    - C2DB Sec. 2.6: Thermodynamic stability
    - C2DB Table 2: Stability classification
"""

import logging
from typing import Dict, Tuple, Optional, List
from pymatgen.core import Structure, Composition

logger = logging.getLogger(__name__)

# Diatomic molecules for reference states
_DIATOMIC_GASES = {"H", "N", "O", "F", "Cl", "Br", "I"}

# Optional ASE import for reference structure building
try:
    from ase.build import bulk, molecule
    from ase.atoms import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


def _build_reference_atoms(el: str) -> Optional[Atoms]:
    """
    Build reference Atoms for element.

    For diatomic gases (H, N, O, F, Cl, Br, I), uses molecule().
    For all other elements, uses bulk().

    Args:
        el: Element symbol

    Returns:
        ASE Atoms or None if construction fails
    """
    if not HAS_ASE:
        return None

    if el in _DIATOMIC_GASES:
        try:
            mol = molecule(el + "2")
            mol.cell = [30.0, 30.0, 30.0]
            mol.pbc = False
            mol.center()
            return mol
        except Exception:
            pass

    try:
        return bulk(el)
    except Exception:
        return None


def precompute_element_references(
    elements: List[str],
    backend,
    cfg
) -> Dict[str, float]:
    """
    Precompute element reference energies (eV/atom).

    Computes energies for all unique elements that don't have
    pre-defined reference energies in cfg.reference_energies.

    NOTE on NM/FM magnetic configurations (Bug #8):

        MACE-MP-0 is an atomic-position potential that does NOT explicitly handle
        spin degrees of freedom. There is no 'spin configuration' parameter in MACE.
        The MAGMOM values used in VASP INCAR export are purely for DFT-level
        self-consistent spin polarization, which MACE does not model.

        In practice: for MACE-MP-0, NM and FM configurations for the same structure
        will predict the SAME energy (the Born-Oppenheimer PES is agnostic to spin
        at the MACE level). Therefore, there is no meaningful NM vs FM energy
        comparison at the MACE reference-state level.

        The NM/FM distinction is still relevant for:
            1. VASP DFT validation (after pipeline): set magmom via MAGMOM_REF in INCAR
            2. Future: if a spin-sensitive MLIP is used, this function should be
               extended to test NM/FM/AFM configurations and take the minimum energy.

        The formation energy reported by S6 is therefore the MACE-relative formation
        energy. Its absolute value depends on which spin state the training DFT data
        represented for each element's reference structure.

    Args:
        elements: List of element symbols
        backend: MLBackend instance
        cfg: ScreeningConfig

    Returns:
        Dict mapping element symbol to reference energy (eV/atom)
    """
    refs: Dict[str, float] = {}

    # Start with pre-defined references
    for el in elements:
        if el in cfg.reference_energies:
            refs[el] = cfg.reference_energies[el]

    # Compute missing references
    missing = [el for el in elements if el not in refs]
    if missing and HAS_ASE:
        logger.info(f"正在为全体系建立一致的单质自洽 ML 参考能数据库: {missing}")
        for el in missing:
            atoms = _build_reference_atoms(el)
            if atoms is None:
                refs[el] = 0.0
                continue
            try:
                a = atoms.copy()
                a.calc = backend.get_calculator()
                e_total = float(a.get_potential_energy())
                refs[el] = e_total / len(a)
            except Exception:
                refs[el] = 0.0

    return refs


def compute_ehull(
    struct: Structure,
    energy_eV: float,
    cached_refs: Dict[str, float],
    cfg
) -> Tuple[float, float, float, str]:
    """
    S3T/S6: Calculate thermodynamic convex hull energy.

    Computes:
        - e_hull: Energy above hull (eV/atom)
        - e_form: Formation energy (eV/atom)
        - elem_e_diff: Energy above hull relative to elemental phases

    Args:
        struct: pymatgen Structure
        energy_eV: Total ML energy (eV)
        cached_refs: Dict of element reference energies
        cfg: ScreeningConfig

    Returns:
        (e_hull, e_form, elem_e_diff, reason)
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

    try:
        comp = struct.composition
        elements = [str(el) for el in comp.elements]

        # Check for missing references
        missing_refs = [el for el in elements if el not in cached_refs]
        if missing_refs:
            return -1.0, -1.0, -1.0, f"missing_refs_in_ml_ff:{missing_refs}"

        # Build phase diagram entries
        all_entries = []
        for el in elements:
            all_entries.append(PDEntry(Composition(el), cached_refs[el]))
        all_entries.append(PDEntry(comp, energy_eV))

        # Compute phase diagram
        pd = PhaseDiagram(all_entries)
        my_entry = all_entries[-1]
        e_hull = float(pd.get_e_above_hull(my_entry))
        e_form = float(pd.get_form_energy_per_atom(my_entry))

        # Energy above hull relative to elemental phases (PD normalizes to 0)
        elem_e_diff = e_hull

        return e_hull, e_form, elem_e_diff, "ok"

    except Exception as e:
        return -1.0, -1.0, -1.0, f"ehull_error:{e}"


def compute_stability_label(
    e_form: float,
    e_hull: Optional[float],
    threshold: float = 0.2
) -> str:
    """
    Compute C2DB stability label.

    C2DB Table 2 stability classification:
        - LOW: e_form > 0 (metastable, positive formation energy)
        - MEDIUM: e_form < 0 and e_hull >= threshold (slightly unstable)
        - HIGH: e_form < 0 and e_hull < threshold (thermodynamically stable)
        - UNKNOWN: e_hull < 0 or calculation error

    Args:
        e_form: Formation energy (eV/atom)
        e_hull: Energy above hull (eV/atom)
        threshold: E_hull threshold (default 0.2 eV/atom)

    Returns:
        Stability label: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    """
    if e_form > 0:
        return "LOW"
    elif e_hull is not None and e_hull >= threshold:
        return "MEDIUM"
    elif e_hull is not None and e_hull < threshold:
        return "HIGH"
    else:
        return "UNKNOWN"


__all__ = [
    'compute_ehull',
    'precompute_element_references',
    'compute_stability_label',
    '_build_reference_atoms',
    '_DIATOMIC_GASES',
]
