"""
Screening Stage S2: Symmetry Check.

Checks:
    - Space group analysis
    - Layer group analysis
    - Crystallographic standard settings

References:
    - C2DB Sec. 2.3: Symmetry and space group conventions
"""

from typing import Tuple, Dict
from pymatgen.core import Structure

# Optional spglib import
try:
    import spglib
    HAS_SPGLIB = True
except ImportError:
    HAS_SPGLIB = False


def check_symmetry_2d(struct: Structure, cfg) -> Tuple[bool, Dict]:
    """
    S2: Crystal symmetry check.

    Analyzes space group and layer group symmetry.
    Uses spglib for symmetry detection.

    Args:
        struct: pymatgen Structure
        cfg: ScreeningConfig

    Returns:
        (passed, info_dict)
    """
    if not HAS_SPGLIB:
        return True, {
            "reason": "spglib_not_available",
            "space_group": "?",
            "layer_group": "?",
        }

    try:
        # Get spglib cell
        cell = (
            struct.lattice.matrix.tolist(),
            struct.frac_coords.tolist(),
            [site.specie.Z for site in struct],
        )

        # Space group analysis
        spacegroup = spglib.get_spacegroup(cell, symprec=cfg.symprec)
        if spacegroup is None:
            spacegroup = "?"

        # Layer group analysis (for 2D materials)
        # Get symmetry dataset to extract hall number
        try:
            dataset = spglib.get_symmetry_dataset(cell, symprec=cfg.symprec)
            hall_number = dataset.hall_number if dataset else None
            layer_group = _hall_to_layer_group(hall_number) if hall_number else "?"
        except Exception:
            # Fallback: try old API or use get_spacegroup result
            layer_group = spacegroup if spacegroup and spacegroup != "?" else "?"

        return True, {
            "reason": "ok",
            "space_group": spacegroup,
            "layer_group": layer_group,
        }

    except Exception as e:
        return True, {
            "reason": f"symmetry_error:{e}",
            "space_group": "?",
            "layer_group": "?",
        }


def _hall_to_layer_group(hall_number: int) -> str:
    """
    Convert Hall number to layer group symbol.

    This is a simplified mapping for common 2D layer groups.
    Full implementation would require layer group database.

    Common 2D layer groups (International notation):
        - p1, p2, pm, pg, cm, pmm, pmg, pgg, cmm
        - p4, p4m, p4g, p3, p3m1, p31m, p6, p6m

    Args:
        hall_number: Hall number from spglib

    Returns:
        Layer group symbol or "?"
    """
    # Simplified mapping - common 2D layer groups
    # Hall numbers are complex; this is a best-effort mapping
    LAYER_GROUPS = {
        1: "p1",
        2: "p-1",
        3: "p2",
        4: "p2_1",
        5: "c2",
        9: "pm",
        10: "pb",
        11: "p2_1/m",
        12: "p2_1/b",
        13: "p2_1/a",
        14: "p2_1/n",
        15: "p2_1/c",
        16: "p2_1/a",
        17: "p2_1/c",
        18: "p2_1/c",
        19: "p2_1/c",
        20: "p2_1/c",
        21: "c2/m",
        22: "c2/e",
        25: "pmm2",
        26: "pmc2_1",
        27: "pcc2",
        28: "pma2",
        29: "pca2_1",
        30: "pnc2",
        31: "pnm2_1",
        32: "pba2",
        33: "pna2_1",
        34: "pnn2",
        35: "cmm2",
        36: "cmc2_1",
        37: "ccc2",
        38: "amm2",
        39: "abm2",
        40: "ama2",
        41: "aba2",
        42: "fmm2",
        43: "fdd2",
        44: "imm2",
        45: "iba2",
        46: "ima2",
    }

    return LAYER_GROUPS.get(hall_number, f"Hall_{hall_number}")


__all__ = ['check_symmetry_2d', 'HAS_SPGLIB']
