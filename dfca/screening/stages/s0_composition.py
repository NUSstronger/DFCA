"""
Screening Stage S0: Composition Check.

Checks:
    - Atom count limits
    - Forbidden elements (radioactive, actinides)
    - Charge neutrality warning

References:
    - C2DB composition criteria
"""

from typing import Tuple, Dict
from pymatgen.core import Structure


# Forbidden atomic numbers (radioactive + actinides)
# Elements: Po(84), At(85), Rn(86), Fr(87), Ra(88), Ac-Lr(89-103)
FORBIDDEN_Z = {84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103}


def check_composition(struct: Structure, cfg) -> Tuple[bool, Dict]:
    """
    S0: Composition fast pre-screening.

    Checks:
        1. Atom count within reasonable range (1-200)
        2. No forbidden elements (radioactive, actinides)
        3. Charge neutrality warning (for ionic materials)

    Args:
        struct: pymatgen Structure object
        cfg: ScreeningConfig instance

    Returns:
        (passed, info_dict)
            - passed: True if structure passes all checks
            - info_dict: Dictionary with check results
    """
    n_atoms = len(struct)

    # 1. Atom count check
    if n_atoms < 1:
        return False, {"reason": f"too_few_atoms:{n_atoms}", "n_atoms": n_atoms}
    if n_atoms > 200:
        return False, {"reason": f"too_many_atoms:{n_atoms}", "n_atoms": n_atoms}

    # 2. Forbidden elements check
    forbidden_in_struct = []
    for site in struct:
        z = site.specie.Z
        if z in FORBIDDEN_Z:
            forbidden_in_struct.append(f"{site.specie.symbol}(Z={z})")

    if forbidden_in_struct:
        return False, {
            "reason": f"forbidden_elements:{forbidden_in_struct}",
            "n_atoms": n_atoms,
            "forbidden": forbidden_in_struct,
        }

    # 3. Charge neutrality check (warning only)
    comp = struct.composition
    total_charge = None
    try:
        total_charge = comp.charge
        if total_charge is not None and abs(total_charge) > 0.1:
            # Non-zero charge - issue warning but don't fail
            charge_warning = f"non_neutral:{total_charge:.2f}"
        else:
            charge_warning = ""
    except Exception:
        charge_warning = ""

    return True, {
        "reason": charge_warning if charge_warning else "ok",
        "n_atoms": n_atoms,
        "charge": total_charge if total_charge is not None else 0.0,
        "formula": comp.reduced_formula,
    }


__all__ = ['check_composition', 'FORBIDDEN_Z']
