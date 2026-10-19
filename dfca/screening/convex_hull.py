# screening/convex_hull.py
# S5': MP API Convex Hull Analysis (Pipeline Terminus)
# v1.3 spec section 5' + 5.4
#
# Physical constraints (fixed bugs):
#   - Bug #11 (this fix): REMOVED dimensionality filtering on MP bulk entries.
#     The dim >= 2.5 threshold via bonding_common.check_dimensionality() was wrong because:
#     (a) Bulk vdW materials (MoS2, h-BN, graphite) have dim=2 by covalent-bond analysis,
#         yet they are the most critical low-energy competing phases for 2D convex hulls.
#     (b) MP entries are already 3D bulk by definition — no filtering needed.
#   - Bug #11: PhaseDiagram construction exceptions now preserve full error messages
#     (e.g. missing-element references) instead of swallowing them as "ehull_mp_error".
#   - Mixed MACE/MP DFT energies: documented limitation (see Bug #6 fix)
#   - Python 3.10 compat: backport NotRequired/Required from typing_extensions
#     (typing.NotRequired was added in Python 3.11; mp-api may import it directly)
#   - pymatgen >= 2024.x compat (step 1): bridge removed pymatgen.core.entries path
#     (mp_api internally hardcodes "from pymatgen.core.entries import ..." which fails
#     after pymatgen moved those classes to pymatgen.entries.computed_entries)
#   - pymatgen >= 2024.x compat (step 2): bridge removed pymatgen.analysis.compatibility path
#     (mp_api hardcodes this when loading multi-element entries like MoS2/BN)

import sys
import typing

if sys.version_info < (3, 11):
    try:
        import typing_extensions
        if not hasattr(typing, "NotRequired"):
            typing.NotRequired = typing_extensions.NotRequired
        if not hasattr(typing, "Required"):
            typing.Required = typing_extensions.Required
    except ImportError:
        pass

# Bridge pymatgen.core.entries -> pymatgen.entries.computed_entries
# Must be done before any mp_api or pymatgen.analysis.phase_diagram import.
try:
    import pymatgen.entries.computed_entries as _computed_entries
    import pymatgen.core as _pmg_core
    _pmg_core.entries = _computed_entries
    sys.modules["pymatgen.core.entries"] = _computed_entries
except Exception:
    pass

# Bridge pymatgen.analysis.compatibility -> pymatgen.entries.compatibility
# mp_api internally hardcodes this import when loading multi-element entries (e.g. MoS2, BN),
# which fails on pymatgen >= 2024.x that moved compatibility to pymatgen.entries.
try:
    import pymatgen.entries.compatibility as _compatibility
    import pymatgen.analysis as _pmg_analysis
    _pmg_analysis.compatibility = _compatibility
    sys.modules["pymatgen.analysis.compatibility"] = _compatibility
except Exception:
    pass

from typing import Tuple, Dict
import os


def compute_ehull_mp(
    struct,
    energy_per_atom,
    mp_api_key="",
    ehull_threshold=0.2,
) -> Tuple[float, bool, Dict]:
    """
    Compute eHull using MP API competing phases (bulk-only).

    Pipeline position: S5' (last stage, after S7b)

    Args:
        struct: pymatgen Structure
        energy_per_atom: eV/atom from MACE-MP-0 (S3.5 relaxed energy)
        mp_api_key: Materials Project API key (or env MP_API_KEY)
        ehull_threshold: C2DB standard 0.2 eV/atom

    Returns:
        (ehull_eV, is_stable, info_dict)
    """
    # Get API key
    key = mp_api_key or os.environ.get("MP_API_KEY", "")
    if not key:
        return -1.0, False, {
            "ehull_eV": -1.0,
            "competing_phases_found": 0,
            "missing_from_MP": [],
            "reason": "no_mp_api_key",
        }

    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        # pymatgen >= 2024.x moved PDEntry to computed_entries
        try:
            from pymatgen.analysis.phase_diagram import PDEntry
        except (ImportError, ModuleNotFoundError):
            from pymatgen.entries.computed_entries import ComputedEntry as PDEntry
        import logging
    except ImportError as e:
        return -1.0, False, {
            "ehull_eV": -1.0,
            "competing_phases_found": 0,
            "missing_from_MP": [],
            "reason": f"mp_api_not_installed:{e}",
        }

    dim_logger = logging.getLogger("Screen2D.convex_hull")

    comp = struct.composition
    elements = [str(el) for el in comp.elements]
    n_atoms = len(struct)

    try:
        with MPRester(key) as mpr:
            # Fetch competing phases for the chemical system.
            # Use try-except for property_data kwarg: newer mp-api versions reject it
            # but older ones require it. Try standard call first, fall back if needed.
            # Ensure structure data is included for dimensionality filtering.
            try:
                entries = mpr.get_entries_in_chemsys(elements, inc_structure="initial")
            except TypeError:
                # Older mp-api version - try without inc_structure
                try:
                    entries = mpr.get_entries_in_chemsys(elements)
                except TypeError:
                    # Older mp-api version that requires property_data
                    entries = mpr.get_entries_in_chemsys(
                        elements,
                        property_data=["formation_energy_per_atom"],
                        inc_structure="initial",
                    )
            except Exception:
                # Fallback: standard call without structure
                entries = mpr.get_entries_in_chemsys(elements)

            # Bug #11 fix: REMOVED dimensionality filtering on MP bulk entries.
            #
            # Previous buggy logic used bonding_common.check_dimensionality() (Larsen covalent-bond
            # graph analysis) with dim >= 2.5 threshold to filter "bulk" vs "2D" competing phases.
            # This is WRONG for convex hull purposes because:
            #   - Bulk materials like MoS2, h-BN, graphite have van-der-Waals interlayer coupling
            #     (no covalent bonds) and are classified as dim=2 by Larsen's algorithm.
            #   - These are precisely the most important low-energy competing phases for 2D
            #     materials on the convex hull.
            #   - The dim >= 2.5 threshold was designed for checking 2D slab connectivity, NOT
            #     for determining whether a structure is a 3D bulk crystal.
            #
            # Fix: MP database entries ARE already 3D bulk crystals by definition (they come from
            # experimental/DFT-optimized bulk structures). We do NOT apply dimensionality filtering
            # to MP competing phases. If dimensionality checking is needed for candidate 2D slabs,
            # that is handled separately in the S-dim stage of the pipeline.
            #
            # Bulk entries are used as-is for convex hull construction.
            bulk_entries = list(entries)

            if not bulk_entries:
                return -1.0, False, {
                    "ehull_eV": -1.0,
                    "competing_phases_found": 0,
                    "missing_from_MP": elements,
                    "reason": "no_bulk_entries_in_MP",
                }

            # Build convex hull with MP bulk entries
            # Note: Bug #6 (known limitation): candidate uses MACE-MP-0 energy,
            # competing phases use MP DFT energy - mixed reference frames.
            # This is documented as a known limitation (not fixed in this version).
            # PDEntry signature: PDEntry(composition, energy, name=None, ...)
            # Use `name` (not `entry_id`) for unique identification to avoid
            # index collision with bulk entries of the same composition.
            my_entry = PDEntry(
                comp,
                energy_per_atom * n_atoms,
                name="candidate_2d",
            )
            all_entries = bulk_entries + [my_entry]

            pd = PhaseDiagram(all_entries)
            ehull = float(pd.get_e_above_hull(my_entry))
            is_stable = bool(ehull <= ehull_threshold)

            # Determine missing elements from MP
            mp_elements = set()
            for e in bulk_entries:
                for el in e.composition.elements:
                    mp_elements.add(str(el))
            missing = [el for el in elements if el not in mp_elements]

            return float(ehull), is_stable, {
                "ehull_eV": round(float(ehull), 4),
                "competing_phases_found": int(len(bulk_entries)),
                "missing_from_MP": missing,
                "reason": "ok" if is_stable else f"ehull={ehull:.4f}>{ehull_threshold}",
            }

    except Exception as e:
        # Bug #11 fix: preserve exception details for debugging instead of swallowing
        # into a generic "ehull_mp_error" label. This helps identify missing element
        # references and other PhaseDiagram construction failures.
        error_msg = str(e)
        # Log to debug channel for diagnostics
        dim_logger.debug(f"PhaseDiagram construction failed for {elements}: {error_msg}", exc_info=True)
        return -1.0, False, {
            "ehull_eV": -1.0,
            "competing_phases_found": 0,
            "missing_from_MP": elements,
            "reason": f"ehull_mp_error:{error_msg}",
        }
