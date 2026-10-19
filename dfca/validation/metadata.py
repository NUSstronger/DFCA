"""
Screen Metadata Utilities.

Functions for loading and processing Screen report metadata.

IMPORTANT:
These functions are diagnostic/provenance only. They do not alter
any GPAW validation pass/fail decisions.

References:
    - DFCA screening pipeline
"""

import csv
import logging
import os
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _safe_screen_float(value, default=0.0):
    """Safely parse a numeric value from the Screen CSV."""
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "" or text.lower() in ("nan", "none"):
            return default
        return float(text)
    except Exception:
        return default


def _safe_screen_int(value, default=0):
    """Safely parse an integer value from the Screen CSV."""
    try:
        return int(round(_safe_screen_float(value, default)))
    except Exception:
        return default


def _safe_screen_bool(value, default=False):
    """Safely parse a boolean value from the Screen CSV."""
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    text = str(value).strip().lower()

    if text in ("true", "1", "yes", "y"):
        return True

    if text in ("false", "0", "no", "n", "", "nan", "none"):
        return False

    return default


def _screen_name_from_gpaw_name(name: str) -> str:
    """
    Convert:
        rank0003_gen_2365 -> gen_2365

    Direct/legacy names are left unchanged.
    """
    m = re.search(r"(gen_\d+)$", str(name))
    if m:
        return m.group(1)

    return str(name)


def load_screen_report_index(report_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load screening_report.csv once and index rows by Screen name.

    Example key:
        gen_2365

    This function only reads metadata. It does not alter any
    GPAW validation decision.
    """
    index = {}

    if not report_path or not os.path.isfile(report_path):
        return index

    try:
        with open(report_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                name = str(row.get("name", "")).strip()

                if not name:
                    continue

                # Duplicate names are unsafe for provenance.
                if name in index:
                    raise RuntimeError(
                        f"Duplicate Screen report name: {name}"
                    )

                index[name] = row
    except Exception as e:
        logger.warning(f"Failed to load screen report index: {e}")

    return index


def get_screen_metadata(
    gpaw_name: str,
    report_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return normalized Screen metadata for one GPAW candidate.

    IMPORTANT:
    Diagnostic/provenance only. No GPAW pass/fail decision is
    changed by this function.
    """
    screen_name = _screen_name_from_gpaw_name(gpaw_name)
    row = report_index.get(screen_name)

    if row is None:
        return {
            "screen_report_found": False,
            "screen_name": screen_name,
        }

    return {
        "screen_report_found": True,
        "screen_name": screen_name,
        "screen_formula": str(row.get("formula", "")),
        "screen_n_atoms": _safe_screen_int(
            row.get("n_atoms")
        ),
        "screen_mag_type": str(
            row.get("mag_type", "")
        ),

        "screen_s4_min_eigenvalue": _safe_screen_float(
            row.get("s4_min_eigenvalue")
        ),
        "screen_s4_stress_residual": _safe_screen_float(
            row.get("s4_stress_residual")
        ),
        "screen_s4_pass": _safe_screen_bool(
            row.get("s4_pass")
        ),

        "screen_s5_gamma_min_THz": _safe_screen_float(
            row.get("s5_gamma_min_THz")
        ),
        "screen_s5_min_freq_THz": _safe_screen_float(
            row.get("s5_min_freq_THz")
        ),
        "screen_s5_imaginary_count": _safe_screen_int(
            row.get("s5_imaginary_count")
        ),
        "screen_s5_pass": _safe_screen_bool(
            row.get("s5_pass")
        ),
        "screen_s5_reason": str(
            row.get("s5_reason", "")
        ),

        "screen_s7a_pass": _safe_screen_bool(
            row.get("s7a_pass")
        ),
        "screen_s7b_pass": _safe_screen_bool(
            row.get("s7b_pass")
        ),

        "screen_s5p_ehull_eV": _safe_screen_float(
            row.get("s5p_ehull_eV")
        ),
        "screen_s5p_pass": _safe_screen_bool(
            row.get("s5p_pass")
        ),
        "screen_s5p_stability_label": str(
            row.get("s5p_stability_label", "")
        ),

        "screen_select_score": _safe_screen_float(
            row.get("select_score")
        ),
        "screen_final_pass": _safe_screen_bool(
            row.get("final_pass")
        ),
    }


__all__ = [
    'load_screen_report_index',
    'get_screen_metadata',
]
