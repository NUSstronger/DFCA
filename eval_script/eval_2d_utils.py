"""
Utility functions for 2D evaluation.

Provides shared utilities for:
- Geometry computation (structure metrics, validity checks)
- Reference data management
- I/O operations
- Scoring utilities
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

try:
    import smact
    from smact.screening import pauling_test

    HAS_SMACT = True
except Exception:
    HAS_SMACT = False


# =============================================================================
# Constants
# =============================================================================

STRUCTURE_CUTOFF_ANG = 0.5
MIN_VACUUM_ANG = 10.0
ANGLE_OUTLIER_MIN_DEG = 60.0
ANGLE_OUTLIER_MAX_DEG = 150.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeometryMetrics:
    """Computed geometry metrics for a crystal structure."""

    atom_count: int
    unique_elems: int
    density: float
    min_pair_dist_ang: float
    slab_thickness_ang: float
    vacuum_thickness_ang: float
    center_error: float
    angle_outlier: bool
    angle_outlier_deviation: float
    reduced_formula: str
    chemical_system: str


class ReferenceData:
    """Container for training/reference data used in evaluation."""

    def __init__(
        self,
        *,
        structures: List[Structure],
        formulas: set[str],
        structures_by_formula: Dict[str, List[Structure]],
        metrics: Dict[str, List[float]],
        angle_outlier_rate: float,
        slab_thickness_p95: Optional[float],
    ) -> None:
        self.structures = structures
        self.formulas = formulas
        self.structures_by_formula = structures_by_formula
        self.metrics = metrics
        self.angle_outlier_rate = angle_outlier_rate
        self.slab_thickness_p95 = slab_thickness_p95


# =============================================================================
# I/O Utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dictionary as JSON to a file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    """Write a DataFrame as CSV to a file."""
    ensure_dir(path.parent)
    rows.to_csv(path, index=False)


def clean_details_for_export(details_df: pd.DataFrame) -> pd.DataFrame:
    """Clean a details DataFrame for CSV export."""
    export_df = details_df.copy()
    if "structure_obj" in export_df.columns:
        export_df = export_df.drop(columns=["structure_obj"])
    if "failure_reasons" in export_df.columns:
        export_df["failure_reasons"] = export_df["failure_reasons"].apply(lambda values: "; ".join(values))
    return export_df


# =============================================================================
# Reference Data Loading
# =============================================================================

def load_cif_strings_from_csv(csv_path: Path) -> List[str]:
    """Load CIF strings from a CSV file with a 'cif' column."""
    rows: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "cif" not in (reader.fieldnames or []):
            raise ValueError(f"'cif' column not found in {csv_path}")
        for row in reader:
            cif = row.get("cif")
            if cif:
                rows.append(cif)
    return rows


def load_generated_structures(gen_dir: Path) -> List[Tuple[str, Path, Optional[Structure], Optional[str]]]:
    """Load all CIF files from a directory as Structure objects."""
    rows: List[Tuple[str, Path, Optional[Structure], Optional[str]]] = []
    for cif_path in sorted(gen_dir.glob("*.cif")):
        try:
            structure = Structure.from_file(cif_path)
            rows.append((cif_path.name, cif_path.resolve(), structure, None))
        except Exception as exc:
            rows.append((cif_path.name, cif_path.resolve(), None, str(exc)))
    return rows


def formula_to_structures(structures: Iterable[Structure]) -> Dict[str, List[Structure]]:
    """Group structures by their reduced formula."""
    grouped: Dict[str, List[Structure]] = defaultdict(list)
    for structure in structures:
        grouped[str(structure.composition.reduced_formula)].append(structure)
    return grouped


def load_reference_data(train_csv: Path) -> ReferenceData:
    """Load and process reference/training data from a CSV file."""
    cif_strings = load_cif_strings_from_csv(train_csv)
    structures: List[Structure] = []
    metrics: Dict[str, List[float]] = defaultdict(list)
    formulas: set[str] = set()
    angle_outliers = 0

    for cif in cif_strings:
        try:
            structure = Structure.from_str(cif, fmt="cif")
        except Exception:
            continue
        structures.append(structure)
        geometry = geometry_metrics(structure)
        formulas.add(geometry.reduced_formula)
        metrics["atom_count"].append(float(geometry.atom_count))
        metrics["unique_elems"].append(float(geometry.unique_elems))
        metrics["slab_thickness_ang"].append(float(geometry.slab_thickness_ang))
        metrics["vacuum_thickness_ang"].append(float(geometry.vacuum_thickness_ang))
        metrics["density"].append(float(geometry.density))
        metrics["center_error"].append(float(geometry.center_error))
        if geometry.angle_outlier:
            angle_outliers += 1

    slab_p95 = None
    if metrics["slab_thickness_ang"]:
        slab_p95 = float(np.percentile(metrics["slab_thickness_ang"], 95))

    return ReferenceData(
        structures=structures,
        formulas=formulas,
        structures_by_formula=formula_to_structures(structures),
        metrics=dict(metrics),
        angle_outlier_rate=(angle_outliers / len(structures)) if structures else 0.0,
        slab_thickness_p95=slab_p95,
    )


# =============================================================================
# Geometry Computation
# =============================================================================

def circular_distance(a: float, b: float) -> float:
    """Compute circular distance on [0, 1] interval."""
    diff = abs((a - b) % 1.0)
    return min(diff, 1.0 - diff)


def occupied_arc(z_values: Sequence[float]) -> Tuple[float, float, float]:
    """
    Find the occupied arc in fractional z-coordinates.

    Returns:
        (span_frac, vacuum_frac, center): Fractional span, vacuum fraction, and center position.
    """
    if not z_values:
        return 0.0, 1.0, 0.5

    points = sorted(float(z) % 1.0 for z in z_values)
    if len(points) == 1:
        return 0.0, 1.0, points[0]

    extended = points + [points[0] + 1.0]
    gaps = [extended[i + 1] - extended[i] for i in range(len(points))]
    max_gap_idx = int(np.argmax(gaps))
    max_gap = gaps[max_gap_idx]
    span = 1.0 - max_gap
    start = extended[max_gap_idx + 1] % 1.0
    center = (start + span / 2.0) % 1.0
    return span, max_gap, center


def min_pairwise_distance(structure: Structure) -> float:
    """Compute minimum pairwise distance between atoms, excluding self."""
    if len(structure) < 2:
        return float("inf")
    dist_mat = structure.distance_matrix
    diag = np.eye(dist_mat.shape[0]) * (STRUCTURE_CUTOFF_ANG + 10.0)
    return float(np.min(dist_mat + diag))


def geometry_metrics(structure: Structure) -> GeometryMetrics:
    """
    Compute geometry metrics for a crystal structure.

    Includes:
    - Atom count and unique elements
    - Slab and vacuum thickness
    - Density
    - Center error (deviation from c=0.5)
    - Angle outlier detection
    """
    zs = [float(coord[2]) % 1.0 for coord in structure.frac_coords]
    span_frac, vacuum_frac, center = occupied_arc(zs)
    lattice = structure.lattice
    slab_thickness = span_frac * lattice.c
    vacuum_thickness = vacuum_frac * lattice.c
    angles = lattice.angles
    angle_deviation = max(abs(angle - 90.0) for angle in angles)
    angle_outlier = any(angle < ANGLE_OUTLIER_MIN_DEG or angle > ANGLE_OUTLIER_MAX_DEG for angle in angles)
    elements = sorted({str(site.specie.symbol) for site in structure.sites})
    chemical_system = "-".join(elements)

    return GeometryMetrics(
        atom_count=len(structure),
        unique_elems=len(elements),
        density=float(structure.density),
        min_pair_dist_ang=min_pairwise_distance(structure),
        slab_thickness_ang=float(slab_thickness),
        vacuum_thickness_ang=float(vacuum_thickness),
        center_error=float(circular_distance(center, 0.5)),
        angle_outlier=bool(angle_outlier),
        angle_outlier_deviation=float(angle_deviation),
        reduced_formula=str(structure.composition.reduced_formula),
        chemical_system=chemical_system,
    )


# =============================================================================
# Validity Checks
# =============================================================================

def structure_validity(structure: Structure) -> Tuple[bool, float]:
    """
    Check if a structure is structurally valid.

    Returns:
        (is_valid, min_distance): Validity flag and minimum pairwise distance.
    """
    min_dist = min_pairwise_distance(structure)
    valid = bool(min_dist > STRUCTURE_CUTOFF_ANG and structure.volume > 0.1)
    return valid, min_dist


def composition_validity(structure: Structure) -> bool:
    """
    Check if a structure's composition is chemically valid using SMACT.

    Returns True if SMACT is not available.
    """
    if not HAS_SMACT:
        return True

    comp = structure.composition
    elements = [str(el) for el in comp.elements]
    stoich = [int(comp[el]) for el in comp.elements]
    if len(elements) <= 1:
        return True

    try:
        smact_elems = [smact.Element(e) for e in elements]
        if all(e.symbol in smact.metals for e in smact_elems):
            return True

        oxidation_sets = [e.oxidation_states for e in smact_elems]
        electronegativities = [e.pauling_eneg for e in smact_elems]
    except Exception:
        return False

    try:
        from itertools import product

        for oxidation_combo in product(*oxidation_sets):
            total_charge = sum(ox * count for ox, count in zip(oxidation_combo, stoich))
            if total_charge != 0:
                continue
            try:
                if pauling_test(oxidation_combo, electronegativities):
                    return True
            except TypeError:
                return True
        return False
    except Exception:
        return False


def two_d_validity(
    geometry: GeometryMetrics,
    *,
    slab_thickness_p95: Optional[float],
    min_vacuum_ang: float = MIN_VACUUM_ANG,
) -> Tuple[bool, List[str]]:
    """
    Check if a structure is valid for 2D materials.

    Returns:
        (is_valid, failure_reasons): Validity flag and list of failure reasons.
    """
    reasons: List[str] = []
    if geometry.vacuum_thickness_ang < min_vacuum_ang:
        reasons.append(f"vacuum<{min_vacuum_ang:.2f}A")
    if slab_thickness_p95 is not None and geometry.slab_thickness_ang > slab_thickness_p95:
        reasons.append(f"slab_thickness>train_p95({slab_thickness_p95:.3f}A)")
    return len(reasons) == 0, reasons


# =============================================================================
# Uniqueness and Novelty
# =============================================================================

def compute_uniqueness_flags(
    details_df: pd.DataFrame,
    *,
    matcher: Optional[StructureMatcher] = None,
) -> pd.Series:
    """Compute uniqueness flags for structures in details_df."""
    if matcher is None:
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True, scale=True)

    flags = pd.Series(False, index=details_df.index, dtype=bool)
    valid_df = details_df[details_df["overall_valid"] == True]
    if valid_df.empty:
        return flags

    unique_by_formula: Dict[str, List[Structure]] = defaultdict(list)
    for idx, row in valid_df.iterrows():
        structure = row["structure_obj"]
        formula = row["reduced_formula"]
        known_structures = unique_by_formula[formula]
        is_unique = True
        for existing in known_structures:
            try:
                if matcher.fit(structure, existing):
                    is_unique = False
                    break
            except Exception:
                continue
        if is_unique:
            flags.at[idx] = True
            known_structures.append(structure)
    return flags


def compute_structure_novelty_flags(
    details_df: pd.DataFrame,
    reference: ReferenceData,
    *,
    matcher: Optional[StructureMatcher] = None,
) -> pd.Series:
    """Compute structure-level novelty flags against reference data."""
    if matcher is None:
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True, scale=True)

    flags = pd.Series(False, index=details_df.index, dtype=bool)
    unique_df = details_df[(details_df["overall_valid"] == True) & (details_df["unique_valid"] == True)]
    if unique_df.empty:
        return flags

    for idx, row in unique_df.iterrows():
        formula = row["reduced_formula"]
        if formula not in reference.formulas:
            flags.at[idx] = True
            continue

        structure = row["structure_obj"]
        is_novel = True
        for train_structure in reference.structures_by_formula.get(formula, []):
            try:
                if matcher.fit(structure, train_structure):
                    is_novel = False
                    break
            except Exception:
                continue
        flags.at[idx] = is_novel
    return flags


# =============================================================================
# Distribution Metrics
# =============================================================================

def robust_wasserstein(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[float]:
    """Compute Wasserstein distance between two distributions."""
    if not values_a or not values_b:
        return None
    return float(wasserstein_distance(list(values_a), list(values_b)))


def iqr_scale(values: Sequence[float]) -> float:
    """Compute IQR-based scale for distance normalization."""
    if not values:
        return 1.0
    q25, q75 = np.percentile(values, [25, 75])
    return float(max(q75 - q25, 1e-6))


def inverse_distance_score(distance: Optional[float], scale: float) -> Optional[float]:
    """Convert a distance to a similarity score using inverse distance."""
    if distance is None:
        return None
    return float(1.0 / (1.0 + distance / max(scale, 1e-6)))


# =============================================================================
# Summary Utilities
# =============================================================================

def summarize_rate(count: int, total: int) -> float:
    """Compute a rate as a fraction."""
    if total <= 0:
        return 0.0
    return float(count / total)


def percentage(value: Optional[float]) -> Optional[float]:
    """Convert a rate to percentage."""
    if value is None:
        return None
    return float(value * 100.0)


def merge_summaries(*summary_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple summary dictionaries into one."""
    merged: Dict[str, Any] = {}
    for summary in summary_dicts:
        merged.update(summary)
    return merged


def paper_row(label: str, summary: Dict[str, Any]) -> pd.DataFrame:
    """Generate a paper-ready summary row."""
    return pd.DataFrame(
        [
            {
                "label": label,
                "overall_validity_%": percentage(summary.get("overall_validity_rate")),
                "uniqueness_%": percentage(summary.get("uniqueness_rate")),
                "novelty_%": percentage(summary.get("novelty_rate_formula_main")),
                "2d_un_%": percentage(summary.get("two_d_un_rate")),
                "wd_slab_thickness_ang": summary.get("wd_slab_thickness_ang"),
                "wd_density": summary.get("wd_density"),
                "stable_at_chgnet_%": percentage(summary.get("stable_at_chgnet_rate")),
                "metastable_at_chgnet_%": percentage(summary.get("metastable_at_chgnet_rate")),
                "two_d_sun_proxy_%": percentage(summary.get("two_d_sun_proxy")),
                "two_d_msun_proxy_%": percentage(summary.get("two_d_msun_proxy")),
            }
        ]
    )


def infer_run_label(gen_dir: Path, explicit_label: Optional[str]) -> str:
    """Infer run label from directory or use explicit label."""
    return explicit_label or gen_dir.name
