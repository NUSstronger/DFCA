from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher

from eval_2d_utils import (
    GeometryMetrics,
    clean_details_for_export,
    composition_validity,
    compute_structure_novelty_flags,
    compute_uniqueness_flags,
    ensure_dir,
    geometry_metrics,
    infer_run_label,
    load_generated_structures,
    load_reference_data,
    paper_row,
    percentage,
    structure_validity,
    summarize_rate,
    two_d_validity,
    write_csv,
    write_json,
)


def _build_detail_row(
    *,
    file_name: str,
    path: Path,
    geometry: Optional[GeometryMetrics],
    parse_error: Optional[str],
    structure_obj: Any,
    slab_thickness_p95: Optional[float],
) -> Dict[str, Any]:
    if geometry is None:
        return {
            "file": file_name,
            "path": str(path),
            "parsed": False,
            "parse_error": parse_error,
            "reduced_formula": "",
            "chemical_system": "",
            "atom_count": 0,
            "unique_elems": 0,
            "density": None,
            "min_pair_dist_ang": None,
            "slab_thickness_ang": None,
            "vacuum_thickness_ang": None,
            "center_error": None,
            "angle_outlier": False,
            "angle_outlier_deviation": None,
            "structural_valid": False,
            "compositional_valid": False,
            "valid_2d": False,
            "overall_valid": False,
            "unique_valid": False,
            "novel_formula_main": False,
            "novel_structure_aux": False,
            "is_2d_un_main": False,
            "is_2d_un_aux_structure": False,
            "failure_reasons": ["parse_failed"],
            "structure_obj": None,
            "train_slab_thickness_p95_ang": slab_thickness_p95,
        }

    structural_valid, min_dist = structure_validity(structure_obj)
    compositional_valid = composition_validity(structure_obj)
    valid_2d, reasons_2d = two_d_validity(geometry, slab_thickness_p95=slab_thickness_p95)

    failure_reasons = []
    if not structural_valid:
        failure_reasons.append("structural_invalid")
    if not compositional_valid:
        failure_reasons.append("compositional_invalid")
    failure_reasons.extend(reasons_2d)

    return {
        "file": file_name,
        "path": str(path),
        "parsed": True,
        "parse_error": parse_error,
        "reduced_formula": geometry.reduced_formula,
        "chemical_system": geometry.chemical_system,
        "atom_count": geometry.atom_count,
        "unique_elems": geometry.unique_elems,
        "density": geometry.density,
        "min_pair_dist_ang": min_dist,
        "slab_thickness_ang": geometry.slab_thickness_ang,
        "vacuum_thickness_ang": geometry.vacuum_thickness_ang,
        "center_error": geometry.center_error,
        "angle_outlier": geometry.angle_outlier,
        "angle_outlier_deviation": geometry.angle_outlier_deviation,
        "structural_valid": structural_valid,
        "compositional_valid": compositional_valid,
        "valid_2d": valid_2d,
        "overall_valid": structural_valid and compositional_valid and valid_2d,
        "unique_valid": False,
        "novel_formula_main": False,
        "novel_structure_aux": False,
        "is_2d_un_main": False,
        "is_2d_un_aux_structure": False,
        "failure_reasons": failure_reasons,
        "structure_obj": structure_obj,
        "train_slab_thickness_p95_ang": slab_thickness_p95,
    }


def evaluate_core(
    *,
    gen_dir: Path,
    train_csv: Path,
    output_dir: Path,
    label: Optional[str] = None,
    skip_structure_novelty: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    label = infer_run_label(gen_dir, label)
    output_dir = ensure_dir(output_dir)
    reference = load_reference_data(train_csv)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True, scale=True)

    detail_rows = []
    for file_name, cif_path, structure, parse_error in load_generated_structures(gen_dir):
        geometry = geometry_metrics(structure) if structure is not None else None
        detail_rows.append(
            _build_detail_row(
                file_name=file_name,
                path=cif_path,
                geometry=geometry,
                parse_error=parse_error,
                structure_obj=structure,
                slab_thickness_p95=reference.slab_thickness_p95,
            )
        )

    details_df = pd.DataFrame(detail_rows)

    uniqueness_flags = compute_uniqueness_flags(details_df, matcher=matcher)
    details_df["unique_valid"] = uniqueness_flags

    details_df.loc[details_df["unique_valid"], "novel_formula_main"] = (
        ~details_df.loc[details_df["unique_valid"], "reduced_formula"].isin(reference.formulas)
    )

    if skip_structure_novelty:
        details_df.loc[details_df["unique_valid"], "novel_structure_aux"] = details_df.loc[
            details_df["unique_valid"], "novel_formula_main"
        ]
    else:
        details_df["novel_structure_aux"] = compute_structure_novelty_flags(details_df, reference, matcher=matcher)

    details_df["is_2d_un_main"] = (
        details_df["overall_valid"] & details_df["unique_valid"] & details_df["novel_formula_main"]
    )
    details_df["is_2d_un_aux_structure"] = (
        details_df["overall_valid"] & details_df["unique_valid"] & details_df["novel_structure_aux"]
    )

    n_total = len(details_df)
    n_struct = int(details_df["structural_valid"].sum())
    n_comp = int(details_df["compositional_valid"].sum())
    n_2d = int(details_df["valid_2d"].sum())
    n_overall = int(details_df["overall_valid"].sum())
    n_unique = int(details_df["unique_valid"].sum())
    n_novel_formula = int(details_df["novel_formula_main"].sum())
    n_novel_structure = int(details_df["novel_structure_aux"].sum())
    n_2d_un = int(details_df["is_2d_un_main"].sum())
    n_2d_un_struct = int(details_df["is_2d_un_aux_structure"].sum())

    overall_valid_df = details_df[details_df["overall_valid"] == True]

    summary: Dict[str, Any] = {
        "label": label,
        "n_total": n_total,
        "n_structural_valid": n_struct,
        "n_compositional_valid": n_comp,
        "n_2d_valid": n_2d,
        "n_overall_valid": n_overall,
        "n_unique": n_unique,
        "n_novel_formula_main": n_novel_formula,
        "n_novel_structure_aux": n_novel_structure,
        "n_2d_un": n_2d_un,
        "n_2d_un_aux_structure": n_2d_un_struct,
        "structural_validity_rate": summarize_rate(n_struct, n_total),
        "compositional_validity_rate": summarize_rate(n_comp, n_total),
        "two_d_validity_rate": summarize_rate(n_2d, n_total),
        "overall_validity_rate": summarize_rate(n_overall, n_total),
        "uniqueness_rate": summarize_rate(n_unique, n_overall),
        "novelty_rate_formula_main": summarize_rate(n_novel_formula, n_unique),
        "novelty_rate_structure_aux": summarize_rate(n_novel_structure, n_unique),
        "two_d_un_rate": summarize_rate(n_2d_un, n_total),
        "two_d_un_rate_aux_structure": summarize_rate(n_2d_un_struct, n_total),
        "train_slab_thickness_p95_ang": reference.slab_thickness_p95,
        "valid_center_error_mean": float(overall_valid_df["center_error"].mean()) if not overall_valid_df.empty else None,
        "valid_center_error_median": float(overall_valid_df["center_error"].median()) if not overall_valid_df.empty else None,
        "valid_slab_thickness_mean_ang": float(overall_valid_df["slab_thickness_ang"].mean()) if not overall_valid_df.empty else None,
        "valid_vacuum_thickness_mean_ang": float(overall_valid_df["vacuum_thickness_ang"].mean()) if not overall_valid_df.empty else None,
    }

    write_json(output_dir / "core_summary.json", summary)
    export_df = clean_details_for_export(details_df)
    write_csv(output_dir / "core_details.csv", export_df)
    write_csv(output_dir / "core_paper_table.csv", paper_row(label, summary))
    return summary, details_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate core 2D crystal generation metrics.")
    parser.add_argument("--gen_dir", type=Path, required=True, help="Directory that contains generated CIF files.")
    parser.add_argument("--train_csv", type=Path, required=True, help="Training CSV with a cif column.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write evaluation outputs.")
    parser.add_argument("--label", type=str, default=None, help="Optional run label.")
    parser.add_argument(
        "--skip_structure_novelty",
        action="store_true",
        help="Skip expensive structure-level novelty checks and copy formula novelty instead.",
    )
    args = parser.parse_args()

    summary, _ = evaluate_core(
        gen_dir=args.gen_dir,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        label=args.label,
        skip_structure_novelty=args.skip_structure_novelty,
    )
    print(f"[core] {summary['label']}")
    print(f"  Overall Validity : {percentage(summary['overall_validity_rate']):.2f}%")
    print(f"  Uniqueness       : {percentage(summary['uniqueness_rate']):.2f}%")
    print(f"  Novelty          : {percentage(summary['novelty_rate_formula_main']):.2f}%")
    print(f"  2D-UN            : {percentage(summary['two_d_un_rate']):.2f}%")


if __name__ == "__main__":
    main()
