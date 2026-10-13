from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from eval_2d_utils import (
    ensure_dir,
    infer_run_label,
    inverse_distance_score,
    iqr_scale,
    load_reference_data,
    paper_row,
    percentage,
    robust_wasserstein,
    write_csv,
    write_json,
)


def evaluate_distribution(
    *,
    details_df: pd.DataFrame,
    train_csv: Path,
    output_dir: Path,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    label = infer_run_label(output_dir, label)
    reference = load_reference_data(train_csv)

    valid_df = details_df[details_df["overall_valid"] == True]

    metric_specs = [
        ("atom_count", "wd_atom_count"),
        ("unique_elems", "wd_unique_elems"),
        ("slab_thickness_ang", "wd_slab_thickness_ang"),
        ("vacuum_thickness_ang", "wd_vacuum_thickness_ang"),
        ("density", "wd_density"),
    ]

    summary: Dict[str, Any] = {"label": label}
    component_scores = []

    for detail_key, summary_key in metric_specs:
        gen_values = [float(v) for v in valid_df[detail_key].dropna().tolist()]
        train_values = [float(v) for v in reference.metrics.get(detail_key, [])]
        wd_value = robust_wasserstein(train_values, gen_values)
        summary[summary_key] = wd_value

        scale = iqr_scale(train_values)
        score = inverse_distance_score(wd_value, scale)
        summary[f"{summary_key}_score"] = score
        if score is not None:
            component_scores.append(score)

    gen_angle_outlier_rate = float(valid_df["angle_outlier"].mean()) if not valid_df.empty else 0.0
    gen_center_error_mean = float(valid_df["center_error"].mean()) if not valid_df.empty else None
    gen_center_error_median = float(valid_df["center_error"].median()) if not valid_df.empty else None

    summary["train_angle_outlier_rate"] = reference.angle_outlier_rate
    summary["gen_angle_outlier_rate"] = gen_angle_outlier_rate
    summary["angle_outlier_rate_gap"] = abs(gen_angle_outlier_rate - reference.angle_outlier_rate)
    summary["gen_center_error_mean"] = gen_center_error_mean
    summary["gen_center_error_median"] = gen_center_error_median
    summary["train_center_error_mean"] = float(np.mean(reference.metrics.get("center_error", [0.0])))
    summary["train_center_error_median"] = float(np.median(reference.metrics.get("center_error", [0.0])))

    angle_score = max(0.0, 1.0 - summary["angle_outlier_rate_gap"])
    summary["angle_outlier_score"] = angle_score
    component_scores.append(angle_score)

    train_center_scale = max(summary["train_center_error_median"], 1e-3)
    center_gap = None
    center_score = None
    if gen_center_error_median is not None:
        center_gap = abs(gen_center_error_median - summary["train_center_error_median"])
        center_score = 1.0 / (1.0 + center_gap / train_center_scale)
        component_scores.append(center_score)
    summary["center_error_gap"] = center_gap
    summary["center_error_score"] = center_score

    summary["two_d_dist_score"] = float(np.mean(component_scores)) if component_scores else None

    write_json(output_dir / "distribution_summary.json", summary)
    write_csv(output_dir / "distribution_paper_table.csv", paper_row(label, summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 2D distribution metrics.")
    parser.add_argument("--details_csv", type=Path, required=True, help="core_details.csv from eval_2d_core.py")
    parser.add_argument("--train_csv", type=Path, required=True, help="Training CSV with cif column.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write outputs.")
    parser.add_argument("--label", type=str, default=None, help="Optional run label.")
    args = parser.parse_args()

    details_df = pd.read_csv(args.details_csv)
    summary = evaluate_distribution(
        details_df=details_df,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        label=args.label,
    )
    print(f"[dist] {summary['label']}")
    print(f"  WD(slab)    : {summary.get('wd_slab_thickness_ang')}")
    print(f"  WD(density) : {summary.get('wd_density')}")
    print(f"  DistScore   : {summary.get('two_d_dist_score')}")


if __name__ == "__main__":
    main()
