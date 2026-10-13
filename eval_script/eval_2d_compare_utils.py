"""
Comparison utilities for 2D evaluation runs.

This module contains functions for comparing multiple evaluation runs,
including metric normalization, winner determination, and report generation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


# =============================================================================
# Metric Definitions
# =============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single evaluation metric."""

    key: str
    label: str
    higher_is_better: bool
    weight: float
    display_as_percent: bool = False
    include_in_overall: bool = True


METRICS: List[MetricSpec] = [
    MetricSpec("structural_validity_rate", "Structural Validity", True, 2.0, True),
    MetricSpec("compositional_validity_rate", "Compositional Validity", True, 2.0, True),
    MetricSpec("two_d_validity_rate", "2D Validity", True, 4.0, True),
    MetricSpec("overall_validity_rate", "Overall Validity", True, 5.0, True),
    MetricSpec("uniqueness_rate", "Uniqueness", True, 2.0, True),
    MetricSpec("novelty_rate_formula_main", "Novelty", True, 2.0, True),
    MetricSpec("two_d_un_rate", "2D-UN", True, 6.0, True),
    MetricSpec("two_d_dist_score", "2D DistScore", True, 4.0, False),
    MetricSpec("wd_atom_count", "WD(atom_count)", False, 1.0, False),
    MetricSpec("wd_unique_elems", "WD(unique_elems)", False, 1.0, False),
    MetricSpec("wd_slab_thickness_ang", "WD(slab_thickness)", False, 3.0, False),
    MetricSpec("wd_vacuum_thickness_ang", "WD(vacuum_thickness)", False, 1.0, False),
    MetricSpec("wd_density", "WD(density)", False, 2.0, False),
    MetricSpec("angle_outlier_rate_gap", "Angle Outlier Gap", False, 1.5, False),
    MetricSpec("center_error_gap", "Center Error Gap", False, 1.5, False),
]


# =============================================================================
# Run Argument Parsing
# =============================================================================


def parse_run_arg(text: str) -> tuple[str, Path]:
    """Parse a run argument in the form 'label=path'."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value '{text}'. Expected the form label=path/to/eval_dir_or_summary.json."
        )
    label, path_text = text.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Missing run label in --run '{text}'.")
    path = Path(path_text.strip())
    if not path:
        raise argparse.ArgumentTypeError(f"Missing run path in --run '{text}'.")
    return label, path


def resolve_summary_path(path: Path) -> Path:
    """Resolve a run path to its summary.json location."""
    if path.is_dir():
        summary_path = path / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found under {path}")
        return summary_path
    if path.is_file():
        return path
    raise FileNotFoundError(f"Run path not found: {path}")


def load_summary(label: str, path: Path) -> Dict[str, Any]:
    """Load a summary JSON file with label and source metadata."""
    summary_path = resolve_summary_path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["comparison_label"] = label
    payload["comparison_source"] = str(summary_path.resolve())
    return payload


# =============================================================================
# Metric Computation
# =============================================================================


def to_float(value: Any) -> float | None:
    """Convert a value to float, returning None for invalid values."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def fmt_value(value: float | None, *, percent: bool) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "N/A"
    if percent:
        return f"{value * 100.0:.2f}%"
    return f"{value:.6f}"


def metric_winners(values: Dict[str, float], higher_is_better: bool) -> List[str]:
    """Determine the winning label(s) for a metric."""
    if not values:
        return []
    best_value = max(values.values()) if higher_is_better else min(values.values())
    tolerance = max(1e-12, abs(best_value) * 1e-9)
    winners = [label for label, value in values.items() if abs(value - best_value) <= tolerance]
    return sorted(winners)


def normalized_scores(values: Dict[str, float], higher_is_better: bool) -> Dict[str, float]:
    """Normalize metric values to [0, 1] range."""
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if math.isclose(min_value, max_value, rel_tol=1e-12, abs_tol=1e-12):
        return {label: 1.0 for label in values}

    result: Dict[str, float] = {}
    span = max_value - min_value
    for label, value in values.items():
        if higher_is_better:
            result[label] = (value - min_value) / span
        else:
            result[label] = (max_value - value) / span
    return result


# =============================================================================
# Run Comparison
# =============================================================================


def compare_runs(run_summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple evaluation runs and compute rankings.

    Returns a dictionary containing:
    - runs: list of run metadata
    - included_metrics: metrics used in overall score
    - metric_rows: per-metric comparison data
    - ranking_rows: overall rankings
    - best_label: label of best-performing run
    - best_score: overall score of best run
    """
    runs = list(run_summaries)
    if len(runs) < 2:
        raise ValueError("At least two runs are required for comparison.")

    run_labels = [run["comparison_label"] for run in runs]
    metric_rows: List[Dict[str, Any]] = []
    overall_scores = {label: 0.0 for label in run_labels}
    metrics_won = {label: 0 for label in run_labels}
    weighted_wins = {label: 0.0 for label in run_labels}
    included_metrics: List[str] = []

    for spec in METRICS:
        available_values = {
            run["comparison_label"]: numeric
            for run in runs
            if (numeric := to_float(run.get(spec.key))) is not None
        }

        row: Dict[str, Any] = {
            "metric_key": spec.key,
            "metric": spec.label,
            "direction": "higher_better" if spec.higher_is_better else "lower_better",
        }
        for run in runs:
            label = run["comparison_label"]
            row[label] = to_float(run.get(spec.key))

        winners = metric_winners(available_values, spec.higher_is_better)
        row["winner"] = ", ".join(winners) if winners else "N/A"
        row["winner_count"] = len(winners)
        metric_rows.append(row)

        for label in winners:
            metrics_won[label] += 1
            weighted_wins[label] += spec.weight

        if spec.include_in_overall and available_values:
            included_metrics.append(spec.key)
            scores = normalized_scores(available_values, spec.higher_is_better)
            for label, score in scores.items():
                overall_scores[label] += score * spec.weight

    ranking_rows = [
        {
            "label": label,
            "overall_score": overall_scores[label],
            "metrics_won": metrics_won[label],
            "weighted_wins": weighted_wins[label],
        }
        for label in run_labels
    ]
    ranking_rows.sort(
        key=lambda row: (row["overall_score"], row["weighted_wins"], row["metrics_won"]),
        reverse=True,
    )
    for index, row in enumerate(ranking_rows, start=1):
        row["rank"] = index

    best_row = ranking_rows[0]
    result = {
        "runs": [
            {
                "label": run["comparison_label"],
                "source": run["comparison_source"],
            }
            for run in runs
        ],
        "included_metrics": included_metrics,
        "metric_rows": metric_rows,
        "ranking_rows": ranking_rows,
        "best_label": best_row["label"],
        "best_score": best_row["overall_score"],
    }
    return result


# =============================================================================
# Report Generation
# =============================================================================


def metric_table(metric_rows: List[Dict[str, Any]], run_labels: List[str]) -> pd.DataFrame:
    """Generate a DataFrame for metric comparison."""
    rows = []
    for row in metric_rows:
        formatted = {
            "metric": row["metric"],
            "direction": row["direction"],
            "winner": row["winner"],
        }
        spec = next(spec for spec in METRICS if spec.key == row["metric_key"])
        for label in run_labels:
            formatted[label] = fmt_value(row.get(label), percent=spec.display_as_percent)
        rows.append(formatted)
    return pd.DataFrame(rows)


def ranking_table(ranking_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate a DataFrame for overall rankings."""
    return pd.DataFrame(
        [
            {
                "rank": row["rank"],
                "label": row["label"],
                "overall_score": round(row["overall_score"], 6),
                "metrics_won": row["metrics_won"],
                "weighted_wins": round(row["weighted_wins"], 2),
            }
            for row in ranking_rows
        ]
    )


def write_markdown_report(
    *,
    path: Path,
    comparison: Dict[str, Any],
    metrics_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> None:
    """Write a markdown comparison report."""
    best_label = comparison["best_label"]
    lines = [
        "# 2D Evaluation Comparison",
        "",
        "## Compared Runs",
        "",
    ]
    for run in comparison["runs"]:
        lines.append(f"- `{run['label']}`: `{run['source']}`")

    lines.extend(
        [
            "",
            "## Overall Ranking",
            "",
            ranking_df.to_markdown(index=False),
            "",
            "## Metric-by-Metric Winners",
            "",
            metrics_df.to_markdown(index=False),
            "",
            "## Conclusion",
            "",
            f"- Overall best: `{best_label}`",
            f"- Included metrics in overall score: {', '.join(comparison['included_metrics'])}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
