"""
Compare multiple 2D evaluation runs and generate reports.

Usage:
    python compare_2d_runs.py \
        --run run1=/path/to/eval1 \
        --run run2=/path/to/eval2 \
        --output_dir ./comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval_2d_compare_utils import (
    compare_runs,
    ensure_dir,
    load_summary,
    metric_table,
    parse_run_arg,
    ranking_table,
    write_markdown_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple 2D evaluation results and output metric winners plus an overall ranking."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run label plus evaluation directory or summary.json path. Repeat this argument for each run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for comparison outputs.",
    )
    args = parser.parse_args()

    parsed_runs = [parse_run_arg(text) for text in args.run]
    run_summaries = [load_summary(label, path) for label, path in parsed_runs]
    comparison = compare_runs(run_summaries)

    output_dir = ensure_dir(args.output_dir)
    run_labels = [label for label, _ in parsed_runs]
    metrics_df = metric_table(comparison["metric_rows"], run_labels)
    ranking_df = ranking_table(comparison["ranking_rows"])

    metrics_df.to_csv(output_dir / "metric_comparison.csv", index=False)
    ranking_df.to_csv(output_dir / "overall_ranking.csv", index=False)
    write_markdown_report(
        path=output_dir / "comparison_report.md",
        comparison=comparison,
        metrics_df=metrics_df,
        ranking_df=ranking_df,
    )

    summary_payload = {
        "best_label": comparison["best_label"],
        "best_score": comparison["best_score"],
        "runs": comparison["runs"],
        "included_metrics": comparison["included_metrics"],
        "ranking_rows": comparison["ranking_rows"],
        "metric_rows": comparison["metric_rows"],
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    print("[compare]")
    print(f"  best_label   : {comparison['best_label']}")
    print(f"  output_dir   : {output_dir.resolve()}")
    print(f"  metric_table : {(output_dir / 'metric_comparison.csv').resolve()}")
    print(f"  ranking      : {(output_dir / 'overall_ranking.csv').resolve()}")


if __name__ == "__main__":
    main()
