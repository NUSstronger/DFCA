"""
Run the full 2D evaluation pipeline.

Usage:
    python run_2d_eval.py \
        --gen_dir ./generated_cifs \
        --train_csv ./train.csv \
        --output_dir ./eval_output \
        --mp_hull_path ./mp_hull.pkl
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from eval_2d_core import evaluate_core
from eval_2d_distribution import evaluate_distribution
from eval_2d_utils import (
    ensure_dir,
    merge_summaries,
    paper_row,
    write_csv,
    write_json,
)


def run_evaluation(
    *,
    gen_dir: Path,
    train_csv: Path,
    output_dir: Path,
    label: str | None,
    skip_structure_novelty: bool,
    skip_stability: bool,
    mp_hull_path: Path | None,
    stability_sample_size: int,
    stability_seed: int,
    stability_device: str,
    stability_max_steps: int,
) -> Dict[str, Any]:
    """
    Run the complete 2D evaluation pipeline.

    Pipeline stages:
    1. Core evaluation (validity, uniqueness, novelty)
    2. Distribution evaluation (Wasserstein distances)
    3. Stability evaluation (CHGNet proxy, optional)

    Returns a merged summary dictionary.
    """
    output_dir = ensure_dir(output_dir)

    # Stage 1: Core evaluation
    core_summary, details_df = evaluate_core(
        gen_dir=gen_dir,
        train_csv=train_csv,
        output_dir=output_dir,
        label=label,
        skip_structure_novelty=skip_structure_novelty,
    )

    # Stage 2: Distribution evaluation
    dist_summary = evaluate_distribution(
        details_df=details_df,
        train_csv=train_csv,
        output_dir=output_dir,
        label=label,
    )

    # Merge summaries
    merged = merge_summaries(core_summary, dist_summary)

    # Stage 3: Stability evaluation (optional)
    if not skip_stability and mp_hull_path is not None:
        from eval_2d_stability import evaluate_stability

        stability_summary = evaluate_stability(
            details_csv=output_dir / "core_details.csv",
            mp_hull_path=mp_hull_path,
            output_dir=output_dir,
            label=label,
            sample_size=stability_sample_size,
            seed=stability_seed,
            device=stability_device,
            max_steps=stability_max_steps,
        )
        merged = merge_summaries(merged, stability_summary)

    # Write final outputs
    write_json(output_dir / "summary.json", merged)
    write_csv(output_dir / "paper_table.csv", paper_row(merged["label"], merged))
    shutil.copyfile(output_dir / "core_details.csv", output_dir / "details.csv")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full 2D evaluation pipeline.")
    parser.add_argument("--gen_dir", type=Path, required=True, help="Directory containing generated CIFs.")
    parser.add_argument("--train_csv", type=Path, required=True, help="Training CSV with cif column.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write outputs.")
    parser.add_argument("--label", type=str, default=None, help="Optional run label.")
    parser.add_argument(
        "--skip_structure_novelty",
        action="store_true",
        help="Skip expensive structure-level novelty checks during core evaluation.",
    )
    parser.add_argument(
        "--skip_stability",
        action="store_true",
        help="Skip CHGNet stability proxy even if mp_hull_path is provided.",
    )
    parser.add_argument("--mp_hull_path", type=Path, default=None, help="Pickled PatchedPhaseDiagram for CHGNet proxy.")
    parser.add_argument("--stability_sample_size", type=int, default=250, help="Sample size for CHGNet stability proxy.")
    parser.add_argument("--stability_seed", type=int, default=42, help="Sampling seed for CHGNet stability proxy.")
    parser.add_argument("--stability_device", type=str, default="cpu", help="Device for CHGNet stability proxy.")
    parser.add_argument("--stability_max_steps", type=int, default=1500, help="Maximum CHGNet relax steps.")
    args = parser.parse_args()

    summary = run_evaluation(
        gen_dir=args.gen_dir,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        label=args.label,
        skip_structure_novelty=args.skip_structure_novelty,
        skip_stability=args.skip_stability,
        mp_hull_path=args.mp_hull_path,
        stability_sample_size=args.stability_sample_size,
        stability_seed=args.stability_seed,
        stability_device=args.stability_device,
        stability_max_steps=args.stability_max_steps,
    )
    print(f"[run] {summary['label']}")
    print(f"  2D-UN       : {summary.get('two_d_un_rate')}")
    print(f"  DistScore   : {summary.get('two_d_dist_score')}")


if __name__ == "__main__":
    main()
