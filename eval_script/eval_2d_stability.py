from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from eval_2d_utils import ensure_dir, infer_run_label, paper_row, percentage, write_csv, write_json


@contextlib.contextmanager
def suppress_output() -> Any:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _load_chgnet_components(device: str):
    from chgnet.model import CHGNet, StructOptimizer

    with suppress_output():
        model = CHGNet.load()
        optimizer = StructOptimizer(model=model, use_device=device)
    return optimizer


def _load_phase_diagram(mp_hull_path: Path):
    mpl_config_dir = (Path.cwd() / ".mplconfig").resolve()
    try:
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    except PermissionError:
        pass

    from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram

    with mp_hull_path.open("rb") as handle:
        phase_diagram = pickle.load(handle)
    if not isinstance(phase_diagram, PatchedPhaseDiagram):
        return phase_diagram
    return phase_diagram


def _extract_relaxation_outputs(relax_result: Any, original_structure: Structure) -> tuple[Structure, Optional[float], Optional[int], Optional[bool]]:
    if isinstance(relax_result, dict):
        final_structure = relax_result.get("final_structure", original_structure)
        trajectory = relax_result.get("trajectory", {})
        energies = trajectory.get("energies") if isinstance(trajectory, dict) else None
        final_energy = None
        if isinstance(energies, (list, tuple)) and energies:
            final_energy = float(energies[-1])
        elif isinstance(relax_result.get("final_energy"), (int, float)):
            final_energy = float(relax_result["final_energy"])
        n_steps = len(energies) if isinstance(energies, (list, tuple)) else relax_result.get("n_steps")
        converged = relax_result.get("converged")
        return final_structure, final_energy, n_steps, converged
    return original_structure, None, None, None


def _energy_above_hull_per_atom(phase_diagram: Any, structure: Structure, total_energy: Optional[float]) -> Optional[float]:
    if total_energy is None or len(structure) == 0:
        return None
    try:
        hull_energy = phase_diagram.get_hull_energy_per_atom(structure.composition)
    except Exception:
        return None
    return float(total_energy / len(structure) - hull_energy)


def evaluate_stability(
    *,
    details_csv: Path,
    mp_hull_path: Path,
    output_dir: Path,
    label: Optional[str] = None,
    sample_size: int = 250,
    seed: int = 42,
    max_steps: int = 1500,
    stability_threshold: float = 0.05,
    metastability_threshold: float = 0.10,
    device: str = "cpu",
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    label = infer_run_label(output_dir, label)

    details_df = pd.read_csv(details_csv)
    candidates = details_df[details_df["is_2d_un_main"] == True].copy()
    if candidates.empty:
        summary = {
            "label": label,
            "stability_pool_size": 0,
            "stability_sample_size": 0,
            "stable_at_chgnet_rate": 0.0,
            "metastable_at_chgnet_rate": 0.0,
            "two_d_sun_proxy": 0.0,
            "two_d_msun_proxy": 0.0,
        }
        write_json(output_dir / "stability_summary.json", summary)
        write_csv(output_dir / "stability_paper_table.csv", paper_row(label, summary))
        return summary

    sampled = candidates.sample(n=min(sample_size, len(candidates)), random_state=seed)

    phase_diagram = _load_phase_diagram(mp_hull_path)
    optimizer = _load_chgnet_components(device)
    rows: List[Dict[str, Any]] = []

    for _, row in sampled.iterrows():
        cif_path = Path(row["path"])
        structure = Structure.from_file(cif_path)
        with suppress_output():
            relax_result = optimizer.relax(structure, steps=max_steps)
        final_structure, final_energy, n_steps, converged = _extract_relaxation_outputs(relax_result, structure)
        e_above_hull = _energy_above_hull_per_atom(phase_diagram, final_structure, final_energy)
        is_stable = e_above_hull is not None and e_above_hull <= stability_threshold
        is_metastable = e_above_hull is not None and e_above_hull <= metastability_threshold
        rows.append(
            {
                "file": row["file"],
                "path": row["path"],
                "reduced_formula": row["reduced_formula"],
                "final_energy_total_ev": final_energy,
                "e_above_hull_ev_per_atom": e_above_hull,
                "n_steps": n_steps,
                "converged": converged,
                "stable_at_chgnet": is_stable,
                "metastable_at_chgnet": is_metastable,
            }
        )

    stability_df = pd.DataFrame(rows)
    stable_rate = float(stability_df["stable_at_chgnet"].mean()) if not stability_df.empty else 0.0
    metastable_rate = float(stability_df["metastable_at_chgnet"].mean()) if not stability_df.empty else 0.0
    two_d_un_rate = float(len(candidates) / len(details_df)) if len(details_df) > 0 else 0.0

    summary = {
        "label": label,
        "stability_pool_size": int(len(candidates)),
        "stability_sample_size": int(len(stability_df)),
        "stable_at_chgnet_rate": stable_rate,
        "metastable_at_chgnet_rate": metastable_rate,
        "two_d_sun_proxy": two_d_un_rate * stable_rate,
        "two_d_msun_proxy": two_d_un_rate * metastable_rate,
        "stability_threshold_ev_per_atom": stability_threshold,
        "metastability_threshold_ev_per_atom": metastability_threshold,
    }

    write_json(output_dir / "stability_summary.json", summary)
    write_csv(output_dir / "stability_details.csv", stability_df)
    write_csv(output_dir / "stability_paper_table.csv", paper_row(label, summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CHGNet stability proxy for 2D-UN structures.")
    parser.add_argument("--details_csv", type=Path, required=True, help="core_details.csv from eval_2d_core.py")
    parser.add_argument("--mp_hull_path", type=Path, required=True, help="Pickled PatchedPhaseDiagram object.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write outputs.")
    parser.add_argument("--label", type=str, default=None, help="Optional run label.")
    parser.add_argument("--sample_size", type=int, default=250, help="Sample size drawn from 2D-UN pool.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--max_steps", type=int, default=1500, help="Maximum CHGNet relaxation steps.")
    parser.add_argument("--device", type=str, default="cpu", help="CHGNet device, e.g. cpu or cuda:0.")
    args = parser.parse_args()

    summary = evaluate_stability(
        details_csv=args.details_csv,
        mp_hull_path=args.mp_hull_path,
        output_dir=args.output_dir,
        label=args.label,
        sample_size=args.sample_size,
        seed=args.seed,
        max_steps=args.max_steps,
        device=args.device,
    )
    print(f"[stable] {summary['label']}")
    print(f"  Stable@CHGNet     : {percentage(summary['stable_at_chgnet_rate']):.2f}%")
    print(f"  MetaStable@CHGNet : {percentage(summary['metastable_at_chgnet_rate']):.2f}%")
    print(f"  2D-SUN_proxy      : {percentage(summary['two_d_sun_proxy']):.2f}%")


if __name__ == "__main__":
    main()
