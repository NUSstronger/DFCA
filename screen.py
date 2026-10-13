#!/usr/bin/env python
"""
Screening Pipeline CLI.

Usage:
    python scripts/screen.py --config configs/screening.yaml

Configuration priority (lowest to highest):
    1. ScreeningConfig dataclass defaults
    2. YAML config file (if --config supplied)
    3. Explicitly supplied CLI arguments

This is a thin entry point that invokes the authoritative ScreeningPipeline.
Scientific calculations are implemented in dfca/screening/.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfca.screening.config import ScreeningConfig
from dfca.screening.pipeline import ScreeningPipeline


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="DFCA 2D Materials Screening Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =================================================================
    # I/O Arguments
    # =================================================================
    p.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory containing CIF files",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for screening results",
    )
    p.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML config file (overrides defaults, CLI overrides YAML)",
    )
    p.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Path to training CSV for novelty filtering",
    )

    # =================================================================
    # Device & ML
    # =================================================================
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for ML relaxation",
    )
    p.add_argument(
        "--mace_model",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        help="MACE model size",
    )

    # =================================================================
    # Geometry (S1)
    # =================================================================
    p.add_argument(
        "--vacuum_min",
        type=float,
        default=None,
        help="Minimum vacuum gap (Å)",
    )
    p.add_argument(
        "--thickness_max",
        type=float,
        default=None,
        help="Maximum slab thickness (Å)",
    )
    p.add_argument(
        "--min_bond",
        type=float,
        default=None,
        help="Minimum bond distance (Å)",
    )
    p.add_argument(
        "--lattice_min",
        type=float,
        default=None,
        help="Minimum lattice a/b parameter (Å)",
    )
    p.add_argument(
        "--lattice_max",
        type=float,
        default=None,
        help="Maximum lattice a/b parameter (Å)",
    )

    # =================================================================
    # Symmetry (S2)
    # =================================================================
    p.add_argument(
        "--ltol",
        type=float,
        default=None,
        help="Lattice tolerance for StructureMatcher",
    )
    p.add_argument(
        "--stol",
        type=float,
        default=None,
        help="Structure tolerance for StructureMatcher",
    )
    p.add_argument(
        "--angle_tol",
        type=float,
        default=None,
        help="Angle tolerance for StructureMatcher (degrees)",
    )
    p.add_argument(
        "--symprec",
        type=float,
        default=None,
        help="Symmetry detection precision (Å)",
    )

    # =================================================================
    # Relaxation (S3.5)
    # =================================================================
    p.add_argument(
        "--relax_fmax",
        type=float,
        default=None,
        help="MACE relaxation convergence (eV/Å)",
    )
    p.add_argument(
        "--relax_steps",
        type=int,
        default=None,
        help="Maximum relaxation steps",
    )
    p.add_argument(
        "--relax_steps_retry",
        type=int,
        default=None,
        help="Steps for retry relaxation",
    )
    p.add_argument(
        "--no_fix_z",
        action="store_true",
        help="Do not fix c-axis during relaxation",
    )

    # =================================================================
    # Elastic Matrix (S4)
    # =================================================================
    p.add_argument(
        "--elastic_delta",
        type=float,
        default=None,
        help="Strain amplitude for elastic matrix calculation",
    )

    # =================================================================
    # Phonon (S5)
    # =================================================================
    p.add_argument(
        "--skip_phonon",
        action="store_true",
        help="Skip phonon stability check (S5)",
    )
    p.add_argument(
        "--phonon_sc",
        type=str,
        default=None,
        help="Phonon supercell size (e.g., 3,3,1)",
    )
    p.add_argument(
        "--phonon_disp",
        type=float,
        default=None,
        help="Phonon displacement (Å)",
    )
    p.add_argument(
        "--phonon_mesh",
        type=str,
        default=None,
        help="Phonon q-point mesh (e.g., 20,20,1)",
    )
    p.add_argument(
        "--imaginary_tol",
        type=float,
        default=None,
        help="Phonon imaginary frequency tolerance (THz)",
    )

    # =================================================================
    # Convex Hull (S5')
    # =================================================================
    p.add_argument(
        "--ehull_thr",
        type=float,
        default=None,
        help="E_hull threshold for thermodynamic stability (eV/atom)",
    )
    p.add_argument(
        "--no_ehull_gate",
        action="store_true",
        help="Disable strict E_hull gate (C2DB default)",
    )
    p.add_argument(
        "--mp_api_key",
        type=str,
        default=None,
        help="Materials Project API key for convex hull",
    )

    # =================================================================
    # MD (S7)
    # =================================================================
    p.add_argument(
        "--enable_md",
        action="store_true",
        help="Enable MD screening (S7a/S7b)",
    )
    p.add_argument(
        "--md_temp",
        type=float,
        default=None,
        help="MD temperature (K)",
    )
    p.add_argument(
        "--md_dt",
        type=float,
        default=None,
        help="MD timestep (fs)",
    )
    p.add_argument(
        "--s7a_steps",
        type=int,
        default=None,
        help="S7a MD steps (short, ~3ps)",
    )
    p.add_argument(
        "--s7b_steps",
        type=int,
        default=None,
        help="S7b MD steps (long, ~10ps)",
    )

    # =================================================================
    # Pipeline Options
    # =================================================================
    p.add_argument(
        "--reject_p1",
        action="store_true",
        help="Reject Phase 1 candidates",
    )
    p.add_argument(
        "--no_export_mag",
        action="store_true",
        help="Disable VASP MAGMOM bucket export",
    )
    p.add_argument(
        "--vasp_vacuum",
        type=float,
        default=None,
        help="VASP export vacuum (Å)",
    )
    p.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Checkpoint save interval",
    )
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from checkpoint",
    )

    # =================================================================
    # Logging
    # =================================================================
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Increase verbosity",
    )

    return p.parse_args()


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def _safe_get_nested(cfg: Dict, *keys, default=None) -> Any:
    """Safely get nested dictionary value."""
    current = cfg
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current


def apply_config_to_screening(
    cfg: ScreeningConfig,
    yaml_config: Dict[str, Any],
    args: argparse.Namespace,
) -> ScreeningConfig:
    """
    Apply configuration with correct priority:
        1. ScreeningConfig defaults (lowest)
        2. YAML config (overrides defaults)
        3. CLI arguments (highest priority)

    Args:
        cfg: Base configuration from defaults
        yaml_config: Configuration loaded from YAML
        args: Parsed command line arguments

    Returns:
        Updated ScreeningConfig with merged settings
    """
    # =================================================================
    # Top-level YAML mappings
    # =================================================================
    if _safe_get_nested(yaml_config, 'input_dir') is not None:
        cfg.input_dir = _safe_get_nested(yaml_config, 'input_dir')
    if _safe_get_nested(yaml_config, 'output_dir') is not None:
        cfg.output_dir = _safe_get_nested(yaml_config, 'output_dir')
    if _safe_get_nested(yaml_config, 'train_csv') is not None:
        cfg.train_csv = _safe_get_nested(yaml_config, 'train_csv')
    if _safe_get_nested(yaml_config, 'device') is not None:
        cfg.device = _safe_get_nested(yaml_config, 'device')
    if _safe_get_nested(yaml_config, 'mace_model') is not None:
        cfg.mace_model = _safe_get_nested(yaml_config, 'mace_model')

    # =================================================================
    # Geometry YAML mappings
    # =================================================================
    geometry = yaml_config.get('geometry', {})
    if geometry:
        if 'vacuum_min_angstrom' in geometry:
            cfg.vacuum_min_angstrom = geometry['vacuum_min_angstrom']
        if 'slab_thickness_max' in geometry:
            cfg.slab_thickness_max = geometry['slab_thickness_max']
        if 'min_bond_distance' in geometry:
            cfg.min_bond_distance = geometry['min_bond_distance']
        if 'lattice_ab_min' in geometry:
            cfg.lattice_ab_min = geometry['lattice_ab_min']
        if 'lattice_ab_max' in geometry:
            cfg.lattice_ab_max = geometry['lattice_ab_max']

    # =================================================================
    # Symmetry YAML mappings
    # =================================================================
    symmetry = yaml_config.get('symmetry', {})
    if symmetry:
        if 'ltol' in symmetry:
            cfg.ltol = symmetry['ltol']
        if 'stol' in symmetry:
            cfg.stol = symmetry['stol']
        if 'angle_tol' in symmetry:
            cfg.angle_tol = symmetry['angle_tol']
        if 'symprec' in symmetry:
            cfg.symprec = symmetry['symprec']

    # =================================================================
    # Relaxation YAML mappings
    # =================================================================
    relaxation = yaml_config.get('relaxation', {})
    if relaxation:
        if 'relax_fmax' in relaxation:
            cfg.relax_fmax = relaxation['relax_fmax']
        if 'relax_steps' in relaxation:
            cfg.relax_steps = relaxation['relax_steps']
        if 'relax_steps_retry' in relaxation:
            cfg.relax_steps_retry = relaxation['relax_steps_retry']
        if 'fix_z_lattice' in relaxation:
            cfg.fix_z_lattice = relaxation['fix_z_lattice']

    # =================================================================
    # Elastic YAML mappings
    # =================================================================
    elastic = yaml_config.get('elastic', {})
    if elastic:
        if 'elastic_delta' in elastic:
            cfg.elastic_delta = elastic['elastic_delta']

    # =================================================================
    # Phonon YAML mappings
    # =================================================================
    phonon = yaml_config.get('phonon', {})
    if phonon:
        if 'skip_phonon' in phonon:
            cfg.skip_phonon = phonon['skip_phonon']
        if 'supercell' in phonon:
            cfg.phonon_supercell = list(phonon['supercell'])
        if 'displacement' in phonon:
            cfg.phonon_displacement = phonon['displacement']
        if 'mesh' in phonon:
            cfg.phonon_mesh = list(phonon['mesh'])
        if 'imaginary_freq_tol' in phonon:
            cfg.imaginary_freq_tol = phonon['imaginary_freq_tol']

    # =================================================================
    # Ehull YAML mappings
    # =================================================================
    ehull = yaml_config.get('ehull', {})
    if ehull:
        if 'threshold_ml' in ehull:
            cfg.ehull_threshold_ml = ehull['threshold_ml']
        if 'enable_hard_gate' in ehull:
            cfg.enable_hard_ehull_gate = ehull['enable_hard_gate']
        if 'mp_api_key' in ehull:
            cfg.mp_api_key = ehull['mp_api_key']

    # =================================================================
    # MD YAML mappings
    # =================================================================
    md = yaml_config.get('md', {})
    if md:
        if 'enable_md' in md:
            cfg.enable_md = md['enable_md']
        if 'temperature' in md:
            cfg.md_temperature = md['temperature']
        if 'dt_fs' in md:
            cfg.md_dt_fs = md['dt_fs']
        if 's7a_steps' in md:
            cfg.md_s7a_steps = md['s7a_steps']
        if 's7b_steps' in md:
            cfg.md_s7b_steps = md['s7b_steps']

    # =================================================================
    # Pipeline YAML mappings
    # =================================================================
    pipeline = yaml_config.get('pipeline', {})
    if pipeline:
        if 'reject_p1' in pipeline:
            cfg.reject_p1 = pipeline['reject_p1']
        if 'verbose' in pipeline:
            cfg.verbose = pipeline['verbose']
        if 'export_mag_buckets' in pipeline:
            cfg.export_mag_buckets = pipeline['export_mag_buckets']
        if 'save_interval' in pipeline:
            cfg.save_interval = pipeline['save_interval']
        if 'vasp_vacuum_angstrom' in pipeline:
            cfg.vasp_vacuum_angstrom = pipeline['vasp_vacuum_angstrom']
        if 'pareto_plot' in pipeline:
            cfg.pareto_plot = pipeline['pareto_plot']
        if 'pareto_plot_top_n' in pipeline:
            cfg.pareto_plot_top_n = pipeline['pareto_plot_top_n']
        if 'reference_energies' in pipeline:
            cfg.reference_energies = pipeline['reference_energies']

    # =================================================================
    # Phase2 YAML mappings
    # =================================================================
    phase2 = yaml_config.get('phase2', {})
    if phase2:
        if 'rerun_phase2' in phase2:
            cfg.rerun_phase2 = phase2['rerun_phase2']
        if 'force_rerun_all' in phase2:
            cfg.force_rerun_all = phase2['force_rerun_all']

    # =================================================================
    # CLI overrides (highest priority)
    # =================================================================
    if args.input is not None:
        cfg.input_dir = args.input
    if args.output is not None:
        cfg.output_dir = args.output
    if args.train_csv is not None:
        cfg.train_csv = args.train_csv
    if args.device is not None:
        cfg.device = args.device
    if args.mace_model is not None:
        cfg.mace_model = args.mace_model

    # Geometry CLI
    if args.vacuum_min is not None:
        cfg.vacuum_min_angstrom = args.vacuum_min
    if args.thickness_max is not None:
        cfg.slab_thickness_max = args.thickness_max
    if args.min_bond is not None:
        cfg.min_bond_distance = args.min_bond
    if args.lattice_min is not None:
        cfg.lattice_ab_min = args.lattice_min
    if args.lattice_max is not None:
        cfg.lattice_ab_max = args.lattice_max

    # Symmetry CLI
    if args.ltol is not None:
        cfg.ltol = args.ltol
    if args.stol is not None:
        cfg.stol = args.stol
    if args.angle_tol is not None:
        cfg.angle_tol = args.angle_tol
    if args.symprec is not None:
        cfg.symprec = args.symprec

    # Relaxation CLI
    if args.relax_fmax is not None:
        cfg.relax_fmax = args.relax_fmax
    if args.relax_steps is not None:
        cfg.relax_steps = args.relax_steps
    if args.relax_steps_retry is not None:
        cfg.relax_steps_retry = args.relax_steps_retry
    if args.no_fix_z:
        cfg.fix_z_lattice = False

    # Elastic CLI
    if args.elastic_delta is not None:
        cfg.elastic_delta = args.elastic_delta

    # Phonon CLI
    if args.skip_phonon:
        cfg.skip_phonon = True
    if args.phonon_sc is not None:
        try:
            cfg.phonon_supercell = [int(x) for x in args.phonon_sc.split(",")]
        except ValueError:
            pass  # Keep YAML value
    if args.phonon_disp is not None:
        cfg.phonon_displacement = args.phonon_disp
    if args.phonon_mesh is not None:
        try:
            cfg.phonon_mesh = [int(x) for x in args.phonon_mesh.split(",")]
        except ValueError:
            pass  # Keep YAML value
    if args.imaginary_tol is not None:
        cfg.imaginary_freq_tol = args.imaginary_tol

    # Ehull CLI
    if args.ehull_thr is not None:
        cfg.ehull_threshold_ml = args.ehull_thr
    if args.no_ehull_gate:
        cfg.enable_hard_ehull_gate = False
    if args.mp_api_key is not None:
        cfg.mp_api_key = args.mp_api_key

    # MD CLI
    if args.enable_md:
        cfg.enable_md = True
    if args.md_temp is not None:
        cfg.md_temperature = args.md_temp
    if args.md_dt is not None:
        cfg.md_dt_fs = args.md_dt
    if args.s7a_steps is not None:
        cfg.md_s7a_steps = args.s7a_steps
    if args.s7b_steps is not None:
        cfg.md_s7b_steps = args.s7b_steps

    # Pipeline CLI
    if args.reject_p1:
        cfg.reject_p1 = True
    if args.no_export_mag:
        cfg.export_mag_buckets = False
    if args.vasp_vacuum is not None:
        cfg.vasp_vacuum_angstrom = args.vasp_vacuum
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval

    return cfg


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Start with defaults
    cfg = ScreeningConfig()

    # Load YAML config if provided
    yaml_config = {}
    if args.config is not None:
        try:
            yaml_config = load_yaml_config(args.config)
            logger.info(f"Loaded config from: {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    else:
        logger.debug("No YAML config provided, using defaults")

    # Apply configuration with correct priority
    cfg = apply_config_to_screening(cfg, yaml_config, args)

    # Create and run pipeline
    logger.info("Initializing screening pipeline...")
    logger.info(f"  Input: {cfg.input_dir}")
    logger.info(f"  Output: {cfg.output_dir}")
    logger.info(f"  Device: {cfg.device}")
    logger.info(f"  MACE model: {cfg.mace_model}")

    pipe = ScreeningPipeline(cfg)

    if not args.no_resume:
        pipe._load_checkpoint()

    logger.info("Running screening pipeline...")
    pipe.run()

    logger.info("Screening complete.")
    logger.info(f"Results: {cfg.output_dir}")


if __name__ == "__main__":
    main()
