#!/usr/bin/env python
"""
DFT Validation CLI.

Usage:
    python scripts/validate_dft.py --input ./screened --output ./validated

This is a thin entry point that invokes the GPAWValidationPipeline.
Requires the HPC environment with GPAW installed.
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="DFCA First-Principles DFT Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing relaxed structures",
    )
    p.add_argument(
        "--output", "-o",
        default="./gpaw_validation",
        help="Output directory for validation results",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=["M1"],
        choices=["M1", "M1+", "M2", "M3", "M4"],
        help="Validation stages to run",
    )
    p.add_argument(
        "--mp_api_key",
        default="",
        help="Materials Project API key (or set MP_API_KEY env var)",
    )
    p.add_argument(
        "--xc",
        default="PBE",
        help="Exchange-correlation functional",
    )
    p.add_argument(
        "--encut",
        type=int,
        default=550,
        help="Plane-wave cutoff energy (eV)",
    )
    p.add_argument(
        "--relax_fmax",
        type=float,
        default=0.01,
        help="Force convergence for relaxation (eV/Å)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for parallel computation",
    )
    p.add_argument(
        "--max_cores",
        type=int,
        default=32,
        help="Maximum cores for parallel computation",
    )
    p.add_argument(
        "--skip_gap_probe",
        action="store_true",
        help="Skip M1+ gap probe",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint",
    )
    return p.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Check GPAW availability
    try:
        from dfca.validation import GPAWValidationPipeline
    except ImportError as e:
        logger.error(f"GPAW validation module not available: {e}")
        logger.error("DFT validation requires the HPC environment with GPAW installed.")
        logger.error("See docs/hpc.md for setup instructions.")
        sys.exit(1)

    # Import GPAW config
    from dfca.validation.config import GPAWConfig

    # Get MP API key from args or environment
    mp_api_key = args.mp_api_key or os.environ.get("MP_API_KEY", "")

    # Build configuration
    cfg = GPAWConfig(
        input_dir=args.input,
        output_dir=args.output,
        xc=args.xc,
        encut=args.encut,
        relax_fmax=args.relax_fmax,
        skip_gap_probe=args.skip_gap_probe,
        parallel={
            'auto_cores': True,
            'max_cores': args.max_cores,
            'mpi': True,
        },
    )

    # Create pipeline
    logger.info("Initializing DFT validation pipeline...")
    pipe = GPAWValidationPipeline(cfg)

    # Run validation
    if hasattr(args, 'resume') and args.resume:
        logger.info("Checkpoint resume enabled")

    try:
        logger.info(f"Running validation stages: {args.stages}")
        pipe.run()

        logger.info("Validation complete.")
        logger.info(f"Results: {args.output}")

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
