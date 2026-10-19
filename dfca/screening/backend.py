"""
ML Backend for screening pipeline.

Wraps MACE-MP as the MLIP engine for 2D materials relaxation.
Uses cfg.mace_model to select model size ('small', 'medium', 'large').
Scientific model family: MACE-MP-0.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MLBackend:
    """
    MACE-MP calculator wrapper for screening pipeline.

    IMPORTANT: MACE-MP-0 is the SOLE MLIP engine used in this pipeline.
    No fallback chain to other potentials.

    Uses cfg.mace_model to select model size:
        - 'small': Smaller/faster model
        - 'medium': Default balanced model (RECOMMENDED)
        - 'large': Larger/more accurate model

    MACE-MP-0 is an atomic-position potential that does NOT explicitly handle
    spin degrees of freedom. The NM/FM distinction only matters for VASP
    INCAR export (MAGMOM values), not for the ML energy calculation itself.
    """

    def __init__(self, device: str = "cpu", model_size: str = "medium"):
        """
        Initialize MACE-MP calculator.

        Args:
            device: Computing device ('cpu' or 'cuda')
            model_size: Model size ('small', 'medium', 'large')
        """
        self.device = device
        self.model_size = model_size
        self.name = f"mace-mp-{model_size}"
        self._calculator = None

    def get_calculator(self):
        """
        Get ASE-compatible MACE-MP calculator.

        Uses cfg.mace_model to select the appropriate MACE-MP variant.

        Returns:
            ASE Calculator instance for MACE-MP
        """
        if self._calculator is None:
            self._calculator = self._load_calculator()
        return self._calculator

    def _load_calculator(self):
        """
        Load MACE-MP calculator from mace.calculators.

        Uses model_size to select the appropriate MACE-MP variant:
            - mace_mp(model="small")
            - mace_mp(model="medium")  # RECOMMENDED
            - mace_mp(model="large")

        All variants are from the MACE-MP-0 scientific model family.
        """
        try:
            from mace.calculators import mace_mp
        except ImportError:
            raise ImportError(
                "MACE not installed. Please install: pip install mace-offential"
            )

        try:
            # Use cfg.mace_model to select the appropriate model size
            # All variants are from the MACE-MP-0 scientific model family
            # mace_mp(model="medium") has been verified to work correctly on server
            calc = mace_mp(
                model=self.model_size,
                device=self.device,
                default_dtype="float32",
            )
            logger.info(f"MACE-MP (model={self.model_size}) loaded on {self.device}")
            return calc
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MACE-MP calculator (model={self.model_size}): {e}"
            )

    def __repr__(self):
        return f"MLBackend(name={self.name}, device={self.device}, model={self.model_size})"


__all__ = ['MLBackend']
