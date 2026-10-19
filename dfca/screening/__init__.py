"""
DFCA Screening Module.

Two-phase 2D materials screening pipeline following C2DB standards.

Screening Stages:
    S0: Composition check
    S1: Geometry check (vacuum, thickness, bond distance)
    S2: Symmetry check (spacegroup + layer group)
    S3: High-performance deduplication
    S3.5: ML relaxation (FIRE → LBFGS retry)
    S-dim: Dimensionality check (get_dimensionality_larsen == 2)
    S3T: Formation energy粗筛
    S3P: Pareto front within cluster
    S4: Elastic matrix (Born stability)
    S5: Phonon stability
    S5': MP convex hull analysis
    S7a: MD fast screening (3ps)
    S7b: MD full screening (10ps)
"""

from dfca.screening.config import ScreeningConfig
from dfca.screening.record import ScreeningRecord
from dfca.screening.backend import MLBackend
from dfca.screening.pipeline import ScreeningPipeline

__all__ = [
    'ScreeningConfig',
    'ScreeningRecord',
    'MLBackend',
    'ScreeningPipeline',
]
