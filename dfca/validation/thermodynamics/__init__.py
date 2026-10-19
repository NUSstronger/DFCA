"""
DFCA Validation Thermodynamics Module.

Functions for convex hull stability calculations and phase diagram analysis.
"""

from dfca.validation.thermodynamics.ehull import (
    compute_ehull_stability,
    compute_ehull_stability_v2,
    fetch_candidate_bulk_structures,
    compute_reference_energy_gpaw,
)

__all__ = [
    'compute_ehull_stability',
    'compute_ehull_stability_v2',
    'fetch_candidate_bulk_structures',
    'compute_reference_energy_gpaw',
]
