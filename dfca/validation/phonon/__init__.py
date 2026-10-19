"""
DFCA Validation Phonon Module.

Functions for phonon dispersion calculations and stability analysis.
"""

from dfca.validation.phonon.dispersion import (
    compute_phonon_dispersion_gpaw,
    get_adaptive_supercell,
)
from dfca.validation.phonon.branches import identify_phonon_branches
from dfca.validation.phonon.stability import (
    detect_za_rotational_rescue_candidate,
    _try_huang_za_rescue,
    _evaluate_phonon_stability_spectrum,
)
from dfca.validation.phonon.plotting import (
    plot_phonon_dispersion,
    plot_phonon_stability_summary,
    plot_thermal_properties,
)

__all__ = [
    'compute_phonon_dispersion_gpaw',
    'get_adaptive_supercell',
    'identify_phonon_branches',
    'detect_za_rotational_rescue_candidate',
    '_try_huang_za_rescue',
    '_evaluate_phonon_stability_spectrum',
    'plot_phonon_dispersion',
    'plot_phonon_stability_summary',
    'plot_thermal_properties',
]
