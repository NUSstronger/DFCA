"""
Phonon Dispersion Plotting Module.

Functions for plotting phonon dispersion curves with branch identification
and stability annotations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


def plot_phonon_dispersion(
    frequencies: np.ndarray,
    qpoints: np.ndarray,
    branch_labels: Optional[List[str]] = None,
    stability_mask: Optional[np.ndarray] = None,
    title: str = "Phonon Dispersion",
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
    color_acoustic: str = "tab:blue",
    color_optical: str = "tab:orange",
    color_unstable: str = "tab:red",
    high_symmetry_points: Optional[List[float]] = None,
    high_symmetry_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Plot phonon dispersion curves with branch identification.

    Args:
        frequencies: Array of phonon frequencies (N_qpoints, N_branches) in THz
        qpoints: Array of q-point coordinates (N_qpoints,) or path distances
        branch_labels: Optional labels for each branch (e.g., 'ZA', 'TA', 'LA', 'LO', 'TO')
        stability_mask: Optional boolean array marking stable branches
        title: Plot title
        output_path: If provided, save figure to this path
        show: If True, display the plot interactively
        figsize: Figure size (width, height) in inches
        dpi: Figure resolution
        color_acoustic: Color for acoustic branches
        color_optical: Color for optical branches
        color_unstable: Color for unstable (negative frequency) branches
        high_symmetry_points: q-point indices of high-symmetry points
        high_symmetry_labels: Labels for high-symmetry points

    Returns:
        Dictionary with plot metadata including figure and axis objects

    Example:
        >>> result = plot_phonon_dispersion(
        ...     frequencies=freqs,
        ...     qpoints=qpath,
        ...     branch_labels=['ZA', 'TA', 'LA', 'LO1', 'LO2'],
        ...     stability_mask=stable,
        ...     output_path='phonon_dispersion.png',
        ... )
        >>> fig = result['figure']
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        logger.warning("matplotlib not available, skipping plot")
        return {'error': 'matplotlib_not_available'}

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        n_qpoints, n_branches = frequencies.shape

        # Determine branch types from labels if provided
        if branch_labels is None:
            branch_labels = [f"B{i}" for i in range(n_branches)]

        # Identify acoustic vs optical branches
        acoustic_indices = []
        optical_indices = []
        for i, label in enumerate(branch_labels):
            label_upper = label.upper()
            if label_upper in ('ZA', 'TA', 'LA'):
                acoustic_indices.append(i)
            else:
                optical_indices.append(i)

        # Plot each branch
        for branch_idx in range(n_branches):
            freqs_branch = frequencies[:, branch_idx]
            label = branch_labels[branch_idx]

            # Determine color based on branch type and stability
            if stability_mask is not None and not stability_mask[branch_idx]:
                color = color_unstable
                linestyle = '--'
                alpha = 0.7
            elif branch_idx in acoustic_indices:
                color = color_acoustic
                linestyle = '-'
                alpha = 0.9
            else:
                color = color_optical
                linestyle = '-'
                alpha = 0.7

            ax.plot(
                qpoints,
                freqs_branch,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                label=label if branch_idx < 10 else None,  # Avoid legend clutter
                linewidth=1.5 if branch_idx in acoustic_indices else 1.0,
            )

        # Mark high-symmetry points
        if high_symmetry_points is not None:
            for point in high_symmetry_points:
                ax.axvline(x=qpoints[int(point)], color='gray', linestyle=':', alpha=0.5)

            if high_symmetry_labels is not None:
                ax.set_xticks([qpoints[int(p)] for p in high_symmetry_points])
                ax.set_xticklabels(high_symmetry_labels)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        # Labels and title
        ax.set_xlabel('q-point', fontsize=12)
        ax.set_ylabel('Frequency (THz)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Clean up y-axis ticks
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        # Legend for acoustic modes only (avoid clutter)
        if acoustic_indices:
            acoustic_handles = []
            for idx in acoustic_indices:
                if idx < len(branch_labels):
                    acoustic_handles.append(
                        plt.Line2D([0], [0], color=color_acoustic, label=branch_labels[idx])
                    )
            if acoustic_handles:
                ax.legend(handles=acoustic_handles, loc='upper right', framealpha=0.9)

        # Set y-axis limits with padding
        ymin, ymax = frequencies.min(), frequencies.max()
        y_padding = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
        # Show some negative frequencies for unstable modes
        ax.set_ylim(bottom=min(ymin - y_padding, 0), top=ymax + y_padding)

        plt.tight_layout()

        # Save if requested
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Phonon dispersion plot saved to {output_path}")

        if show:
            plt.show()

        return {
            'figure': fig,
            'axes': ax,
            'frequencies': frequencies,
            'qpoints': qpoints,
            'branch_labels': branch_labels,
            'n_qpoints': n_qpoints,
            'n_branches': n_branches,
        }


def plot_phonon_stability_summary(
    branch_stabilities: Dict[str, bool],
    acoustic_criteria: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Create a summary bar chart of phonon branch stabilities.

    Args:
        branch_stabilities: Dictionary mapping branch names to stability status
        acoustic_criteria: Optional dictionary of acoustic mode criteria values
        output_path: If provided, save figure to this path
        show: If True, display the plot interactively
        figsize: Figure size (width, height) in inches
        dpi: Figure resolution

    Returns:
        Dictionary with plot metadata
    """
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        logger.warning("matplotlib not available, skipping plot")
        return {'error': 'matplotlib_not_available'}

    if not HAS_MATPLOTLIB:
        return {'error': 'matplotlib_not_available'}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    branches = list(branch_stabilities.keys())
    stabilities = list(branch_stabilities.values())

    colors = ['tab:green' if s else 'tab:red' for s in stabilities]

    bars = ax.barh(branches, [1 if s else 0 for s in stabilities], color=colors, alpha=0.7)

    # Add value annotations
    for bar, stable in zip(bars, stabilities):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            'Stable' if stable else 'Unstable',
            ha='left',
            va='center',
            fontsize=10,
        )

    ax.set_xlabel('Stability Status', fontsize=12)
    ax.set_ylabel('Phonon Branch', fontsize=12)
    ax.set_title('Phonon Branch Stability Summary', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Stability summary plot saved to {output_path}")

    if show:
        plt.show()

    return {
        'figure': fig,
        'axes': ax,
        'branch_stabilities': branch_stabilities,
    }


def plot_thermal_properties(
    temperatures: np.ndarray,
    free_energy: np.ndarray,
    entropy: np.ndarray,
    heat_capacity: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Plot phonon thermal properties (free energy, entropy, heat capacity).

    Args:
        temperatures: Array of temperatures in K
        free_energy: Helmholtz free energy in eV
        entropy: Entropy in eV/K
        heat_capacity: Heat capacity in eV/K
        output_path: If provided, save figure to this path
        show: If True, display the plot interactively
        figsize: Figure size (width, height) in inches
        dpi: Figure resolution

    Returns:
        Dictionary with plot metadata
    """
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        logger.warning("matplotlib not available, skipping plot")
        return {'error': 'matplotlib_not_available'}

    if not HAS_MATPLOTLIB:
        return {'error': 'matplotlib_not_available'}

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    # Free energy
    axes[0, 0].plot(temperatures, free_energy, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Temperature (K)')
    axes[0, 0].set_ylabel('Free Energy (eV)')
    axes[0, 0].set_title('Helmholtz Free Energy')
    axes[0, 0].grid(True, alpha=0.3)

    # Entropy
    axes[0, 1].plot(temperatures, entropy, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Temperature (K)')
    axes[0, 1].set_ylabel('Entropy (eV/K)')
    axes[0, 1].set_title('Vibrational Entropy')
    axes[0, 1].grid(True, alpha=0.3)

    # Heat capacity
    axes[1, 0].plot(temperatures, heat_capacity, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Temperature (K)')
    axes[1, 0].set_ylabel('Heat Capacity (eV/K)')
    axes[1, 0].set_title('Vibrational Heat Capacity')
    axes[1, 0].grid(True, alpha=0.3)

    # Combined
    axes[1, 1].plot(temperatures, free_energy, 'b-', linewidth=2, label='Free Energy')
    twin = axes[1, 1].twinx()
    twin.plot(temperatures, entropy, 'r-', linewidth=2, label='Entropy')
    axes[1, 1].set_xlabel('Temperature (K)')
    axes[1, 1].set_ylabel('Free Energy (eV)', color='b')
    twin.set_ylabel('Entropy (eV/K)', color='r')
    axes[1, 1].set_title('Free Energy and Entropy')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Thermal properties plot saved to {output_path}")

    if show:
        plt.show()

    return {
        'figure': fig,
        'axes': axes,
        'temperatures': temperatures,
    }


__all__ = [
    'plot_phonon_dispersion',
    'plot_phonon_stability_summary',
    'plot_thermal_properties',
]
