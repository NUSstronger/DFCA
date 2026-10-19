"""
DFCA Models Module.

Provides transformer-based models for 2D crystal structure generation.

Submodules:
    - attention: Custom attention mechanisms (DualStream, RelativePositionBias)
    - dit: CrystalDiT model architecture
"""

from dfca.models.attention import (
    BondDecoupledMLP,
    RelativePositionBias,
    DualStream_DFCA_Attention,
    modulate,
)

from dfca.models.dit import (
    TimestepEmbedder,
    CrystalEmbedder,
    SimpleDiTBlock,
    FinalLayer,
    CrystalDiT,
)

__all__ = [
    # Attention
    'BondDecoupledMLP',
    'RelativePositionBias',
    'DualStream_DFCA_Attention',
    'modulate',
    # DiT
    'TimestepEmbedder',
    'CrystalEmbedder',
    'SimpleDiTBlock',
    'FinalLayer',
    'CrystalDiT',
]
