"""
DFCA CrystalDiT Model Module.

CrystalDiT: Transformer-based model for 2D crystal structure generation.
This module implements the crystal-specific embedding and DiT architecture.

IMPORTANT: This module preserves the exact original implementation from crystal_dit_phy.py
for checkpoint compatibility. Any changes must maintain state_dict key compatibility.

Scientific Contribution:
    - CrystalEmbedder: Lattice and atom features → hidden space
    - CrystalDiT: Full model with XY/Z anisotropic treatment
    - TimestepEmbedder: Sinusoidal timestep embeddings

Model Architecture:
    1. CrystalEmbedder: Separate XY (linear) and Z (MLP) embeddings + type/position embeddings
    2. TimestepEmbedder: Timestep → vector
    3. DiT Blocks: Dual-stream attention with cross-attention
    4. FinalLayer: Separate atom and lattice output projections

Ablation Switches:
    - use_fusion: Controls MLP fusion layer
    - cross_attn_mode: Controls cross-attention mode
"""

import torch
import torch.nn as nn
import numpy as np
import math

from diffusion.model import modulate, DiTBlock, TimestepEmbedder


# Custom 1D positional encoding function
def get_1d_sincos_pos_embed(embed_dim, length):
    """Generate 1D sinusoidal positional embeddings"""
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    positions = np.arange(length)
    dim_t = np.arange(embed_dim // 2, dtype=np.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / embed_dim)

    pos_x = positions[:, np.newaxis] / dim_t
    pos_embed = np.zeros((length, embed_dim))

    pos_embed[:, 0::2] = np.sin(pos_x)
    pos_embed[:, 1::2] = np.cos(pos_x)

    return pos_embed


class CrystalEmbedder(nn.Module):
    """Embed crystal structure (lattice vectors and atom features) into hidden space"""

    def __init__(self, hidden_size, max_atoms):
        super().__init__()
        self.max_atoms = max_atoms

        self.hidden_size = hidden_size

        self.lattice_embedder_xy = nn.Linear(2, hidden_size // 2)
        self.lattice_embedder_z = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 2)
        )

        # Atom feature embedder (period, group, x, y, z)
        self.atom_embedder_xy = nn.Linear(4, hidden_size // 2)
        self.atom_embedder_z = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 2)
        )

        # Type embeddings to distinguish lattice and atom tokens
        self.lattice_type_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        self.atom_type_embedding = nn.Parameter(torch.zeros(1, hidden_size))

        # Positional embeddings for lattice vectors
        self.lattice_pos_embed = nn.Parameter(torch.zeros(1, 3, hidden_size), requires_grad=False)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.lattice_embedder_xy.weight)
        nn.init.constant_(self.lattice_embedder_xy.bias, 0)
        nn.init.xavier_uniform_(self.lattice_embedder_z[0].weight)
        nn.init.constant_(self.lattice_embedder_z[0].bias, 0)
        nn.init.xavier_uniform_(self.lattice_embedder_z[2].weight)
        nn.init.constant_(self.lattice_embedder_z[2].bias, 0)
        nn.init.xavier_uniform_(self.atom_embedder_xy.weight)
        nn.init.constant_(self.atom_embedder_xy.bias, 0)
        nn.init.xavier_uniform_(self.atom_embedder_z[0].weight)
        nn.init.constant_(self.atom_embedder_z[0].bias, 0)
        nn.init.xavier_uniform_(self.atom_embedder_z[2].weight)
        nn.init.constant_(self.atom_embedder_z[2].bias, 0)

        # Initialize lattice positional embeddings with sinusoidal encoding
        pos_embed = get_1d_sincos_pos_embed(self.lattice_pos_embed.shape[-1], 3)
        self.lattice_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.lattice_type_embedding, std=0.02)
        nn.init.normal_(self.atom_type_embedding, std=0.02)

    def forward(self, lattice_vectors, atom_features):
        """
        Args:
            lattice_vectors: [batch_size, 3, 3] - lattice vectors
            atom_features: [batch_size, max_atoms, 5] - atom features (period, group, x, y, z)
        Returns:
            lattice_emb: [batch_size, 3, hidden_size] - lattice embeddings
            atom_emb: [batch_size, max_atoms, hidden_size] - atom embeddings
        """
        lattice_xy = lattice_vectors[..., :2]
        lattice_z = lattice_vectors[..., 2:3]
        lattice_emb = torch.cat([
            self.lattice_embedder_xy(lattice_xy),
            self.lattice_embedder_z(lattice_z)
        ], dim=-1)
        lattice_emb = lattice_emb + self.lattice_pos_embed + self.lattice_type_embedding

        atom_z_inputs = torch.cat([atom_features[..., :2], atom_features[..., 4:5]], dim=-1)
        atom_emb = torch.cat([
            self.atom_embedder_xy(atom_features[..., :4]),
            self.atom_embedder_z(atom_z_inputs)
        ], dim=-1)
        atom_emb = atom_emb + self.atom_type_embedding

        return lattice_emb, atom_emb


class SimpleDiTBlock(nn.Module):
    """
    DiT block that concatenates atom and lattice features for self-attention.

    Args:
        use_fusion: enable MLP fusion layers in DiT blocks (default: True)
        cross_attn_mode: cross-attention mode ('bidirectional' | 'xy_to_z' | 'none')
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, max_atoms=20,
                 use_fusion=True, cross_attn_mode='bidirectional'):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_atoms = max_atoms
        self.dit_block = DiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio,
            use_fusion=use_fusion, cross_attn_mode=cross_attn_mode
        )

    def forward(self, atom_features, lattice_features, c, padding_mask=None, xy_coords=None, lattice_xy_basis=None):
        """
        Args:
            atom_features: [batch_size, max_atoms, hidden_size]
            lattice_features: [batch_size, 3, hidden_size]
            c: [batch_size, hidden_size] - timestep conditioning
            xy_coords: [batch_size, max_atoms+3, 2] - xy coords for relative position bias
            lattice_xy_basis: [batch_size, 2, 2] - lattice xy basis vectors
        Returns:
            Updated atom and lattice features
        """
        # Mask key direction only (query must still attend to padding tokens).
        # Shape [B, 1, 1, N] broadcasts to [B, H, N_q, N_k] affecting only the key dimension.
        B = atom_features.shape[0]
        if padding_mask is not None:
            lat = torch.zeros(B, 3, dtype=torch.bool, device=atom_features.device)
            full_pad = torch.cat([padding_mask, lat], dim=1)  # [B, N]
            attn_mask = torch.zeros(B, 1, 1, self.max_atoms + 3, device=atom_features.device)
            attn_mask = attn_mask.masked_fill(full_pad.unsqueeze(1).unsqueeze(2), float('-inf'))
        else:
            attn_mask = None

        # Concatenate features
        combined_features = torch.cat([atom_features, lattice_features], dim=1)

        # Apply DiT block with xy_coords for relative position bias
        combined_features = self.dit_block(combined_features, c, attn_mask=attn_mask, xy_coords=xy_coords, lattice_xy_basis=lattice_xy_basis)

        # Split back
        atom_features_out = combined_features[:, :self.max_atoms, :]
        lattice_features_out = combined_features[:, self.max_atoms:, :]

        return atom_features_out, lattice_features_out


class FinalLayer(nn.Module):
    """Final layer for DiT model output"""

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final_lattice = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final_atom = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.lattice_linear = nn.Linear(hidden_size, 3, bias=True)
        self.atom_linear = nn.Linear(hidden_size, 5, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def forward(self, atom_features, lattice_features, c):
        shift_atom, scale_atom, shift_lattice, scale_lattice = self.adaLN_modulation(c).chunk(4, dim=1)

        atom_features = modulate(self.norm_final_atom(atom_features), shift_atom, scale_atom)
        lattice_features = modulate(self.norm_final_lattice(lattice_features), shift_lattice, scale_lattice)

        lattice_out = self.lattice_linear(lattice_features)
        atom_out = self.atom_linear(atom_features)

        return lattice_out, atom_out


class CrystalDiT(nn.Module):
    """
    Crystal DiT model using concatenated self-attention over atom and lattice tokens.

    Conditioning:
        use_fusion: enable MLP fusion layers in DiT blocks (default: True)
        cross_attn_mode: cross-attention routing ('bidirectional' | 'xy_to_z' | 'none')
    """

    def __init__(
        self,
        max_atoms=20,
        hidden_size=512,
        depth=18,
        num_heads=8,
        mlp_ratio=4.0,
        use_fusion=True,
        cross_attn_mode='bidirectional',
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_fusion = use_fusion
        self.cross_attn_mode = cross_attn_mode

        self.crystal_embedder = CrystalEmbedder(hidden_size, max_atoms)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            SimpleDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, max_atoms=max_atoms,
                use_fusion=use_fusion, cross_attn_mode=cross_attn_mode
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize modulation layers to zero
        for block in self.blocks:
            nn.init.constant_(block.dit_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.dit_block.adaLN_modulation[-1].bias, 0)

        # Initialize final layer to zero
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.lattice_linear.weight, 0)
        nn.init.constant_(self.final_layer.lattice_linear.bias, 0)
        nn.init.constant_(self.final_layer.atom_linear.weight, 0)
        nn.init.constant_(self.final_layer.atom_linear.bias, 0)

    def forward(self, lattice_vectors, atom_features, t, padding_mask=None):
        """
        Args:
            lattice_vectors: [batch_size, 3, 3] - lattice vectors
            atom_features: [batch_size, max_atoms, 5] - atom features (period, group, x, y, z)
            t: [batch_size] - diffusion timesteps
        Returns:
            lattice_out: [batch_size, 3, 3] - predicted lattice noise
            atom_out: [batch_size, max_atoms, 5] - predicted atom noise
        """
        lattice_emb, atom_emb = self.crystal_embedder(lattice_vectors, atom_features)
        c = self.t_embedder(t)

        atom_xy = atom_features[:, :, 2:4]
        if padding_mask is not None:
            atom_xy = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(atom_xy), atom_xy)
        lattice_token_xy = torch.zeros(atom_xy.shape[0], 3, 2, device=atom_xy.device, dtype=atom_xy.dtype)
        xy_coords = torch.cat([atom_xy, lattice_token_xy], dim=1)
        lattice_xy_basis = lattice_vectors[:, :2, :2]

        atom_feat = atom_emb
        lattice_feat = lattice_emb

        for block in self.blocks:
            atom_feat, lattice_feat = block(atom_feat, lattice_feat, c, padding_mask=padding_mask, xy_coords=xy_coords, lattice_xy_basis=lattice_xy_basis)

        lattice_out, atom_out = self.final_layer(atom_feat, lattice_feat, c)
        lattice_out = lattice_out.view(-1, 3, 3)

        return lattice_out, atom_out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    'TimestepEmbedder',
    'CrystalEmbedder',
    'SimpleDiTBlock',
    'FinalLayer',
    'CrystalDiT',
    'count_parameters',
]
