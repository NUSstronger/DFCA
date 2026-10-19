"""
DFCA Model Attention Module.

Contains custom attention mechanisms and MLP layers for the DFCA model.
This module implements the dual-stream attention architecture with XY/Z anisotropic treatment.

Scientific Contribution:
    - DualStream_DFCA_Attention: XY and Z streams with bidirectional cross-attention
    - BondDecoupledMLP: MLP with optional fusion layer for XY/Z streams
    - RelativePositionBias: Distance-based bias for XY plane periodicity

Ablation Switches:
    - use_fusion: Controls whether fusion layer is applied
    - cross_attn_mode: Controls cross-attention mode ('bidirectional', 'xy_to_z', 'none')
"""

import torch
import torch.nn as nn
import math


class BondDecoupledMLP(nn.Module):
    """
    MLP with decoupled XY and Z streams.

    Ablation module A: use_fusion controls the fusion layer.
    - use_fusion=True: xy and z streams interact through a fusion layer (enhanced version)
    - use_fusion=False: xy and z streams are concatenated directly after separate processing (baseline for ablation)
    """
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., use_fusion=True):
        super().__init__()
        self.half_in = in_features // 2
        self.use_fusion = use_fusion

        try:
            from timm.models.vision_transformer import Mlp
            self.mlp_xy = Mlp(in_features=self.half_in, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
            self.mlp_z = Mlp(in_features=self.half_in, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
        except ImportError:
            # Fallback if timm is not available
            hidden_dim = hidden_features or in_features * 4
            self.mlp_xy = nn.Sequential(
                nn.Linear(self.half_in, hidden_dim),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, self.half_in),
                nn.Dropout(drop)
            )
            self.mlp_z = nn.Sequential(
                nn.Linear(self.half_in, hidden_dim),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, self.half_in),
                nn.Dropout(drop)
            )

        # Fusion layer: enables xy and z features to interact after separate processing (controlled by use_fusion).
        if self.use_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(in_features, in_features),
                act_layer(),
                nn.Linear(in_features, in_features),
                nn.Dropout(drop)
            )

    def forward(self, x):
        x_xy, x_z = torch.chunk(x, 2, dim=-1)

        out_xy = self.mlp_xy(x_xy)
        out_z = self.mlp_z(x_z)

        out = torch.cat([out_xy, out_z], dim=-1)

        # Ablation switch: whether to apply the fusion layer.
        if self.use_fusion:
            out = self.fusion(out)
        return out


class RelativePositionBias(nn.Module):
    """Relative-position bias for the XY plane, enhancing in-plane periodicity awareness."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Maps distance to per-head bias terms.
        self.distance_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, num_heads)
        )

    def forward(self, xy_coords, lattice_xy_basis=None):
        """
        Args:
            xy_coords: [B, N, 2] fractional xy coordinates
            lattice_xy_basis: [B, 2, 2] lattice xy basis vectors
        Returns:
            bias: [B, num_heads, N, N]
        """
        if xy_coords is None:
            return None
        B, N, _ = xy_coords.shape
        atom_tokens = max(N - 3, 0)
        bias = torch.zeros(B, N, N, self.num_heads, device=xy_coords.device, dtype=xy_coords.dtype)
        if atom_tokens == 0:
            return bias.permute(0, 3, 1, 2)
        atom_xy = xy_coords[:, :atom_tokens, :]
        diff_frac = atom_xy.unsqueeze(2) - atom_xy.unsqueeze(1)
        diff_frac = diff_frac - torch.round(diff_frac)
        if lattice_xy_basis is not None:
            diff_cart = torch.einsum('bijd,bdk->bijk', diff_frac, lattice_xy_basis)
        else:
            diff_cart = diff_frac
        distances = torch.norm(diff_cart, dim=-1, keepdim=True)
        atom_bias = self.distance_embed(distances)
        bias[:, :atom_tokens, :atom_tokens, :] = atom_bias
        return bias.permute(0, 3, 1, 2)


class DualStream_DFCA_Attention(nn.Module):
    """
    Dual-stream attention with XY/Z anisotropic treatment.

    Ablation module B: cross_attn_mode controls the cross-attention mode.

    Three modes for ablation:
    - 'bidirectional': bidirectional cross-attention (enhanced; z_from_xy is the dominant direction)
    - 'xy_to_z': unidirectional (XY queries Z; baseline)
    - 'none': no cross-attention (pure dual-stream self-attention)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 z_from_xy_init=0.782159, xy_from_z_init=0.1, cross_attn_mode='bidirectional'):
        super().__init__()
        self.half_dim = dim // 2
        self.num_heads_xy = num_heads // 2
        self.num_heads_z = num_heads // 2
        head_dim = (dim // 2) // (num_heads // 2)
        self.scale = head_dim ** -0.5

        # XY-Stream: QKV
        self.qkv_xy = nn.Linear(self.half_dim, self.half_dim * 3, bias=qkv_bias)

        # Z-Stream: QKV
        self.qkv_z = nn.Linear(self.half_dim, self.half_dim * 3, bias=qkv_bias)

        # Relative position bias (XY-Stream only).
        self.rel_pos_bias = RelativePositionBias(self.num_heads_xy)

        # Ablation switch: cross-attention mode.
        self.cross_attn_mode = cross_attn_mode

        # Bidirectional cross-attention gates.
        if cross_attn_mode == 'bidirectional':
            # z_from_xy: Z attends to XY (layer thickness determined by in-plane arrangement) — dominant direction.
            self.gate_z_from_xy = nn.Parameter(torch.tensor(z_from_xy_init))
            # xy_from_z: XY attends to Z — auxiliary direction.
            self.gate_xy_from_z = nn.Parameter(torch.tensor(xy_from_z_init))
        elif cross_attn_mode == 'xy_to_z':
            # Unidirectional baseline: XY queries Z.
            self.cross_attn_scale = nn.Parameter(torch.tensor(0.5))
        # cross_attn_mode == 'none': no cross-attention.

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, xy_coords=None, lattice_xy_basis=None):
        """
        Args:
            x: [B, N, C] input features
            attn_mask: attention mask
            xy_coords: [B, N, 2] fractional xy coordinates for relative position bias
            lattice_xy_basis: [B, 2, 2] lattice xy basis vectors
        """
        B, N, C = x.shape
        x_xy, x_z = torch.chunk(x, 2, dim=-1)

        # === XY-Stream self-attention + relative position bias ===
        qkv_xy = self.qkv_xy(x_xy).reshape(B, N, 3, self.num_heads_xy, -1).permute(2, 0, 3, 1, 4)
        q_xy, k_xy, v_xy = qkv_xy[0], qkv_xy[1], qkv_xy[2]

        attn_xy = (q_xy @ k_xy.transpose(-2, -1)) * self.scale

        # Apply relative position bias.
        rel_bias = self.rel_pos_bias(xy_coords, lattice_xy_basis=lattice_xy_basis)
        if rel_bias is not None:
            attn_xy = attn_xy + rel_bias

        if attn_mask is not None:
            attn_xy = attn_xy + attn_mask
        attn_xy = attn_xy.softmax(dim=-1)
        attn_xy = self.attn_drop(attn_xy)
        out_xy = (attn_xy @ v_xy).transpose(1, 2).reshape(B, N, self.half_dim)

        # === Z-Stream self-attention ===
        qkv_z = self.qkv_z(x_z).reshape(B, N, 3, self.num_heads_z, -1).permute(2, 0, 3, 1, 4)
        q_z, k_z, v_z = qkv_z[0], qkv_z[1], qkv_z[2]

        attn_z = (q_z @ k_z.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_z = attn_z + attn_mask
        attn_z = attn_z.softmax(dim=-1)
        attn_z = self.attn_drop(attn_z)
        out_z = (attn_z @ v_z).transpose(1, 2).reshape(B, N, self.half_dim)

        # === Cross-attention (ablation switch) ===
        if self.cross_attn_mode == 'bidirectional':
            # Bidirectional cross-attention (enhanced).
            # 1. z_from_xy: Z queries XY (dominant: in-plane arrangement determines inter-layer thickness).
            attn_z_from_xy = (q_z @ k_xy.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_z_from_xy = attn_z_from_xy + attn_mask
            attn_z_from_xy = attn_z_from_xy.softmax(dim=-1)
            attn_z_from_xy = self.attn_drop(attn_z_from_xy)
            cross_z_from_xy = (attn_z_from_xy @ v_xy).transpose(1, 2).reshape(B, N, self.half_dim)

            # 2. xy_from_z: XY queries Z (auxiliary direction).
            attn_xy_from_z = (q_xy @ k_z.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_xy_from_z = attn_xy_from_z + attn_mask
            attn_xy_from_z = attn_xy_from_z.softmax(dim=-1)
            attn_xy_from_z = self.attn_drop(attn_xy_from_z)
            cross_xy_from_z = (attn_xy_from_z @ v_z).transpose(1, 2).reshape(B, N, self.half_dim)

            out_z = out_z + self.gate_z_from_xy * cross_z_from_xy
            out_xy = out_xy + self.gate_xy_from_z * cross_xy_from_z

        elif self.cross_attn_mode == 'xy_to_z':
            # Unidirectional baseline: XY queries Z.
            attn_cross = (q_xy @ k_z.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_cross = attn_cross + attn_mask
            attn_cross = attn_cross.softmax(dim=-1)
            attn_cross = self.attn_drop(attn_cross)
            cross_out = (attn_cross @ v_z).transpose(1, 2).reshape(B, N, self.half_dim)
            out_xy = out_xy + self.cross_attn_scale * cross_out
        # cross_attn_mode == 'none': no cross-attention.

        # Merge streams.
        out = torch.cat([out_xy, out_z], dim=-1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


__all__ = [
    'BondDecoupledMLP',
    'RelativePositionBias',
    'DualStream_DFCA_Attention',
    'modulate',
]
