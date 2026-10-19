"""
Electronic Parameters Adaptation.

Functions for adaptive electronic parameters based on band gap interpolation.

References:
    - C2DB GPAW calculation standards
"""

from typing import Dict, Any

import numpy as np


def adaptive_electronic_params(
    gap_eV: float,
    dos_at_ef: float,
    config,
) -> Dict[str, float]:
    """
    根据探测出的真实能隙和费米面态密度，用连续函数插值出这个具体材料
    该用多密的 k 点、多窄的展宽、多大的超胞，不做任何离散分类。

    设计原则：
      - gap_eV 趋近于 0 (金属/半金属) → 参数趋向最精细(金属级)设置
      - gap_eV 较大 (>1eV, 正常半导体/绝缘体) → 参数自动衰减回默认设置，
        不做任何加密，避免浪费算力
      - dos_at_ef 用于区分"能隙为0但DOS也趋于0"的狄拉克点材料(如石墨烯，
        费米面附近态密度稀疏，实际上比普通金属更需要密 k 点才能准确采样
        费米面附近的少量态)和"能隙为0且DOS很大"的常规金属

    参数:
        gap_eV: DFT 探测出的 HOMO-LUMO 能隙 (eV)
        dos_at_ef: 费米能级态密度 (态/eV/atom)
        config: GPAWConfig 配置

    返回:
        {
            'kpoint_multiplier': float,   # 相对默认 dk_per_ang 的加密倍数
            'fermi_smearing': float,      # 该用的展宽宽度 (eV)
            'supercell_min_length': float,# M3 超胞最小边长 (Å)
        }
    """
    # 能隙插值因子：gap 越小，因子越接近 1；gap 越大，因子越接近 0
    # gap_interp_scale 是插值曲线的特征尺度(单位 eV)，暴露成 config 可调，
    # 不是硬编码判据阈值，是插值函数的形状参数
    gap_factor = 1.0 / (1.0 + gap_eV / config.gap_interp_scale)

    # 狄拉克点修正：DOS(E_F) 很小但 gap 也接近 0 的情况(石墨烯类)，
    # 需要额外加密，用 DOS 的倒数做一个附加因子(DOS 越小，附加因子越大)
    dirac_factor = 1.0 / (1.0 + dos_at_ef / config.dos_interp_scale)

    # 两个效应叠加：gap→0 时 combined_factor → 2（gap_factor→1 + dirac_factor→1）
    # gap→∞ 时 combined_factor → 1（gap_factor→0，dirac_factor→0 但留一项）
    # 这确保半导体(大gap)时基础加密仍有一点点保留，而金属时达到最大值
    combined_factor = gap_factor * (1.0 + dirac_factor)

    kpoint_multiplier = 1.0 + config.k_max_boost * combined_factor
    fermi_smearing = config.fermi_smearing_semiconductor - \
        (config.fermi_smearing_semiconductor - config.fermi_smearing_metal) * gap_factor
    supercell_min_length = config.supercell_min_length_base + \
        config.supercell_min_length_boost * gap_factor

    return {
        'kpoint_multiplier': kpoint_multiplier,
        'fermi_smearing': fermi_smearing,
        'supercell_min_length': supercell_min_length,
    }


def adaptive_kpoints(
    atoms,
    dk: float = 0.03,
    max_nk: int = 10,
) -> tuple:
    """
    根据晶格常数自适应计算 k 点网格（2D Gamma-centered）。

    对齐 C2DB 标准：Δk ≤ 0.03 Å⁻¹

    参数:
        atoms: ASE Atoms 对象
        dk: 最大 k 点间距 Å⁻¹（金属/狄拉克体系传入更小的dk实现加密）
        max_nk: 面内单方向 k 点上限，防止自适应加密导致网格爆炸

    返回: (kpts, koffset)
    """
    cell = atoms.get_cell()
    a, b, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])

    # 计算面内倒格矢长度
    b1_length = 2 * np.pi / a if a > 0 else 1.0
    b2_length = 2 * np.pi / b if b > 0 else 1.0

    # k 点数 = 倒格矢长度 / Δk
    # 下限为 2 确保 Γ 点采样质量
    # 上限为 max_nk 防止加密倍数过大导致 k 点爆炸
    nk1 = min(max(2, int(np.ceil(b1_length / dk))), max_nk)
    nk2 = min(max(2, int(np.ceil(b2_length / dk))), max_nk)

    # c 方向固定为 1（真空层）
    return [nk1, nk2, 1], [0, 0, 0]  # Gamma-centered


def adaptive_kpoints_for_supercell(
    atoms,
    dk: float = 0.03,
    max_nk: int = 4,
) -> tuple:
    """
    根据超胞晶格常数自适应计算 k 点网格。

    与 adaptive_kpoints 相同，但专为超胞优化：
    - 允许更少的 k 点（下限 1）
    - 上限为 max_nk（防止超大超胞产生过多 k 点）

    物理原理：
        超胞的面内晶格常数扩大 N 倍，导致倒格矢长度缩短为 1/N，
        因此 k 点数自然减少为 1/N，保持 Δk 恒定。

    参数:
        atoms: ASE Atoms 对象（超胞）
        dk: 最大 k 点间距 Å⁻¹
        max_nk: 面内单方向 k 点上限

    返回:
        (kpts, koffset)
    """
    cell = atoms.get_cell()
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])

    # 倒格矢长度
    b1_length = 2 * np.pi / a if a > 0 else 1.0
    b2_length = 2 * np.pi / b if b > 0 else 1.0

    # k 点数 = 倒格矢长度 / Δk
    # 下限设为 1（超胞可以很少 k 点）
    # 上限为 max_nk
    nk1 = min(max(1, int(np.ceil(b1_length / dk))), max_nk)
    nk2 = min(max(1, int(np.ceil(b2_length / dk))), max_nk)

    # c 方向固定为 1（真空层）
    return [nk1, nk2, 1], [0, 0, 0]


__all__ = [
    'adaptive_electronic_params',
    'adaptive_kpoints',
    'adaptive_kpoints_for_supercell',
]
