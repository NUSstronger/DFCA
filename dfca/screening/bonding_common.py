#!/usr/bin/env python3
"""
Shared Bonding & Dimensionality Utilities
=========================================
2D Materials Screening Pipeline v3.2 — Single Source of Truth

本模块是全流水线唯一、成键判定和维度分析的核心实现。
所有模块（dimensionality_check、S5' convex_hull、S7 MD 诊断）
必须调用本模块的函数，而非各自维护独立的半径表或公差公式。

C2DB Sec. 2.2 标准（来源：Gjerding et al. 2021）：
  - 使用 ASE natural_cutoffs 共价半径 × 1.35 倍截断构建成键图
  - 使用 CutOffDictNN 构建 StructureGraph（支持同种及异种元素成键）
  - 使用 get_dimensionality_larsen(sg) 计算拓扑维度
  - 验证有且仅有 1 个独立的 2D 连通网络

统一判据（C2DB Sec. 2.2，两条件同时满足才算通过）：
  1. get_dimensionality_larsen(sg) == 2
  2. nx.number_connected_components(sg.graph.to_undirected()) == 1

依赖：
  - networkx: 连通分量分析
  - pymatgen[analysis]: StructureGraph 和维度分析
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import networkx as nx
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.local_env import CutOffDictNN, StructureGraph
from pymatgen.core import Structure

logger = logging.getLogger("Screen2D.bonding")

# ─────────────────────────────────────────────────────────────────
# §1 共价半径数据库（C2DB Sec. 2.2: ASE natural_cutoffs）
# ─────────────────────────────────────────────────────────────────

_COVALENT_CUTOFF_FACTOR = 1.35  # C2DB Sec. 2.2 规定的截断倍数

_ASE_COVALENT_RADII = {
    "H": 0.37, "He": 0.32, "Li": 1.34, "Be": 0.90, "B": 0.82,
    "C": 0.77, "N": 0.71, "O": 0.63, "F": 0.59, "Ne": 0.62,
    "Na": 1.54, "Mg": 1.36, "Al": 1.18, "Si": 1.11, "P": 1.07,
    "S": 1.02, "Cl": 0.99, "Ar": 1.01, "K": 1.96, "Ca": 1.74,
    "Sc": 1.44, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19,
    "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.15, "Zn": 1.09,
    "Ga": 1.07, "Ge": 1.07, "As": 1.09, "Se": 1.09, "Br": 1.10,
    "Kr": 1.09, "Rb": 2.11, "Sr": 1.90, "Y": 1.63, "Zr": 1.48,
    "Nb": 1.37, "Mo": 1.30, "Tc": 1.27, "Ru": 1.25, "Rh": 1.24,
    "Pd": 1.20, "Ag": 1.28, "Cd": 1.40, "In": 1.36, "Sn": 1.30,
    "Sb": 1.33, "Te": 1.37, "I": 1.40, "Xe": 1.40, "Cs": 2.32,
    "Ba": 1.98, "La": 1.69, "Ce": 1.63, "Pr": 1.76, "Nd": 1.74,
    "Pm": 1.73, "Sm": 1.72, "Eu": 1.68, "Gd": 1.69, "Tb": 1.68,
    "Dy": 1.67, "Ho": 1.66, "Er": 1.65, "Tm": 1.64, "Yb": 1.70,
    "Lu": 1.56, "Hf": 1.54, "Ta": 1.46, "W": 1.37, "Re": 1.35,
    "Os": 1.37, "Ir": 1.32, "Pt": 1.23, "Au": 1.24, "Hg": 1.40,
    "Tl": 1.44, "Pb": 1.44, "Bi": 1.49, "Po": 1.50, "At": 1.50,
    "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80,
}


# ─────────────────────────────────────────────────────────────────
# §2 共价半径查询与截断距离计算
# ─────────────────────────────────────────────────────────────────

def get_covalent_radii() -> Dict[str, float]:
    """返回共价半径数据库（只读字典）"""
    return _ASE_COVALENT_RADII.copy()


def compute_bond_cutoff(el1: str, el2: str) -> float:
    """
    计算两个元素之间的成键截断距离。

    C2DB Sec. 2.2 公式：
        cutoff = 1.35 × (r_cov(el1) + r_cov(el2))

    Args:
        el1, el2: 元素符号

    Returns:
        截断距离（Å）
    """
    r1 = _ASE_COVALENT_RADII.get(el1, 1.2)
    r2 = _ASE_COVALENT_RADII.get(el2, 1.2)
    return _COVALENT_CUTOFF_FACTOR * (r1 + r2)


# ─────────────────────────────────────────────────────────────────
# §3 StructureGraph 构建（C2DB Sec. 2.2 统一方法）
# ─────────────────────────────────────────────────────────────────

def build_cut_dict(struct: Structure) -> Dict[Tuple[str, str], float]:
    """
    构建包含所有元素对（含同种元素）的截断距离字典。

    用于 CutOffDictNN 初始化。
    """
    elements = sorted(list({str(s.specie) for s in struct}))
    cut_dict: Dict[Tuple[str, str], float] = {}

    for idx, el1 in enumerate(elements):
        for el2 in elements[idx:]:
            cutoff = compute_bond_cutoff(el1, el2)
            cut_dict[(el1, el2)] = cutoff
            cut_dict[(el2, el1)] = cutoff

    return cut_dict


def build_structure_graph(struct: Structure) -> Tuple[Optional[StructureGraph], int]:
    """
    使用 CutOffDictNN + C2DB Sec. 2.2 截断距离构建 StructureGraph。

    Returns:
        (sg, bond_count): StructureGraph 对象和成键数量；失败时返回 (None, 0)
    """
    try:
        cut_dict = build_cut_dict(struct)
        nn_strategy = CutOffDictNN(cut_off_dict=cut_dict)
        sg = StructureGraph.with_local_env_strategy(struct, nn_strategy)
        bond_count = len(sg.graph.edges)
        return sg, bond_count
    except Exception as e:
        logger.debug(f"StructureGraph 构建失败: {e}")
        return None, 0


# ─────────────────────────────────────────────────────────────────
# §4 维度与连通性分析（C2DB Sec. 2.2 统一判据）
# ─────────────────────────────────────────────────────────────────

def check_dimensionality(
    struct: Structure,
) -> Tuple[bool, Dict[str, Any]]:
    """
    检查结构是否为二维单连通网络（C2DB Sec. 2.2 统一判据）。

    判据（两条件同时满足）：
        1. get_dimensionality_larsen(sg) == 2
        2. 连通分量数 == 1

    Returns:
        (is_2d_connected, info_dict)

    info_dict 字段：
        dimensionality : int  - 拓扑维度（0/1/2/3/-1）
        bond_count    : int  - 成键数量
        n_components  : int  - 连通分量数
        reason        : str - 诊断字符串
    """
    sg, bond_count = build_structure_graph(struct)

    if sg is None:
        return False, {
            "dimensionality": -1,
            "bond_count": 0,
            "n_components": -1,
            "reason": "structure_graph_build_failed",
        }

    if bond_count == 0:
        return False, {
            "dimensionality": -1,
            "bond_count": 0,
            "n_components": -1,
            "reason": "no_bonds_found",
        }

    # 维度计算
    try:
        overall_dim = int(get_dimensionality_larsen(sg))
    except Exception as e:
        logger.debug(f"维度计算失败: {e}")
        return False, {
            "dimensionality": -1,
            "bond_count": bond_count,
            "n_components": -1,
            "reason": f"dimensionality_calc_failed:{e}",
        }

    # 连通分量数
    undirected = sg.graph.to_undirected()
    n_components = int(nx.number_connected_components(undirected))

    result = {
        "dimensionality": overall_dim,
        "bond_count": bond_count,
        "n_components": n_components,
        "reason": "",
    }

    # C2DB Sec. 2.2 判据
    if overall_dim == 2 and n_components == 1:
        result["reason"] = "ok"
        return True, result
    elif overall_dim != 2:
        result["reason"] = f"not_2d_overall(dim={overall_dim})"
        return False, result
    else:  # n_components != 1
        result["reason"] = f"multiple_components(n={n_components})"
        return False, result


def is_2d_connected(struct: Structure) -> bool:
    """快捷封装：仅返回布尔值"""
    ok, _ = check_dimensionality(struct)
    return ok


def get_bond_info(struct: Structure) -> Dict[str, Any]:
    """
    获取详细成键信息（调试/诊断用）。

    Returns:
        dict，包含键列表、维度、连通分量等
    """
    sg, bond_count = build_structure_graph(struct)
    if sg is None:
        return {"n_atoms": len(struct), "bond_count": 0, "bonds": [], "connectivity": {}}

    bonds_detail = []
    for edge in sg.graph.edges:
        i, j = edge[0], edge[1]
        el_i = str(struct[i].specie)
        el_j = str(struct[j].specie)
        d = struct.distance(i, j)
        cutoff = compute_bond_cutoff(el_i, el_j)
        bonds_detail.append({
            "i": i, "j": j,
            "el_i": el_i, "el_j": el_j,
            "distance": round(d, 3),
            "cutoff": round(cutoff, 3),
        })

    undirected = sg.graph.to_undirected()
    n_components = int(nx.number_connected_components(undirected))
    try:
        overall_dim = int(get_dimensionality_larsen(sg))
    except Exception:
        overall_dim = -1

    return {
        "n_atoms": len(struct),
        "bond_count": bond_count,
        "bonds": bonds_detail,
        "connectivity": {
            "dimensionality": overall_dim,
            "n_components": n_components,
            "pass": (overall_dim == 2 and n_components == 1),
        },
    }
