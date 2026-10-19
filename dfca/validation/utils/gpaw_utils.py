"""
GPAW Validation Utilities.

Missing utility functions from the original gpaw_validation.py that were not
migrated to the modular structure.
"""

import os
import time
import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Type hints
try:
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Atoms = None

# Import GPAW classes with fallback
try:
    import gpaw
    from gpaw import PW, FermiDirac
    HAS_GPAW = True
except ImportError:
    HAS_GPAW = False
    # Placeholder classes for when GPAW is not installed
    class PW:
        def __init__(self, *args, **kwargs): pass
    class FermiDirac:
        def __init__(self, *args, **kwargs): pass

__all__ = [
    'PW',
    'FermiDirac',
    '_check_dependencies',
    '_timer',
    '_find_structure_file',
    'compute_dft_energy_per_atom',
    'generate_band_qpoints',
    'get_2d_high_symmetry_path',
    'print_elastic_analysis',
    'parameters_to_fc',
    'spectrum_from_fc',
]


# ─────────────────────────────────────────────────────────────────────────────
# GPAW Classes
# ─────────────────────────────────────────────────────────────────────────────

# PW and FermiDirac are imported from gpaw above
# They are re-exported here for backward compatibility


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Checking
# ─────────────────────────────────────────────────────────────────────────────

def _check_dependencies(check_gpaw: bool = True) -> None:
    """
    检查所有必需的依赖，缺失时给出友好的安装提示。

    Args:
        check_gpaw: 是否检查 GPAW（测试模式下可跳过）
    """
    missing = []

    try:
        import ase
    except ImportError:
        missing.append("ase (pip install ase)")

    if check_gpaw:
        try:
            import gpaw
        except ImportError:
            missing.append("gpaw (conda install gpaw -c conda-forge)")

    try:
        import phonopy
    except ImportError:
        missing.append("phonopy (pip install phonopy)")

    try:
        from pymatgen.core import Structure, Element, Composition
    except ImportError:
        missing.append("pymatgen (pip install pymatgen)")

    try:
        from ase.optimize import FIRE, BFGS
    except ImportError:
        missing.append("ase.optimize (upgrade ase)")

    try:
        from ase.constraints import ExpCellFilter, FrechetCellFilter
    except ImportError:
        try:
            from ase.filters import ExpCellFilter, FrechetCellFilter
        except ImportError:
            missing.append("ase.constraints/FrechetCellFilter")

    if missing:
        print("=" * 70)
        print("ERROR: Missing required dependencies, cannot run GPAW validation pipeline:")
        print("=" * 70)
        for dep in missing:
            print("  - " + dep)
        print("=" * 70)
        print("Installation suggestions:")
        print("  conda install gpaw mpi4py -c conda-forge")
        print("  pip install ase pymatgen phonopy matplotlib")
        print("=" * 70)
        import sys
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Timing Utilities
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _timer(label: str):
    """
    简单计时上下文管理器。

    Usage:
        with _timer("computation"):
            # do work
            pass
        # elapsed time is available via the yielded function

    Yields:
        A function that returns elapsed time in seconds when called.
    """
    t0 = time.time()
    yield lambda: time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Structure File Finding
# ─────────────────────────────────────────────────────────────────────────────

def _find_structure_file(directory: str) -> Optional[str]:
    """
    在给定目录中查找结构文件。

    优先级: POSCAR > CONTCAR > *.cif > *.vasp

    Args:
        directory: 要搜索的目录路径

    Returns:
        文件路径，若未找到则返回 None
    """
    candidates = [
        'POSCAR', 'CONTCAR', 'POSCAR_relaxed',
        'structure.vasp', 'final.vasp',
    ]

    for fname in candidates:
        path = os.path.join(directory, fname)
        if os.path.isfile(path):
            return path

    # 回退：查找目录下任何 cif 或 vasp 文件
    for entry in os.scandir(directory):
        if entry.is_file():
            if entry.name.lower().endswith(('.cif', '.vasp')):
                return entry.path

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Energy Calculations
# ─────────────────────────────────────────────────────────────────────────────

def compute_dft_energy_per_atom(atoms: Atoms) -> float:
    """
    计算每个原子的平均能量（eV/atom）。

    Args:
        atoms: ASE Atoms 对象，必须已经 attached 了 calculator

    Returns:
        每个原子的平均能量 (eV/atom)
    """
    if atoms.calc is None:
        raise ValueError("Atoms object has no calculator attached")
    return atoms.get_potential_energy() / len(atoms)


# ─────────────────────────────────────────────────────────────────────────────
# Band Structure Path Generation
# ─────────────────────────────────────────────────────────────────────────────

def get_2d_high_symmetry_path(atoms: Atoms) -> Tuple[List[List[float]], List[str]]:
    """
    根据 2D 材料晶系自动生成高对称路径。

    使用 pymatgen 的 seekpath 或晶格参数判断晶系，
    返回对应的高对称 q 点路径和标签。

    支持的晶系：
    - Hexagonal (六方): Γ-M-K-Γ
    - Tetragonal (四方): Γ-X-M-Y-Γ
    - Orthorhombic (正交): Γ-X-S-Y-Γ 或 Γ-X-U-Z-Γ
    - 通用: Γ-X-M-Γ (降级方案)

    Args:
        atoms: ASE Atoms 对象

    Returns:
        (path_qpoints, labels)
        - path_qpoints: 高对称点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        - labels: 对应标签 ['Γ', 'M', 'K', 'Γ']
    """
    try:
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        from pymatgen.io.ase import AseAtomsAdaptor

        struct = AseAtomsAdaptor().get_structure(atoms)
        kpath = HighSymmKpath(struct)
        bp = kpath.get_kpoints(line_mode=True, path=kpath.path)

        # 提取路径和标签
        path = bp['kpoints']
        labels = bp['labels']
        return path, labels
    except Exception:
        # 降级方案：根据晶格矢量计算真实 γ 角判断晶系
        cell = atoms.get_cell()
        a_vec, b_vec = np.array(cell[0]), np.array(cell[1])
        a_len = np.linalg.norm(a_vec)
        b_len = np.linalg.norm(b_vec)

        # 通过向量点积计算真实晶格夹角 γ
        cos_gamma = np.dot(a_vec, b_vec) / (a_len * b_len)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)  # 防止数值误差
        gamma_deg = np.degrees(np.arccos(cos_gamma))

        # 六方晶系：γ ≈ 120° 或 60°
        if abs(gamma_deg - 120.0) < 10.0 or abs(gamma_deg - 60.0) < 10.0:
            return [
                [0.0, 0.0, 0.0],    # Γ
                [0.5, 0.0, 0.0],    # M
                [1/3, 1/3, 0.0],    # K
                [0.0, 0.0, 0.0],    # Γ
            ], ['Γ', 'M', 'K', 'Γ']

        # 四方/正交晶系
        if abs(gamma_deg - 90.0) < 5.0:
            # 如果 a ≈ b，使用 M；否则使用通用 X-M
            if abs(a_len - b_len) / max(a_len, b_len) < 0.05:
                return [
                    [0.0, 0.0, 0.0],    # Γ
                    [0.5, 0.0, 0.0],    # X
                    [0.5, 0.5, 0.0],    # M
                    [0.0, 0.0, 0.0],    # Γ
                ], ['Γ', 'X', 'M', 'Γ']

        # 通用降级方案
        return [
            [0.0, 0.0, 0.0],    # Γ
            [0.5, 0.0, 0.0],    # X
            [0.5, 0.5, 0.0],    # M
            [0.0, 0.0, 0.0],    # Γ
        ], ['Γ', 'X', 'M', 'Γ']


def generate_band_qpoints(
    path_qpoints: List[List[float]],
    npoints_per_segment: int = 51,
) -> Tuple[np.ndarray, List[int]]:
    """
    生成用于声子/电子能带绘制的密集 q 点路径。

    在高对称点之间进行线性插值，生成平滑的能带曲线。

    Args:
        path_qpoints: 高对称点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        npoints_per_segment: 每个分段内的点数（默认 51）

    Returns:
        (all_qpoints, special_indices)
        - all_qpoints: 所有插值后的 q 点 (N_qpoints, 3)
        - special_indices: 高对称点在 all_qpoints 中的索引
    """
    all_qpoints = []
    special_indices = [0]

    for i in range(len(path_qpoints) - 1):
        start = np.array(path_qpoints[i])
        end = np.array(path_qpoints[i + 1])

        for j in range(npoints_per_segment):
            t = j / (npoints_per_segment - 1)
            q = start + t * (end - start)
            all_qpoints.append(q.tolist())

        # 记录高对称点索引（每段的终点）
        if i < len(path_qpoints) - 2:
            special_indices.append(len(all_qpoints) - 1)

    special_indices.append(len(all_qpoints) - 1)
    return np.array(all_qpoints), special_indices


# ─────────────────────────────────────────────────────────────────────────────
# Elastic Analysis Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_elastic_analysis(
    c2d: Dict[str, float],
    noise_floor: float,
    config: Any,
) -> None:
    """
    打印弹性常数分析结果，用于判断 C66 vs (C11-C12)/2 偏离是否在噪声置信区间内。

    对齐 C2DB Sec 3.1：六方晶系理论约束 C66 = (C11 - C12) / 2

    Args:
        c2d: 弹性常数字典 {'C11': ..., 'C12': ..., 'C66': ...}
        noise_floor: 本征噪声尺度 (N/m)
        config: GPAWConfig 配置对象
    """
    c11 = c2d.get('C11', 0.0)
    c12 = c2d.get('C12', 0.0)
    c66 = c2d.get('C66', 0.0)

    # 六方晶系理论值
    c66_theory = (c11 - c12) / 2.0
    deviation = abs(c66 - c66_theory)

    # 3σ 置信区间
    confidence_interval = config.symmetry_noise_sigma * noise_floor

    print("\n" + "=" * 60)
    print("  Elastic Matrix Analysis (C2DB Sec 3.1)")
    print("=" * 60)
    print(f"  C11 = {c11:.2f} N/m")
    print(f"  C12 = {c12:.2f} N/m")
    print(f"  C66 = {c66:.2f} N/m")
    print(f"  C66_theory = (C11-C12)/2 = {c66_theory:.2f} N/m")
    print("-" * 60)
    print(f"  偏离量 |C66 - C66_theory| = {deviation:.2f} N/m")
    print(f"  本征噪声尺度 (noise_floor) = {noise_floor:.4f} N/m")
    print(f"  {config.symmetry_noise_sigma}σ 置信区间 = {confidence_interval:.4f} N/m")
    print("-" * 60)

    if abs(c66_theory) < 1e-10:
        print("  [INFO] C66_theory ≈ 0，跳过理论值比较")
    elif deviation <= confidence_interval:
        print(f"  [PASS] 偏离在 {config.symmetry_noise_sigma}σ 置信区间内")
        print(f"          数值对称性保持良好，无显著非零剪切耦合")
    else:
        print(f"  [WARN] 偏离超出 {config.symmetry_noise_sigma}σ 置信区间")
        print(f"          可能存在非零剪切耦合或对称性偏离")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Phonon Force Constants Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parameters_to_fc(
    cs: Any,
    parameters: Any,
    ideal_supercell: Any,
) -> np.ndarray:
    """
    将 ForceConstantPotential 参数转换为 force constants 数组。

    Args:
        cs: ClusterSpace 对象
        parameters: 拟合得到的力常数参数
        ideal_supercell: 理想超胞 Atoms 对象

    Returns:
        force_constants: 形状为 (N_atoms, N_atoms, 3, 3) 的力常数数组
    """
    from hiphive import ForceConstantPotential

    fcp = ForceConstantPotential(cs, parameters)
    fc_obj = fcp.get_force_constants(ideal_supercell)

    fc = np.asarray(
        fc_obj.get_fc_array(order=2, format="phonopy"),
        dtype=float,
    )
    return fc


def spectrum_from_fc(
    force_constants: np.ndarray,
    atoms: Any,
    q_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从力常数计算声子谱。

    Args:
        force_constants: 力常数数组
        atoms: ASE Atoms 对象
        q_points: q 点数组 (N_q, 3)

    Returns:
        (frequencies, eigenvectors)
        - frequencies: 声子频率 (N_q, N_modes) in THz
        - eigenvectors: 声子本征向量 (N_q, N_modes, N_atoms, 3)
    """
    try:
        import phonopy
    except ImportError:
        raise ImportError("phonopy is required for spectrum_from_fc")

    # Create phonopy object
    ph = phonopy.Phonopy(
        atoms,
        supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    # Set force constants
    ph.force_constants = force_constants

    # Calculate dynamical matrix and get frequencies
    frequencies = []
    eigenvectors = []

    for q in q_points:
        dm = ph.get_dynamical_matrix_at_q(q)
        freqs, eigvecs = np.linalg.eigvalsh(dm)
        frequencies.append(np.sqrt(np.abs(freqs)) * 1e12 / (2 * np.pi))  # Convert to THz
        eigenvectors.append(eigvecs)

    return np.array(frequencies), np.array(eigenvectors)
