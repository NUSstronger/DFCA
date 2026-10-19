"""
Thermodynamic Stability Module.

Functions for convex hull stability calculations using Materials Project API.

References:
    - C2DB convex hull standards
    - Materials Project phase diagram API
"""

import logging
import os
from typing import Dict, List, Tuple, Optional, Any

from dfca.validation.mpi import MPI_RANK
from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms

logger = logging.getLogger(__name__)

# MP API availability check
try:
    from mp_api import MpApi
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False


def compute_ehull_stability(
    struct,
    dft_energy_per_atom: float,
    mp_api_key: str = "",
) -> Tuple[float, bool, str]:
    """
    计算 DFT 能量相对于凸包的能量差。

    参数:
        struct: pymatgen Structure 对象
        dft_energy_per_atom: eV/atom
        mp_api_key: Materials Project API key

    返回:
        (ehull_eV, is_stable, reason)
        - is_stable=True 时，reason 为 "ok" 或具体数值
        - is_stable=False 时，reason 必为非空字符串，说明具体原因
        - 任何情况下 reason 都不会是空字符串
    """
    global HAS_MP_API

    # 检查 mp_api 是否可用
    if not HAS_MP_API:
        return -1.0, False, "mp_api_unavailable_requires_install_typing_extensions"

    key = mp_api_key or os.environ.get("MP_API_KEY", "")
    if not key:
        return -1.0, False, "no_mp_api_key_provided"

    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        try:
            from pymatgen.analysis.phase_diagram import PDEntry
        except ImportError:
            from pymatgen.entries.computed_entries import ComputedEntry as PDEntry
    except ImportError as e:
        return -1.0, False, f"mp_api_or_pymatgen_import_error_{e}"

    try:
        comp = struct.composition
        elements = [str(el) for el in comp.elements]
        n_atoms = len(struct)

        with MPRester(key) as mpr:
            # 获取化学系统的所有条目
            try:
                entries = mpr.get_entries_in_chemsys(elements)
            except TypeError:
                entries = mpr.get_entries_in_chemsys(
                    elements,
                    property_data=["formation_energy_per_atom"],
                )

            if not entries:
                return -1.0, False, "no_reference_entries_in_chemsys_check_mp_api_key"

            # 构建目标材料条目
            my_entry = PDEntry(comp, dft_energy_per_atom * n_atoms)
            all_entries = entries + [my_entry]

            # 边界检查：entries 可能只包含材料自身（没有任何参考相可比较）
            if len(all_entries) <= 1:
                return -1.0, False, "only_target_material_no_reference_phases_available"

            # 构建凸包（隔离 try 块，提供更精确的错误诊断）
            try:
                pd = PhaseDiagram(all_entries)
                ehull = pd.get_e_above_hull(my_entry)
            except ValueError as e:
                # PhaseDiagram 内部可能因化学式不匹配/entries 无效等抛出 ValueError
                return -1.0, False, f"phase_diagram_construction_failed_ValueError_{e}"
            except Exception as e:
                return -1.0, False, f"phase_diagram_error_{type(e).__name__}_{e}"

            # 验证 PhaseDiagram 结果的有效性
            if ehull is None:
                return -1.0, False, "phase_diagram_returned_none_e_above_hull"

            # 判断稳定性：阈值 0.2 eV/atom (C2DB 标准)
            threshold = 0.2
            is_stable = bool(ehull <= threshold)

            if is_stable:
                # 材料在凸包上或附近（热力学稳定或亚稳态）
                if ehull <= 1e-6:
                    reason = "on_hull_thermodynamically_stable"
                else:
                    reason = f"ok_e_above_hull={ehull:.4f}_within_threshold"
            else:
                # 材料在凸包上方（热力学不稳定）
                reason = f"unstable_e_above_hull={ehull:.4f}_exceeds_{threshold}"

            return float(ehull), is_stable, reason

    except Exception as e:
        return -1.0, False, f"ehull_computation_failed_{type(e).__name__}_{e}"


def fetch_candidate_bulk_structures(
    elements: List[str],
    mp_api_key: str,
    pool_size: int = 3,
) -> List[Dict]:
    """
    只从Materials Project拉取该化学体系内候选bulk结构（不使用MP的能量数据），
    按MP自身标注的 energy_above_hull 排序，每个化学式只保留最稳定的前
    pool_size 个，控制总计算量。

    返回:
        [{'structure': Structure对象, 'formula': str, 'mp_id': str}, ...]
    """
    from mp_api.client import MPRester
    from collections import defaultdict

    with MPRester(mp_api_key) as mpr:
        docs = mpr.summary.search(
            chemsys="-".join(sorted(elements)),
            fields=["structure", "energy_above_hull", "formula_pretty", "material_id"]
        )

    # 按 formula_pretty 分组，每组按 energy_above_hull 升序，取前 pool_size 个
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.formula_pretty].append(doc)

    selected = []
    for formula, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x.energy_above_hull)
        for item in items_sorted[:pool_size]:
            selected.append({
                'structure': item.structure,
                'formula': formula,
                'mp_id': str(item.material_id),
            })
    return selected


def compute_reference_energy_gpaw(
    struct,
    config,
) -> Optional[float]:
    """
    对候选bulk结构用与M1/M2/M3相同的GPAW参数（encut/xc/收敛设置）
    重新做3D周期性全弛豫（不加2D的c轴/真空层约束，也不施加M1那种
    FixSymmetry约束——bulk参考相不需要保持特定对称性，直接自由弛豫
    获得基态能量即可），返回能量(eV/atom)。计算失败返回None，调用方
    负责跳过该条目而不是让整体计算崩溃。

    磁性处理：复用 classify_magnetic() 判断初始自旋极化设置，与M1保持
    一致的种子磁矩策略；不需要做FM/NM比较，bulk参考相只需要一个合理
    的下界能量估计，不需要严格判定磁性基态。
    """
    from ase.optimize import FIRE
    from ase.constraints import FrechetCellFilter
    from pymatgen.io.ase import AseAtomsAdaptor

    from dfca.validation.calculators import build_gpaw_calculator

    try:
        atoms = AseAtomsAdaptor.get_atoms(struct)
        elements = atoms.get_chemical_symbols()
        mag_type, is_magnetic = classify_magnetic(elements)
        if is_magnetic:
            magmoms = build_initial_magmoms(atoms)
            atoms.set_initial_magnetic_moments(magmoms)

        calc = build_gpaw_calculator(
            atoms, config, mag_type, is_magnetic,
            for_supercell=False,
        )
        atoms.calc = calc
        # 全晶胞自由弛豫，bulk无mask限制，不施加FixSymmetry
        cell_filter = FrechetCellFilter(atoms)
        opt = FIRE(cell_filter, logfile=None)
        opt.run(fmax=config.relax_fmax, steps=config.relax_steps)

        e_per_atom = atoms.get_potential_energy() / len(atoms)
        return float(e_per_atom)
    except Exception as e:
        if MPI_RANK == 0:
            logger.warning(f"参考相能量计算失败: {e}")
        return None


def compute_ehull_stability_v2(
    struct,
    dft_energy_per_atom: float,
    config,
    mp_api_key: str,
    ref_cache: Dict[str, float],
) -> Tuple[float, bool, str]:
    """
    方案B：候选参考结构来自MP，能量本地GPAW统一重算，保证凸包内所有
    条目都用同一套xc/赝势/收敛参数算出来，才能和目标材料的能量直接相减。

    ref_cache: 跨材料复用的参考能量缓存 {cache_key: energy_per_atom}，
    避免重复计算相同的单质/化合物，调用方负责传入并持久化这个缓存。
    """
    global HAS_MP_API

    if not mp_api_key:
        return -1.0, False, "no_mp_api_key_provided_v2"
    if not HAS_MP_API:
        return -1.0, False, "mp_api_unavailable_requires_install_typing_extensions_v2"

    elements = [str(el) for el in struct.composition.elements]

    # 导入所需模块
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        try:
            from pymatgen.analysis.phase_diagram import PDEntry
        except ImportError:
            from pymatgen.entries.computed_entries import ComputedEntry as PDEntry
    except ImportError as e:
        return -1.0, False, f"mp_api_or_pymatgen_import_error_v2_{e}"

    try:
        candidates = fetch_candidate_bulk_structures(elements, mp_api_key,
                                                      pool_size=config.ehull_ref_pool_size)
    except Exception as e:
        return -1.0, False, f"mp_structure_fetch_failed_v2: {e}"

    if not candidates:
        return -1.0, False, "no_candidate_structures_found_v2"

    entries = []
    for cand in candidates:
        cand_struct = cand['structure']
        cache_key = f"{cand['formula']}_{cand['mp_id']}"
        if cache_key not in ref_cache:
            e = compute_reference_energy_gpaw(cand_struct, config)
            if e is None:
                continue  # 跳过计算失败的候选，不让整体流程崩溃
            ref_cache[cache_key] = e
        e_per_atom = ref_cache[cache_key]
        entries.append(PDEntry(cand_struct.composition, e_per_atom * len(cand_struct)))

    if not entries:
        return -1.0, False, "all_reference_energy_calculations_failed_v2"

    my_entry = PDEntry(struct.composition, dft_energy_per_atom * len(struct))
    entries.append(my_entry)

    # 边界检查：entries 可能只包含材料自身（没有任何参考相可比较）
    if len(entries) <= 1:
        return -1.0, False, "only_target_material_no_reference_phases_v2"

    try:
        pd = PhaseDiagram(entries)
        ehull = pd.get_e_above_hull(my_entry)
    except ValueError as e:
        return -1.0, False, f"phase_diagram_construction_failed_v2_ValueError_{e}"
    except Exception as e:
        return -1.0, False, f"phase_diagram_construction_failed_v2: {e}"

    if ehull is None:
        return -1.0, False, "phase_diagram_returned_none_e_above_hull_v2"

    threshold = config.ehull_threshold
    is_stable = bool(ehull <= threshold)

    if is_stable:
        reason = f"ok_gpaw_local_ref(ehull={ehull:.4f})" if ehull > 1e-6 else "on_hull_gpaw_local_ref"
    else:
        reason = f"unstable_gpaw_local_ref(ehull={ehull:.4f}>{threshold})"

    return float(ehull), is_stable, reason


__all__ = [
    'compute_ehull_stability',
    'compute_ehull_stability_v2',
    'fetch_candidate_bulk_structures',
    'compute_reference_energy_gpaw',
]
