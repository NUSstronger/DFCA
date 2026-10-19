"""
GPAW DFT Validation Pipeline.

Orchestrates the complete M1-M4 validation workflow with:
- M1: Structure relaxation with 2D constraints
- M2: Elastic constants (Born stability criteria)
- M3: Phonon dispersion (dynamic stability)
- M4: Convex hull stability (Materials Project API)

Features:
- Fine-grained checkpoint/resume (stage-level)
- Immediate disk persistence (v2.0)
- MPI-safe distributed execution
- Adaptive electronic parameters based on band gap

References:
    - C2DB validation standards
    - GPAW documentation
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np

from dfca.validation.calculators import GPAW_DEFAULTS
from dfca.validation.mpi import (
    MPI_RANK,
    MPI_SIZE,
    mpi_broadcast_object,
    mpi_world_sum,
    mpi_trace,
)
from dfca.validation.config import GPAWConfig
from dfca.validation.record import ValidationRecord
from dfca.validation.electronic.gap import probe_electronic_gap
from dfca.validation.electronic.symmetry import get_spacegroup_info
from dfca.validation.electronic.params import adaptive_electronic_params
from dfca.validation.magnetic.comparison import compare_magnetic_ground_states
from dfca.validation.utils.magnetic import classify_magnetic, build_initial_magmoms
from dfca.validation.mechanical.elastic import compute_elastic_matrix_2d_gpaw
from dfca.validation.phonon.dispersion import compute_phonon_dispersion_gpaw
from dfca.validation.phonon.plotting import plot_phonon_dispersion
from dfca.validation.relaxation.relax_2d import relax_structure_2d
from dfca.validation.thermodynamics.ehull import compute_ehull_stability, compute_ehull_stability_v2
from dfca.validation.io import read_structure, scan_output_directory
from dfca.validation.metadata import load_screen_report_index, get_screen_metadata

logger = logging.getLogger(__name__)

# Availability checks
try:
    import gpaw
    HAS_GPAW = True
except ImportError:
    HAS_GPAW = False

try:
    from phonopy import Phonopy
    HAS_PHONOPY = True
except ImportError:
    HAS_PHONOPY = False

try:
    from mp_api import MpApi
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False


# MPI broadcast sequence counter for debugging
_MPI_OBJECT_BCAST_SEQ = 0


def _gpaw_broadcast(obj, root=0, comm=None):
    """
    Broadcast a Python object using GPAW's MPI communicator.

    Args:
        obj: Object to broadcast (only used on root rank)
        root: Root rank for broadcasting
        comm: GPAW communicator (defaults to world)

    Returns:
        Broadcasted object on all ranks
    """
    global _MPI_OBJECT_BCAST_SEQ
    _MPI_OBJECT_BCAST_SEQ += 1

    if not HAS_GPAW:
        return obj

    if comm is None:
        from gpaw import mpi as _gpaw_mpi
        comm = _gpaw_mpi.world

    try:
        return comm.broadcast(obj, root)
    except Exception:
        # Fallback to custom broadcast
        from dfca.validation.mpi import _mpi_broadcast_python_object
        return _mpi_broadcast_python_object(obj, root, comm)


class GPAWValidationPipeline:
    """
    GPAW 第一性原理验证流水线 v2.0 (支持断点续算与即时落盘)

    Orchestrates the complete M1-M4 validation workflow:
    - M1: Structure relaxation with 2D constraints
    - M2: Elastic constants (Born stability criteria)
    - M3: Phonon dispersion (dynamic stability)
    - M4: Convex hull stability (Materials Project API)

    Features:
    - Fine-grained checkpoint/resume (stage-level)
    - Immediate disk persistence (v2.0)
    - MPI-safe distributed execution
    - Adaptive electronic parameters based on band gap
    """

    def __init__(self, config: GPAWConfig):
        """Initialize the validation pipeline.

        Args:
            config: GPAWConfig instance with all validation parameters
        """
        self.config = config
        self.records: List[ValidationRecord] = []
        self.done_names: set = set()  # 已完成结构名称集合
        self.ehull_check_available: bool = HAS_MP_API  # 全局 mp_api 可用性标志
        self.ref_energy_cache: Dict[str, float] = {}  # 方案B: 跨材料复用的参考能量缓存

        # ── Screen -> GPAW metadata index ───────────────────────────
        # Screen metadata is diagnostic/provenance only at this stage.
        # It MUST NOT alter any M1/M2/M3/M4 pass/fail decision.
        self.screen_report_path = os.path.join(
            config.input_dir,
            "screening_report.csv",
        )

        if MPI_RANK == 0:
            try:
                self.screen_report_index = load_screen_report_index(
                    self.screen_report_path
                )

                if self.screen_report_index:
                    logger.info(
                        f"Screen report loaded: "
                        f"{len(self.screen_report_index)} rows | "
                        f"{self.screen_report_path}"
                    )
                else:
                    logger.warning(
                        f"Screen report unavailable or empty: "
                        f"{self.screen_report_path}"
                    )

            except Exception as screen_exc:
                logger.warning(
                    f"Screen report load failed: {screen_exc}"
                )
                self.screen_report_index = {}
        else:
            self.screen_report_index = None

        # Rank 0 is the authoritative owner of the CSV read.
        # All ranks receive the identical index so metadata state cannot
        # diverge during MPI execution.
        self.screen_report_index = mpi_broadcast_object(
            self.screen_report_index if MPI_RANK == 0 else None,
            root=0,
        )

        # ── 加载参考能量缓存 ────────────────────────────────────────
        cache_path = os.path.join(config.output_dir, "ref_energies_cache.json")
        if MPI_RANK == 0 and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.ref_energy_cache = json.load(f)
                logger.info(f"从 {cache_path} 加载参考能量缓存: {len(self.ref_energy_cache)} 条目")
            except Exception as e:
                logger.warning(f"参考能量缓存加载失败: {e}，使用空缓存")
        # ─────────────────────────────────────────────────────────

        # ── Stage 控制与细粒度断点续算 ───────────────────────────
        self.stage: str = 'resume_failed'  # 'all', 'm1', 'm2', 'm3', 'm4', 'resume_failed'
        self.force_stage: bool = False  # 强制重跑指定 stage
        # ───────────────────────────────────────────────────────

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 报告文件路径
        self.csv_path = os.path.join(config.output_dir, "gpaw_final_report.csv")
        self.json_path = os.path.join(config.output_dir, "summary.json")
        self.phonon_dir = os.path.join(config.output_dir, "phonon_spectra")
        self.checkpoint_path = os.path.join(config.output_dir, "_checkpoint.json")

        if config.save_phonon_pdf:
            os.makedirs(self.phonon_dir, exist_ok=True)

        # ── v2.0 断点续算：加载历史记录 ───────────────────────────
        self._load_checkpoint()

    def _load_checkpoint(self):
        """
        加载断点文件，恢复已完成的计算状态。

        支持两种断点源：
        1. _checkpoint.json (详细断点，包含所有 record)
        2. gpaw_final_report.csv (简略断点，仅包含结构名称)

        逐条恢复记录，单条失败不影响其他记录。
        未知字段自动过滤，避免因 checkpoint 版本升级导致的 TypeError。

        MPI broadcast: rank 0 读取后通过 world.broadcast 同步到所有 rank，
        保证所有进程对"已完成材料列表"的判断一致。
        """
        from dataclasses import fields as dc_fields
        known_fields = {f.name for f in dc_fields(ValidationRecord)}

        # ── 仅 rank 0 执行文件 I/O ─────────────────────────────────
        _local_records = []  # rank 0 的本地 records
        _local_done_names = set()
        checkpoint_found = False

        if MPI_RANK == 0:
            # 优先尝试 _checkpoint.json
            if os.path.exists(self.checkpoint_path):
                try:
                    with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                        ckpt = json.load(f)

                    if 'records' in ckpt:
                        for rec_dict in ckpt['records']:
                            name = rec_dict.get('name', '<unknown>')
                            try:
                                filtered = {k: v for k, v in rec_dict.items() if k in known_fields}
                                dropped = set(rec_dict.keys()) - known_fields
                                if dropped:
                                    logger.warning(
                                        f"[{name}] 忽略旧checkpoint中已废弃的字段: {dropped}"
                                    )
                                rec = ValidationRecord(**filtered)
                                _local_records.append(rec)
                                _local_done_names.add(rec.name)
                            except Exception as e:
                                logger.error(
                                    f"[{name}] 单条记录恢复失败，将视为全新材料重跑: {e}",
                                    exc_info=True,
                                )
                        checkpoint_found = True
                        logger.info(f"从断点恢复: {len(_local_records)} 个已完成结构")

                except Exception as e:
                    logger.warning(f"断点文件加载失败: {e}，将尝试 CSV 恢复")

            # 回退到 CSV 断点
            if not checkpoint_found and os.path.exists(self.csv_path):
                try:
                    import csv
                    with open(self.csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            _local_done_names.add(row.get('name', ''))
                    if _local_done_names:
                        logger.info(f"从 CSV 恢复: {len(_local_done_names)} 个已完成结构")
                        checkpoint_found = True
                except Exception as e:
                    logger.warning(f"CSV 断点加载失败: {e}")

            if checkpoint_found:
                logger.info(f"检出历史记录，自动跳过 {len(_local_done_names)} 个已完成结构")
            else:
                logger.info("未发现历史记录，从头开始计算")

        # ── Broadcast 到所有 rank ──────────────────────────────────
        # 注意：GPAW world.broadcast(a, root) 只能广播连续 numpy 数组
        # checkpoint 数据 (list/set/bool) 是 Python 对象，必须用 gpaw.mpi.broadcast()
        # 广播失败必须 abort——各 rank 跳过不同结构会导致数据错乱
        try:
            _local_records = _gpaw_broadcast(_local_records if MPI_RANK == 0 else None, root=0)
            _local_done_names = _gpaw_broadcast(_local_done_names if MPI_RANK == 0 else None, root=0)
            checkpoint_found = _gpaw_broadcast(checkpoint_found if MPI_RANK == 0 else None, root=0)
        except Exception as bc_err:
            if MPI_RANK == 0:
                logger.error(f"Checkpoint 数据 MPI 广播同步失败(致命): {bc_err}")
                logger.error("各 rank 跳过不同结构将导致数据错乱，必须终止")
            if HAS_GPAW:
                from gpaw import mpi as _gpaw_mpi
                _gpaw_mpi.world.barrier()
            sys.exit(1)

        # 所有 rank 使用相同的数据
        self.records = _local_records
        self.done_names = _local_done_names

    def _find_relaxed_poscar(self, name: str, mag_type: str) -> Optional[str]:
        """
        查找已弛豫的 POSCAR 文件。

        优先级：
        1. ./dft_relaxed/{mag_type}/POSCAR_{name}
        2. ./dft_relaxed/{mag_type}_nm/POSCAR_{name}

        Args:
            name: 结构名称
            mag_type: 磁性分类

        Returns:
            文件路径，若不存在则返回 None
        """
        # 尝试不同目录
        for mt in [mag_type, f"{mag_type}_nm", "nonmag"]:
            for suffix in ['', '_nm']:
                dft_dir = os.path.join(self.config.output_dir, "dft_relaxed", mt + suffix)
                poscar_path = os.path.join(dft_dir, f"POSCAR_{name}")
                if os.path.exists(poscar_path):
                    if MPI_RANK == 0:
                        logger.info(f"找到已弛豫结构: {poscar_path}")
                    return poscar_path

        # Fallback: recursively search dft_relaxed.
        # The magnetic branch directory may differ from the current
        # Screen-derived mag_type after an earlier FM/NM comparison.
        relaxed_root = os.path.join(
            self.config.output_dir,
            "dft_relaxed",
        )

        if os.path.isdir(relaxed_root):
            target = f"POSCAR_{name}"

            for root, dirs, files in os.walk(relaxed_root):
                dirs.sort()

                if target in files:
                    poscar_path = os.path.join(root, target)

                    if MPI_RANK == 0:
                        logger.info(
                            f"递归找到已弛豫结构: {poscar_path}"
                        )

                    return poscar_path

        # 尝试 checkpoint 中记录的路径
        for rec in self.records:
            if (
                rec.name == name
                and rec.dft_poscar_path
                and os.path.exists(rec.dft_poscar_path)
            ):
                if MPI_RANK == 0:
                    logger.info(
                        f"从 checkpoint 找到已弛豫结构: "
                        f"{rec.dft_poscar_path}"
                    )
                return rec.dft_poscar_path

        return None

    def _get_failed_stages(self, rec: ValidationRecord) -> List[str]:
        """
        获取未通过的 stage 列表。

        Args:
            rec: ValidationRecord

        Returns:
            未通过的 stage 列表 ['m1', 'm2', 'm3', 'm4']
        """
        failed = []
        if not rec.m1_pass:
            failed.append('m1')
        if rec.m1_pass and not rec.m2_elastic_pass:
            failed.append('m2')
        if rec.m1_pass and not rec.m3_phonon_pass:
            failed.append('m3')
        # M4 仅记录，不阻断流程
        return failed

    def validate_single_structure(
        self,
        input_mag_type: str,
        name: str,
        poscar_path: str,
        mp_api_key: str = "",
        existing_rec: Optional[ValidationRecord] = None,
    ) -> ValidationRecord:
        """对单个结构执行验证流程（支持细粒度断点续算）。

        Stage 控制：
        - 'all': 始终重跑所有 stage
        - 'm1/m2/m3/m4': 仅重跑指定 stage
        - 'resume_failed': 根据 checkpoint 状态，仅重跑未通过的 stage

        Args:
            input_mag_type: Magnetic type from Screen classification
            name: Structure name
            poscar_path: Path to input POSCAR file
            mp_api_key: Materials Project API key for M4
            existing_rec: Existing record from checkpoint (if any)

        Returns:
            ValidationRecord with all stage results
        """
        # ── 安全初始化：确保整个函数作用域内 mag_type 始终有定义 ───────
        mag_type = input_mag_type  # 防御性赋值，消除 UnboundLocalError 风险
        # ─────────────────────────────────────────────────────────────────

        if existing_rec is not None:
            rec = existing_rec
        else:
            rec = ValidationRecord()
            rec.name = name
            rec.mag_type = input_mag_type

        # ── Attach Screen metadata ──────────────────────────────────
        # This is intentionally refreshed even for records restored
        # from an older checkpoint. Therefore adding Screen metadata
        # does not require deleting or rebuilding existing GPAW results.
        screen_meta = get_screen_metadata(
            name,
            self.screen_report_index,
        )

        for screen_key, screen_value in screen_meta.items():
            if hasattr(rec, screen_key):
                setattr(
                    rec,
                    screen_key,
                    screen_value,
                )

        # IMPORTANT:
        # Do NOT overwrite rec.mag_type / rec.is_magnetic from the CSV.
        # The Screen bucket is authoritative for the initial branch,
        # and M1/FM-NM DFT remains authoritative afterwards.

        if MPI_RANK == 0:
            if rec.screen_report_found:
                logger.info(
                    f"[{name}] Screen metadata | "
                    f"id={rec.screen_name} | "
                    f"formula={rec.screen_formula} | "
                    f"S5_min={rec.screen_s5_min_freq_THz:+.4f} THz | "
                    f"S5_imag={rec.screen_s5_imaginary_count} | "
                    f"S7a={rec.screen_s7a_pass} | "
                    f"S7b={rec.screen_s7b_pass} | "
                    f"Ehull={rec.screen_s5p_ehull_eV:.4f} eV/atom | "
                    f"stability={rec.screen_s5p_stability_label}"
                )
            else:
                logger.warning(
                    f"[{name}] No matching Screen metadata "
                    f"(lookup={rec.screen_name})"
                )

        rec.error_type = ""
        rec.raw_error = ""
        # ─────────────────────────────────────────────────────────────────

        # Stage 执行决策
        #
        # Explicit stage modes have literal semantics:
        #   --stage m1  -> M1 only
        #   --stage m2  -> M2 only (requires a previous M1 PASS)
        #   --stage m3  -> M3 only (requires a previous M1 PASS)
        #   --stage m4  -> M4 only (requires a previous M1 PASS)
        #   --stage all -> M1 + M2 + M3 + M4
        #
        # resume_failed retains checkpoint-driven behaviour.
        if existing_rec is None:
            if self.stage == 'all':
                run_m1 = True
                run_m2 = True
                run_m3 = True
                run_m4 = True

            elif self.stage == 'm12':
                # Round-1 production validation:
                # high-accuracy M1 relaxation + GAP probe + M2 elastic.
                # M3 phonons are intentionally deferred to soft-mode
                # candidates selected after this round.
                run_m1 = True
                run_m2 = True
                run_m3 = False
                run_m4 = False

            elif self.stage == 'm1':
                run_m1 = True
                run_m2 = False
                run_m3 = False
                run_m4 = False

            elif self.stage in ('m2', 'm3', 'm4'):
                # A downstream-only stage is not allowed to silently use
                # the Screen structure as if it were DFT-relaxed.
                run_m1 = False
                run_m2 = (self.stage == 'm2')
                run_m3 = (self.stage == 'm3')
                run_m4 = (self.stage == 'm4')

            else:
                # A genuinely new structure in resume_failed starts at M1
                # and cascades through downstream stages in the SAME run.
                #
                # Dependency guards below still stop the pipeline immediately
                # if M1 or M2 fails, so enabling downstream stages here does
                # not allow invalid structures to bypass earlier stages.
                run_m1 = True
                run_m2 = True
                run_m3 = True
                run_m4 = True
        else:
            if self.stage == 'resume_failed':
                # Cascade from the earliest unfinished blocking stage.
                #
                # Example:
                #   M1=False              -> M1 -> M2 -> M3 -> M4
                #   M1=True, M2=False     ->      M2 -> M3 -> M4
                #   M1=True, M2=True,
                #       M3=False          ->            M3 -> M4
                #
                # M4 is non-blocking in this pipeline.  If M1/M2/M3 are
                # already complete, do not wake an otherwise completed
                # material merely because M4 is unavailable or failed.
                if not existing_rec.m1_pass:
                    run_m1 = True
                    run_m2 = True
                    run_m3 = True
                    run_m4 = True

                elif not existing_rec.m2_elastic_pass:
                    run_m1 = False
                    run_m2 = True
                    run_m3 = True
                    run_m4 = True

                elif not existing_rec.m3_completed:
                    run_m1 = False
                    run_m2 = False
                    run_m3 = True
                    run_m4 = True

                else:
                    run_m1 = False
                    run_m2 = False
                    run_m3 = False
                    run_m4 = False

            elif self.stage == 'm12':
                # Round-1 checkpoint-aware resume.
                #
                # M1 incomplete -> resume/run M1, then continue to M2.
                # M1 complete but M2 incomplete -> reuse relaxed M1
                # structure and run only M2.
                # M1 and M2 complete -> skip this material completely.
                if not existing_rec.m1_pass:
                    run_m1 = True
                    run_m2 = True
                elif not existing_rec.m2_elastic_pass:
                    run_m1 = False
                    run_m2 = True
                else:
                    run_m1 = False
                    run_m2 = False

                run_m3 = False
                run_m4 = False

            else:
                # Other explicit stage modes keep literal semantics.
                run_m1 = (
                    self.stage == 'all'
                    or self.stage == 'm1'
                )
                run_m2 = (
                    self.stage == 'all'
                    or self.stage == 'm2'
                )
                # Explicit M3 is checkpoint-aware unless --force_stage
                # is requested.  A completed physical FAIL must not be
                # recomputed on every PBS resume.
                run_m3 = (
                    self.stage == 'all'
                    or (
                        self.stage == 'm3'
                        and not existing_rec.m3_completed
                    )
                )
                run_m4 = (
                    self.stage == 'all'
                    or self.stage == 'm4'
                )

        # force_stage 强制重跑指定 stage
        if self.force_stage:
            run_m1 = run_m1 or (self.stage in ['all', 'm1'])
            run_m2 = run_m2 or (self.stage in ['all', 'm2'])
            run_m3 = run_m3 or (self.stage in ['all', 'm3'])
            run_m4 = run_m4 or (self.stage in ['all', 'm4'])

        # -------------------------------------------------------------
        # Recover an M1 result that was durably written to dft_relaxed/
        # but whose whole-material checkpoint record was never committed.
        #
        # This occurs when PBS walltime kills the job after M1 completed
        # (and its relaxed POSCAR was saved) but before M2/material-level
        # checkpoint completion.
        #
        # For m12 only, the relaxed POSCAR is a durable M1 completion
        # artifact.  Reuse it instead of repeating an expensive M1.
        # GAP will be recomputed below because its scalar result may not
        # have reached the material checkpoint.
        artifact_relaxed_path = None

        if (
            self.stage == 'm12'
            and existing_rec is None
            and run_m1
        ):
            artifact_relaxed_path = self._find_relaxed_poscar(
                name,
                mag_type,
            )

            if artifact_relaxed_path:
                run_m1 = False
                run_m2 = True
                run_m3 = False
                run_m4 = False

                # Mark M1 as completed for the downstream dependency guard.
                # Geometry is loaded from the durable relaxed POSCAR below.
                rec.m1_pass = True
                rec.m1_relax_converged = True
                rec.m1_reason = "recovered_from_relaxed_poscar"

                if MPI_RANK == 0:
                    logger.info(
                        f"[{name}] M1 artifact recovery: "
                        f"{artifact_relaxed_path} -> skip M1, continue M2"
                    )

        if MPI_RANK == 0:
            logger.info(f"[{name}] Stage: m1={run_m1}, m2={run_m2}, m3={run_m3}, m4={run_m4} (mode={self.stage})")

        # Nothing requested for this material.  This is especially important
        # for checkpoint-aware m12 resume: an M1+M2-complete material must be
        # returned immediately instead of entering relaxed-structure recovery,
        # which could otherwise turn run_m1 back on when no saved POSCAR exists.
        if not any((run_m1, run_m2, run_m3, run_m4)):
            if MPI_RANK == 0:
                logger.info(
                    f"[{name}] Requested stages already complete -> checkpoint skip"
                )
            return rec

        try:
            # Step 0: 读取结构
            atoms_input, struct = read_structure(poscar_path)

            # Screen -> GPAW magnetic-state inheritance.
            # The bucket controls the INITIAL DFT magnetic branch.
            # GPAW FM/NM comparison may still revise the final state later.
            atoms_input.info['screen_mag_type'] = mag_type

            atoms_to_use = None

            # Two valid ways to reach a downstream-only stage:
            #
            # 1. normal checkpoint recovery: existing_rec says M1 passed;
            # 2. artifact recovery: M1 relaxed POSCAR exists although the
            #    whole-material checkpoint record was not committed.
            relaxed_path = None

            if not run_m1:
                if artifact_relaxed_path:
                    relaxed_path = artifact_relaxed_path
                elif (
                    existing_rec is not None
                    and existing_rec.m1_pass
                ):
                    relaxed_path = self._find_relaxed_poscar(
                        name,
                        mag_type,
                    )

            if relaxed_path:
                try:
                    atoms_to_use, _ = read_structure(relaxed_path)
                    atoms_to_use.set_constraint(None)
                    if MPI_RANK == 0:
                        logger.info(
                            f"[{name}] 从已弛豫结构加载: {relaxed_path}"
                        )
                except Exception as e:
                    if MPI_RANK == 0:
                        logger.warning(
                            f"[{name}] 加载失败，回退: {e}"
                        )
                    atoms_to_use = None

            atoms = atoms_to_use if atoms_to_use else atoms_input

            # Re-attach Screen metadata after reading a relaxed POSCAR,
            # because ASE structure files do not preserve atoms.info.
            atoms.info['screen_mag_type'] = mag_type

            if not run_m1 and atoms_to_use is None:
                run_m1 = True

            rec.n_atoms = len(atoms)
            rec.formula = struct.composition.reduced_formula
            elements = atoms.get_chemical_symbols()

            # Initial magnetic state comes from the Screen bucket.
            # Direct/legacy inputs still use the element classifier.
            if mag_type in ('nonmag', 'mag_3d', 'mag_4d5d', 'mag_re'):
                is_magnetic = (mag_type != 'nonmag')
            else:
                _, is_magnetic = classify_magnetic(elements)

            rec.mag_type = mag_type
            rec.is_magnetic = is_magnetic

            if is_magnetic:
                rec.magmom_config = build_initial_magmoms(atoms)
                rec.magnetic_moment = sum(rec.magmom_config)
            else:
                rec.magmom_config = [0.0] * len(atoms)
                rec.magnetic_moment = 0.0

            if MPI_RANK == 0:
                logger.info(f"[{rec.formula}] GPAW DFT 验证 (mag_type={mag_type})")

            # M1: 结构弛豫
            if run_m1:
                atoms.set_constraint(None)
                mpi_trace(f"[{rec.name}] FLOW BEFORE M1")
                atoms_relaxed, m1_result = relax_structure_2d(
                    atoms.copy(), self.config, max_retries=self.config.scf_retry_max, name=rec.name)
                mpi_trace(
                    f"[{rec.name}] FLOW AFTER M1 "
                    f"pass={m1_result.get('pass')} "
                    f"reason={m1_result.get('reason')} "
                    f"obj_bcast_seq={_MPI_OBJECT_BCAST_SEQ}"
                )
                rec.m1_relax_converged = m1_result.get('converged', False)
                rec.m1_fmax_reached = m1_result.get('fmax_reached', 0.0)
                rec.m1_steps = m1_result.get('steps', 0)
                rec.m1_energy_eV = m1_result.get('energy_eV', 0.0)
                rec.m1_energy_per_atom = m1_result.get('energy_per_atom', 0.0)
                rec.m1_time_s = m1_result.get('time_s', 0.0)
                rec.m1_pass = m1_result.get('pass', False)
                rec.m1_reason = m1_result.get('reason', '')
                rec.retry_count = m1_result.get('retry_count', 0)
                rec.m1_zero_step_flag = m1_result.get('zero_step_verification_triggered', False)
                rec.m1_zero_step_verified = m1_result.get('zero_step_verified', False)
                rec.m1_zero_step_energy_delta = m1_result.get('zero_step_energy_delta', 0.0)
                rec.m1_zero_step_verify_passed = m1_result.get('zero_step_verify_passed', False)
                rec.m1_resumed_from_partial = m1_result.get('resumed_from_partial', False)
                rec.m1_spacegroup_before = m1_result.get('spacegroup_before', '')
                rec.m1_spacegroup_after = m1_result.get('spacegroup_after', '')

            if not rec.m1_pass:
                rec.error_type = 'relaxation'
                rec.raw_error = m1_result.get('error', m1_result.get('reason', '')) if run_m1 else (existing_rec.raw_error if existing_rec else 'unknown')
                rec.all_dft_pass = False
                return rec

            # M1 结果更新工作结构
            atoms_working = atoms_relaxed if run_m1 else atoms

            if run_m1:
                # Synchronize the magnetic state returned by M1.
                # Screen determines the INITIAL magnetic branch;
                # M1/FM-NM DFT determines the state propagated to M2/M3.
                rec.mag_type = m1_result.get(
                    'mag_type',
                    rec.mag_type,
                )
                rec.is_magnetic = bool(
                    m1_result.get(
                        'is_magnetic',
                        rec.is_magnetic,
                    )
                )

                actual_mag = m1_result.get(
                    'final_magnetic_moment',
                    0.0,
                )
                rec.magnetic_moment = float(abs(actual_mag))

                is_truly_magnetic = (
                    abs(actual_mag) > 0.01
                    or rec.is_magnetic
                )

                if is_truly_magnetic and HAS_GPAW and self.config.fm_nm_compare_before_m2:
                    try:
                        fm_nm_result = compare_magnetic_ground_states(
                            atoms_relaxed, self.config, force_nm_relax=False)
                        rec.m1_ground_state = fm_nm_result['ground_state']
                        rec.m1_fm_nm_gap_eV = fm_nm_result['fm_nm_gap_eV']
                        if fm_nm_result['should_use_nm']:
                            fm_nm_result = compare_magnetic_ground_states(
                                atoms_relaxed, self.config, force_nm_relax=True)
                            if fm_nm_result['atoms_nm'] is not None:
                                atoms_working = fm_nm_result['atoms_nm']
                                rec.m1_energy_per_atom = fm_nm_result['nm_final_energy']
                                rec.magnetic_moment = fm_nm_result['nm_final_magmom']
                                rec.is_magnetic = False
                                rec.mag_type = 'nonmag'  # 同步更新，与 is_magnetic 保持一致
                                rec.magmom_config = fm_nm_result['nm_magmom_config']
                                rec.m1_reason = "nm_relaxed_overwrites_fm"
                            else:
                                # NM 弛豫失败，reason 字符串更明确，记录异常详情
                                failure_detail = fm_nm_result.get('nm_relax_failure_detail', 'unknown')
                                rec.m1_reason += f"; nm_relax_failed({failure_detail})"
                                rec.m1_nm_relax_failure_detail = failure_detail
                        if MPI_RANK == 0:
                            logger.info(f"[{rec.formula}] 基态: {rec.m1_ground_state}")
                    except Exception as e:
                        if MPI_RANK == 0:
                            logger.warning(f"FM/NM 比较失败: {e}")
                        rec.m1_ground_state = "comparison_failed"
                elif is_truly_magnetic:
                    rec.m1_ground_state = "comparison_disabled"
                else:
                    rec.m1_ground_state = "not_magnetic"
            else:
                if existing_rec:
                    rec.m1_ground_state = existing_rec.m1_ground_state
                    rec.m1_fm_nm_gap_eV = existing_rec.m1_fm_nm_gap_eV
                    # 补充继承磁性状态，防止被 classify_magnetic 覆盖
                    rec.is_magnetic = existing_rec.is_magnetic
                    rec.mag_type = existing_rec.mag_type  # 同步继承，与 is_magnetic 保持一致
                    rec.magnetic_moment = existing_rec.magnetic_moment
                    rec.magmom_config = existing_rec.magmom_config

            # -------------------------------------------------------------
            # Persist the FINAL M1 relaxed geometry immediately.
            #
            # This is intentionally after FM/NM ground-state selection:
            # atoms_working may have been replaced by the NM-relaxed
            # geometry above.  Downstream checkpoint resume must use this
            # final M1 geometry rather than the original Screen structure.
            # -------------------------------------------------------------
            if run_m1 and rec.m1_pass and self.config.save_poscar:
                if MPI_RANK == 0:
                    final_m1_mag_type = (
                        rec.mag_type
                        if rec.is_magnetic
                        else f"{rec.mag_type}_nm"
                    )

                    dft_dir = os.path.join(
                        self.config.output_dir,
                        "dft_relaxed",
                        final_m1_mag_type,
                    )
                    os.makedirs(dft_dir, exist_ok=True)

                    dft_path = os.path.join(
                        dft_dir,
                        f"POSCAR_{rec.name}",
                    )

                    try:
                        from ase.io import write
                        write(
                            dft_path,
                            atoms_working,
                            direct=True,
                        )
                        rec.dft_poscar_path = dft_path
                        logger.info(
                            f"[{rec.name}] M1 relaxed POSCAR saved: "
                            f"{dft_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[{rec.name}] M1 relaxed POSCAR save failed: {e}"
                        )

                mpi_trace(
                    f"[{rec.name}] BEFORE POSCAR_PATH_BCAST "
                    f"seq={_MPI_OBJECT_BCAST_SEQ}"
                )
                rec.dft_poscar_path = mpi_broadcast_object(
                    rec.dft_poscar_path if MPI_RANK == 0 else None,
                    root=0,
                )
                mpi_trace(
                    f"[{rec.name}] AFTER POSCAR_PATH_BCAST "
                    f"seq={_MPI_OBJECT_BCAST_SEQ} "
                    f"type={type(rec.dft_poscar_path).__name__}"
                )

            # M1+: 能隙探测
            #
            # Normal checkpoint resume can inherit the saved GAP result when
            # M1 is not rerun.  Artifact recovery has no committed material
            # record, so GAP must be recomputed from the recovered M1 geometry.
            need_gap_probe = (
                run_m1
                or artifact_relaxed_path is not None
            )

            if self.config.skip_gap_probe or not need_gap_probe:
                if (
                    not need_gap_probe
                    and existing_rec is not None
                ):
                    rec.probed_gap_eV = existing_rec.probed_gap_eV
                    rec.probed_dos_at_ef = existing_rec.probed_dos_at_ef
                    rec.probed_probe_success = existing_rec.probed_probe_success

                adaptive_params = (
                    adaptive_electronic_params(
                        0.0,
                        999.0,
                        self.config,
                    )
                    if self.config.skip_gap_probe
                    else None
                )
            else:
                mpi_trace(
                    f"[{rec.name}] PRE_GAP_SEQ "
                    f"seq={_MPI_OBJECT_BCAST_SEQ}"
                )
                mpi_trace(
                    f"[{rec.name}] FLOW BEFORE GAP "
                    f"is_magnetic={rec.is_magnetic} "
                    f"mag_type={mag_type} rec.mag_type={rec.mag_type}"
                )
                probe_result = probe_electronic_gap(
                    atoms_working,
                    self.config,
                    rec.mag_type,
                    rec.is_magnetic,
                )
                mpi_trace(
                    f"[{rec.name}] FLOW AFTER GAP "
                    f"result={probe_result}"
                )
                rec.probed_gap_eV = probe_result['gap_eV']
                rec.probed_dos_at_ef = probe_result['dos_at_ef']
                rec.probed_probe_success = probe_result['probe_success']
                if probe_result['probe_success']:
                    adaptive_params = adaptive_electronic_params(probe_result['gap_eV'], probe_result['dos_at_ef'], self.config)
                else:
                    adaptive_params = adaptive_electronic_params(0.0, 999.0, self.config)
            rec.used_kpoint_multiplier = adaptive_params['kpoint_multiplier'] if adaptive_params else 1.0
            rec.used_fermi_smearing = adaptive_params['fermi_smearing'] if adaptive_params else 0.05
            rec.used_supercell_min_length = adaptive_params['supercell_min_length'] if adaptive_params else 10.0

            # M2: 弹性常数
            if run_m2:
                atoms_working.set_constraint(None)
                # 传递 M1+FM/NM比较后的真实磁性状态，而非重新猜测
                mpi_trace(
                    f"[{rec.name}] FLOW BEFORE M2 "
                    f"is_magnetic={rec.is_magnetic} rec.mag_type={rec.mag_type}"
                )
                c2d, m2_stable, m2_reason, stress_res, noise_floor = compute_elastic_matrix_2d_gpaw(
                    atoms_working, self.config,
                    is_magnetic=rec.is_magnetic, mag_type=rec.mag_type, name=rec.name)
                mpi_trace(
                    f"[{rec.name}] FLOW AFTER M2 "
                    f"pass={m2_stable} reason={m2_reason}"
                )
                rec.m2_c11 = c2d.get('C11', 0.0)
                rec.m2_c12 = c2d.get('C12', 0.0)
                rec.m2_c22 = c2d.get('C22', 0.0)
                rec.m2_c66 = c2d.get('C66', 0.0)
                rec.m2_stress_residual = stress_res
                rec.m2_noise_floor = noise_floor
                rec.m2_elastic_pass = m2_stable
                rec.m2_reason = m2_reason
            else:
                if existing_rec:
                    rec.m2_c11 = existing_rec.m2_c11
                    rec.m2_c12 = existing_rec.m2_c12
                    rec.m2_c22 = existing_rec.m2_c22
                    rec.m2_c66 = existing_rec.m2_c66
                    rec.m2_stress_residual = existing_rec.m2_stress_residual
                    rec.m2_noise_floor = existing_rec.m2_noise_floor
                    rec.m2_elastic_pass = existing_rec.m2_elastic_pass
                    rec.m2_reason = existing_rec.m2_reason

            if not rec.m2_elastic_pass:
                rec.error_type = 'elastic'
                rec.raw_error = rec.m2_reason
                rec.all_dft_pass = False
                return rec

            # M3: 声子谱
            if run_m3:
                # 传递 M1+FM/NM比较后的真实磁性状态，而非重新猜测
                mpi_trace(
                    f"[{rec.name}] FLOW BEFORE M3 "
                    f"is_magnetic={rec.is_magnetic} rec.mag_type={rec.mag_type}"
                )
                (gamma_min, min_freq, imag_count, ph_stable, ph_reason, phonon_obj,
                 noise_floor_THz, za_tol_dyn, boundary_tol_dyn, branch_info, calc_info) = \
                    compute_phonon_dispersion_gpaw(
                        atoms_working, self.config, adaptive_params,
                        is_magnetic=rec.is_magnetic, mag_type=rec.mag_type, name=rec.name)
                mpi_trace(
                    f"[{rec.name}] FLOW AFTER M3 "
                    f"pass={ph_stable} reason={ph_reason}"
                )
                rec.m3_gamma_min_THz = gamma_min
                rec.m3_min_freq_THz = min_freq
                rec.m3_imaginary_count = imag_count
                rec.m3_phonon_pass = ph_stable
                rec.m3_reason = ph_reason

                # Reaching here means the M3 phonon calculation itself
                # completed normally.  Physical FAIL is still completed.
                rec.m3_completed = True
                rec.m3_noise_floor_THz = noise_floor_THz
                rec.m3_gamma_tol_dynamic = za_tol_dyn
                rec.m3_boundary_tol_dynamic = boundary_tol_dyn
                rec.m3_za_branch_idx = branch_info['za_branch_idx']
                rec.m3_ta_branch_idx = branch_info['ta_branch_idx']
                rec.m3_la_branch_idx = branch_info['la_branch_idx']
                rec.m3_za_gamma_freq = branch_info['za_gamma_freq']
                rec.m3_ta_gamma_freq = branch_info['ta_gamma_freq']
                rec.m3_la_gamma_freq = branch_info['la_gamma_freq']
                rec.m3_optical_gamma_min = branch_info['optical_gamma_min']
                rec.m3_za_tol_dynamic = branch_info['za_tol_dynamic']
                rec.m3_optical_tol_dynamic = branch_info['optical_tol_dynamic']
                rec.m3_za_ok = branch_info['za_ok']
                rec.m3_ta_ok = branch_info['ta_ok']
                rec.m3_la_ok = branch_info['la_ok']
                rec.m3_optical_ok = branch_info['optical_ok']
                rec.m3_used_displacement = calc_info['used_displacement']
                rec.m3_used_supercell = str(calc_info['used_supercell'])
                rec.m3_is_small_lattice = calc_info['is_small_lattice']
                rec.m3_noise_estimation_method = calc_info.get('noise_estimation_method', '')

                if self.config.save_phonon_pdf and HAS_PHONOPY and ph_stable and phonon_obj:
                    if MPI_RANK == 0:
                        try:
                            pdf_path = os.path.join(self.phonon_dir, f"phonon_{rec.formula}_{rec.name}.pdf")
                            plot_path = plot_phonon_dispersion(phonon_obj, atoms_working, pdf_path, self.config)
                            rec.m3_phonon_pdf_path = plot_path
                        except Exception as e:
                            logger.warning(f"声子谱绘图失败: {e}")
            else:
                if existing_rec:
                    for field in ['m3_gamma_min_THz', 'm3_min_freq_THz', 'm3_imaginary_count',
                                  'm3_phonon_pass', 'm3_reason', 'm3_noise_floor_THz',
                                  'm3_gamma_tol_dynamic', 'm3_boundary_tol_dynamic',
                                  'm3_za_branch_idx', 'm3_ta_branch_idx', 'm3_la_branch_idx',
                                  'm3_za_gamma_freq', 'm3_ta_gamma_freq', 'm3_la_gamma_freq',
                                  'm3_optical_gamma_min', 'm3_za_tol_dynamic', 'm3_optical_tol_dynamic',
                                  'm3_za_ok', 'm3_ta_ok', 'm3_la_ok', 'm3_optical_ok',
                                  'm3_used_displacement', 'm3_used_supercell', 'm3_is_small_lattice']:
                        setattr(rec, field, getattr(existing_rec, field))

            if not rec.m3_phonon_pass:
                rec.error_type = 'phonon'
                rec.raw_error = rec.m3_reason
                rec.all_dft_pass = False
                return rec

            # M4: 凸包稳定性
            if run_m4 and mp_api_key and HAS_MP_API:
                rec.m4_used_local_ref = False  # 默认旧方案

                if self.config.use_local_ref_ehull:
                    # 方案B: 优先尝试本地GPAW统一重算
                    ehull, m4_pass, m4_reason = compute_ehull_stability_v2(
                        struct, rec.m1_energy_per_atom, self.config, mp_api_key,
                        self.ref_energy_cache)
                    if m4_reason.startswith("no_") or "failed" in m4_reason:
                        # 方案B失败，降级到旧方案，明确标注精度等级更低
                        ehull, m4_pass, m4_reason_fallback = compute_ehull_stability(
                            struct, rec.m1_energy_per_atom, mp_api_key)
                        m4_reason = f"{m4_reason_fallback}_MP_energy_approximate_not_gpaw_consistent"
                        rec.m4_used_local_ref = False  # 实际使用旧方案
                    else:
                        rec.m4_used_local_ref = True  # 方案B成功
                else:
                    # 直接使用旧方案
                    ehull, m4_pass, m4_reason = compute_ehull_stability(
                        struct, rec.m1_energy_per_atom, mp_api_key)
                    rec.m4_used_local_ref = False

                rec.m4_ehull_eV = ehull
                rec.m4_convex_hull_pass = m4_pass
                rec.m4_reason = m4_reason
            elif existing_rec:
                rec.m4_ehull_eV = existing_rec.m4_ehull_eV
                rec.m4_convex_hull_pass = existing_rec.m4_convex_hull_pass
                rec.m4_reason = existing_rec.m4_reason
                rec.m4_used_local_ref = getattr(existing_rec, 'm4_used_local_ref', False)

            if not rec.m4_convex_hull_pass:
                if MPI_RANK == 0:
                    logger.warning(f"[{rec.formula}] M4 凸包不稳定 (ehull={rec.m4_ehull_eV:.3f} eV)")

            # 方案B参考能量缓存持久化：每次成功计算一个新参考相能量后写回磁盘
            if rec.m4_used_local_ref and self.ref_energy_cache:
                self._save_ref_energy_cache()

            # 保存 DFT-Refined POSCAR (MPI: rank 0 only)
            rec.all_dft_pass = True
            if self.config.save_poscar and MPI_RANK == 0:
                dft_dir = os.path.join(self.config.output_dir, "dft_relaxed",
                                       mag_type if rec.is_magnetic else f"{mag_type}_nm")
                os.makedirs(dft_dir, exist_ok=True)
                dft_path = os.path.join(dft_dir, f"POSCAR_{rec.name}")
                try:
                    from ase.io import write
                    write(dft_path, atoms_working, direct=True)
                    rec.dft_poscar_path = dft_path
                except Exception as e:
                    if MPI_RANK == 0:
                        logger.warning(f"POSCAR 保存失败: {e}")

        except Exception as e:
            rec.error_type = 'unknown'
            rec.raw_error = f"{type(e).__name__}: {str(e)}"
            rec.all_dft_pass = False
            if MPI_RANK == 0:
                logger.error(f"[{rec.formula}] 验证异常: {e}")

        return rec

    def run(self, mp_api_key: str = ""):
        """执行验证流水线 (v3.0: 细粒度断点续算 + Stage-wise Execution)"""
        if MPI_RANK == 0:
            logger.info("=" * 70)
        if MPI_RANK == 0:
            logger.info("GPAW DFT Validation Pipeline v3.0")
        if MPI_RANK == 0:
            logger.info(f"Stage Mode: {self.stage}")
        if MPI_RANK == 0:
            logger.info(f"Force Stage: {self.force_stage}")
        if MPI_RANK == 0:
            logger.info(f"Input:  {self.config.input_dir}")
        if MPI_RANK == 0:
            logger.info(f"Output: {self.config.output_dir}")
        if MPI_RANK == 0:
            logger.info("=" * 70)

        # 扫描输入目录
        structures = scan_output_directory(self.config.input_dir)

        if not structures:
            if MPI_RANK == 0:
                logger.warning(f"未找到任何候选结构: {self.config.input_dir}")
            return

        if MPI_RANK == 0:
            logger.info(f"找到 {len(structures)} 个候选结构")

        # ── 细粒度断点续算：构建待处理列表 ───────────────────────
        todo_structures = []
        for mt, name, path in structures:
            # 检查该结构是否需要处理
            matched_rec = None
            for rec in self.records:
                if rec.name == name:
                    matched_rec = rec
                    break

            if self.stage == 'all':
                # 强制重跑所有结构
                todo_structures.append((mt, name, path, None))
            elif self.stage == 'resume_failed':
                # 检查哪些 stage 未通过
                if matched_rec is None:
                    # 无 checkpoint，从头开始
                    todo_structures.append((mt, name, path, None))
                elif not matched_rec.all_dft_pass:
                    # 有未通过的 stage，需要续算
                    todo_structures.append((mt, name, path, matched_rec))
                else:
                    # 所有 stage 都已通过，跳过
                    if MPI_RANK == 0:
                        logger.info(f"跳过已通过结构: {name}")
            else:
                # 单 stage 模式（如 'm3'），对所有结构执行该 stage
                todo_structures.append((mt, name, path, matched_rec))

        if MPI_RANK == 0:
            logger.info(f"待处理: {len(todo_structures)} 个结构")

        if not todo_structures:
            if MPI_RANK == 0:
                logger.info("所有结构均已完成，无需继续计算")
            self.print_summary()
            return

        # 处理每个结构
        for i, (mag_type, name, poscar_path, existing_rec) in enumerate(todo_structures, 1):
            if MPI_RANK == 0:
                logger.info(f"\n[{i}/{len(todo_structures)}] 处理: {name}")

            rec = self.validate_single_structure(
                mag_type, name, poscar_path, mp_api_key, existing_rec
            )

            # 更新或添加记录
            updated = False
            for idx, r in enumerate(self.records):
                if r.name == name:
                    self.records[idx] = rec
                    updated = True
                    break
            if not updated:
                self.records.append(rec)

            # ── v2.0 即时落盘：每完成一个结构立即保存 ─────────────
            self.save_reports()
            self._save_checkpoint()

            # 进度报告
            status = "PASS" if rec.all_dft_pass else "FAIL"
            if MPI_RANK == 0:
                logger.info(f"    {status} | {rec.formula}")
            if not rec.all_dft_pass:
                if MPI_RANK == 0:
                    logger.info(f"    失败原因: {rec.error_type} - {rec.raw_error[:80]}")

        # 最终保存
        self.print_summary()
        if MPI_RANK == 0:
            logger.info("计算完成！所有结果已保存。")

    def _save_checkpoint(self):
        """
        保存断点文件（含完整 records）。

        用于程序中断后恢复计算状态。

        MPI safety: 仅 rank 0 写入文件，避免多进程并发写入损坏数据。
        """
        if MPI_RANK != 0:
            return

        try:
            ckpt = {
                'version': '2.0',
                'timestamp': datetime.now().isoformat(),
                'records': [asdict(r) for r in self.records],
            }
            # 序列化 numpy 类型
            for rec_dict in ckpt['records']:
                for k, v in rec_dict.items():
                    if isinstance(v, (np.integer, np.floating)):
                        rec_dict[k] = v.item()
                    elif isinstance(v, np.ndarray):
                        rec_dict[k] = v.tolist()
                    elif isinstance(v, list) and v and isinstance(v[0], (np.integer, np.floating)):
                        rec_dict[k] = [x.item() if hasattr(x, 'item') else x for x in v]

            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(ckpt, f, indent=2, ensure_ascii=False)

        except Exception as e:
            if MPI_RANK == 0:
                logger.warning(f"断点保存失败: {e}")

    def _save_ref_energy_cache(self):
        """
        持久化参考能量缓存到磁盘（方案B专用）。

        每次成功计算一个新参考相能量后追加写回磁盘。
        MPI safety: 仅 rank 0 写入文件，避免多进程并发写入损坏数据。
        写入后通过 world.broadcast 同步到其他 rank，保持所有 rank 的缓存一致。
        """
        cache_path = os.path.join(self.config.output_dir, "ref_energies_cache.json")

        if MPI_RANK == 0:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.ref_energy_cache, f, indent=2, ensure_ascii=False)
                logger.info(f"参考能量缓存已保存: {len(self.ref_energy_cache)} 条目 -> {cache_path}")
            except Exception as e:
                logger.warning(f"参考能量缓存保存失败: {e}")

        # Broadcast 到所有 rank，同步缓存内容
        # 注意：ref_energy_cache 是 dict (Python对象)，必须用 mpi_broadcast_object()
        # broadcast失败不能warning后继续，必须raise
        try:
            self.ref_energy_cache = mpi_broadcast_object(
                self.ref_energy_cache if MPI_RANK == 0 else None, root=0
            )
        except Exception as bc_err:
            mpi_trace("[MPI_FATAL] ref_energy_cache broadcast failed: " + str(bc_err))
            raise

    def save_reports(self):
        """
        保存 CSV 和 JSON 报告 (v2.0: 规范化字段与 numpy 序列化)。

        MPI safety: 仅 rank 0 写入文件，避免多进程并发写入损坏数据。

        CSV 完整字段：
        - 标识: name, formula, mag_type, n_atoms
        - 磁性: is_magnetic, magnetic_moment, magmom_config
        - M1弛豫: m1_relax_converged, m1_fmax_reached, m1_steps,
                   m1_energy_eV, m1_energy_per_atom, m1_time_s, m1_pass, m1_reason
        - M2弹性: m2_c11, m2_c12, m2_c22, m2_c66, m2_stress_residual,
                   m2_elastic_pass, m2_reason
        - M3声子: m3_gamma_min_THz, m3_min_freq_THz, m3_imaginary_count,
                   m3_phonon_pass, m3_reason, m3_phonon_pdf_path
        - M4凸包: m4_ehull_eV, m4_convex_hull_pass, m4_reason
        - 综合: all_dft_pass, dft_poscar_path, error_type, raw_error, retry_count
        """
        if MPI_RANK != 0:
            return

        if not self.records:
            return

        import csv

        # 规范化 CSV 字段顺序（按模块分组）
        fieldnames = [
            # 标识
            'name', 'formula', 'mag_type', 'n_atoms',
            # 磁性
            'is_magnetic', 'magnetic_moment', 'magmom_config',
            # M1 弛豫
            'm1_relax_converged', 'm1_fmax_reached', 'm1_steps',
            'm1_energy_eV', 'm1_energy_per_atom', 'm1_time_s',
            'm1_pass', 'm1_reason',
            'm1_zero_step_flag', 'm1_zero_step_verified', 'm1_zero_step_energy_delta',
            'm1_zero_step_verify_passed', 'm1_nm_relax_failure_detail',
            # M2 弹性
            'm2_c11', 'm2_c12', 'm2_c22', 'm2_c66',
            'm2_stress_residual', 'm2_elastic_pass', 'm2_reason',
            # M3 声子
            'm3_gamma_min_THz', 'm3_min_freq_THz', 'm3_imaginary_count',
            'm3_phonon_pass', 'm3_reason', 'm3_phonon_pdf_path',
            'm3_noise_floor_THz', 'm3_noise_estimation_method',
            # M4 凸包
            'm4_ehull_eV', 'm4_convex_hull_pass', 'm4_reason',
            # 综合
            'all_dft_pass', 'dft_poscar_path', 'error_type', 'raw_error', 'retry_count',
        ]

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for rec in self.records:
                row = {}
                rec_dict = asdict(rec)
                for field in fieldnames:
                    v = rec_dict.get(field)
                    # numpy 类型序列化
                    if isinstance(v, (np.bool_,)):
                        v = bool(v)
                    elif isinstance(v, np.integer):
                        v = int(v)
                    elif isinstance(v, np.floating):
                        v = float(v)
                    elif isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, list) and v:
                        v = [x.item() if hasattr(x, 'item') else x for x in v]
                    row[field] = v if v is not None else ''
                writer.writerow(row)

        # JSON 摘要
        summary = {
            'pipeline': 'GPAW DFT Validation',
            'timestamp': datetime.now().isoformat(),
            'input_dir': self.config.input_dir,
            'output_dir': self.config.output_dir,
            'total_structures': len(self.records),
            'passed': sum(1 for r in self.records if r.all_dft_pass),
            'failed': sum(1 for r in self.records if not r.all_dft_pass),
            'pass_rate': f"{sum(1 for r in self.records if r.all_dft_pass) / max(len(self.records), 1):.1%}",
            'magnetic_breakdown': {
                'nonmag': sum(1 for r in self.records if r.mag_type == 'nonmag'),
                'mag_3d': sum(1 for r in self.records if r.mag_type == 'mag_3d'),
                'mag_4d5d': sum(1 for r in self.records if r.mag_type == 'mag_4d5d'),
                'mag_re': sum(1 for r in self.records if r.mag_type == 'mag_re'),
            },
            'failure_breakdown': {
                'relaxation': sum(1 for r in self.records if r.error_type == 'relaxation'),
                'elastic': sum(1 for r in self.records if r.error_type == 'elastic'),
                'phonon': sum(1 for r in self.records if r.error_type == 'phonon'),
                'ehull': sum(1 for r in self.records if r.error_type == 'ehull'),
                'unknown': sum(1 for r in self.records if r.error_type == 'unknown'),
            },
            'ehull_check_available': self.ehull_check_available,  # M4 凸包检验是否可用
        }

        # 如果凸包检验不可用，在摘要中增加警告
        if not self.ehull_check_available:
            summary['warnings'] = [
                "M4 convex hull stability check was NOT performed for any structure.",
                "Do NOT rely on all_dft_pass for true thermodynamic stability assessment.",
                "To enable: pip install typing_extensions OR pip install mp-api<0.41"
            ]

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """打印最终摘要"""
        n_total = len(self.records)
        n_pass = sum(1 for r in self.records if r.all_dft_pass)

        # 如果凸包检验不可用，在开头醒目打印警告
        if not self.ehull_check_available:
            print("\n" + "=" * 70)
            print("⚠️  警告：本次运行未进行任何凸包稳定性检验")
            print("⚠️  请勿依赖 all_dft_pass 判断材料的真实稳定性")
            print("⚠️  解决方案: pip install typing_extensions  或  pip install mp-api<0.41")
            print("=" * 70)

        print("\n" + "=" * 70)
        print("  GPAW DFT Validation Summary")
        print("=" * 70)
        print(f"  Total:    {n_total}")
        print(f"  Passed:   {n_pass} ({n_pass/max(n_total,1):.1%})")
        print(f"  Failed:   {n_total - n_pass} ({(n_total-n_pass)/max(n_total,1):.1%})")
        print("=" * 70)
        print("  Failure Breakdown:")
        for err_type in ['relaxation', 'elastic', 'phonon', 'ehull', 'unknown']:
            n = sum(1 for r in self.records if r.error_type == err_type)
            if n > 0:
                print(f"    {err_type:12s}: {n}")
        print("=" * 70)

        # 列出通过的结构
        passed = [r for r in self.records if r.all_dft_pass]
        if passed:
            print("\n  DFT-Validated Structures:")
            for rec in passed:
                print(f"    {rec.formula:20s} | {rec.mag_type:10s} | "
                      f"E={rec.m1_energy_per_atom:.4f} eV/atom | "
                      f"C11={rec.m2_c11:.1f} N/m")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# §9 CLI 接口
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(
        description="GPAW First-Principles DFT Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法
  python gpaw_validation_pipeline.py --input ./dft_ready --output ./gpaw_validation

  # 指定参数
  python gpaw_validation_pipeline.py -i ./dft_ready -o ./gpaw_out \\
      --encut 600 --relax_fmax 0.02 --dk_per_ang 0.05

  # MPI 并行（需要安装 mpi4py）
  mpirun -np 4 python gpaw_validation_pipeline.py -i ./dft_ready -o ./gpaw_out

  # 指定 MP API key
  python gpaw_validation_pipeline.py -i ./dft_ready -o ./gpaw_out \\
      --mp_api_key YOUR_API_KEY
        """
    )

    # 输入输出
    p.add_argument('-i', '--input', default=None,
                   help='筛选流水线输出目录 (required)')
    p.add_argument('-o', '--output', default='./gpaw_validation',
                   help='GPAW 验证结果输出目录 (default: ./gpaw_validation)')

    # GPAW 参数
    gpaw_grp = p.add_argument_group('GPAW DFT Parameters')
    gpaw_grp.add_argument('--encut', type=int, default=550,
                          help='平面波截断能 (eV, default: 550)')
    gpaw_grp.add_argument('--xc', default='PBE',
                          help='交换关联泛函 (default: PBE)')
    gpaw_grp.add_argument('--fermi', type=float, default=0.05,
                          help='Fermi-Dirac 展宽 (eV, default: 0.05)')
    gpaw_grp.add_argument('--dk', type=float, default=0.03,
                          help='最大 k 点间距 (Å⁻¹, default: 0.03)')
    gpaw_grp.add_argument('--spinpol', choices=['auto', 'true', 'false'],
                          default='auto', help='自旋极化 (default: auto)')

    # 自适应电子参数（能隙连续插值）
    adaptive_grp = p.add_argument_group('Adaptive Electronic Parameters (能隙连续插值)')
    adaptive_grp.add_argument('--gap_interp_scale', type=float, default=0.5,
                             help='能隙插值特征尺度 (eV, default: 0.5)')
    adaptive_grp.add_argument('--dos_interp_scale', type=float, default=1.0,
                             help='态密度插值特征尺度 (态/eV/atom, default: 1.0)')
    adaptive_grp.add_argument('--k_max_boost', type=float, default=2.0,
                             help='金属极限 k 密度最大加密倍数 (default: 2.0)')
    adaptive_grp.add_argument('--fermi_metal', type=float, default=0.02,
                             help='金属极限 Fermi 展宽下限 (eV, default: 0.02)')
    adaptive_grp.add_argument('--sc_min_base', type=float, default=10.0,
                             help='超胞最小尺寸基础值 (Å, default: 10.0)')
    adaptive_grp.add_argument('--sc_min_boost', type=float, default=5.0,
                             help='超胞最小尺寸金属极限增量 (Å, default: 5.0)')
    adaptive_grp.add_argument('--skip_gap_probe', action='store_true', default=False,
                             help='跳过能隙探测，始终使用金属级最保守设置')

    # 弛豫参数
    relax_grp = p.add_argument_group('Relaxation Parameters')
    relax_grp.add_argument('--relax_fmax', type=float, default=0.01,
                            help='力收敛阈值 (eV/Å, default: 0.01)')
    relax_grp.add_argument('--relax_steps', type=int, default=200,
                           help='最大弛豫步数 (default: 200)')
    relax_grp.add_argument('--optimizer', choices=['FIRE', 'BFGS'], default='FIRE',
                           help='优化器 (default: FIRE)')
    relax_grp.add_argument('--fix_cell', choices=['ab', 'full', 'none'], default='ab',
                           help='固定晶胞模式 (default: ab)')
    # A: 对称性约束参数
    relax_grp.add_argument('--symmetry_tol', type=float, default=0.01,
                           help='spglib 对称性识别容差 (default: 0.01)')
    # B1: 始终自旋极化
    relax_grp.add_argument('--always_spinpol', action='store_true', default=True,
                           help='始终使用自旋极化弛豫 (default: True)')
    relax_grp.add_argument('--no_always_spinpol', action='store_true',
                           help='关闭始终自旋极化弛豫')
    # B2: FM/NM 基态比较参数
    relax_grp.add_argument('--fm_nm_tol', type=float, default=1e-3,
                           help='FM/NM 简并阈值 (eV/atom, default: 1e-3)')
    relax_grp.add_argument('--no_fm_nm_compare', action='store_true',
                           help='关闭 FM/NM 基态比较和 NM 重弛豫')
    # 零步收敛验证参数
    relax_grp.add_argument('--no_zero_step_verify', action='store_true',
                           help='关闭零步收敛二次验证')
    relax_grp.add_argument('--zero_step_disp', type=float, default=0.05,
                           help='零步验证原子位移幅度 (Å, default: 0.05)')
    relax_grp.add_argument('--zero_step_thr', type=float, default=0.05,
                           help='零步验证能量差异阈值 (eV/atom, default: 0.05)')

    # 弹性参数
    elastic_grp = p.add_argument_group('Elastic Constants Parameters (C: 动态噪声容差)')
    elastic_grp.add_argument('--elastic_delta', type=float, default=0.005,
                              help='应变幅度 (default: 0.005 = ±0.5%%)')
    elastic_grp.add_argument('--elastic_fmax', type=float, default=0.01,
                              help='弹性计算力收敛阈值 (eV/Å, default: 0.01)')
    elastic_grp.add_argument('--elastic_relax_steps', type=int, default=100,
                              help='弹性计算内部弛豫最大步数 (default: 100)')
    # C: 动态噪声容差参数
    elastic_grp.add_argument('--born_noise_sigma', type=float, default=3.0,
                              help='Born判据噪声置信倍数 (default: 3.0)')
    elastic_grp.add_argument('--symmetry_noise_sigma', type=float, default=3.0,
                              help='六方对称检查噪声置信倍数 (default: 3.0)')

    # 声子参数（C: 动态噪声容差）
    phonon_grp = p.add_argument_group('Phonon Parameters (C: Dynamic Noise Tolerance)')
    phonon_grp.add_argument('--phonon_sc', default='3,3,1',
                             help='声子超胞尺寸 (default: 3,3,1)')
    phonon_grp.add_argument('--phonon_disp', type=float, default=0.01,
                            help='原子位移量 (Å, default: 0.01)')
    # C: 动态噪声容差参数
    phonon_grp.add_argument('--phonon_noise_sigma', type=float, default=3.0,
                            help='Γ点容差置信倍数 (default: 3.0)')
    phonon_grp.add_argument('--phonon_noise_floor_max', type=float, default=1.0,
                            help='噪声估计上限 THz，超过此值视为方法失效 (default: 1.0)')
    phonon_grp.add_argument('--phonon_boundary_tighten', type=float, default=5.0,
                            help='BZ边界容差收紧倍数 (default: 5.0)')
    # 静态兜底值
    phonon_grp.add_argument('--phonon_gamma_tol', type=float, default=-0.1,
                            help='Γ点静态兜底容差 (THz, default: -0.1)')
    phonon_grp.add_argument('--phonon_boundary_tol', type=float, default=-0.1,
                            help='BZ边界静态兜底容差 (THz, default: -0.1)')
    phonon_grp.add_argument('--no_phonon_pdf', action='store_true',
                            help='禁用声子谱 PDF 生成')

    # 凸包参数
    ehull_grp = p.add_argument_group('Convex Hull Parameters')
    ehull_grp.add_argument('--ehull_thr', type=float, default=0.2,
                            help='凸包能量阈值 (eV/atom, default: 0.2)')
    ehull_grp.add_argument('--ehull_ref_pool_size', type=int, default=3,
                            help='方案B: 每个化学子式取MP里最稳定的前N个候选 (default: 3)')
    ehull_grp.add_argument('--no_local_ref_ehull', action='store_true',
                            help='禁用方案B(本地GPAW统一重算)，降级到旧方案(MP能量近似)')
    ehull_grp.add_argument('--mp_api_key', default='',
                            help='Materials Project API key (或设置 MP_API_KEY 环境变量)')

    # 重试与并行
    retry_grp = p.add_argument_group('Retry & Parallel')
    retry_grp.add_argument('--scf_retry', type=int, default=2,
                            help='SCF 不收敛重试次数 (default: 2)')
    retry_grp.add_argument('--max_cores', type=int, default=32,
                            help='最大并行核数 (default: 32)')

    # ── Stage 控制与细粒度断点续算 ─────────────────────────────────
    stage_grp = p.add_argument_group('Stage Control & Fine-grained Checkpoint')
    stage_grp.add_argument('--stage', choices=['all', 'm1', 'm2', 'm3', 'm4', 'm12', 'resume_failed'],
                          default='resume_failed',
                          help='执行模式: all=从头执行全部, m1/m2/m3/m4=仅执行指定模块, m12=M1+GAP+M2, '
                               'resume_failed=细粒度续算，仅重跑未通过的模块 (default: resume_failed)')
    stage_grp.add_argument('--force_stage', action='store_true',
                           help='强制重跑指定 stage，忽略该 stage 的已有结果')
    stage_grp.add_argument('--list_stages', action='store_true',
                           help='仅列出 checkpoint 中各结构的 stage 状态，不执行计算')
    # ─────────────────────────────────────────────────────────────

    # 输出控制
    p.add_argument('--no_poscar', action='store_true',
                   help='禁用 DFT-Refined POSCAR 输出')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='详细输出')

    return p.parse_args()


def main():
    """主入口"""
    args = parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 构建配置
    config = GPAWConfig(
        input_dir=args.input,
        output_dir=args.output,
        encut=args.encut,
        xc=args.xc,
        fermi_smearing=args.fermi,
        dk_per_ang=args.dk,
        spinpol={'auto': None, 'true': True, 'false': False}[args.spinpol],
        # 自适应电子参数（能隙连续插值）
        gap_interp_scale=args.gap_interp_scale,
        dos_interp_scale=args.dos_interp_scale,
        k_max_boost=args.k_max_boost,
        fermi_smearing_metal=args.fermi_metal,
        supercell_min_length_base=args.sc_min_base,
        supercell_min_length_boost=args.sc_min_boost,
        skip_gap_probe=args.skip_gap_probe,
        # 弛豫参数
        relax_fmax=args.relax_fmax,
        relax_steps=args.relax_steps,
        optimizer=args.optimizer,
        fix_cell=args.fix_cell,
        symmetry_tol=args.symmetry_tol,
        always_spinpol=(not args.no_always_spinpol),
        # B2: FM/NM 基态比较参数
        fm_nm_degenerate_tol_eV=args.fm_nm_tol,
        fm_nm_compare_before_m2=(not args.no_fm_nm_compare),
        # 零步收敛验证参数
        zero_step_verify=(not args.no_zero_step_verify),
        zero_step_displacement=args.zero_step_disp,
        zero_step_verify_threshold=args.zero_step_thr,
        elastic_delta=args.elastic_delta,
        elastic_fmax=args.elastic_fmax,
        elastic_relax_steps=args.elastic_relax_steps,
        born_noise_sigma=args.born_noise_sigma,
        symmetry_noise_sigma=args.symmetry_noise_sigma,
        phonon_supercell=[int(x) for x in args.phonon_sc.split(',')],
        phonon_displacement=args.phonon_disp,
        # C: 动态噪声容差参数
        phonon_noise_sigma=args.phonon_noise_sigma,
        phonon_noise_floor_max_THz=args.phonon_noise_floor_max,
        phonon_boundary_tighten_factor=args.phonon_boundary_tighten,
        # 静态兜底值
        phonon_gamma_tol=args.phonon_gamma_tol,
        phonon_boundary_tol=args.phonon_boundary_tol,
        ehull_threshold=args.ehull_thr,
        ehull_ref_pool_size=args.ehull_ref_pool_size,
        use_local_ref_ehull=not args.no_local_ref_ehull,
        save_poscar=not args.no_poscar,
        save_phonon_pdf=not args.no_phonon_pdf,
        scf_retry_max=args.scf_retry,
    )

    # MP API key
    mp_key = args.mp_api_key or os.environ.get('MP_API_KEY', '')

    # 执行流水线
    pipeline = GPAWValidationPipeline(config)

    # ── Stage 控制与细粒度断点续算 ───────────────────────────────
    # 传递 stage 参数给流水线
    pipeline.stage = args.stage
    pipeline.force_stage = args.force_stage
    # ───────────────────────────────────────────────────────────

    # 执行流水线
    pipeline.run(mp_api_key=mp_key)


# ═════════════════════════════════════════════════════════════════════════════
# §10 单元测试
# ═════════════════════════════════════════════════════════════════════════════

def _quick_test():
    """快速单元测试"""
    print("\n" + "=" * 70)
    print("  GPAW Validation Pipeline - Quick Test")
    print("=" * 70)

    # 测试磁性判定
    print("\n[Magnetic Classification Test]")
    test_cases = [
        (['Mo', 'S', 'S'], 'mag_4d5d', True),
        (['Fe', 'O'], 'mag_3d', True),
        (['Ti', 'C'], 'mag_3d', True),
        (['B', 'N'], 'nonmag', False),
        (['Gd', 'N'], 'mag_re', True),
        (['Si'], 'nonmag', False),
    ]

    for elements, expected_type, expected_mag in test_cases:
        mag_type, is_mag = classify_magnetic(elements)
        status = '✓' if mag_type == expected_type and is_mag == expected_mag else '✗'
        print(f"  {status} {elements} -> type={mag_type}, is_mag={is_mag}")

    # 测试 ValidationRecord 新字段存在性
    print("\n[ValidationRecord m1_resumed_from_partial Field Test]")
    rec_test = ValidationRecord()
    assert hasattr(rec_test, 'm1_resumed_from_partial'), "缺少 m1_resumed_from_partial 字段"
    assert rec_test.m1_resumed_from_partial == False, "默认值应为 False"
    print(f"  ✓ m1_resumed_from_partial 字段存在，默认值={rec_test.m1_resumed_from_partial}")

    # 测试初态磁矩配置
    print("\n[Initial Magnetic Moments Test]")
    from ase import Atoms
    test_atoms = Atoms('Fe2O3', positions=[(0,0,0), (1,0,0), (0,1,0), (0.5,0.5,0), (0.5,0,0)],
                        cell=[3, 3, 20], pbc=[True, True, True])
    magmoms = build_initial_magmoms(test_atoms)
    print(f"  Fe2O3 magmoms: {magmoms}")
    assert len(magmoms) == 5, "磁矩配置数量错误"
    assert magmoms[0] == 4.0, "Fe 磁矩应为 4.0"

    # 测试 k 点自适应
    print("\n[Adaptive k-points Test]")
    from dfca.validation.electronic.symmetry import adaptive_kpoints
    kpts, koffset = adaptive_kpoints(test_atoms, dk=0.03)
    print(f"  kpts: {kpts}, offset: {koffset}")
    assert kpts[2] == 1, "c 方向 k 点应为 1"

    # 测试自适应电子参数插值函数
    print("\n[Adaptive Electronic Parameters Test]")
    config_test = GPAWConfig()

    # 常规金属极限：gap=0, DOS 较大 (dirac_factor -> 0)
    metal_params = adaptive_electronic_params(
        gap_eV=0.0, dos_at_ef=999.0, config=config_test
    )
    print(f"  常规金属极限 (gap=0, DOS=999.0):")
    print(f"    k_mult={metal_params['kpoint_multiplier']:.2f} (期望≈3.0)")
    print(f"    smearing={metal_params['fermi_smearing']:.4f} eV (期望≈0.02)")
    print(f"    sc_min={metal_params['supercell_min_length']:.1f} Å (期望≈15.0)")
    assert abs(metal_params['kpoint_multiplier'] - 3.0) < 0.1, "常规金属极限 k_mult 应接近 3.0"
    assert abs(metal_params['fermi_smearing'] - 0.02) < 0.01, "常规金属极限 smearing 应接近 0.02"

    # 半导体：gap 很大
    semi_params = adaptive_electronic_params(
        gap_eV=5.0, dos_at_ef=0.0, config=config_test
    )
    print(f"  半导体 (gap=5.0 eV):")
    print(f"    k_mult={semi_params['kpoint_multiplier']:.2f} (期望≈1.0)")
    print(f"    smearing={semi_params['fermi_smearing']:.4f} eV (期望≈0.05)")
    print(f"    sc_min={semi_params['supercell_min_length']:.1f} Å (期望≈10.0)")
    assert abs(semi_params['kpoint_multiplier'] - 1.0) < 0.5, "半导体 k_mult 应接近 1.0"
    assert abs(semi_params['fermi_smearing'] - 0.05) < 0.01, "半导体 smearing 应接近 0.05"

    # 中间情况：gap=1eV（接近 MoS2）
    mos2_params = adaptive_electronic_params(
        gap_eV=1.6, dos_at_ef=0.0, config=config_test
    )
    print(f"  MoS2 类 (gap=1.6 eV):")
    print(f"    k_mult={mos2_params['kpoint_multiplier']:.2f}")
    print(f"    smearing={mos2_params['fermi_smearing']:.4f} eV")
    print(f"    sc_min={mos2_params['supercell_min_length']:.1f} Å")

    # BN (gap=5.5eV)
    bn_params = adaptive_electronic_params(
        gap_eV=5.5, dos_at_ef=0.0, config=config_test
    )
    print(f"  BN 类 (gap=5.5 eV):")
    print(f"    k_mult={bn_params['kpoint_multiplier']:.2f}")
    print(f"    smearing={bn_params['fermi_smearing']:.4f} eV")
    print(f"    sc_min={bn_params['supercell_min_length']:.1f} Å")

    # 石墨烯 (gap=0, dos 趋于 0)
    graphene_params = adaptive_electronic_params(
        gap_eV=0.0, dos_at_ef=0.001, config=config_test
    )
    print(f"  石墨烯类 (gap=0, DOS→0):")
    print(f"    k_mult={graphene_params['kpoint_multiplier']:.2f} (狄拉克点，额外加密)")
    print(f"    smearing={graphene_params['fermi_smearing']:.4f} eV")
    print(f"    sc_min={graphene_params['supercell_min_length']:.1f} Å")
    assert graphene_params['kpoint_multiplier'] > metal_params['kpoint_multiplier'], \
        "狄拉克点材料应比普通金属更密"

    print("\n[Summary Table]")
    print(f"  {'Material':<15} {'gap(eV)':<10} {'DOS':<8} {'k_mult':<10} {'smearing':<12} {'sc_min':<10}")
    print(f"  {'-'*65}")
    print(f"  {'Metal':<15} {'0.0':<10} {'999':<8} {metal_params['kpoint_multiplier']:<10.2f} {metal_params['fermi_smearing']:<12.4f} {metal_params['supercell_min_length']:<10.1f}")
    print(f"  {'BN (5.5eV)':<15} {'5.5':<10} {'0.0':<8} {bn_params['kpoint_multiplier']:<10.2f} {bn_params['fermi_smearing']:<12.4f} {bn_params['supercell_min_length']:<10.1f}")
    print(f"  {'MoS2 (1.6eV)':<15} {'1.6':<10} {'0.0':<8} {mos2_params['kpoint_multiplier']:<10.2f} {mos2_params['fermi_smearing']:<12.4f} {mos2_params['supercell_min_length']:<10.1f}")
    print(f"  {'Graphene':<15} {'0.0':<10} {'0.001':<8} {graphene_params['kpoint_multiplier']:<10.2f} {graphene_params['fermi_smearing']:<12.4f} {graphene_params['supercell_min_length']:<10.1f}")

    print("\n" + "=" * 70)
    print("  Quick Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _quick_test()
    else:
        main()
