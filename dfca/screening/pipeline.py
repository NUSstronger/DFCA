"""
Screening Pipeline Orchestrator.

Two-phase screening pipeline for 2D materials.
Phase 1: S0 → S2 → S3 → S3.5 → S-dim → S3T → S3P (Pareto)
Phase 2: S4 → S5 → S7a → S7b → S5' (expensive physics)

References:
    - C2DB screening protocol v3.2
"""

import os
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import asdict

import numpy as np

from dfca.screening.config import ScreeningConfig
from dfca.screening.record import ScreeningRecord
from dfca.screening.backend import MLBackend

# Import screening stages
from dfca.screening.stages.s0_composition import check_composition
from dfca.screening.stages.s1_geometry import check_geometry_2d
from dfca.screening.stages.s2_symmetry import check_symmetry_2d
from dfca.screening.stages.s3_dedup import BucketedDeduplicator, build_deduplicator
from dfca.screening.stages.s35_relaxation import relax_2d_with_retry
from dfca.screening.stages.s_thermo import compute_ehull, precompute_element_references
from dfca.screening.stages.s5_phonon import check_phonon_stability

# Import external modules
from pymatgen.core import Structure

# Optional imports
try:
    from pymatgen.analysis.dimensionality import get_dimensionality_larsen
    HAS_DIM_CHECK = True
except ImportError:
    HAS_DIM_CHECK = False

try:
    from dfca.screening.elastic_matrix import compute_elastic_matrix
    HAS_ELASTIC = True
except ImportError:
    HAS_ELASTIC = False

try:
    from dfca.screening.md_screening import run_md_s7a, run_md_s7b
    HAS_MD = True
except ImportError:
    HAS_MD = False

try:
    from dfca.screening.convex_hull import compute_ehull_mp
    HAS_CONVEX_HULL = True
except ImportError:
    HAS_CONVEX_HULL = False

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


def _json_default(obj):
    """JSON serialization fallback."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ScreeningPipeline:
    """
    Two-phase screening pipeline for 2D materials.

    Phase 1 (fast): S0 → S2 → S3 → S3.5 → S-dim → S3T → S3P (Pareto)
    Phase 2 (expensive): S4 → S5 → S7a → S7b → S5'

    Features:
        - Checkpoint/restart support
        - Phase 2 recovery mode
        - VASP job export
        - Pareto front visualization
    """

    def __init__(self, cfg: ScreeningConfig):
        """Initialize pipeline with configuration."""
        self.cfg = cfg
        self.backend: Optional[MLBackend] = None
        self.dedup: Optional[BucketedDeduplicator] = None
        self.records: List[ScreeningRecord] = []
        self._cif_cache: Dict[str, Structure] = {}
        self._relaxed_cifs_dir: Path = Path(cfg.output_dir) / "relaxed_cifs"
        self._elem_refs: Dict[str, float] = dict(cfg.reference_energies)
        self._known_formulas: set = set()

        # Setup directories
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(self._relaxed_cifs_dir, exist_ok=True)

        # Paths
        self._ckpt_path = os.path.join(cfg.output_dir, "_checkpoint.json")
        self._report_path = os.path.join(cfg.output_dir, "screening_report.csv")
        self._summary_path = os.path.join(cfg.output_dir, "summary.json")

        # Load training formulas if provided
        if cfg.train_csv:
            self._load_train_formulas(cfg.train_csv)

    def _ensure_backend(self):
        """Ensure ML backend is initialized."""
        if self.backend is None:
            logger.info("正在挂载并调配后端机器学习模型PES环境...")
            self.backend = MLBackend(self.cfg.device, self.cfg.mace_model)

    def _ensure_dedup(self):
        """Ensure deduplicator is initialized."""
        if self.dedup is None:
            self.dedup = BucketedDeduplicator(
                build_deduplicator(self.cfg),
                symprec=self.cfg.symprec
            )

    def _load_train_formulas(self, csv_path: str):
        """Load training set formulas for novelty filtering."""
        import pandas as pd

        try:
            df = pd.read_csv(csv_path)
            formula_col = None
            candidate_cols = [
                "pretty_formula", "formula", "reduced_formula",
                "Composition", "composition", "formula_pretty", "Formula"
            ]
            for col in candidate_cols:
                if col in df.columns:
                    formula_col = col
                    break

            if formula_col is not None:
                raw_formulas = df[formula_col].dropna().unique()
                std_formulas = set()
                for f in raw_formulas:
                    try:
                        std_formulas.add(
                            Structure.from_str(str(f), fmt="cif").composition.reduced_formula
                        )
                    except Exception:
                        std_formulas.add(
                            Composition(str(f).strip()).reduced_formula
                            if 'Composition' in dir() else str(f).strip()
                        )
                self._known_formulas = std_formulas
                logger.info(f"[novelty] 从列 '{formula_col}' 成功加载 {len(self._known_formulas):,} 个已知 formula")

        except Exception as e:
            logger.warning(f"加载训练集 formula 失败: {e}，跳过 novelty 过滤")

    def _warmup_references(self, all_elements: List[str]):
        """Precompute element reference energies."""
        need = [el for el in set(all_elements) if el not in self._elem_refs]
        if not need:
            return
        self._ensure_backend()
        new_refs = precompute_element_references(need, self.backend, self.cfg)
        self._elem_refs.update(new_refs)
        logger.info(f"成功预缓存自洽单质基准参考能数据库: {sorted(self._elem_refs.keys())}")

    def _is_phase2_complete(self, rec: ScreeningRecord) -> bool:
        """Check if record has completed Phase 2."""
        if rec.final_pass:
            return True
        if rec.failure_type in ('physical', 'numerical', 'born_unstable', 'phonon_unstable'):
            return True
        if rec.s5_pass is True:
            return True
        if rec.s4_pass is True:
            return True
        if rec.s5_reason and rec.s5_reason != "skipped_uncertain_convergence":
            return True
        if rec.s4_reason and "elastic_matrix" in rec.s4_reason:
            return True
        if rec.s5p_pass is not False and rec.s5p_reason:
            return True
        return False

    def load_structures(self) -> Tuple[List[Structure], List[str]]:
        """Load structures from input directory."""
        cif_files = sorted(Path(self.cfg.input_dir).glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(
                f"未能在指定的目标输入路径下检出有效 .cif 结构文件: {self.cfg.input_dir}"
            )
        structs, names, failed = [], [], 0
        for f in tqdm(cif_files, desc="全量结构载入") if HAS_TQDM else cif_files:
            try:
                struct = Structure.from_file(str(f))
                structs.append(struct)
                names.append(f.stem)
            except Exception:
                failed += 1
        return structs, names

    def _save_checkpoint(self):
        """Save checkpoint with records and relaxed structures."""
        relaxed_cifs = {}
        if self._relaxed_cifs_dir.exists():
            for cif_file in self._relaxed_cifs_dir.glob("*.cif"):
                try:
                    struct = Structure.from_file(str(cif_file))
                    relaxed_cifs[cif_file.stem] = struct.to(fmt="cif")
                except Exception:
                    pass

        checkpoint_data = {
            "records": [asdict(r) for r in self.records],
            "relaxed_structures": relaxed_cifs,
        }

        tmp_path = self._ckpt_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(checkpoint_data, fp, indent=2, default=_json_default)
        os.replace(tmp_path, self._ckpt_path)

    def _load_checkpoint(self) -> bool:
        """Load checkpoint from file."""
        if not os.path.exists(self._ckpt_path):
            return False

        with open(self._ckpt_path, encoding="utf-8") as fp:
            content = fp.read()

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            # Recovery from partial JSON
            logger.warning(f"Checkpoint JSON 不完整: {e}，尝试 raw_decode 提取...")
            try:
                parsed, end_idx = json.JSONDecoder().raw_decode(content)
                raw = parsed if isinstance(parsed, dict) else {"records": parsed, "relaxed_structures": {}}
            except Exception:
                return False

        if isinstance(raw, dict):
            records_data = raw.get("records", raw)
            ckpt_cifs = raw.get("relaxed_structures", {})
        else:
            records_data = raw
            ckpt_cifs = {}

        self.records = []
        for d in records_data:
            rec = ScreeningRecord(name=d["name"])
            rec.__dict__.update(d)
            self.records.append(rec)

        # Load relaxed structures
        loaded_count = 0
        if self._relaxed_cifs_dir.exists():
            for cif_file in self._relaxed_cifs_dir.glob("*.cif"):
                if cif_file.stem not in self._cif_cache:
                    try:
                        self._cif_cache[cif_file.stem] = Structure.from_file(str(cif_file))
                        loaded_count += 1
                    except Exception:
                        pass

        for name, cif_str in ckpt_cifs.items():
            if name not in self._cif_cache:
                try:
                    self._cif_cache[name] = Structure.from_str(cif_str, fmt="cif")
                    loaded_count += 1
                except Exception:
                    pass

        logger.info(f"检出断点续算记录。成功自动载入 {len(self.records)} 条历史数据，"
                    f"已弛豫结构: {loaded_count} 个")
        return True

    def _screen_one_phase1(self, struct: Structure, name: str
                          ) -> Tuple[ScreeningRecord, Optional[Structure]]:
        """Execute Phase 1 screening stages."""
        from pymatgen.core import Composition

        t0 = time.time()
        rec = ScreeningRecord(name=name, formula=struct.composition.reduced_formula, n_atoms=len(struct))

        # Novelty filter
        if self._known_formulas:
            formula = struct.composition.reduced_formula
            if formula in self._known_formulas:
                rec.s0_pass = False
                rec.s0_reason = "known_formula"
                rec.elapsed_s = time.time() - t0
                return rec, None

        # S0 Composition
        s0_ok, s0_info = check_composition(struct, self.cfg)
        rec.s0_pass = s0_ok
        rec.s0_reason = s0_info.get("reason", "")
        if not s0_ok:
            rec.elapsed_s = time.time() - t0
            return rec, None

        # S1 Geometry
        s1_ok, s1_info = check_geometry_2d(struct, self.cfg)
        rec.s1_vacuum_angstrom = s1_info.get("vacuum_angstrom", 0.0)
        rec.s1_thickness_angstrom = s1_info.get("thickness_angstrom", 0.0)
        rec.s1_min_bond = s1_info.get("min_bond_angstrom", 0.0)
        rec.s1_gamma = s1_info.get("gamma", 0.0)
        rec.s1_aspect_ratio = s1_info.get("aspect_ratio", 0.0)
        rec.s1_pass = s1_ok
        rec.s1_reason = s1_info.get("reason", "")
        if not s1_ok:
            rec.elapsed_s = time.time() - t0
            return rec, None

        # S2 Symmetry
        s2_ok, s2_info = check_symmetry_2d(struct, self.cfg)
        rec.s2_space_group = s2_info.get("space_group", "?")
        rec.s2_layer_group = s2_info.get("layer_group", "?")
        rec.s2_pass = s2_ok
        rec.s2_reason = s2_info.get("reason", "")
        if not s2_ok:
            rec.elapsed_s = time.time() - t0
            return rec, None

        # S3 Deduplication
        self._ensure_dedup()
        is_dup, dup_of, cluster_id = self.dedup.is_duplicate(struct, name)
        rec.s3_is_duplicate = is_dup
        rec.s3_duplicate_of = dup_of
        if is_dup:
            rec.elapsed_s = time.time() - t0
            return rec, None

        # S3.5 ML Relaxation
        self._ensure_backend()
        converged, rel_struct, ml_energy, s35_reason, conv_status = relax_2d_with_retry(
            struct, self.backend, self.cfg
        )
        rec.s35_ml_energy_eV = round(ml_energy, 4)
        rec.s35_ml_energy_per_atom = round(ml_energy / len(struct), 4)
        rec.s35_relax_converged = converged
        rec.s35_convergence_status = conv_status

        if conv_status == "uncertain":
            rec.s35_pass = True
            rec.s35_reason = s35_reason
            rec.elapsed_s = time.time() - t0
            return rec, None

        work_struct = rel_struct if converged else struct

        # Post-relaxation geometry check
        post_ok, post_info = check_geometry_2d(work_struct, self.cfg)
        rec.s35_post_geom_ok = post_ok
        if not post_ok:
            rec.s35_pass = False
            rec.s35_reason = f"post_relax_geom_fail:{post_info.get('reason')}"
            rec.elapsed_s = time.time() - t0
            return rec, None

        # Post-relaxation deduplication
        is_dup2, dup_of2, cluster_id2 = self.dedup.is_duplicate(work_struct, name, exclude_name=name)
        if is_dup2:
            rec.s3_is_duplicate = True
            rec.s3_duplicate_of = dup_of2
            rec.s35_pass = False
            rec.s35_reason = f"post_relax_dedup_failed:duplicate_of_{dup_of2}"
            rec.elapsed_s = time.time() - t0
            return rec, None

        # S-dim connectivity check
        if HAS_DIM_CHECK:
            sdim_ok, sdim_info = check_is_2d_connected(work_struct)
            rec.sdim_dimensionality = sdim_info.get("dimensionality", -1)
            rec.sdim_pass = sdim_ok
            rec.sdim_reason = sdim_info.get("reason", "")
            if not sdim_ok:
                rec.s35_pass = True
                rec.elapsed_s = time.time() - t0
                return rec, None
        else:
            rec.sdim_pass = True
            rec.sdim_reason = "module_not_available"

        rec.s35_pass = True
        rec.s35_reason = "ok" if converged else "not_converged_but_geom_ok"

        # S3T Thermodynamics
        try:
            e_hull, e_form, elem_e_diff, s3t_reason = compute_ehull(
                work_struct, ml_energy, self._elem_refs, self.cfg
            )
            rec.s3t_formation_energy_per_atom = round(e_form, 4)
            rec.s3t_e_above_hull_elem = round(elem_e_diff, 4) if elem_e_diff is not None else None
            rec.s3t_reason = s3t_reason

            # Stability labeling
            if e_form > 0:
                rec.s3t_reason = f"{s3t_reason}; stability=LOW"
            elif e_hull is not None and e_hull >= 0 and e_hull > 0.2:
                rec.s3t_reason = f"{s3t_reason}; stability=MEDIUM"
            elif e_hull is not None and e_hull >= 0 and e_hull <= 0.2:
                rec.s3t_reason = f"{s3t_reason}; stability=HIGH"

            rec.s3t_pass = True
            rec.s6_formation_energy = round(e_form * len(struct), 4)
            rec.s6_formation_energy_per_atom = round(e_form, 4)
            rec.s6_reason = s3t_reason
            rec.s6_pass = True

            self.dedup.add_to_bucket(work_struct, name, e_form, cluster_id=cluster_id2)

        except Exception as e:
            rec.s3t_pass = False
            rec.s3t_reason = f"compute_ehull_error:{e}"
            rec.elapsed_s = time.time() - t0
            return rec, None

        # Save relaxed CIF
        try:
            rel_cif_path = self._relaxed_cifs_dir / f"{name}.cif"
            work_struct.to(filename=str(rel_cif_path))
        except Exception as save_ex:
            logger.warning(f"弛豫 CIF 物理落盘失败 {name}: {save_ex}")

        rec.elapsed_s = time.time() - t0
        return rec, work_struct

    def _screen_one_phase2(self, rec: ScreeningRecord, work_struct: Structure
                          ) -> Tuple[ScreeningRecord, Optional[Structure]]:
        """Execute Phase 2 screening stages."""
        t0 = time.time()
        t_phase1 = rec.elapsed_s

        # S-dim (if not already done)
        if not rec.sdim_pass and HAS_DIM_CHECK:
            sdim_ok, sdim_info = check_is_2d_connected(work_struct)
            rec.sdim_dimensionality = sdim_info.get("dimensionality", -1)
            rec.sdim_pass = sdim_ok
            rec.sdim_reason = sdim_info.get("reason", "")
            if not sdim_ok:
                rec.failure_type = "physical"
                rec.raw_error = 'sdim_fail:' + sdim_info.get('reason', '')
                rec.elapsed_s = t_phase1 + (time.time() - t0)
                return rec, None

        # S4 Elastic Matrix
        if HAS_ELASTIC:
            def _classify_failure(err_str):
                err_lower = err_str.lower()
                if any(k in err_lower for k in ['nan', 'diverge', 'explosion', 'instability', 'overflow']):
                    return 'physical'
                if any(k in err_lower for k in ['not converged', 'singular', 'matrix', 'failed', 'error', 'unable']):
                    return 'numerical'
                return 'unknown'

            s4_passed = False
            for attempt in range(2):
                try:
                    d = self.cfg.elastic_delta
                    c2d, s4_stable, s4_reason, s4_stress_res, s4_energy_traj = compute_elastic_matrix(
                        work_struct, self.backend.get_calculator(),
                        fmax=self.cfg.relax_fmax,
                        max_steps=self.cfg.relax_steps,
                        strain_amps=(-2 * d, -d, d, 2 * d),
                    )
                    rec.s4_c11 = c2d.get("C11", 0.0)
                    rec.s4_c12 = c2d.get("C12", 0.0)
                    rec.s4_c22 = c2d.get("C22", 0.0)
                    rec.s4_c66 = c2d.get("C66", 0.0)
                    rec.s4_stress_residual = s4_stress_res
                    rec.s4_energy_traj = s4_energy_traj
                    rec.s4_pass = s4_stable
                    rec.s4_reason = s4_reason
                    if s4_stable:
                        s4_passed = True
                        break
                    else:
                        rec.failure_type = 'physical'
                        rec.raw_error = f"born_unstable:{s4_reason}"
                        rec.elapsed_s = t_phase1 + (time.time() - t0)
                        return rec, None
                except Exception as e:
                    err_str = str(e)
                    ftype = _classify_failure(err_str)
                    rec.failure_type = ftype
                    rec.raw_error = err_str
                    if ftype == 'physical' or attempt == 1:
                        rec.s4_pass = False
                        rec.s4_reason = f"elastic_matrix_error:{e}"
                        rec.elapsed_s = t_phase1 + (time.time() - t0)
                        return rec, None
                    rec.s4_reason = f"elastic_matrix_retry({attempt+1}):{e}"
        else:
            rec.s4_pass = False
            rec.s4_reason = "module_not_available"
            rec.elapsed_s = t_phase1 + (time.time() - t0)
            return rec, None

        # S5 Phonon
        if rec.s35_convergence_status == "uncertain":
            rec.s5_pass = None
            rec.s5_reason = "skipped_uncertain_convergence"
        else:
            ph_ok, ph_info = check_phonon_stability(work_struct, self.backend, self.cfg)
            rec.s5_gamma_min_THz = ph_info.get("gamma_min_THz", 0.0)
            rec.s5_min_freq_THz = ph_info.get("min_freq_THz", 0.0)
            rec.s5_imaginary_count = ph_info.get("imaginary_count", 0)
            rec.s5_pass = ph_ok
            rec.s5_reason = ph_info.get("reason", "")
            if not ph_ok:
                rec.elapsed_s = t_phase1 + (time.time() - t0)
                return rec, None

        # S7a MD Fast
        if self.cfg.enable_md and HAS_MD:
            def _classify_md_failure(err_str):
                err_lower = err_str.lower()
                if any(k in err_lower for k in ['explosion', 'nan', 'overflow', 'instability']):
                    return 'physical'
                return 'numerical'

            s7a_passed = False
            for attempt in range(2):
                dt_current = self.cfg.md_dt_fs / (2 ** attempt)
                try:
                    s7a_ok, s7a_info = run_md_s7a(
                        work_struct, self.backend.get_calculator(),
                        temperature=self.cfg.md_temperature,
                        dt_fs=dt_current,
                        n_steps=self.cfg.md_s7a_steps,
                    )
                    rec.s7a_struct_integrity_ok = s7a_info.get("struct_integrity_ok", False)
                    rec.s7a_max_z_drift = s7a_info.get("max_z_drift", 0.0)
                    rec.s7a_fragment_count = s7a_info.get("fragment_count", 0)
                    rec.s7a_pass = s7a_ok
                    rec.s7a_reason = s7a_info.get("reason", "")
                    if s7a_ok:
                        s7a_passed = True
                        break
                    else:
                        rec.failure_type = 'physical'
                        rec.raw_error = f"md_integrity:{s7a_info.get('reason', '')}"
                        rec.elapsed_s = t_phase1 + (time.time() - t0)
                        return rec, None
                except Exception as e:
                    err_str = str(e)
                    ftype = _classify_md_failure(err_str)
                    rec.failure_type = ftype
                    rec.raw_error = err_str
                    if ftype == 'physical' or attempt == 1:
                        rec.s7a_pass = False
                        rec.s7a_reason = f"md_s7a_error:{e}"
                        rec.elapsed_s = t_phase1 + (time.time() - t0)
                        return rec, None
                    rec.s7a_reason = f"md_s7a_retry({attempt+1}):{e}"
        else:
            rec.s7a_pass = True
            rec.s7a_reason = "module_not_available" if self.cfg.enable_md else "md_disabled"

        # S7b MD Full
        if self.cfg.enable_md and HAS_MD:
            s7b_passed = False
            for attempt in range(2):
                dt_current = self.cfg.md_dt_fs / (2 ** attempt)
                try:
                    s7b_ok, s7b_info = run_md_s7b(
                        work_struct, self.backend.get_calculator(),
                        temperature=self.cfg.md_temperature,
                        dt_fs=dt_current,
                        n_steps=self.cfg.md_s7b_steps,
                    )
                    rec.s7b_struct_integrity_ok = s7b_info.get("struct_integrity_ok", False)
                    rec.s7b_energy_trend_ok = s7b_info.get("energy_trend_ok", False)
                    rec.s7b_temperature_ok = s7b_info.get("temperature_ok", False)
                    rec.s7b_t_max = s7b_info.get("T_max", 0.0)
                    rec.s7b_energy_curve_2ps = s7b_info.get("energy_curve_2ps", [])
                    rec.s7b_atom_snapshot = s7b_info.get("atom_snapshot", "")
                    rec.s7b_pass = s7b_ok
                    rec.s7b_reason = s7b_info.get("reason", "")
                    if s7b_ok:
                        s7b_passed = True
                        break
                    else:
                        rec.failure_type = 'physical'
                        rec.raw_error = f"md_s7b_criteria:{s7b_info.get('reason', '')}"
                        rec.elapsed_s = t_phase1 + (time.time() - t0)
                        return rec, None
                except Exception as e:
                    err_str = str(e)
                    ftype = _classify_md_failure(err_str)
                    rec.failure_type = ftype
                    rec.raw_error = err_str
                    if ftype == 'physical' or attempt == 1:
                        rec.s7b_pass = False
                        rec.s7b_reason = f"md_s7b_error:{e}"
                        rec.elapsed_s = t_phase1 + (time.time() - t0)
                        return rec, None
                    rec.s7b_reason = f"md_s7b_retry({attempt+1}):{e}"
        else:
            rec.s7b_pass = True
            rec.s7b_reason = "module_not_available" if self.cfg.enable_md else "md_disabled"

        # S5' Convex Hull
        if HAS_CONVEX_HULL:
            _, s5p_stable, s5p_info = compute_ehull_mp(
                work_struct,
                energy_per_atom=rec.s35_ml_energy_per_atom,
                mp_api_key=self.cfg.mp_api_key,
                ehull_threshold=self.cfg.ehull_threshold_ml,
            )
            rec.s5p_ehull_eV = s5p_info.get("ehull_eV", -1.0)
            rec.s5p_competing_phases_found = s5p_info.get("competing_phases_found", 0)
            rec.s5p_missing_from_MP = s5p_info.get("missing_from_MP", [])
            rec.s5p_reason = s5p_info.get("reason", "")

            ehull = rec.s5p_ehull_eV
            if ehull < 0:
                rec.s5p_stability_label = "HIGH"
                rec.s5p_pass = True
            elif ehull < self.cfg.ehull_threshold_ml:
                rec.s5p_stability_label = "MEDIUM"
                rec.s5p_pass = True
            else:
                rec.s5p_stability_label = "LOW"
                rec.s5p_pass = not self.cfg.enable_hard_ehull_gate

            if not rec.s5p_pass:
                rec.elapsed_s = t_phase1 + (time.time() - t0)
                return rec, None
        else:
            rec.s5p_pass = False
            rec.s5p_reason = "module_not_available"
            rec.elapsed_s = t_phase1 + (time.time() - t0)
            return rec, None

        # Final check
        if rec.s35_convergence_status == "uncertain":
            rec.uncertain_final = True
            rec.final_pass = False
            rec.elapsed_s = t_phase1 + (time.time() - t0)
            return rec, work_struct

        rec.final_pass = True
        rec.elapsed_s = t_phase1 + (time.time() - t0)
        return rec, work_struct

    def run(self):
        """Execute two-phase screening pipeline."""
        structs, names = self.load_structures()
        n_total = len(structs)
        if n_total == 0:
            return

        all_elements = []
        for s in structs:
            all_elements += [str(el) for el in s.composition.elements]
        self._ensure_backend()
        self._warmup_references(all_elements)

        done_names = {r.name for r in self.records}
        todo = [(s, n) for s, n in zip(structs, names) if n not in done_names]

        self._ensure_dedup()

        # Phase 1
        if todo and not self.cfg.rerun_phase2:
            logger.info(f"[Phase1 新结构] 仍有 {len(todo)} 个结构待处理")
            iterator = tqdm(todo, desc="Phase1: S0→S3T", unit="struct") if HAS_TQDM else todo

            phase1_candidates = []
            for i, (struct, name) in enumerate(iterator):
                try:
                    rec, work_struct = self._screen_one_phase1(struct, name)
                except Exception as e:
                    logger.error(f"Phase1 严重物理异常阻断材料: {name}, info: {e}")
                    rec = ScreeningRecord(name=name, formula=struct.composition.reduced_formula)
                    work_struct = None

                self.records.append(rec)
                if rec.s3t_pass and work_struct is not None:
                    phase1_candidates.append((rec, work_struct))
                    self._cif_cache[rec.name] = work_struct

                if (i + 1) % self.cfg.save_interval == 0:
                    self._save_checkpoint()

        self._save_checkpoint()

        # Pareto front computation
        dominated_records = set()
        if phase1_candidates:
            self.dedup.compute_pareto_all_clusters()
            for rec, work_struct in phase1_candidates:
                cluster_id = self.dedup.get_cluster_id(rec.name)
                if cluster_id is None:
                    continue
                dominated, dom_by = self.dedup.is_pareto_dominated(cluster_id, rec.name)
                rec.s3_pareto_dominated = dominated
                rec.s3_pareto_dominated_by = dom_by
                if dominated:
                    dominated_records.add(rec.name)

        # Phase 2
        phase2_survivors = [
            (rec, work_struct)
            for rec, work_struct in phase1_candidates
            if rec.name not in dominated_records
        ]

        if not phase2_survivors:
            self._save_summary()
            self._print_funnel()
            return

        todo_phase2 = [
            (rec, work_struct)
            for rec, work_struct in phase2_survivors
            if not self._is_phase2_complete(rec)
        ]

        logger.info(f"[Phase2 开始] {len(todo_phase2)} 个 Pareto 幸存结构进入高耗时阶段")

        iterator2 = tqdm(todo_phase2, desc="Phase2: S-dim→S5'") if HAS_TQDM else todo_phase2

        for rec, work_struct in iterator2:
            try:
                rec, rel_struct = self._screen_one_phase2(rec, work_struct)
            except Exception as e:
                logger.error(f"Phase2 严重物理异常阻断材料: {rec.name}, info: {e}")
                rec.failure_type = 'numerical'
                rec.raw_error = str(e)
                rel_struct = None

            for idx, r in enumerate(self.records):
                if r.name == rec.name:
                    self.records[idx] = rec
                    break

            if rec.final_pass and rel_struct is not None:
                self._cif_cache[rec.name] = rel_struct

            self._save_checkpoint()

        self._save_checkpoint()
        self._save_summary()
        self._print_funnel()

    def _save_report(self):
        """Save CSV report of all records."""
        if not self.records:
            return
        rows = [asdict(r) for r in self.records]
        if HAS_PANDAS:
            pd.DataFrame(rows).to_csv(self._report_path, index=False)
        else:
            import csv
            with open(self._report_path, "w", newline="") as fp:
                w = csv.DictWriter(fp, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    def _save_summary(self):
        """Save summary statistics."""
        n = len(self.records)
        if n == 0:
            return
        cnt = lambda f: sum(1 for r in self.records if f(r))

        s = {
            "total_input": n,
            "pipeline_version": "v3.2_two_phase_pareto",
            "S0_composition": {"pass": cnt(lambda r: r.s0_pass)},
            "S1_geometry": {"pass": cnt(lambda r: r.s1_pass)},
            "S2_symmetry": {"pass": cnt(lambda r: r.s2_pass)},
            "S3.5_ml_relaxation": {"pass": cnt(lambda r: r.s35_pass)},
            "final_dft_candidates": cnt(lambda r: r.final_pass),
        }
        with open(self._summary_path, "w") as fp:
            json.dump(s, fp, indent=2, default=_json_default)

    def _print_funnel(self):
        """Print screening funnel statistics."""
        n = len(self.records)
        if n == 0:
            return
        cnt = lambda f: sum(1 for r in self.records if f(r))

        print("\n" + "═" * 64)
        print("  2D Materials Screening Funnel  (v3.2 C2DB Standard)")
        print("═" * 64)

        stages = [
            ("Input", n),
            ("S0 Composition", cnt(lambda r: r.s0_pass)),
            ("S1 Geometry", cnt(lambda r: r.s1_pass)),
            ("S2 Symmetry", cnt(lambda r: r.s2_pass)),
            ("S3.5 ML Relax", cnt(lambda r: r.s35_pass)),
            ("S4 Elastic", cnt(lambda r: r.s4_pass)),
            ("S5 Phonon", cnt(lambda r: r.s5_pass)),
            ("Final DFT", cnt(lambda r: r.final_pass)),
        ]

        for label, c in stages:
            bar = "█" * int(c / n * 40) + "░" * (40 - int(c / n * 40))
            pct = f"{c}/{n} ({c/n*100:.1f}%)"
            print(f"  {label:<24} {bar} {pct}")

        print("═" * 56 + "\n")


# Helper function for dimensionality check
def check_is_2d_connected(struct: Structure):
    """Check if structure is 2D connected."""
    try:
        dim = get_dimensionality_larsen(struct)
        if dim == 2:
            return True, {"dimensionality": dim, "reason": "ok"}
        else:
            return False, {"dimensionality": dim, "reason": f"not_2d:dimensionality={dim}"}
    except Exception as e:
        return False, {"dimensionality": -1, "reason": f"dim_check_error:{e}"}


__all__ = ['ScreeningPipeline', 'check_is_2d_connected']
