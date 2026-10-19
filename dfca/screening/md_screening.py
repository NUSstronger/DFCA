# screening/md_screening.py
# S7a/S7b: Two-tier MD Screening
# ============================================================================
# v3.2 C2DB Compliance:
#   - Auto supercell expansion: a or b < 15 A, max 150 atoms
#   - BOND_TOL: 30% drift fraction (was 50% in v3.0)
#   - Bond cutoff: unified with C2DB Sec. 2.2 via bonding_common (1.35x radii)
#   - Z-drift: MIC unwrapping to handle PBC crossings
#   - Enhanced diagnostics: min_size_achieved, supercell reason tags
#   - No FrechetCellFilter: NVTBerendsen fixes volume
#   - Correct units: ase.units.fs (~0.0982269)
#   - MaxwellBoltzmannDistribution: mass-aware velocities
# ============================================================================

import numpy as np
from typing import Tuple, Dict
from ase import units
from ase.md.nvtberendsen import NVTBerendsen
from ase.neighborlist import natural_cutoffs

COLLECT_INTERVAL = 10  # steps between energy/temperature snapshots
MIN_CELL_SIZE = 15.0  # Minimum in-plane size (A)
MAX_ATOMS = 150  # Maximum supercell atoms to prevent performance collapse
# Bond drift tolerance: fraction of reference bond length
# NOTE: This is drift fraction, NOT a bond cutoff multiplier.
# The actual bond cutoff for connectivity uses bonding_common (1.35x covalent radii).
BOND_TOL = 0.5  # 50% drift fraction threshold
Z_DRIFT_TOL = 3.0  # A

# Bonding standard: import from unified C2DB Sec. 2.2 module
# Bug #13 fix: use same multi-level import pattern as other screening submodules
# to ensure bonding_common is found regardless of working directory.
try:
    from dfca.screening.bonding_common import compute_bond_cutoff, build_structure_graph
    _USE_UNIFIED_BONDING = True
except ImportError:
    try:
        import bonding_common as _bc
        compute_bond_cutoff = _bc.compute_bond_cutoff
        build_structure_graph = _bc.build_structure_graph
        _USE_UNIFIED_BONDING = True
    except ImportError:
        _USE_UNIFIED_BONDING = False
        compute_bond_cutoff = None
        build_structure_graph = None

# NetworkX for graph-based connectivity analysis
try:
    import networkx as nx
except ImportError:
    nx = None


def _bfs_connected_components(adj):
    """Pure-Python BFS fallback when networkx is not available."""
    n = len(adj)
    visited = [False] * n
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            for j in range(n):
                if adj[i, j] and not visited[j]:
                    stack.append(j)
    return components


def _adaptive_supercell(atoms, min_size=MIN_CELL_SIZE, max_atoms=MAX_ATOMS):
    """
    Automatically expand small unit cells with atom count cap.

    v3.2: Fixed boundary failures — now returns diagnostic info so callers
    can distinguish three cases:
      1. Expansion successful AND min_size achieved
      2. Expansion performed but min_size NOT achieved (too few atoms)
      3. No expansion possible (n_orig > max_atoms)

    Args:
        atoms: ASE Atoms object
        min_size: Minimum in-plane size (A)
        max_atoms: Maximum total atoms in supercell

    Returns:
        Tuple[Atoms, Dict]: (expanded/unchanged Atoms, diagnostic info dict)
        diagnostic dict fields:
            expanded       : bool  - whether any repetition was applied
            min_size_achieved : bool - whether final a/b >= min_size
            final_cell_a  : float - final a-axis length (A)
            final_cell_b  : float - final b-axis length (A)
            n_atoms       : int   - final atom count
            reason        : str   - one of:
                         "ok" | "expanded_below_min_size" | "atoms_exceed_cap_no_expansion"
    """
    cell = atoms.get_cell()
    a_len = float(np.linalg.norm(cell[0]))
    b_len = float(np.linalg.norm(cell[1]))
    n_orig = len(atoms)

    # Case 3: single structure exceeds max_atoms — cannot expand at all
    if n_orig > max_atoms:
        return atoms, {
            "expanded": False,
            "min_size_achieved": False,
            "final_cell_a": round(a_len, 3),
            "final_cell_b": round(b_len, 3),
            "n_atoms": n_orig,
            "reason": "atoms_exceed_cap_no_expansion",
        }

    # Calculate required expansion factors
    na_needed = max(1, int(np.ceil(min_size / a_len)))
    nb_needed = max(1, int(np.ceil(min_size / b_len)))

    # Try to find a valid (nia, nib) within max_atoms budget
    for nia in range(na_needed, 0, -1):
        for nib in range(nb_needed, 0, -1):
            n_total = n_orig * nia * nib
            if n_total <= max_atoms:
                if nia > 1 or nib > 1:
                    expanded = atoms.repeat((nia, nib, 1))
                    new_a = float(np.linalg.norm(expanded.cell[0]))
                    new_b = float(np.linalg.norm(expanded.cell[1]))
                    size_ok = (new_a >= min_size) and (new_b >= min_size)
                    return expanded, {
                        "expanded": True,
                        "min_size_achieved": size_ok,
                        "final_cell_a": round(new_a, 3),
                        "final_cell_b": round(new_b, 3),
                        "n_atoms": len(expanded),
                        "reason": "ok" if size_ok else "expanded_below_min_size",
                    }
                # nia == nib == 1: no repetition needed
                return atoms, {
                    "expanded": False,
                    "min_size_achieved": (a_len >= min_size and b_len >= min_size),
                    "final_cell_a": round(a_len, 3),
                    "final_cell_b": round(b_len, 3),
                    "n_atoms": n_orig,
                    "reason": "ok",
                }

    # Fallback: find largest expansion that fits within max_atoms
    target_n = max_atoms // n_orig
    na = max(1, int(np.sqrt(target_n)))
    nb = max(1, target_n // na)  # preserve rectangular shape better than sqrt+sqrt
    if na * nb * n_orig > max_atoms:
        nb = max(1, max_atoms // (na * n_orig))
    if na > 1 or nb > 1:
        expanded = atoms.repeat((na, nb, 1))
        new_a = float(np.linalg.norm(expanded.cell[0]))
        new_b = float(np.linalg.norm(expanded.cell[1]))
        size_ok = (new_a >= min_size) and (new_b >= min_size)
        return expanded, {
            "expanded": True,
            "min_size_achieved": size_ok,
            "final_cell_a": round(new_a, 3),
            "final_cell_b": round(new_b, 3),
            "n_atoms": len(expanded),
            "reason": "ok" if size_ok else "expanded_below_min_size",
        }

    # Should not reach here, but defensive fallback
    return atoms, {
        "expanded": False,
        "min_size_achieved": (a_len >= min_size and b_len >= min_size),
        "final_cell_a": round(a_len, 3),
        "final_cell_b": round(b_len, 3),
        "n_atoms": n_orig,
        "reason": "ok",
    }


def _check_struct_integrity(
    atoms_cur,
    atoms_ref,
    slab_thickness,
    bond_tol=BOND_TOL,
    z_drift_tol=Z_DRIFT_TOL,
    min_ref_dist=0.1,
):
    """
    Comprehensive structural integrity check.

    v3.0: Enhanced diagnostics with bond drift fraction.

    Returns:
        (is_ok, max_z_drift, fragments, reason)
    """
    # Bug #13 fix: add shape assertion to guard against mismatched atom counts
    # between atoms_cur and atoms_ref, which can occur if the supercell expansion
    # logic or caller passes inconsistent objects. Without this check,
    # drift[bonded] would raise IndexError on the boolean mask indexing.
    n_cur = len(atoms_cur)
    n_ref = len(atoms_ref)
    if n_cur != n_ref:
        return False, 0.0, -1, f"atom_count_mismatch(cur={n_cur},ref={n_ref})"

    dist_cur = atoms_cur.get_all_distances(mic=True)
    dist_ref = atoms_ref.get_all_distances(mic=True)

    # Bonded pairs: use unified C2DB Sec. 2.2 bond thresholds (1.35x covalent radii)
    # to be consistent with the connectivity check below, which uses
    # bonding_common.build_structure_graph / compute_bond_cutoff.
    # Fixes Bug #12: natural_cutoffs(atoms_ref) used 1.0x radii which under-counts
    # bonds in polar/weak-covalent 2D materials (halides, low-valent chalcogenides),
    # causing bonded.sum() == 0 to trigger prematurely and skip bond drift checks.
    if _USE_UNIFIED_BONDING and compute_bond_cutoff is not None:
        symbols = atoms_ref.get_chemical_symbols()
        bond_thresh = np.zeros((n_ref, n_ref))
        for i in range(n_ref):
            for j in range(i + 1, n_ref):
                c_val = compute_bond_cutoff(symbols[i], symbols[j])
                bond_thresh[i, j] = bond_thresh[j, i] = c_val
    else:
        cutoffs = np.array(natural_cutoffs(atoms_ref))
        bond_thresh = cutoffs[:, None] + cutoffs[None, :]
    bonded = dist_ref < bond_thresh
    np.fill_diagonal(bonded, False)

    if bonded.sum() == 0:
        bond_ok = True
        max_bond_drift_frac = 0.0
    else:
        drift = np.abs(dist_cur - dist_ref)
        ref_safe = np.clip(dist_ref, min_ref_dist, None)
        bond_drift_frac = drift[bonded] / ref_safe[bonded]
        max_bond_drift_frac = float(bond_drift_frac.max())
        bond_ok = bool((bond_drift_frac < bond_tol).all())

    # Z-axis drift with MIC (Minimum Image Convention) unwrapping
    # PBC wrap: if an atom crosses z=0 or z=c, its absolute coordinate jumps by ~c.
    # We apply MIC along c-axis to unwrap the displacement.
    com_cur = atoms_cur.positions.mean(axis=0)
    com_ref = atoms_ref.positions.mean(axis=0)
    pos_cur = atoms_cur.positions - com_cur
    pos_ref = atoms_ref.positions - com_ref
    c_len = atoms_cur.cell[2, 2]
    dz = pos_cur[:, 2] - pos_ref[:, 2]
    # MIC: subtract integer multiples of c to bring into [-c/2, c/2]
    dz_unwrapped = dz - np.round(dz / c_len) * c_len
    max_z_drift = float(np.max(np.abs(dz_unwrapped)))
    z_ok = max_z_drift <= (slab_thickness + z_drift_tol)

    # Connectivity: use unified C2DB Sec. 2.2 StructureGraph via bonding_common
    # This handles PBC translation vectors natively (CutOffDictNN constructs periodic bonds),
    # fixing the issue where get_all_distances(mic=True) loses topological translation info.
    if _USE_UNIFIED_BONDING:
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            pmg_struct = AseAtomsAdaptor().get_structure(atoms_cur)
            sg, bond_count = build_structure_graph(pmg_struct)
            if sg is not None:
                ug = sg.graph.to_undirected()
                # Count connected components via BFS on the StructureGraph
                if nx is not None:
                    fragments = int(nx.number_connected_components(ug))
                else:
                    # Build adjacency for BFS fallback
                    # Bug #14 fix: use dynamic bond_thresh instead of hardcoded 2.5 A
                    adj_fallback = np.zeros((n_ref, n_ref), dtype=bool)
                    for i in range(n_ref):
                        for j in range(i + 1, n_ref):
                            if dist_cur[i, j] <= bond_thresh[i, j]:
                                adj_fallback[i, j] = adj_fallback[j, i] = True
                    fragments = _bfs_connected_components(adj_fallback)
                connectivity_ok = (fragments == 1)
                bond_count_diag = bond_count
            else:
                connectivity_ok = False
                fragments = -1
                bond_count_diag = 0
        except Exception:
            # Fallback to ASE natural_cutoffs if conversion fails
            # Bug #14 fix: use dynamic bond_thresh instead of hardcoded 2.5 A
            adj_fallback = np.zeros((n_ref, n_ref), dtype=bool)
            for i in range(n_ref):
                for j in range(i + 1, n_ref):
                    if dist_cur[i, j] <= bond_thresh[i, j]:
                        adj_fallback[i, j] = adj_fallback[j, i] = True
            fragments = _bfs_connected_components(adj_fallback)
            connectivity_ok = (fragments == 1)
            bond_count_diag = int(adj_fallback.sum() / 2)
    else:
        # Bug #14 fix: use dynamic bond_thresh instead of hardcoded 2.5 A
        adj_fallback = np.zeros((n_ref, n_ref), dtype=bool)
        for i in range(n_ref):
            for j in range(i + 1, n_ref):
                if dist_cur[i, j] <= bond_thresh[i, j]:
                    adj_fallback[i, j] = adj_fallback[j, i] = True
        fragments = _bfs_connected_components(adj_fallback)
        connectivity_ok = (fragments == 1)
        bond_count_diag = int(adj_fallback.sum() / 2)

    is_ok = bool(bond_ok and z_ok and connectivity_ok)

    if is_ok:
        reason = "ok"
    else:
        parts = []
        if not bond_ok:
            parts.append(f"bond({max_bond_drift_frac:.0%}>{bond_tol:.0%})")
        if not z_ok:
            parts.append(f"z_drift({max_z_drift:.2f}A)")
        if not connectivity_ok:
            parts.append(f"frag={fragments}")
        reason = "integrity_fail(" + "|".join(parts) + ")"

    return is_ok, max_z_drift, fragments, reason


def _linear_slope(arr, collect_interval=1):
    """
    Linear regression slope in eV/(atom * real_MD_step).

    Bug #15 fix: x-axis is now real MD step count, not sample index.

    Previously x = np.arange(len(arr)) was used, which makes the slope
    have units eV/(atom * COLLECT_INTERVAL_steps) — 10x larger than claimed.
    The correct approach is to multiply x by the sampling interval.

    Args:
        arr: list/array of potential energies (eV)
        collect_interval: number of real MD steps between consecutive samples (default 1)

    Returns:
        slope in eV/(atom * real_MD_step)
    """
    if len(arr) < 2:
        return 0.0
    # x = real MD step number, not sample index
    x = np.arange(len(arr), dtype=float) * collect_interval
    y = np.array(arr, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = np.sum((x - x_mean) * (y - y_mean))
    var = np.sum((x - x_mean) ** 2)
    return cov / var if var > 1e-12 else 0.0


def run_md_s7a(
    struct, calc,
    temperature=300.0, dt_fs=1.0, n_steps=3000,
    min_cell_size=MIN_CELL_SIZE, max_atoms=MAX_ATOMS,
):
    """
    S7a: MD Fast Screen - 3ps, NVT 300K.

    v3.2: Uses updated _adaptive_supercell with diagnostic info.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    try:
        atoms = AseAtomsAdaptor().get_atoms(struct)
    except Exception:
        atoms = struct

    # Auto supercell expansion FIRST (may create new Atoms via repeat()).
    # CRITICAL: atoms.calc must be set AFTER _adaptive_supercell returns,
    # because repeat() creates a new Atoms object with calc=None.
    atoms, sc_info = _adaptive_supercell(atoms, min_size=min_cell_size, max_atoms=max_atoms)
    atoms.calc = calc
    expanded = sc_info["expanded"]
    min_size_ok = sc_info["min_size_achieved"]
    result_info = {**sc_info}

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)

    dyn = NVTBerendsen(
        atoms,
        timestep=dt_fs * units.fs,
        taut=100 * dt_fs * units.fs,
        temperature_K=temperature,
    )

    atoms_ref = atoms.copy()
    slab_thickness = float(atoms_ref.positions[:, 2].max() - atoms_ref.positions[:, 2].min())

    try:
        dyn.run(n_steps)
    except Exception as e:
        return False, {
            "struct_integrity_ok": False,
            "max_z_drift": 0.0, "fragment_count": 0,
            "reason": f"md_explosion:{e}",
            **result_info,
        }

    ok, max_z_drift, fragments, reason = _check_struct_integrity(
        atoms, atoms_ref, slab_thickness, bond_tol=BOND_TOL
    )

    # Merge supercell diagnostic info with MD results
    result = {
        **result_info,  # includes expanded, min_size_achieved, final_cell_a/b, reason
        "struct_integrity_ok": ok,
        "max_z_drift": max_z_drift,
        "fragment_count": fragments,
        "bond_tol": BOND_TOL,
    }
    # Don't overwrite reason if it came from sdim diagnostic
    if "atoms_exceed_cap" in result.get("reason", "") or "expanded_below" in result.get("reason", ""):
        result["reason"] = "supercell_" + result["reason"] + "|" + reason
    else:
        result["reason"] = reason

    return ok, result


def run_md_s7b(
    struct, calc,
    temperature=300.0, dt_fs=1.0, n_steps=10000,
    min_cell_size=MIN_CELL_SIZE, max_atoms=MAX_ATOMS,
):
    """
    S7b: MD Full - 10ps, NVT 300K.

    v3.0: Auto supercell with max_atoms cap, tightened bond_tol.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    try:
        atoms = AseAtomsAdaptor().get_atoms(struct)
    except Exception:
        atoms = struct

    # Auto supercell expansion FIRST (may create new Atoms via repeat()).
    # CRITICAL: atoms.calc must be set AFTER _adaptive_supercell returns,
    # because repeat() creates a new Atoms object with calc=None.
    atoms, sc_info = _adaptive_supercell(atoms, min_size=min_cell_size, max_atoms=max_atoms)
    atoms.calc = calc
    expanded = sc_info["expanded"]
    min_size_ok = sc_info["min_size_achieved"]
    result_info = {**sc_info}

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)

    dyn = NVTBerendsen(
        atoms,
        timestep=dt_fs * units.fs,
        taut=100 * dt_fs * units.fs,
        temperature_K=temperature,
    )

    atoms_ref = atoms.copy()
    slab_thickness = float(atoms_ref.positions[:, 2].max() - atoms_ref.positions[:, 2].min())
    n_atoms = len(atoms)

    energies = []
    temperatures_list = []

    def collect():
        energies.append(float(atoms.get_potential_energy()))
        try:
            temperatures_list.append(float(atoms.get_temperature()))
        except Exception:
            pass

    dyn.attach(collect, interval=COLLECT_INTERVAL)

    try:
        dyn.run(n_steps)
    except Exception as e:
        return False, {
            "struct_integrity_ok": False,
            "energy_trend_ok": False, "temperature_ok": False,
            "T_max": 0.0, "T_min": 0.0, "T_mean": 0.0,
            "energy_curve_2ps": [], "atom_snapshot": "",
            "reason": f"md_explosion:{e}",
            **result_info,
        }

    # Criterion 1: Structural integrity (tightened tolerance)
    ok_integrity, max_z_drift, fragments, _ = _check_struct_integrity(
        atoms, atoms_ref, slab_thickness, bond_tol=BOND_TOL
    )

    # Criterion 2: Energy trend (last 5ps)
    # Bug #15 fix: _linear_slope now returns eV/(atom * real_MD_step) directly
    # (x-axis = real step count, not sample index). COLLECT_INTERVAL is passed as
    # collect_interval so the slope is correctly in eV/(atom * step).
    # Symmetric threshold: abs() prevents both runaway endothermic drift AND exothermic
    # collapse / MD explosion, which was previously unchecked with a one-sided test.
    ENERGY_SLOPE_MAX = 0.0005  # eV/(atom * step) threshold
    mid = max(1, len(energies) // 2)
    slope_per_atom = _linear_slope(energies[mid:], collect_interval=COLLECT_INTERVAL) / n_atoms
    energy_trend_ok = abs(slope_per_atom) < ENERGY_SLOPE_MAX

    # Criterion 3: Temperature health (equilibrium mean)
    if temperatures_list:
        t_arr = np.array(temperatures_list)
        equil_start = max(1, len(t_arr) // 5)
        t_mean = float(t_arr[equil_start:].mean())
        t_max = float(t_arr.max())
        t_min = float(t_arr.min())
        temperature_ok = bool(abs(t_mean - temperature) <= 60.0)
    else:
        t_mean = t_max = t_min = 0.0
        temperature_ok = False

    ok = bool(ok_integrity and energy_trend_ok and temperature_ok)

    fail_parts = []
    if not ok_integrity:
        fail_parts.append("integrity")
    if not energy_trend_ok:
        fail_parts.append(f"energy(slope={slope_per_atom:.6f}eV/atom/step)")
    if not temperature_ok:
        fail_parts.append(f"T_mean={t_mean:.0f}K")

    reason = "ok" if ok else "s7b_fail(" + "|".join(fail_parts) + ")"

    # Merge supercell diagnostic info with MD results
    result = {
        **result_info,
        "struct_integrity_ok": ok_integrity,
        "energy_trend_ok": energy_trend_ok,
        "temperature_ok": temperature_ok,
        "T_max": t_max, "T_min": t_min, "T_mean": t_mean,
        "energy_curve_2ps": energies[-2000:] if len(energies) > 2000 else list(energies),
        "atom_snapshot": str(atoms.get_chemical_formula()),
        "reason": reason,
        "bond_tol": BOND_TOL,
    }
    # Prepend supercell diagnostic to reason if relevant
    if "atoms_exceed_cap" in result_info.get("reason", "") or "expanded_below" in result_info.get("reason", ""):
        result["reason"] = "supercell_" + result_info.get("reason", "") + "|" + reason

    return ok, result
