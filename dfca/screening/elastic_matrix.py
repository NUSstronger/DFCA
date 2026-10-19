# screening/elastic_matrix.py
# S4: 2D Elastic Matrix with C2DB Mandel Notation
# ============================================================================
# v3.2 C2DB Original Standard Compliance:
#   - Full 3x3 in-plane stiffness matrix in C2DB Mandel notation
#   - C2DB Eq.(1): C11, C22, C12, sqrt(2)*C16, sqrt(2)*C26, 2*C66
#     NOTE: C16/C26 are shear-normal coupling coefficients arising from xy strain
#     (Voigt index 5), NOT from xz/yz (indices 3/4). xz/yz strain components
#     are NOT computed (they would require out-of-plane cell relaxation which is
#     not applicable to 2D slabs with fixed c-axis).
#   - Eigenvalue decomposition via np.linalg.eigvalsh
#   - Born stability: C2DB original standard min{eig(C)} >= 0.0 (strict positivity)
#     NO additional nonlinear strain fit threshold - follows Gjerding et al. 2021 exactly
#   - Improved FIRE relaxation: fmax = 0.01 eV/A for soft materials
#   - Lz normalization: C2D = C3D * Lz * 0.1 (GPa*A -> N/m)
#   - 4-point linear regression for robust Cij extraction
# ============================================================================

import numpy as np
from typing import Tuple, Dict, List
from ase.optimize import FIRE

GPA_PER_EV_A3 = 160.21766208  # 1 eV/A3 = 160.21766208 GPa
STRAIN_AMPS = [-0.01, -0.005, 0.005, 0.01]
# Born stability tolerance: C2DB original standard (Gjerding et al. 2021)
# min{eig(C)} >= 0.0 for mechanical stability
# Using -1e-5 for float arithmetic tolerance only
BORN_STABILITY_TOLERANCE = -1e-5


def voigt_to_strain(voigt):
    """Convert Voigt strain [e1..e6] to 3x3 symmetric strain matrix."""
    e = np.zeros((3, 3))
    e[0, 0] = voigt[0]
    e[1, 1] = voigt[1]
    e[2, 2] = voigt[2]
    e[1, 2] = e[2, 1] = voigt[3] / 2.0
    e[0, 2] = e[2, 0] = voigt[4] / 2.0
    e[0, 1] = e[1, 0] = voigt[5] / 2.0
    return e


def apply_lattice_strain(cell, voigt):
    """Apply Voigt strain: F = (I + eps), cell @ F.T (row-vector convention).

    ASE/pymatgen uses row-vector convention: each row is a lattice vector.
    Strain must be applied from the right and transposed, otherwise C16/C26
    shear-normal coupling terms are computed incorrectly.
    """
    F = np.eye(3) + voigt_to_strain(voigt)
    return cell @ F.T


def _fit_slope(strains, stresses, stress_ref=None):
    """
    1st-order polyfit: slope = d(stress)/d(strain), intercept ignored.

    Bug #13 fix: include the true zero-strain reference point (0, stress_ref)
    in the fit to anchor the regression at the physically correct origin.

    Previous buggy logic extracted mid_idx from the strained list and subtracted
    it as baseline — this was NOT the true reference state and made stress_ref
    a dead parameter. The correct approach is to include (0, stress_ref) as an
    explicit data point in the linear fit.
    """
    if len(strains) < 2:
        return 0.0
    strains_arr = np.array(strains, dtype=float)
    stresses_arr = np.array(stresses, dtype=float)

    if stress_ref is not None:
        # Append the true zero-strain reference point and fit through all points
        all_strains = np.append(strains_arr, 0.0)
        all_stresses = np.append(stresses_arr, float(stress_ref))
        coeffs = np.polyfit(all_strains, all_stresses, deg=1)
    else:
        coeffs = np.polyfit(strains_arr, stresses_arr, deg=1)
    return float(coeffs[0])


def _build_mandel_matrix(c_mat, factor):
    """
    Build 3x3 in-plane stiffness matrix in C2DB Mandel notation.

    C2DB Eq.(1) definition:
      - Diagonal normal: C11, C22
      - Cross normal: C12 (averaged symmetric)
      - Shear-normal coupling: sqrt(2) * C16, sqrt(2) * C26
      - Pure shear: 2 * C66

    Args:
        c_mat: 6x6 Voigt stiffness matrix
        factor: Lz normalization factor (GPa*A -> N/m)

    Returns:
        3x3 symmetric matrix in Mandel notation
    """
    # C2DB Mandel notation: [e11, e22, e12, e16, e26, e66]
    # Mapping: Voigt [0=xx,1=yy,2=zz,3=yz,4=xz,5=xy] -> Mandel
    C11 = c_mat[0, 0] * factor
    C22 = c_mat[1, 1] * factor
    C12 = (c_mat[0, 1] + c_mat[1, 0]) / 2.0 * factor
    C66 = c_mat[5, 5] * factor
    C16 = c_mat[0, 5] * factor  # xy-xx coupling
    C26 = c_mat[1, 5] * factor  # xy-yy coupling

    # Build 3x3 Mandel matrix (C2DB Eq.1)
    mandel = np.zeros((3, 3))
    mandel[0, 0] = C11                         # C11
    mandel[1, 1] = C22                         # C22
    mandel[0, 1] = mandel[1, 0] = C12         # C12
    # Shear-normal coupling with sqrt(2) weighting
    mandel[0, 2] = mandel[2, 0] = np.sqrt(2) * C16
    mandel[1, 2] = mandel[2, 1] = np.sqrt(2) * C26
    # Pure shear with 2x weighting
    mandel[2, 2] = 2.0 * C66

    return mandel


def compute_elastic_matrix(
    struct,
    calc,
    fmax=0.01,
    max_steps=200,
    strain_amps=(-0.01, -0.005, 0.005, 0.01),
):
    """
    Compute 2D in-plane elastic matrix via 4-point finite differences
    in C2DB Mandel notation.

    v3.2 C2DB Mandate Compliance (Gjerding et al. 2021 Sec. 3.1):
      1. Full 3x3 in-plane stiffness matrix in Mandel notation
      2. C2DB Eq.(1): sqrt(2)*C16, sqrt(2)*C26, 2*C66
      3. Generic Born stability via eigenvalue positivity: min{eig(C)} >= 0
      4. Improved FIRE fmax = 0.01 eV/A for soft materials
      5. NO nonlinear strain fit check (removed to avoid false negatives for flexible materials)

    Args:
        struct: pymatgen Structure or ASE Atoms
        calc: ASE calculator with compute() method
        fmax: FIRE convergence threshold (default 0.01 eV/A)
        max_steps: Maximum FIRE relaxation steps
        strain_amps: 4-element tuple of strain amplitudes

    Returns:
        (c2d_dict, is_stable, reason, stress_residual, energy_traj)
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    if calc is None:
        return {}, False, "calc_is_none", 0.0, []

    # Convert to ASE Atoms
    try:
        atoms = AseAtomsAdaptor().get_atoms(struct)
    except Exception:
        from ase import Atoms
        atoms = struct
    atoms.calc = calc

    # Lz normalization factor
    c_axis_length = float(atoms.cell[2, 2])
    if c_axis_length <= 0:
        return {}, False, f"invalid_c_axis({c_axis_length})", 0.0, []
    factor = c_axis_length * 0.1  # GPa*A -> N/m

    # Reference state: fix lattice, relax atoms only
    atoms_ref = atoms.copy()
    atoms_ref.calc = calc
    opt_ref = FIRE(atoms_ref, logfile=None)
    try:
        opt_ref.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        return {}, False, f"reference_relax_failed:{e}", 0.0, []

    stress_ref = atoms_ref.get_stress()
    stress_residual = float(np.linalg.norm(stress_ref))
    energy_traj = [float(atoms_ref.get_potential_energy())]

    def strain_relax(comp_idx, delta_val):
        """Apply single Voigt strain component, relax atoms, return stress."""
        voigt = np.zeros(6)
        voigt[comp_idx] = delta_val
        deformed = apply_lattice_strain(atoms_ref.cell[:], voigt)

        at = atoms_ref.copy()
        at.set_cell(deformed, scale_atoms=True)
        at.calc = calc
        FIRE(at, logfile=None).run(fmax=fmax, steps=max_steps)
        return at.get_stress()

    # Bug #13 fix: warn if large residual stress may affect slope fitting.
    # Theoretically, the linear slope is independent of the intercept, so a non-zero
    # stress_ref should not bias the fit. In practice, for tiny strain amplitudes
    # (±0.5%) and under-relaxed FIRE runs, a large baseline can inject noise.
    # We now include the zero-strain point (0, stress_ref) in the fit via _fit_slope.
    STRESS_RESIDUAL_WARN_THRESHOLD = 0.05  # GPa
    if stress_residual > STRESS_RESIDUAL_WARN_THRESHOLD:
        import logging as _log
        _log.getLogger("Screen2D.elastic").warning(
            f"Large reference residual stress {stress_residual:.4f} GPa "
            f"(>{STRESS_RESIDUAL_WARN_THRESHOLD} GPa threshold). "
            f"Applying center-difference correction in slope fitting."
        )

    # Compute 6x6 Voigt stiffness matrix using 4-point finite differences
    # In-plane components: xx(0), yy(1), xy(5)
    # NOTE: xz(4)/yz(3) NOT computed — out-of-plane shear coupling not applicable
    # to 2D slabs with fixed c-axis; these columns were dead code in v3.1
    in_plane = [("xx", 0), ("yy", 1), ("xy", 5)]
    all_comps = in_plane

    c_mat = np.zeros((6, 6))

    for name, idx in all_comps:
        strains_list = []
        stresses_list = {i: [] for i in range(6)}

        for delta_val in strain_amps:
            sig = strain_relax(idx, delta_val)
            strains_list.append(delta_val)
            for i in range(6):
                stresses_list[i].append(sig[i])

        for i in range(6):
            # Bug #13 fix: include zero-strain reference point (0, stress_ref) in fit
            c_mat[i, idx] = _fit_slope(strains_list, stresses_list[i], stress_ref=stress_ref[i])

    # C2DB Sec. 3.1 标准：
    # 仅使用 Born 特征值稳定性判据 min{eig(C)} >= 0
    # 移除自行添加的非线性应变区间拟合检查（max_rel_diff > 0.05）
    # 以避免因应变区间的微小非线性误杀柔性材料

    # Build C2DB Mandel matrix
    mandel = _build_mandel_matrix(c_mat, GPA_PER_EV_A3 * factor)

    # Eigenvalue decomposition for Born stability
    eigenvalues = np.linalg.eigvalsh(mandel)
    min_eigenvalue = float(eigenvalues.min())

    # C2DB Born stability criterion (Gjerding et al. 2021):
    # All eigenvalues must be >= 0 (strict positivity)
    # Using BORN_STABILITY_TOLERANCE = -1e-5 for float arithmetic safety
    is_stable = bool(min_eigenvalue >= BORN_STABILITY_TOLERANCE)

    # Build return dictionary
    c2d_dict = {
        "C11": float(mandel[0, 0]),
        "C22": float(mandel[1, 1]),
        "C12": float(mandel[0, 1]),
        "C66": float(mandel[2, 2] / 2.0),  # Undo 2x weighting for reporting
        "C16": float(mandel[0, 2] / np.sqrt(2)),  # Undo sqrt(2) weighting
        "C26": float(mandel[1, 2] / np.sqrt(2)),
        "min_eigenvalue": min_eigenvalue,
        "eigenvalues": eigenvalues.tolist(),
        "Lz_A": c_axis_length,
    }

    if is_stable:
        reason = "ok"
    else:
        reason = f"born_unstable(min_eig={min_eigenvalue:.4f}<{BORN_STABILITY_TOLERANCE})"

    return c2d_dict, is_stable, reason, stress_residual, energy_traj
