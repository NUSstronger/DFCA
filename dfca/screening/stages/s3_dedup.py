"""
Screening Stage S3: High-Performance Deduplication.

Features:
    - Formula-based bucketing for scalability
    - StructureMatcher for topological comparison
    - Pareto front within each cluster (C2DB Sec. 2.3)

References:
    - C2DB Sec. 2.3: Structural uniqueness and Pareto optimality
"""

import logging
from typing import Tuple, Optional, List, Dict, Set
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def _compute_pareto_front(ndh_points: List[Tuple[int, float]]) -> List[int]:
    """
    Compute Pareto front indices from (N, ΔH) points.

    Pareto optimal: point i dominates point j if:
        N_i <= N_j AND ΔH_i <= ΔH_j
        AND at least one strict inequality

    Args:
        ndh_points: List of (n_atoms, formation_energy) tuples

    Returns:
        List of indices that are on the Pareto front
    """
    n = len(ndh_points)
    if n == 0:
        return []

    if n == 1:
        return [0]

    is_pareto = [True] * n

    for i in range(n):
        for j in range(n):
            if i == j or not is_pareto[i]:
                continue

            n_i, dh_i = ndh_points[i]
            n_j, dh_j = ndh_points[j]

            # i dominates j if: n_i <= n_j AND dh_i <= dh_j AND at least one strict
            if n_i <= n_j and dh_i <= dh_j and (n_i < n_j or dh_i < dh_j):
                is_pareto[j] = False

    return [i for i in range(n) if is_pareto[i]]


def build_deduplicator(cfg) -> 'StructureMatcherBuilder':
    """
    Build StructureMatcher with C2DB-standard tolerances.

    Args:
        cfg: ScreeningConfig

    Returns:
        StructureMatcherBuilder instance
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(
        ltol=cfg.ltol,  # 0.30
        stol=cfg.stol,  # 0.30
        angle_tol=cfg.angle_tol,  # 10.0 degrees
        scale=False,
        attempt_supercell=False,
        comparator=None,
    )


class BucketedDeduplicator:
    """
    High-performance deduplication with formula-based bucketing.

    Structures are bucketed by reduced_formula, then deduplicated
    within each bucket using StructureMatcher. Pareto front is
    computed within each cluster (same formula).

    References:
        - C2DB Sec. 2.3: Structural uniqueness and Pareto optimality
    """

    def __init__(self, builder, symprec: float = 0.1):
        """
        Initialize BucketedDeduplicator.

        Args:
            builder: StructureMatcher instance
            symprec: Symmetry precision for spglib
        """
        self.matcher = builder
        self.symprec = symprec

        # Formula-based buckets: {formula: {name: struct}}
        self._buckets: Dict[str, Dict[str, Structure]] = {}

        # Formation energies: {name: e_form}
        self._e_form: Dict[str, float] = {}

        # Cluster IDs: {name: cluster_id}
        self._cluster_ids: Dict[str, str] = {}

        # Structure to name mapping for matching
        self._struct_to_names: Dict[str, str] = {}

    def is_duplicate(
        self,
        struct: Structure,
        name: str,
        exclude_name: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check if structure is duplicate of existing one.

        Args:
            struct: Structure to check
            name: Name of the structure
            exclude_name: Name to exclude from comparison (for post-relax dedup)

        Returns:
            (is_duplicate, duplicate_of, cluster_id)
        """
        formula = struct.composition.reduced_formula
        cluster_id = formula  # Use formula as cluster ID

        # Initialize bucket if needed
        if formula not in self._buckets:
            self._buckets[formula] = {}

        # Check against all existing structures in bucket
        for existing_name, existing_struct in self._buckets[formula].items():
            if existing_name == exclude_name:
                continue

            try:
                if self.matcher.fit(struct, existing_struct):
                    # Found duplicate
                    self._cluster_ids[name] = cluster_id
                    return True, existing_name, cluster_id
            except Exception:
                continue

        # Not a duplicate - add to bucket
        self._buckets[formula][name] = struct
        self._cluster_ids[name] = cluster_id
        self._struct_to_names[str(id(struct))] = name

        return False, "", cluster_id

    def add_to_bucket(
        self,
        struct: Structure,
        name: str,
        e_form: float,
        cluster_id: Optional[str] = None
    ) -> None:
        """
        Add structure to bucket for Pareto computation.

        Args:
            struct: Structure to add
            name: Structure name
            e_form: Formation energy (eV/atom)
            cluster_id: Optional cluster ID (defaults to formula)
        """
        if cluster_id is None:
            cluster_id = struct.composition.reduced_formula

        self._e_form[name] = e_form
        self._cluster_ids[name] = cluster_id

        # Add to bucket if not already present
        if cluster_id not in self._buckets:
            self._buckets[cluster_id] = {}
        if name not in self._buckets[cluster_id]:
            self._buckets[cluster_id][name] = struct

    def compute_pareto_all_clusters(self) -> None:
        """
        Compute Pareto front for all clusters.

        For each cluster (formula), computes which structures are
        Pareto-optimal based on (n_atoms, formation_energy).
        """
        for cluster_id, structures in self._buckets.items():
            if len(structures) <= 1:
                continue

            # Build (N, ΔH) points
            ndh_points = []
            names_list = []
            for name, struct in structures.items():
                n_atoms = len(struct)
                e_form = self._e_form.get(name, 0.0)
                ndh_points.append((n_atoms, e_form))
                names_list.append(name)

            # Compute Pareto front
            pareto_indices = _compute_pareto_front(ndh_points)
            pareto_names = {names_list[i] for i in pareto_indices}

            # Mark dominated structures
            for name in names_list:
                if name not in pareto_names:
                    # This structure is Pareto-dominated
                    pass  # Dominance is tracked via is_pareto_dominated()

    def is_pareto_dominated(
        self,
        cluster_id: str,
        name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if structure is Pareto-dominated within cluster.

        Args:
            cluster_id: Cluster ID (formula)
            name: Structure name

        Returns:
            (is_dominated, dominating_name)
        """
        if cluster_id not in self._buckets:
            return False, None

        structures = self._buckets[cluster_id]
        if name not in self._e_form:
            return False, None

        n_self = structures[name].composition.num_atoms
        dh_self = self._e_form[name]

        for other_name, other_struct in structures.items():
            if other_name == name:
                continue

            if other_name not in self._e_form:
                continue

            n_other = len(other_struct)
            dh_other = self._e_form[other_name]

            # Check if other dominates self
            # other dominates self if: n_other <= n_self AND dh_other <= dh_self
            # AND at least one strict inequality
            if n_other <= n_self and dh_other <= dh_self:
                if n_other < n_self or dh_other < dh_self:
                    # other dominates self
                    return True, other_name

        return False, None

    def get_cluster_id(self, name: str) -> Optional[str]:
        """Get cluster ID for a structure name."""
        return self._cluster_ids.get(name)

    def get_all_clusters(self) -> Dict[str, List]:
        """
        Get all clusters with their (name, e_form) pairs.

        Returns:
            Dict mapping cluster_id to list of (name, e_form)
        """
        result = {}
        for cluster_id, structures in self._buckets.items():
            pairs = []
            for name in structures:
                e_form = self._e_form.get(name, 0.0)
                pairs.append((name, e_form))
            result[cluster_id] = pairs
        return result


__all__ = [
    'BucketedDeduplicator',
    'build_deduplicator',
    '_compute_pareto_front',
]
