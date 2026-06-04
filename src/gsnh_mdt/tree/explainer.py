"""
Explainer methods for ExpertGSNHTree.

Extracted from tree/builder.py lines 1181-1412.
AXp extraction, weak AXp checking, path satisfiability (numeric + affine).

Theorem-strict compliance extensions:
- Path-level certificate checking
- Backend metadata recording
- NonTheoremPathError for uncertified paths
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from gsnh_mdt.literals.binary import GSNHBinaryLiteral
from gsnh_mdt.literals.compare import CompareLiteral
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.predicates import Square2CNFPredicate
from gsnh_mdt.sat.exact_solver import ExactSATSolver
from gsnh_mdt.types import LanguageFamily, LiteralPolarity


# ================================================================
# Backend metadata for theorem compliance tracking
# ================================================================

@dataclass
class AXpBackendMetadata:
    """Metadata recorded for each AXp/path SAT call.

    Attributes
    ----------
    axp_backend : str
        One of: structural_horn, structural_antihorn, two_sat,
        affine, interval_dfs_fallback, prototype_case_split,
        rejected_non_theorem, empty_path
    theorem_certified : bool
        True iff the path check used a Coq-verified polynomial backend.
    path_certificate : str
        One of: horn, antihorn, 2cnf, none
    reason : str, optional
        Explanation if the path was rejected.
    """
    axp_backend: str = "unknown"
    theorem_certified: bool = False
    path_certificate: str = "none"
    reason: str = ""
    theorem_mode_used: bool = False


def _enumerate_paths(tree):
    """Enumerate all root-to-leaf paths once.

    Returns a list of (path_edges, predicted_class) tuples where
    path_edges is a list of (predicate, branch_bool) pairs.

    This is the O(2^depth) traversal that was previously repeated
    inside every ``weak_axp_check`` call.  Caching the result and
    passing it explicitly eliminates ~7 M redundant recursive calls
    on large benchmarks.
    """
    paths = []

    def _walk(node, current_path):
        if node is None:
            return
        if node.get('is_leaf', True) or node.get('predicate') is None:
            pred_y = 1 if node.get('proba', 0.5) >= 0.5 else 0
            paths.append((current_path, pred_y))
            return
        _walk(node.get('left'), current_path + [(node['predicate'], True)])
        _walk(node.get('right'), current_path + [(node['predicate'], False)])

    _walk(tree.root_, [])
    return paths


def weak_axp_check(tree, x, y, S, paths=None):
    """Check if partial instance x_S guarantees prediction y using CSP.

    Parameters
    ----------
    tree : ExpertGSNHTree
        The fitted tree.
    x : np.ndarray
        Full instance vector.
    y : int
        Target prediction to verify.
    S : set
        Set of fixed feature indices.
    paths : list, optional
        Pre-computed list of (path_edges, leaf_y) tuples from
        ``_enumerate_paths(tree)``.  When provided, the expensive
        recursive traversal is skipped entirely.  Backward compatible:
        if ``None``, paths are enumerated on the fly (old behavior).
    """
    if paths is None:
        paths = _enumerate_paths(tree)

    for path_edges, leaf_y in paths:
        if leaf_y != y:
            if _is_sat_path(tree, path_edges, x, S):
                return False
    return True


from gsnh_mdt.sat.threshold_encoder import encode_horn_path, encode_antihorn_path
from gsnh_mdt.sat.path_certificate import (
    NonTheoremPathError,
    build_ordered_selected_path_cnf,
    classify_cnf_fragment,
    is_polynomial_safe_path,
)


def _path_sat_structural_horn(path_edges, x, S, family):
    # Check for unsupported literals before encoding
    for pred, _ in path_edges:
        if pred.is_xor:
            raise NotImplementedError("XOR literals cannot be mixed with Horn/AntiHorn structural backend")
        if isinstance(pred, Square2CNFPredicate) or pred.language_family == LanguageFamily.CONJ_UI:
            raise NotImplementedError(f"{pred.language_family} cannot be routed to Horn/AntiHorn structural backend")
        for lit in pred.literals:
            if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                raise NotImplementedError("CompareLiteral/GSNHBinaryLiteral not supported in structural backend")

    if family == LanguageFamily.HORN:
        clauses = encode_horn_path(path_edges, x, S)
        return ExactSATSolver.horn_sat(clauses)

    if family == LanguageFamily.ANTI_HORN:
        clauses = encode_antihorn_path(path_edges, x, S)
        return ExactSATSolver.antihorn_sat(clauses)

    raise NotImplementedError


def _check_unsupported_literals_theorem_mode(path_edges):
    """Check if any predicate on the path uses unsupported literal types.

    In theorem_strict mode, only GSNHLiteral (threshold atoms) are supported.
    CompareLiteral, GSNHBinaryLiteral, XOR, and affine literals are rejected.
    """
    unsupported = []
    for pred, _ in path_edges:
        if pred.is_xor:
            unsupported.append("XOR/Affine")
        if isinstance(pred, Square2CNFPredicate):
            # Square2CNF allowed only if certified
            continue
        for lit in pred.literals:
            if isinstance(lit, CompareLiteral):
                unsupported.append("CompareLiteral")
            elif isinstance(lit, GSNHBinaryLiteral):
                unsupported.append("GSNHBinaryLiteral")
    return unsupported


def _is_sat_path(tree, path_edges, x, S):
    """
    Route to correct SAT checker.

    In theorem_strict mode, every path must be certified as Horn,
    AntiHorn, or 2CNF. Uncertified paths raise NonTheoremPathError
    or record rejected_non_theorem metadata.
    """
    theorem_strict = getattr(tree, 'theorem_strict', False)
    metadata = AXpBackendMetadata()
    metadata.theorem_mode_used = bool(theorem_strict)

    if not path_edges:
        metadata.axp_backend = "empty_path"
        metadata.theorem_certified = True
        metadata.path_certificate = "horn"  # trivially Horn
        tree.explainer_backend_ = "empty_path"
        _record_metadata(tree, metadata)
        return True

    has_xor = any(pred.is_xor for pred, _ in path_edges)

    # Affine/GF(2) remains auxiliary in Python theorem metadata until a
    # verified GF(2) certificate checker is implemented.  In theorem-strict
    # mode, reject unsupported/non-Boolean threshold structure instead of
    # silently treating affine_sat as a theorem certificate.
    if theorem_strict and has_xor:
        metadata.axp_backend = "affine"
        metadata.theorem_certified = False
        metadata.path_certificate = "affine_unverified"
        guard_error = _affine_theorem_guard_error(tree, path_edges)
        if guard_error:
            metadata.reason = guard_error
            _record_metadata(tree, metadata)
            raise NonTheoremPathError(guard_error)
        if not all(pred.is_xor for pred, _ in path_edges):
            metadata.reason = "Mixed affine paths are not theorem-certified by the Python checker"
            _record_metadata(tree, metadata)
            raise NonTheoremPathError(metadata.reason)
        metadata.reason = (
            "Affine GF(2) solving is auxiliary in Python until verified "
            "certificate checking is implemented"
        )
        _record_metadata(tree, metadata)
        return _affine_path_sat(path_edges, x, S)

    # Theorem-strict mode: check for unsupported literals
    if theorem_strict:
        unsupported = _check_unsupported_literals_theorem_mode(path_edges)
        if unsupported:
            metadata.axp_backend = "rejected_non_theorem"
            metadata.theorem_certified = False
            metadata.reason = f"Unsupported literals: {', '.join(set(unsupported))}"
            _record_metadata(tree, metadata)
            raise NonTheoremPathError(metadata.reason)

    has_square2cnf = any(isinstance(pred, Square2CNFPredicate) for pred, _ in path_edges)

    if theorem_strict and has_square2cnf:
        try:
            _, cnf = build_ordered_selected_path_cnf(path_edges, x, S)
        except NonTheoremPathError as e:
            metadata.axp_backend = "rejected_non_theorem"
            metadata.theorem_certified = False
            metadata.path_certificate = "none"
            metadata.reason = str(e)
            _record_metadata(tree, metadata)
            raise

        if any(len(clause) > 2 for clause in cnf):
            metadata.axp_backend = "rejected_non_theorem"
            metadata.theorem_certified = False
            metadata.path_certificate = "none"
            metadata.reason = "Square2CNF path did not certify as 2-CNF"
            _record_metadata(tree, metadata)
            raise NonTheoremPathError(metadata.reason)

        tree.explainer_backend_ = "two_sat"
        metadata.axp_backend = "two_sat"
        metadata.theorem_certified = True
        metadata.path_certificate = "2cnf"
        metadata.reason = ""
        _record_metadata(tree, metadata)
        return ExactSATSolver.two_sat(cnf)

    # Determine homogeneous family on path, ignoring leaves/affine
    fams = {pred.language_family for pred, _ in path_edges if not pred.is_xor}

    if fams == {LanguageFamily.HORN}:
        tree.explainer_backend_ = "structural_horn"
        metadata.axp_backend = "structural_horn"
        metadata.theorem_certified = True
        metadata.path_certificate = "horn"
        _record_metadata(tree, metadata)
        return _path_sat_structural_horn(path_edges, x, S, LanguageFamily.HORN)

    if fams == {LanguageFamily.ANTI_HORN}:
        tree.explainer_backend_ = "structural_antihorn"
        metadata.axp_backend = "structural_antihorn"
        metadata.theorem_certified = True
        metadata.path_certificate = "antihorn"
        _record_metadata(tree, metadata)
        return _path_sat_structural_horn(path_edges, x, S, LanguageFamily.ANTI_HORN)

    # Affine-only path
    if path_edges and all(pred.is_xor for pred, _ in path_edges):
        tree.explainer_backend_ = "affine"
        metadata.axp_backend = "affine"
        metadata.theorem_certified = False
        metadata.path_certificate = "none"
        metadata.reason = "Affine backend is auxiliary unless separately certified"
        _record_metadata(tree, metadata)
        return _affine_path_sat(path_edges, x, S)

    # Theorem-strict BEST_PER_NODE: path-level certificate check
    if theorem_strict:
        try:
            is_safe, certificate = is_polynomial_safe_path(path_edges, x, S)
        except NonTheoremPathError as e:
            metadata.axp_backend = "rejected_non_theorem"
            metadata.theorem_certified = False
            metadata.path_certificate = "none"
            metadata.reason = str(e)
            _record_metadata(tree, metadata)
            raise

        if not is_safe:
            metadata.axp_backend = "rejected_non_theorem"
            metadata.theorem_certified = False
            metadata.path_certificate = certificate
            metadata.reason = (
                f"Path CNF classified as '{certificate}'; "
                "only horn/antihorn/2cnf are theorem-compliant"
            )
            _record_metadata(tree, metadata)
            raise NonTheoremPathError(metadata.reason)

        # Route to certified solver based on certificate.
        # Always use build_ordered_selected_path_cnf for CNF construction
        # because encode_horn_path/encode_antihorn_path do not understand
        # Square2CNFPredicate clause structure and would flatten the
        # literals into a single oversized clause.
        _, cnf = build_ordered_selected_path_cnf(path_edges, x, S)

        if certificate == "horn":
            tree.explainer_backend_ = "structural_horn"
            metadata.axp_backend = "structural_horn"
            metadata.theorem_certified = True
            metadata.path_certificate = "horn"
            _record_metadata(tree, metadata)
            return ExactSATSolver.horn_sat(cnf)

        if certificate == "antihorn":
            tree.explainer_backend_ = "structural_antihorn"
            metadata.axp_backend = "structural_antihorn"
            metadata.theorem_certified = True
            metadata.path_certificate = "antihorn"
            _record_metadata(tree, metadata)
            return ExactSATSolver.antihorn_sat(cnf)

        if certificate == "2cnf":
            tree.explainer_backend_ = "two_sat"
            metadata.axp_backend = "two_sat"
            metadata.theorem_certified = True
            metadata.path_certificate = "2cnf"
            _record_metadata(tree, metadata)
            return ExactSATSolver.two_sat(cnf)

    # Non-strict fallback for empirical families (ConjUI, Square2CNF, BEST_PER_NODE, etc)
    if LanguageFamily.CONJ_UI in fams or LanguageFamily.SQUARE_2CNF in fams or len(fams) > 1:
        if LanguageFamily.SQUARE_2CNF in fams:
            tree.explainer_backend_ = "prototype_case_split"
            metadata.axp_backend = "prototype_case_split"
        else:
            tree.explainer_backend_ = "interval_dfs_fallback"
            metadata.axp_backend = "interval_dfs_fallback"
        metadata.theorem_certified = False
        metadata.path_certificate = "none"

    _record_metadata(tree, metadata)

    if not _path_sat_numeric(path_edges, x, S):
        return False
        
    has_xor = any(pred.is_xor for pred, branch in path_edges)
    if has_xor:
        if not _affine_path_sat(path_edges, x, S):
            return False
            
    return True



def _affine_theorem_guard_error(tree, path_edges) -> str:
    """Return a rejection reason for affine paths outside Python theorem scope."""
    thresholds_by_feature = {}

    for pred, _ in path_edges:
        if not pred.is_xor:
            continue
        for lit in pred.literals:
            if not isinstance(lit, GSNHLiteral):
                return f"Unsupported affine literal type: {type(lit).__name__}"
            thresholds_by_feature.setdefault(int(lit.feature), set()).add(float(lit.threshold))

    for feature, thresholds in thresholds_by_feature.items():
        if len(thresholds) != 1:
            return (
                "Affine theorem mode requires one Boolean threshold atom per "
                f"feature; feature {feature} has {len(thresholds)} thresholds"
            )

    binner = getattr(tree, "binner_", None)
    bin_edges = getattr(binner, "bin_edges_", None)
    if bin_edges is not None:
        for feature in thresholds_by_feature:
            try:
                edges = bin_edges[feature]
            except (IndexError, KeyError, TypeError):
                continue
            if len(edges) > 3:
                return (
                    "Affine theorem mode requires Boolean-compatible binned "
                    f"features; feature {feature} has {len(edges) - 1} bins"
                )
    elif getattr(tree, "n_bins", None) is not None and int(getattr(tree, "n_bins")) > 2:
        return (
            "Affine theorem mode requires Boolean-compatible domains; "
            f"tree.n_bins={getattr(tree, 'n_bins')}"
        )

    return ""

def _record_metadata(tree, metadata: AXpBackendMetadata):
    """Record backend metadata on the tree object."""
    if hasattr(tree, 'axp_metadata_') and isinstance(tree.axp_metadata_, list):
        tree.axp_metadata_.append(metadata)


# ================================================================
# NUMERIC INTERVAL-BASED PATH SATISFIABILITY (journal-faithful)
# ================================================================
# This is the critical fix: we never booleanize features at 0.5.
# Instead we work with the actual thresholds stored in each literal.
#
# Algorithm:
#   1. Translate False-branch edges into conjunctive interval constraints
#      (negation of each literal → per-feature interval intersection).
#   2. Collect True-branch edges as OR-disjunctions to satisfy.
#   3. Check fixed features (in S) against intervals.
#   4. Solve remaining OR-clauses via DFS case-splitting.
#
# Square2CNF extension:
#   True branch → add both clauses as OR constraints.
#   False branch → De Morgan: ¬[(a∨b)∧(c∨d)] = (¬a∧¬b) ∨ (¬c∧¬d)
#   → case split: try negating all lits of clause 1, or clause 2.
# ================================================================

def _path_sat_numeric(path_edges, x, S):
    """
    Exact numeric satisfiability for UI-family paths.

    Determines whether there exists x' such that:
      - For all f in S: x'[f] = x[f]
      - x' satisfies every edge constraint along the path

    Uses interval arithmetic with the ACTUAL thresholds, not booleanization.
    Supports Horn, AntiHorn, ConjUI, and Square2CNF predicates.
    """
    # Per-feature intervals: intervals[f] = (lo, hi)
    # Semantics: lo <= x'[f] < hi  (GE gives [t,+inf), LT gives (-inf,t))
    intervals = {}
    or_clauses = []  # List of lists of GSNHLiteral (disjunctions to satisfy)
    case_splits = []  # List of list-of-lists: each inner list is a mandatory set

    for pred, branch in path_edges:
        if pred.is_xor:
            # XOR predicates handled by affine solver, skip in numeric
            continue

        # ── Square2CNF predicates ──
        if isinstance(pred, Square2CNFPredicate):
            _encode_square_2cnf_edge(pred, branch, intervals, or_clauses,
                                     case_splits)
            continue

        is_and_semantics = (pred.language_family == LanguageFamily.CONJ_UI)

        # -----------------------------------------------------------
        # When do we need an OR-clause (Disjunction)?
        # 1. True branch of Horn (OR) -> at least one literal must be True
        # 2. False branch of ConjUI (AND) -> at least one literal must be
        #    False (De Morgan: ¬(l₁∧l₂) = ¬l₁ ∨ ¬l₂)
        # -----------------------------------------------------------
        if (branch and not is_and_semantics) or (not branch and is_and_semantics):
            clause_lits = []
            for lit in pred.literals:
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    raise NotImplementedError(
                        f"Unsupported literal {type(lit).__name__} cannot be ignored in path SAT"
                    )
                # If it's a False branch, the literal must be negated
                actual_lit = lit if branch else lit.negate()
                clause_lits.append(actual_lit)
            if clause_lits:
                or_clauses.append(clause_lits)

        # -----------------------------------------------------------
        # When do we need a mandatory intersection?
        # 1. False branch of Horn (OR) -> ALL literals must be False
        # 2. True branch of ConjUI (AND) -> ALL literals must be True
        # -----------------------------------------------------------
        else:
            for lit in pred.literals:
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    raise NotImplementedError(
                        f"Unsupported literal {type(lit).__name__} cannot be ignored in path SAT"
                    )
                # If it's a False branch, the literal must be negated
                actual_lit = lit if branch else lit.negate()
                
                lo_new, hi_new = actual_lit.to_interval()
                f = actual_lit.feature
                lo_old, hi_old = intervals.get(f, (-np.inf, np.inf))
                
                lo_merged = max(lo_old, lo_new)
                hi_merged = min(hi_old, hi_new)
                
                if lo_merged >= hi_merged:
                    return False  # Empty interval → UNSAT
                intervals[f] = (lo_merged, hi_merged)

    # Check fixed features against intervals
    for f in S:
        lo, hi = intervals.get(f, (-np.inf, np.inf))
        val = x[f]
        if val < lo or val >= hi:
            return False  # Fixed value outside feasible interval

    # Now solve the OR-clauses and case-splits
    if case_splits:
        return _solve_with_case_splits(or_clauses, case_splits, intervals, x, S)
    else:
        return _solve_or_clauses_dfs(or_clauses, intervals, x, S, 0)


def _encode_square_2cnf_edge(pred, branch, intervals, or_clauses, case_splits):
    """Encode a Square2CNF edge into constraints.

    True branch (P is True):
        P = (a∨b) ∧ (c∨d) → add each clause as an OR constraint.
    False branch (P is False):
        ¬P = ¬[(a∨b)∧(c∨d)] = (¬a∧¬b) ∨ (¬c∧¬d)
        → case split: one of the clauses must be entirely False.
    """
    if branch:
        # True branch: both clauses must be satisfied
        for a, b in pred.clauses:
            or_clauses.append([a, b])
    else:
        # False branch: De Morgan → at least one clause must be fully False
        # That means: (¬a ∧ ¬b) ∨ (¬c ∧ ¬d)
        # = case_split: try each clause being fully negated
        alternatives = []
        for clause in pred.clauses:
            # This alternative: negate ALL literals in this clause
            negated = [lit.negate() for lit in clause]
            alternatives.append(negated)
        case_splits.append(alternatives)


def _solve_with_case_splits(or_clauses, case_splits, intervals, x, S):
    """Solve constraints with case splits from Square2CNF false branches.

    For each case split, we try each alternative. If ANY alternative leads
    to a satisfiable state, the path is satisfiable.
    """
    if not case_splits:
        return _solve_or_clauses_dfs(or_clauses, intervals, x, S, 0)

    # Process first case split, recurse on remaining
    current_split = case_splits[0]
    remaining_splits = case_splits[1:]

    for alternative in current_split:
        # Each alternative is a list of literals that must all be True
        intervals_copy = dict(intervals)
        feasible = True
        for lit in alternative:
            lo_new, hi_new = lit.to_interval()
            f = lit.feature
            lo_old, hi_old = intervals_copy.get(f, (-np.inf, np.inf))
            lo_merged = max(lo_old, lo_new)
            hi_merged = min(hi_old, hi_new)
            if lo_merged >= hi_merged:
                feasible = False
                break
            intervals_copy[f] = (lo_merged, hi_merged)

        if not feasible:
            continue

        # Check fixed features
        fixed_ok = True
        for f in S:
            lo, hi = intervals_copy.get(f, (-np.inf, np.inf))
            val = x[f]
            if val < lo or val >= hi:
                fixed_ok = False
                break
        if not fixed_ok:
            continue

        # Recurse on remaining case splits
        if _solve_with_case_splits(or_clauses, remaining_splits,
                                   intervals_copy, x, S):
            return True

    return False


def _solve_or_clauses_dfs(or_clauses, intervals, x, S, idx):
    """
    DFS over disjunction list: for each OR-clause, try each literal
    as the witness, add its interval constraint, and recurse.
    """
    # Skip clauses that are already satisfied by current intervals
    while idx < len(or_clauses):
        clause = or_clauses[idx]
        # Check if any literal in this clause is already forced true
        # by the current intervals and fixed values
        already_sat = False
        for lit in clause:
            f = lit.feature
            lo_ivl, hi_ivl = intervals.get(f, (-np.inf, np.inf))
            lit_lo, lit_hi = lit.to_interval()
            # If the entire feasible interval for f is within the literal's range
            if lo_ivl >= lit_lo and hi_ivl <= lit_hi:
                already_sat = True
                break
            # If f is fixed, check directly
            if f in S:
                val = x[f]
                if lit.polarity == LiteralPolarity.GE:
                    if val >= lit.threshold:
                        already_sat = True
                        break
                else:
                    if val < lit.threshold:
                        already_sat = True
                        break
        if already_sat:
            idx += 1
            continue
        break

    if idx >= len(or_clauses):
        return True  # All clauses satisfied

    clause = or_clauses[idx]

    # Try each literal as the witness for this clause
    for lit in clause:
        f = lit.feature
        lo_old, hi_old = intervals.get(f, (-np.inf, np.inf))
        lit_lo, lit_hi = lit.to_interval()

        # Intersect current interval with literal's interval
        lo_new = max(lo_old, lit_lo)
        hi_new = min(hi_old, lit_hi)

        if lo_new >= hi_new:
            continue  # This literal can't be satisfied

        # Check fixed feature compatibility
        if f in S:
            val = x[f]
            if val < lo_new or val >= hi_new:
                continue  # Fixed value not in interval

        # Accept this witness: save state, recurse
        intervals_copy = dict(intervals)
        intervals_copy[f] = (lo_new, hi_new)

        if _solve_or_clauses_dfs(or_clauses, intervals_copy, x, S, idx + 1):
            return True

    return False  # No witness works for this clause


# ================================================================
# AFFINE PATH SATISFIABILITY (GF(2) — uses stored thresholds)
# ================================================================

def _affine_path_sat(path_edges, x, S):
    """Exact GF(2) Gaussian elimination for Affine XOR paths.
    Uses each literal's actual stored threshold, not a hardcoded 0.5.

    FIX (STEP 7): Uses (feature, threshold) tuples as variable keys
    instead of bare feature indices, preventing collapse when the same
    feature appears at different thresholds in different XOR constraints.
    """
    equations = []

    # Collect all (feature, threshold) pairs used across XOR predicates
    feature_threshold_map = {}  # feature → threshold (for fixed-value lookup)
    for pred, branch in path_edges:
        if not pred.is_xor:
            continue
        for lit in pred.literals:
            if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                continue
            var_key = (int(lit.feature), float(lit.threshold))
            feature_threshold_map[var_key] = lit.threshold

    # Unit constraints for fixed features
    for f in S:
        for var_key, thr in feature_threshold_map.items():
            if var_key[0] == f:
                val = 1 if x[f] >= thr else 0
                equations.append((set([var_key]), val))

    for pred, branch in path_edges:
        if not pred.is_xor:
            continue
        c = 1 if branch else 0
        vs = set()
        for lit in pred.literals:
            if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                continue
            var_key = (int(lit.feature), float(lit.threshold))
            vs ^= {var_key}
            if lit.polarity == LiteralPolarity.LT:
                c ^= 1
        equations.append((vs, c & 1))

    return ExactSATSolver.affine_sat(equations)


def extract_axp(tree, x):
    """Extract a single minimal AXp for an instance. Returns set of features.

    Deletion-based algorithm: start with all features, try removing each
    in deterministic order (0, 1, ..., n_features-1). Remove feature f
    if weak_axp_check(S - {f}) remains True.

    Performance note: paths are enumerated once and reused across all
    ``n_features`` calls to ``weak_axp_check``, avoiding O(d × 2^depth)
    redundant recursive traversals.
    """
    from gsnh_mdt.tree.prediction import predict

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[0]
    y = predict(tree, x.reshape(1, -1))[0]
    S = set(range(tree.n_features_))

    # Clear metadata for this extraction
    if hasattr(tree, 'axp_metadata_'):
        tree.axp_metadata_ = []

    # Enumerate paths ONCE — this was the 67% bottleneck
    paths = _enumerate_paths(tree)

    # Deterministic deletion order: 0, 1, ..., n_features-1
    for f in range(tree.n_features_):
        S.remove(f)
        if not weak_axp_check(tree, x, y, S, paths=paths):
            S.add(f)

    return S
