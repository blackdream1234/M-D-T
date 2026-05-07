"""
Square 2CNF search — real paper-style (Carbonnel 2025).

Generates predicates of the form:
    (l₁ ∨ l₂) ∧ (l₃ ∨ l₄)

First implementation: correctness-first, direct mask evaluation.
No optimized prefix sums — uses top-k features and limited thresholds.

Each clause is a disjunction of 2 unary interval literals.
The conjunction of 2 such clauses forms the split predicate.
"""

import numpy as np

from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.predicates import Square2CNFPredicate
from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.scoring.penalties import penalized_gain
from gsnh_mdt.types import LiteralPolarity


def _make_literal(feature, threshold, polarity):
    """Create a GSNHLiteral with given parameters."""
    return GSNHLiteral(
        feature=feature,
        threshold=threshold,
        polarity=polarity,
    )


def _generate_literals(features, edges, n_bins):
    """Generate candidate literals from features and bin edges.

    For each feature, generate GE and LT literals at each bin edge.
    Returns list of GSNHLiteral objects.
    """
    lits = []
    for f in features:
        f_edges = edges[f]
        nb = n_bins[f]
        # Use a subset of thresholds to control runtime
        step = max(1, nb // 8)
        for bi in range(step, nb, step):
            if bi < len(f_edges):
                t = float(f_edges[bi])
                lits.append(_make_literal(f, t, LiteralPolarity.GE))
                lits.append(_make_literal(f, t, LiteralPolarity.LT))
    return lits


def search_square_2cnf(X, y, features, edges, n_bins, min_leaf,
                       max_candidates=500, n_classes=2, penalty=True):
    """Search for the best Square2CNF split.

    Generates candidate (l₁ ∨ l₂) ∧ (l₃ ∨ l₄) predicates and
    evaluates them by direct mask computation.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix (already binned or raw).
    y : np.ndarray, shape (n,)
        Binary labels.
    features : list[int]
        Feature indices to consider (top-k).
    edges : dict[int, np.ndarray]
        Bin edges per feature.
    n_bins : dict[int, int]
        Number of bins per feature.
    min_leaf : int
        Minimum samples per branch.
    max_candidates : int
        Maximum number of 2-clause combinations to evaluate.

    Returns
    -------
    best_gain : float
        Best BIC-penalized information gain (or -1 if none found).
    best_pred : Square2CNFPredicate or None
        The best predicate found.
    """
    n = len(y)
    total_pos = float(y.sum())
    total_neg = float(n - total_pos)

    if total_pos < 1 or total_neg < 1:
        return -1.0, None

    # Generate candidate literals
    lits = _generate_literals(features, edges, n_bins)
    if len(lits) < 4:
        return -1.0, None

    # Pre-compute masks for all literals
    lit_masks = np.array([lit.evaluate(X) for lit in lits], dtype=bool)

    best_gain = -1.0
    best_pred = None
    n_lits = len(lits)
    evaluated = 0

    # Generate all 2-literal disjunctive clauses
    # Then combine pairs of clauses into 2-clause conjunctions
    # For efficiency: limit search to top combinations

    # Generate clauses: (i, j) pairs where i < j, representing (lits[i] ∨ lits[j])
    clause_indices = []
    for i in range(n_lits):
        for j in range(i + 1, n_lits):
            # Skip if same feature + same polarity + same threshold
            if (lits[i].feature == lits[j].feature
                    and lits[i].polarity == lits[j].polarity
                    and lits[i].threshold == lits[j].threshold):
                continue
            clause_indices.append((i, j))

    n_clauses = len(clause_indices)
    if n_clauses < 2:
        return -1.0, None

    # Pre-compute clause masks
    clause_masks = np.array([
        lit_masks[i] | lit_masks[j]
        for i, j in clause_indices
    ], dtype=bool)

    # Combine pairs of clauses: (c1, c2) where c1 < c2
    for ci in range(n_clauses):
        if evaluated >= max_candidates:
            break
        for cj in range(ci + 1, n_clauses):
            if evaluated >= max_candidates:
                break

            # Ensure clauses use at least partially different features
            # to avoid trivially redundant predicates
            i1, j1 = clause_indices[ci]
            i2, j2 = clause_indices[cj]
            feats_c1 = {lits[i1].feature, lits[j1].feature}
            feats_c2 = {lits[i2].feature, lits[j2].feature}

            # Skip if both clauses use the exact same feature pair
            if feats_c1 == feats_c2:
                continue

            # AND of two clause masks
            mask = clause_masks[ci] & clause_masks[cj]
            in_total = int(mask.sum())
            out_total = n - in_total

            if in_total < min_leaf or out_total < min_leaf:
                evaluated += 1
                continue

            in_pos = float(y[mask].sum())
            in_neg = float(in_total - in_pos)

            gain = information_gain(total_pos, total_neg, in_pos, in_neg)
            
            if penalty and gain > 0:
                gain = penalized_gain(gain, arity=2, n_bins=max(4, 2), n_samples=n, n_classes=n_classes)

            if gain > best_gain and gain > 0:
                best_gain = gain
                clause1 = (lits[i1], lits[j1])
                clause2 = (lits[i2], lits[j2])
                best_pred = Square2CNFPredicate(
                    clauses=(clause1, clause2),
                    information_gain=gain,
                )

            evaluated += 1

    return best_gain, best_pred
