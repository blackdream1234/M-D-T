"""
Path-level CNF certificate checker for theorem-strict compliance.

Implements the core functions required by the Coq-verified theorem:
1. build_ordered_selected_path_cnf — constructs the full CNF from a path
2. classify_cnf_fragment — classifies as Horn/AntiHorn/2CNF/none
3. is_polynomial_safe_path — checks if the CNF falls in a tractable class

These correspond directly to the formal definitions in GSNH_Threshold_AXp.v:
  - structural_order_clause
  - is_horn_clause / is_antihorn_clause
  - path_clauses / edge_clauses
"""

from typing import Dict, List, Optional, Set, Tuple

from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.binary import GSNHBinaryLiteral
from gsnh_mdt.literals.compare import CompareLiteral
from gsnh_mdt.literals.predicates import Square2CNFPredicate
from gsnh_mdt.sat.threshold_encoder import (
    ThresholdEncoding,
    atom_id,
    encode_literal,
    negate_encoded_lit,
    add_structural_order_clauses,
    add_fixed_assignment_clauses,
)
from gsnh_mdt.types import LanguageFamily, LiteralPolarity

# Type aliases matching the Coq specification
Clause = List[Tuple[int, bool]]  # List of (variable_id, is_positive)
CNF = List[Clause]


class NonTheoremPathError(Exception):
    """Raised when a path cannot be certified as theorem-compliant.

    In theorem_strict mode, this is raised when a path's CNF does not
    fall into Horn, AntiHorn, or 2CNF, meaning no polynomial-time
    decision procedure from the Coq-verified theorem applies.
    """
    pass


def build_ordered_selected_path_cnf(
    path_edges: list,
    x,
    selected_features: set,
) -> Tuple[ThresholdEncoding, CNF]:
    """Build the ordered selected-path CNF as specified by the Coq proof.

    Produces:
        selected_feature_unit_clauses
        + structural_order_clauses
        + path_clauses

    This mirrors the formal definition in GSNH_Threshold_AXp.v §4-5.

    Parameters
    ----------
    path_edges : list of (predicate, branch_bool) pairs
    x : array-like — the explained instance
    selected_features : set of feature indices in S

    Returns
    -------
    encoding : ThresholdEncoding — the atom↔variable mapping
    cnf : list of clauses — the full CNF
    """
    encoding = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])

    # 1. Encode path clauses
    for edge_id, (pred, branch) in enumerate(path_edges):
        # Skip predicates with unsupported literal types
        for lit in _get_literals(pred):
            if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                raise NonTheoremPathError(
                    f"Unsupported literal type in theorem mode: {type(lit).__name__}"
                )

        if isinstance(pred, Square2CNFPredicate):
            _encode_square2cnf_path_clauses(pred, branch, encoding, edge_id)
        else:
            if branch:
                # True branch: disjunction → single clause
                clause = []
                for lit in pred.literals:
                    if isinstance(lit, GSNHLiteral):
                        clause.append(encode_literal(lit, encoding))
                if clause:
                    encoding.clauses.append(clause)
            else:
                # False branch: negation → singleton clauses
                # ¬(l1 ∨ l2 ∨ ... ∨ lk) = ¬l1 ∧ ¬l2 ∧ ... ∧ ¬lk
                for lit in pred.literals:
                    if isinstance(lit, GSNHLiteral):
                        enc_lit = encode_literal(lit, encoding)
                        encoding.clauses.append([negate_encoded_lit(enc_lit)])

    # 2. Add structural order clauses
    add_structural_order_clauses(encoding)

    # 3. Add fixed assignment clauses for selected features
    add_fixed_assignment_clauses(encoding, x, selected_features)

    return encoding, encoding.clauses


def _get_literals(pred) -> list:
    """Extract all literals from a predicate."""
    if isinstance(pred, Square2CNFPredicate):
        return list(pred.iter_literals())
    return list(pred.literals)


def _encode_square2cnf_path_clauses(pred: Square2CNFPredicate, branch: bool,
                                     encoding: ThresholdEncoding, edge_id: int):
    """Encode Square2CNF predicate path clauses.

    True branch:  (l1∨l2) ∧ (l3∨l4) → two 2-literal clauses
    False branch: ¬[(l1∨l2)∧(l3∨l4)] =
                  (¬l1∨¬l3) ∧ (¬l1∨¬l4) ∧ (¬l2∨¬l3) ∧ (¬l2∨¬l4)
    """
    if len(pred.clauses) != 2:
        raise NonTheoremPathError(
            "Theorem-certified Square2CNF requires exactly two 2-literal clauses: "
            "(l1 OR l2) AND (l3 OR l4)."
        )
    for clause in pred.clauses:
        if len(clause) != 2:
            raise NonTheoremPathError(
                "Theorem-certified Square2CNF requires every clause to have exactly two literals."
            )

    if branch:
        # True branch: each clause becomes a 2-literal disjunction
        for a, b in pred.clauses:
            clause = []
            if isinstance(a, GSNHLiteral):
                clause.append(encode_literal(a, encoding))
            if isinstance(b, GSNHLiteral):
                clause.append(encode_literal(b, encoding))
            if clause:
                encoding.clauses.append(clause)
    else:
        # False branch: exact Square2CNF complement over original literals.
        # ¬[(l1∨l2)∧(l3∨l4)] =
        #   (¬l1∨¬l3) ∧ (¬l1∨¬l4) ∧ (¬l2∨¬l3) ∧ (¬l2∨¬l4)
        c1, c2 = pred.clauses
        l1, l2 = c1
        l3, l4 = c2

        encoded = [
            encode_literal(l1, encoding),
            encode_literal(l2, encoding),
            encode_literal(l3, encoding),
            encode_literal(l4, encoding),
        ]
        e1, e2, e3, e4 = [negate_encoded_lit(lit) for lit in encoded]
        encoding.clauses.extend([
            [e1, e3],
            [e1, e4],
            [e2, e3],
            [e2, e4],
        ])


def classify_cnf_fragment(cnf: CNF) -> str:
    """Classify a CNF fragment into a tractable class.

    Returns one of: "horn", "antihorn", "2cnf", "none"

    Implemented structural SAT-fragment rules:
    - horn: every clause has ≤ 1 positive literal
    - antihorn: every clause has ≤ 1 negative literal
    - 2cnf: every clause has length ≤ 2
    - none: does not satisfy any of the above

    This classifier intentionally checks ordinary Horn/AntiHorn SAT shape;
    it does not validate the stronger star-nested Horn/AntiHorn language
    condition unless a separate star-nested checker is added.
    """
    if not cnf:
        return "horn"  # empty CNF is trivially all three

    is_horn = True
    is_antihorn = True
    is_2cnf = True

    for clause in cnf:
        n_positive = sum(1 for _, sign in clause if sign)
        n_negative = sum(1 for _, sign in clause if not sign)

        if n_positive > 1:
            is_horn = False
        if n_negative > 1:
            is_antihorn = False
        if len(clause) > 2:
            is_2cnf = False

    if is_horn:
        return "horn"
    if is_antihorn:
        return "antihorn"
    if is_2cnf:
        return "2cnf"
    return "none"


def is_polynomial_safe_path(
    path_edges: list,
    x,
    selected_features: set,
) -> Tuple[bool, str]:
    """Check if a path's ordered selected-path CNF is polynomial-safe.

    Returns (is_safe, certificate) where certificate is one of:
    "horn", "antihorn", "2cnf", "none"

    A path is polynomial-safe for the implemented Python checker iff its CNF
    has an accepted ordinary Horn, AntiHorn, or 2CNF SAT-fragment certificate.
    This is a structural SAT-fragment certificate, not a validation of the
    stronger star-nested Horn/AntiHorn language condition.
    """
    try:
        _, cnf = build_ordered_selected_path_cnf(path_edges, x, selected_features)
    except NonTheoremPathError:
        return False, "none"

    certificate = classify_cnf_fragment(cnf)
    is_safe = certificate in ("horn", "antihorn", "2cnf")
    return is_safe, certificate
