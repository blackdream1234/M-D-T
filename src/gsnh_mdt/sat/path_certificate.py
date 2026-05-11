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
    False branch: ¬[(l1∨l2)∧(l3∨l4)] = (¬l1∧¬l2) ∨ (¬l3∧¬l4)
                  Encoded with auxiliary variable s:
                  (¬s ∨ ¬l1), (¬s ∨ ¬l2), (s ∨ ¬l3), (s ∨ ¬l4)
    """
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
        # False branch: 2-CNF with auxiliary switch variable
        if len(pred.clauses) != 2:
            raise NonTheoremPathError("Square2CNF false branch must have exactly 2 clauses.")
        
        c1, c2 = pred.clauses
        l1, l2 = c1
        l3, l4 = c2
        
        # Auxiliary variable
        aux_atom = ("square2cnf_false_aux", edge_id)
        if aux_atom not in encoding.atom_to_var:
            idx = len(encoding.var_to_atom)
            encoding.atom_to_var[aux_atom] = idx
            encoding.var_to_atom.append(aux_atom)
        s_idx = encoding.atom_to_var[aux_atom]
        
        enc_l1 = encode_literal(l1, encoding) if isinstance(l1, GSNHLiteral) else None
        enc_l2 = encode_literal(l2, encoding) if isinstance(l2, GSNHLiteral) else None
        enc_l3 = encode_literal(l3, encoding) if isinstance(l3, GSNHLiteral) else None
        enc_l4 = encode_literal(l4, encoding) if isinstance(l4, GSNHLiteral) else None
        
        # (¬s ∨ ¬l1), (¬s ∨ ¬l2), (s ∨ ¬l3), (s ∨ ¬l4)
        if enc_l1: encoding.clauses.append([(s_idx, False), negate_encoded_lit(enc_l1)])
        if enc_l2: encoding.clauses.append([(s_idx, False), negate_encoded_lit(enc_l2)])
        if enc_l3: encoding.clauses.append([(s_idx, True), negate_encoded_lit(enc_l3)])
        if enc_l4: encoding.clauses.append([(s_idx, True), negate_encoded_lit(enc_l4)])


def classify_cnf_fragment(cnf: CNF) -> str:
    """Classify a CNF fragment into a tractable class.

    Returns one of: "horn", "antihorn", "2cnf", "none"

    Rules (matching Coq definitions):
    - horn: every clause has ≤ 1 positive literal
    - antihorn: every clause has ≤ 1 negative literal
    - 2cnf: every clause has length ≤ 2
    - none: does not satisfy any of the above
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

    A path is polynomial-safe iff its CNF is Horn, AntiHorn, or 2CNF.

    This is the key theorem-compliance function: the Coq proof establishes
    that Horn, AntiHorn, and 2CNF are all solvable in polynomial time,
    while general CNF is NP-complete.
    """
    try:
        _, cnf = build_ordered_selected_path_cnf(path_edges, x, selected_features)
    except NonTheoremPathError:
        return False, "none"

    certificate = classify_cnf_fragment(cnf)
    is_safe = certificate in ("horn", "antihorn", "2cnf")
    return is_safe, certificate
