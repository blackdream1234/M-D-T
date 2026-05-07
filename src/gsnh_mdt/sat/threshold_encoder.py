from dataclasses import dataclass
from typing import Dict, Tuple, List, Set
from gsnh_mdt.literals.base import GSNHLiteral, LiteralPolarity

Atom = Tuple[int, float]  # (feature, threshold)
Clause = List[Tuple[Atom, bool]]  # Literal representation for the SAT solver using Atoms

@dataclass
class ThresholdEncoding:
    atom_to_var: Dict[Atom, int]
    var_to_atom: List[Atom]
    clauses: List[List[Tuple[int, bool]]]

def atom_id(encoding: ThresholdEncoding, feature: int, threshold: float) -> int:
    atom = (int(feature), float(threshold))
    if atom not in encoding.atom_to_var:
        idx = len(encoding.var_to_atom)
        encoding.atom_to_var[atom] = idx
        encoding.var_to_atom.append(atom)
    return encoding.atom_to_var[atom]

def encode_literal(lit: GSNHLiteral, encoding: ThresholdEncoding) -> Tuple[int, bool]:
    if not isinstance(lit, GSNHLiteral):
        raise NotImplementedError(f"Unsupported literal type: {type(lit)}")
    
    f = lit.feature
    t = lit.threshold
    var_idx = atom_id(encoding, f, t)
    
    # B(f, t) := x[f] >= t
    # GE is encoded as B(f, t) -> sign=True
    # LT is encoded as ¬B(f, t) -> sign=False
    sign = lit.polarity == LiteralPolarity.GE
    return (var_idx, sign)

def negate_encoded_lit(encoded_lit: Tuple[int, bool]) -> Tuple[int, bool]:
    return (encoded_lit[0], not encoded_lit[1])

def add_structural_order_clauses(encoding: ThresholdEncoding):
    # Group atoms by feature
    atoms_by_feature: Dict[int, List[float]] = {}
    for f, t in encoding.var_to_atom:
        atoms_by_feature.setdefault(f, []).append(t)
        
    for f, thresholds in atoms_by_feature.items():
        sorted_t = sorted(thresholds)
        # For t1 < t2, B(f, t2) => B(f, t1)
        # which is ¬B(f, t2) ∨ B(f, t1)
        for i in range(len(sorted_t) - 1):
            for j in range(i + 1, len(sorted_t)):
                t1 = sorted_t[i]
                t2 = sorted_t[j]
                var_t1 = encoding.atom_to_var[(f, t1)]
                var_t2 = encoding.atom_to_var[(f, t2)]
                clause = [(var_t2, False), (var_t1, True)]
                encoding.clauses.append(clause)

def add_fixed_assignment_clauses(encoding: ThresholdEncoding, x, S: set):
    for f in S:
        val = x[f]
        # Find all thresholds for this fixed feature
        for var_idx, (atom_f, atom_t) in enumerate(encoding.var_to_atom):
            if atom_f == f:
                # If x[f] >= t, we add B(f, t) -> (var_idx, True)
                # Else, ¬B(f, t) -> (var_idx, False)
                if val >= atom_t:
                    encoding.clauses.append([(var_idx, True)])
                else:
                    encoding.clauses.append([(var_idx, False)])

def encode_horn_path(path_edges: list, x, S: set) -> List[List[Tuple[int, bool]]]:
    encoding = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    
    # Collect all predicates and encode
    for pred, branch in path_edges:
        if branch:
            # True branch: l1 ∨ ... ∨ lk
            clause = []
            for lit in pred.literals:
                clause.append(encode_literal(lit, encoding))
            encoding.clauses.append(clause)
        else:
            # False branch: ¬l1 ∧ ... ∧ ¬lk
            # Equivalent to multiple unit clauses: ¬l1, ¬l2, ...
            for lit in pred.literals:
                enc_lit = encode_literal(lit, encoding)
                encoding.clauses.append([negate_encoded_lit(enc_lit)])
                
    add_structural_order_clauses(encoding)
    add_fixed_assignment_clauses(encoding, x, S)
    return encoding.clauses

def encode_antihorn_path(path_edges: list, x, S: set) -> List[List[Tuple[int, bool]]]:
    # Same logic for paths, we just produce the encoding clauses.
    # The ExactSATSolver.antihorn_sat will handle the Horn-reduction by flipping everything.
    encoding = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    
    for pred, branch in path_edges:
        if branch:
            clause = []
            for lit in pred.literals:
                clause.append(encode_literal(lit, encoding))
            encoding.clauses.append(clause)
        else:
            for lit in pred.literals:
                enc_lit = encode_literal(lit, encoding)
                encoding.clauses.append([negate_encoded_lit(enc_lit)])
                
    add_structural_order_clauses(encoding)
    add_fixed_assignment_clauses(encoding, x, S)
    return encoding.clauses


def build_ordered_selected_path_cnf(path_edges: list, x, selected_features: set):
    """Build ordered selected-path CNF: selected units + order + path clauses."""
    enc = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    for pred, branch in path_edges:
        if not hasattr(pred, "literals"):
            return []
        if branch:
            clause = [encode_literal(lit, enc) for lit in pred.literals]
            enc.clauses.append(clause)
        else:
            for lit in pred.literals:
                enc.clauses.append([negate_encoded_lit(encode_literal(lit, enc))])
    add_structural_order_clauses(enc)
    add_fixed_assignment_clauses(enc, x, selected_features)
    return enc.clauses
