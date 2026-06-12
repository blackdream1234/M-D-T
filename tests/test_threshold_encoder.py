import pytest
import numpy as np
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.predicates import GSNHPredicate
from gsnh_mdt.types import LanguageFamily
from gsnh_mdt.sat.threshold_encoder import (
    ThresholdEncoding, encode_horn_path, encode_antihorn_path, atom_id,
    add_structural_order_clauses
)
from gsnh_mdt.sat.exact_solver import ExactSATSolver
from gsnh_mdt.tree.explainer import weak_axp_check
from gsnh_mdt.types import LiteralPolarity

def make_lit(f, t, is_ge):
    return GSNHLiteral(feature=f, threshold=float(t), polarity=LiteralPolarity.GE if is_ge else LiteralPolarity.LT)

def test_horn_structural_contradiction():
    # Test 1: x0 >= 5 and x0 < 2
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 5.0, True),), information_gain=0.1, language_family=LanguageFamily.HORN), True),
        (GSNHPredicate(literals=(make_lit(0, 2.0, False),), information_gain=0.1, language_family=LanguageFamily.HORN), True)
    ]
    x = np.array([0.0])
    S = set()
    clauses = encode_horn_path(path_edges, x, S)
    assert not ExactSATSolver.horn_sat(clauses)

def test_horn_structural_order_implication():
    # Test 2: B(f, 10) => B(f, 5).
    # Path has x0 >= 10. We check if SAT with fixed S.
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 10.0, True),), information_gain=0.1, language_family=LanguageFamily.HORN), True)
    ]
    x_valid = np.array([12.0])
    clauses = encode_horn_path(path_edges, x_valid, S={0})
    assert ExactSATSolver.horn_sat(clauses)

    x_invalid = np.array([4.0])
    clauses_inv = encode_horn_path(path_edges, x_invalid, S={0})
    assert not ExactSATSolver.horn_sat(clauses_inv)

def test_horn_path_true_branch():
    # Test 3: Horn path true branch: (x0 < 3 OR x1 >= 2)
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 3.0, False), make_lit(1, 2.0, True)), information_gain=0.1, language_family=LanguageFamily.HORN), True)
    ]
    x = np.array([0.0, 0.0])
    clauses = encode_horn_path(path_edges, x, S=set())
    assert ExactSATSolver.horn_sat(clauses)

def test_horn_path_false_branch():
    # Test 4: False branch of Horn: not(x0 < 3 OR x1 >= 2)
    # Equivalent: x0 >= 3 AND x1 < 2
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 3.0, False), make_lit(1, 2.0, True)), information_gain=0.1, language_family=LanguageFamily.HORN), False)
    ]
    x = np.array([4.0, 1.0])
    clauses = encode_horn_path(path_edges, x, S={0, 1})
    assert ExactSATSolver.horn_sat(clauses)
    
    x_invalid = np.array([1.0, 1.0])
    clauses_inv = encode_horn_path(path_edges, x_invalid, S={0, 1})
    assert not ExactSATSolver.horn_sat(clauses_inv)

def test_antihorn_mirror():
    # Test 5: AntiHorn mirror
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 3.0, True), make_lit(1, 2.0, False)), information_gain=0.1, language_family=LanguageFamily.ANTI_HORN), True)
    ]
    x = np.array([0.0, 0.0])
    clauses = encode_antihorn_path(path_edges, x, S=set())
    assert ExactSATSolver.antihorn_sat(clauses)

def test_no_dfs_call_in_journal_horn(monkeypatch):
    # Test 6: Monkeypatch _solve_or_clauses_dfs to raise RuntimeError
    import gsnh_mdt.tree.explainer as explainer
    def mock_dfs(*args, **kwargs):
        raise RuntimeError("DFS should not be called")
    monkeypatch.setattr(explainer, "_solve_or_clauses_dfs", mock_dfs)
    
    # Run weak_axp_check on a Horn tree mock
    class MockTree:
        n_features_ = 2
        explainer_backend_ = ""
        def __init__(self):
            pass

    tree = MockTree()
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 3.0, False),), information_gain=0.1, language_family=LanguageFamily.HORN), True)
    ]
    x = np.array([1.0, 1.0])
    y = 1
    paths = [(path_edges, 0)]  # Target is 1, leaf is 0 -> it will check sat
    
    result = weak_axp_check(tree, x, y, S=set(), paths=paths)
    # SAT path, so it can reach wrong leaf, so it's NOT a weak AXP
    assert result is False
    assert tree.explainer_backend_ == "structural_horn"

def test_affine_different_thresholds():
    # Test 7: Affine different thresholds
    path_edges = [
        (GSNHPredicate(literals=(make_lit(0, 2.0, True),), information_gain=0.1, language_family=LanguageFamily.AFFINE, is_xor=True), True),
        (GSNHPredicate(literals=(make_lit(0, 5.0, True),), information_gain=0.1, language_family=LanguageFamily.AFFINE, is_xor=True), True)
    ]
    x = np.array([3.0])
    import gsnh_mdt.tree.explainer as explainer
    res = explainer._affine_path_sat(path_edges, x, S=set())
    # XOR(x0>=2) = 1  AND XOR(x0>=5) = 1 -> SAT (without fixed features)
    assert res is True

def test_empty_path_sat():
    import gsnh_mdt.tree.explainer as explainer
    class T:
        pass
    tree = T()
    assert explainer._is_sat_path(tree, [], np.array([0.0]), set()) is True
    assert tree.explainer_backend_ == "empty_path"


def test_structural_order_all_pairs_count():
    enc3 = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    for t in (2.0, 5.0, 8.0):
        atom_id(enc3, 0, t)
    add_structural_order_clauses(enc3)
    assert len(enc3.clauses) == 3

    enc4 = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    for t in (1.0, 2.0, 5.0, 8.0):
        atom_id(enc4, 0, t)
    add_structural_order_clauses(enc4)
    assert len(enc4.clauses) == 6


def test_atom_id_canonicalizes_near_equal_thresholds():
    enc = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])

    assert atom_id(enc, 0, 0.1 + 0.2) == atom_id(enc, 0, 0.3)
    assert len(enc.var_to_atom) == 1


def test_structural_order_uses_canonical_thresholds():
    enc = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])

    atom_id(enc, 0, 0.1 + 0.2)
    atom_id(enc, 0, 0.3)
    add_structural_order_clauses(enc)

    assert len(enc.var_to_atom) == 1
    assert len(enc.clauses) == 0
