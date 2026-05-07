import numpy as np
import pytest

from gsnh_mdt.api.classifier import GSNHClassifier
from gsnh_mdt.sat.threshold_encoder import ThresholdEncoding, atom_id, encode_literal, build_ordered_selected_path_cnf
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.types import LiteralPolarity, LanguageFamily
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.tree.explainer import NonTheoremPathError


def test_journal_any_rejected_api():
    X = np.array([[0.0], [1.0], [0.2], [0.8]])
    y = np.array([0, 1, 0, 1])
    clf = GSNHClassifier(mode='journal', language=LanguageFamily.ANY, use_calibration=False, use_pruning=False)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_lt_encodes_negated_B_atom():
    enc = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    lit = GSNHLiteral(0, 0.5, LiteralPolarity.LT)
    var, sign = encode_literal(lit, enc)
    assert sign is False


def test_false_branch_encoded_as_singleton_negations():
    l1 = GSNHLiteral(0, 0.5, LiteralPolarity.GE)
    l2 = GSNHLiteral(1, 0.3, LiteralPolarity.LT)
    pred = type('P', (), {'literals': (l1, l2)})()
    cnf = build_ordered_selected_path_cnf([(pred, False)], np.array([0.2, 0.2]), {0})
    unit_count = sum(1 for c in cnf if len(c) == 1)
    assert unit_count >= 2


def test_order_implications_complete_for_three_thresholds():
    enc = ThresholdEncoding(atom_to_var={}, var_to_atom=[], clauses=[])
    atom_id(enc, 0, 0.1)
    atom_id(enc, 0, 0.5)
    atom_id(enc, 0, 0.9)
    from gsnh_mdt.sat.threshold_encoder import add_structural_order_clauses
    add_structural_order_clauses(enc)
    # all ordered pairs: (0.9->0.5), (0.9->0.1), (0.5->0.1)
    assert len(enc.clauses) == 3


def test_mixed_bestpn_without_certificate_rejected_in_theorem_mode():
    X = np.random.RandomState(0).rand(40, 4)
    y = (X[:, 0] > 0.5).astype(int)
    tree = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=2, min_samples_leaf=2, min_samples_split=4),
        language=LanguageFamily.BEST_PER_NODE,
        theorem_strict=True,
        search_3d=False,
    )
    tree.fit(X, y)
    with pytest.raises(NonTheoremPathError):
        tree.extract_axp(X[0])
