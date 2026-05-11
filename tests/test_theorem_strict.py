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
    """A deliberately non-certified mixed path must be rejected in theorem mode.

    Do not rely on fitting a random BEST_PER_NODE tree here: the learned path may
    legitimately be Horn/AntiHorn/2CNF-certified. This test constructs a path
    whose ordered CNF is neither Horn, AntiHorn, nor 2-CNF.
    """
    import numpy as np
    import pytest

    from gsnh_mdt.literals.base import GSNHLiteral
    from gsnh_mdt.literals.predicates import GSNHPredicate
    from gsnh_mdt.sat.path_certificate import NonTheoremPathError
    from gsnh_mdt.tree.explainer import _is_sat_path
    from gsnh_mdt.types import LanguageFamily, LiteralPolarity

    def lit(f, t, pol):
        return GSNHLiteral(feature=f, threshold=float(t), polarity=pol)

    # Clause with 3 positive literals: not Horn, but AntiHorn.
    pos3 = GSNHPredicate(
        literals=(
            lit(0, 0.2, LiteralPolarity.GE),
            lit(1, 0.2, LiteralPolarity.GE),
            lit(2, 0.2, LiteralPolarity.GE),
        ),
        information_gain=0.1,
        language_family=LanguageFamily.CONJ_UI,
    )

    # Clause with 3 negative literals: Horn, but not AntiHorn.
    neg3 = GSNHPredicate(
        literals=(
            lit(0, 0.8, LiteralPolarity.LT),
            lit(1, 0.8, LiteralPolarity.LT),
            lit(2, 0.8, LiteralPolarity.LT),
        ),
        information_gain=0.1,
        language_family=LanguageFamily.CONJ_UI,
    )

    class MockTree:
        theorem_strict = True
        explainer_backend_ = ""
        axp_metadata_ = []

    tree = MockTree()
    path_edges = [(pos3, True), (neg3, True)]

    with pytest.raises(NonTheoremPathError):
        _is_sat_path(tree, path_edges, np.array([0.0, 0.0, 0.0]), set())

