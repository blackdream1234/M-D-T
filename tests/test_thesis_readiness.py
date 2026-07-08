"""Regression tests for thesis-readiness correctness boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from gsnh_mdt.api.classifier import GSNHClassifier
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.predicates import GSNHPredicate, Square2CNFPredicate
from gsnh_mdt.sat.path_certificate import NonTheoremPathError
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.explainer import _is_sat_path
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.types import LanguageFamily, LiteralPolarity


def _lit(feature: int, threshold: float = 0.5, positive: bool = True) -> GSNHLiteral:
    return GSNHLiteral(
        feature,
        threshold,
        LiteralPolarity.GE if positive else LiteralPolarity.LT,
    )


def _tree_stub(theorem_strict: bool = False):
    return SimpleNamespace(
        theorem_strict=theorem_strict,
        explainer_backend_="",
        axp_metadata_=[],
        n_bins=2,
    )


def _horn_pred() -> GSNHPredicate:
    return GSNHPredicate((_lit(0, positive=True),), language_family=LanguageFamily.HORN)


def _antihorn_pred() -> GSNHPredicate:
    return GSNHPredicate((_lit(0, positive=True),), language_family=LanguageFamily.ANTI_HORN)


def _xor_pred() -> GSNHPredicate:
    return GSNHPredicate(
        (_lit(1, positive=True), _lit(2, positive=True)),
        language_family=LanguageFamily.AFFINE,
        is_xor=True,
    )


def test_mixed_horn_xor_non_strict_falls_back_without_crashing():
    tree = _tree_stub(theorem_strict=False)
    path = [(_horn_pred(), True), (_xor_pred(), True)]

    result = _is_sat_path(tree, path, np.array([1.0, 1.0, 0.0]), {0, 1, 2})

    assert isinstance(result, (bool, np.bool_))
    assert tree.explainer_backend_ != "structural_horn"
    assert tree.axp_metadata_[-1].axp_backend == "interval_dfs_fallback"
    assert tree.axp_metadata_[-1].theorem_certified is False


def test_mixed_horn_xor_theorem_strict_rejected_not_structural_horn():
    tree = _tree_stub(theorem_strict=True)
    path = [(_horn_pred(), True), (_xor_pred(), True)]

    with pytest.raises(NonTheoremPathError):
        _is_sat_path(tree, path, np.array([1.0, 1.0, 0.0]), {0, 1, 2})

    assert tree.explainer_backend_ != "structural_horn"
    assert tree.axp_metadata_[-1].axp_backend == "affine"
    assert tree.axp_metadata_[-1].theorem_certified is False


def test_pure_horn_path_still_uses_structural_horn():
    tree = _tree_stub(theorem_strict=False)

    assert _is_sat_path(tree, [(_horn_pred(), True)], np.array([1.0]), {0}) is True
    assert tree.explainer_backend_ == "structural_horn"
    assert tree.axp_metadata_[-1].path_certificate == "horn"
    assert tree.axp_metadata_[-1].theorem_certified is True


def test_pure_antihorn_path_still_uses_structural_antihorn():
    tree = _tree_stub(theorem_strict=False)

    assert _is_sat_path(tree, [(_antihorn_pred(), True)], np.array([1.0]), {0}) is True
    assert tree.explainer_backend_ == "structural_antihorn"
    assert tree.axp_metadata_[-1].path_certificate == "antihorn"
    assert tree.axp_metadata_[-1].theorem_certified is True


def test_pure_square2cnf_theorem_path_still_uses_two_sat():
    tree = _tree_stub(theorem_strict=True)
    pred = Square2CNFPredicate(
        clauses=(
            (_lit(0, positive=True), _lit(1, positive=True)),
            (_lit(0, positive=False), _lit(1, positive=True)),
        ),
        information_gain=0.1,
    )

    assert _is_sat_path(tree, [(pred, True)], np.array([1.0, 1.0]), {0, 1}) is True
    assert tree.explainer_backend_ == "two_sat"
    assert tree.axp_metadata_[-1].axp_backend == "two_sat"
    assert tree.axp_metadata_[-1].path_certificate == "2cnf"
    assert tree.axp_metadata_[-1].theorem_certified is True


def _pruning_dataset(n: int = 600):
    rng = np.random.RandomState(0)
    X = rng.rand(n, 4)
    y = (X[:, 0] + 0.25 * X[:, 1] > 0.55).astype(np.int32)
    return X, y


def test_expert_tree_internal_pruning_uses_current_node_schema_and_predicts():
    X, y = _pruning_dataset()
    tree = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
        ),
        language=LanguageFamily.HORN,
        mode="heuristic",
        use_supervised_binning=False,
        search_3d=False,
        prune=True,
        prune_alpha=0.01,
        random_state=0,
    )

    tree.fit(X, y)
    pred = tree.predict(X[:25])

    assert pred.shape == (25,)
    assert tree.root_ is not None
    assert "is_leaf" in tree.root_


def test_api_classifier_pruning_uses_current_node_schema_and_predicts():
    X, y = _pruning_dataset()
    clf = GSNHClassifier(
        model_type="single",
        use_pruning=True,
        pruning_alpha=0.01,
        use_calibration=False,
        max_depth=3,
        min_samples_leaf=5,
        language=LanguageFamily.HORN,
        mode="heuristic",
        random_state=0,
        verbose=False,
    )

    clf.fit(X, y)
    pred = clf.predict(X[:25])

    assert pred.shape == (25,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_bestpn_root_split_is_labelled_with_actual_searched_family():
    X = np.tile(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), (20, 1))
    y = (X[:, 0] >= 0.5).astype(np.int32)
    tree = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=1, min_samples_split=2, min_samples_leaf=1),
        language=LanguageFamily.BEST_PER_NODE,
        mode="journal",
        use_supervised_binning=False,
        search_3d=False,
        allow_affine_in_bestpn=False,
    )

    tree.fit(X, y)

    assert tree.root_["predicate"] is not None
    assert tree.root_["language"] != LanguageFamily.BEST_PER_NODE.value
    assert tree.root_["predicate"].language_family != LanguageFamily.BEST_PER_NODE
    assert tree.root_["predicate"].language_family in {
        LanguageFamily.HORN,
        LanguageFamily.ANTI_HORN,
        LanguageFamily.CONJ_UI,
    }
    assert LanguageFamily.SQUARE_2CNF.value not in tree.language_counts_
    assert LanguageFamily.AFFINE.value not in tree.language_counts_


def test_bestpn_can_include_affine_only_when_explicitly_enabled():
    X = np.tile(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), (20, 1))
    y = np.tile(np.array([0, 1, 1, 0], dtype=np.int32), 20)

    without_affine = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=1, min_samples_split=2, min_samples_leaf=1),
        language=LanguageFamily.BEST_PER_NODE,
        mode="journal",
        use_supervised_binning=False,
        search_3d=False,
        allow_affine_in_bestpn=False,
    ).fit(X, y)
    with_affine = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=1, min_samples_split=2, min_samples_leaf=1),
        language=LanguageFamily.BEST_PER_NODE,
        mode="journal",
        use_supervised_binning=False,
        search_3d=False,
        allow_affine_in_bestpn=True,
    ).fit(X, y)

    assert LanguageFamily.AFFINE.value not in without_affine.language_counts_
    assert with_affine.root_["predicate"].language_family == LanguageFamily.AFFINE
    assert LanguageFamily.SQUARE_2CNF.value not in with_affine.language_counts_


def test_bestpn_is_local_empirical_pool_not_theorem_oracle():
    X = np.tile(np.array([[0.0, 0.0], [1.0, 1.0]]), (30, 1))
    y = np.tile(np.array([0, 1], dtype=np.int32), 30)
    tree = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=1, min_samples_split=2, min_samples_leaf=1),
        language=LanguageFamily.BEST_PER_NODE,
        mode="journal",
        use_supervised_binning=False,
        search_3d=False,
    ).fit(X, y)

    assert tree.language == LanguageFamily.BEST_PER_NODE
    assert set(tree.language_counts_).issubset(
        {
            LanguageFamily.HORN.value,
            LanguageFamily.ANTI_HORN.value,
            LanguageFamily.CONJ_UI.value,
        }
    )
    assert tree.root_["language"] in tree.language_counts_


def test_affine_3d_i_loop_audit_matches_range_semantics_when_not_divisible():
    for ni, step in [(5, 2), (7, 3), (8, 3)]:
        historical_i_loop = [step + i * step for i in range(ni // step)]
        explicit_range_loop = list(range(step, ni + 1, step))
        assert historical_i_loop == explicit_range_loop
