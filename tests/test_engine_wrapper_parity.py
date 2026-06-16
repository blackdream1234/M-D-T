import numpy as np
import pytest

from gsnh_mdt.engine import GSNHEngineClassifier

RUST_FAMILIES = ["ConjUI", "Horn", "AntiHorn", "Affine", "Square2CNF"]


def majority_leaf_data():
    return (
        np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float),
        np.asarray([1, 1, 1, 0, 0], dtype=int),
    )


def python_depth_zero_classifier(family):
    return GSNHEngineClassifier(
        engine="python",
        family=family,
        max_depth=0,
        max_arity=2,
        min_samples_leaf=1,
        min_samples_split=1,
        use_supervised_binning=False,
        verbose=False,
    )


def rust_depth_zero_classifier(family):
    return GSNHEngineClassifier(
        engine="rust",
        family=family,
        max_depth=0,
        max_arity=2,
        min_samples_leaf=1,
        min_samples_split=1,
    )


def test_python_engine_remains_default_for_parity_wrapper():
    clf = GSNHEngineClassifier()
    assert clf.engine == "python"


def test_rust_engine_remains_explicit_opt_in():
    clf = GSNHEngineClassifier(engine="rust", family="ConjUI")
    assert clf.engine == "rust"


@pytest.mark.parametrize("family", RUST_FAMILIES)
def test_depth_zero_majority_leaf_predictions_scores_and_summary_match_real_rust(family):
    pytest.importorskip("_rust_gsnh")
    X, y = majority_leaf_data()

    py_clf = python_depth_zero_classifier(family).fit(X, y)
    rust_clf = rust_depth_zero_classifier(family).fit(X, y)

    py_predictions = py_clf.predict(X).astype(int).tolist()
    rust_predictions = rust_clf.predict(X)

    assert rust_predictions == py_predictions == [1, 1, 1, 1, 1]
    assert rust_clf.score(X, y) == pytest.approx(py_clf.score(X, y))

    rust_summary = rust_clf.summary()
    assert rust_summary["n_nodes"] == rust_summary["n_leaves"] + rust_summary["n_internal_nodes"]
    assert rust_summary["max_depth"] == 0


def test_depth_zero_majority_leaf_parity_skips_cleanly_without_real_rust():
    rust = pytest.importorskip("_rust_gsnh")
    assert rust is not None


def test_wrapper_parity_test_scope_is_deliberately_depth_zero_without_stubbed_rust():
    X, y = majority_leaf_data()
    py_clf = python_depth_zero_classifier("ConjUI").fit(X, y)
    assert py_clf.predict(X).astype(int).tolist() == [1, 1, 1, 1, 1]
    assert py_clf.score(X, y) == pytest.approx(3 / 5)
