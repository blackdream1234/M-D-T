import numpy as np
import pytest

from gsnh_mdt.engine import GSNHEngineClassifier


def simple_conjui_and_data():
    return (
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float),
        np.asarray([0, 0, 0, 1], dtype=int),
    )


def simple_horn_or_data():
    return (
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float),
        np.asarray([1, 0, 1, 1], dtype=int),
    )


def simple_antihorn_or_data():
    return (
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float),
        np.asarray([1, 1, 0, 1], dtype=int),
    )


def simple_affine_xor_data():
    return (
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float),
        np.asarray([0, 1, 1, 0], dtype=int),
    )


def simple_square2cnf_or_data():
    # y = (x0 >= 0.5) OR (x1 >= 0.5); parity is prediction/score only.
    return (
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float),
        np.asarray([0, 1, 1, 1], dtype=int),
    )


def python_depth_one_classifier(family):
    return GSNHEngineClassifier(
        engine="python",
        family=family,
        max_depth=1,
        max_arity=2,
        min_samples_leaf=1,
        min_samples_split=2,
        use_supervised_binning=False,
        search_2d=True,
        search_3d=False,
        mode="journal",
        verbose=False,
    )


def rust_depth_one_classifier(family):
    return GSNHEngineClassifier(
        engine="rust",
        family=family,
        max_depth=1,
        max_arity=2,
        min_samples_leaf=1,
        min_samples_split=2,
    )


@pytest.mark.parametrize("family", ["ConjUI"])
def test_depth_one_conjui_and_predictions_scores_and_summary_match_real_rust(family):
    pytest.importorskip("_rust_gsnh")
    X, y = simple_conjui_and_data()

    py_clf = python_depth_one_classifier(family).fit(X, y)
    rust_clf = rust_depth_one_classifier(family).fit(X, y)

    py_predictions = py_clf.predict(X).astype(int).tolist()
    rust_predictions = rust_clf.predict(X)

    assert py_predictions == [0, 0, 0, 1]
    assert rust_predictions == py_predictions
    assert rust_clf.score(X, y) == pytest.approx(py_clf.score(X, y))

    rust_summary = rust_clf.summary()
    assert rust_summary["n_nodes"] == rust_summary["n_leaves"] + rust_summary["n_internal_nodes"]
    assert rust_summary["max_depth"] <= 1


@pytest.mark.parametrize("family", ["Horn"])
def test_depth_one_horn_or_predictions_scores_and_summary_match_real_rust(family):
    pytest.importorskip("_rust_gsnh")
    X, y = simple_horn_or_data()

    py_clf = python_depth_one_classifier(family).fit(X, y)
    rust_clf = rust_depth_one_classifier(family).fit(X, y)

    py_predictions = py_clf.predict(X).astype(int).tolist()
    rust_predictions = rust_clf.predict(X)

    assert py_predictions == [1, 0, 1, 1]
    assert rust_predictions == py_predictions
    assert rust_clf.score(X, y) == pytest.approx(py_clf.score(X, y))

    rust_summary = rust_clf.summary()
    assert rust_summary["n_nodes"] == rust_summary["n_leaves"] + rust_summary["n_internal_nodes"]
    assert rust_summary["max_depth"] <= 1


@pytest.mark.parametrize("family", ["AntiHorn"])
def test_depth_one_antihorn_or_predictions_scores_and_summary_match_real_rust(family):
    pytest.importorskip("_rust_gsnh")
    X, y = simple_antihorn_or_data()

    py_clf = python_depth_one_classifier(family).fit(X, y)
    rust_clf = rust_depth_one_classifier(family).fit(X, y)

    py_predictions = py_clf.predict(X).astype(int).tolist()
    rust_predictions = rust_clf.predict(X)

    assert py_predictions == [1, 1, 0, 1]
    assert rust_predictions == py_predictions
    assert rust_clf.score(X, y) == pytest.approx(py_clf.score(X, y))

    rust_summary = rust_clf.summary()
    assert rust_summary["n_nodes"] == rust_summary["n_leaves"] + rust_summary["n_internal_nodes"]
    assert rust_summary["max_depth"] <= 1


@pytest.mark.parametrize("family", ["Affine"])
def test_depth_one_affine_xor_predictions_scores_and_summary_match_real_rust(family):
    pytest.importorskip("_rust_gsnh")
    X, y = simple_affine_xor_data()

    py_clf = python_depth_one_classifier(family).fit(X, y)
    rust_clf = rust_depth_one_classifier(family).fit(X, y)

    py_predictions = py_clf.predict(X).astype(int).tolist()
    rust_predictions = rust_clf.predict(X)

    assert py_predictions == [0, 1, 1, 0]
    assert rust_predictions == py_predictions
    assert rust_clf.score(X, y) == pytest.approx(py_clf.score(X, y))

    rust_summary = rust_clf.summary()
    assert rust_summary["n_nodes"] == rust_summary["n_leaves"] + rust_summary["n_internal_nodes"]
    assert rust_summary["max_depth"] <= 1


@pytest.mark.parametrize("family", ["Square2CNF"])
def test_depth_one_square2cnf_or_predictions_scores_and_summary_match_real_rust(family):
    pytest.importorskip("_rust_gsnh")
    # Do not assert split-object equality: equally scoring formulas may differ.
    X, y = simple_square2cnf_or_data()

    py_clf = python_depth_one_classifier(family).fit(X, y)
    rust_clf = rust_depth_one_classifier(family).fit(X, y)

    py_predictions = py_clf.predict(X).astype(int).tolist()
    rust_predictions = rust_clf.predict(X)

    assert py_predictions == [0, 1, 1, 1]
    assert rust_predictions == py_predictions
    assert rust_clf.score(X, y) == pytest.approx(py_clf.score(X, y))

    rust_summary = rust_clf.summary()
    assert rust_summary["n_nodes"] == rust_summary["n_leaves"] + rust_summary["n_internal_nodes"]
    assert rust_summary["max_depth"] <= 1
