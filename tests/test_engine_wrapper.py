import sys
import types

import numpy as np
import pytest

from gsnh_mdt.engine import GSNHEngineClassifier


def tiny_and_data():
    return np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
    ), np.asarray([0, 0, 0, 1], dtype=int)


def test_default_engine_is_python():
    clf = GSNHEngineClassifier()
    assert clf.engine == "python"


def test_python_engine_fits_predicts_scores_and_summarizes():
    X, y = tiny_and_data()
    clf = GSNHEngineClassifier(
        engine="python",
        family="ConjUI",
        max_depth=0,
        min_samples_leaf=1,
        min_samples_split=1,
        use_supervised_binning=False,
        verbose=False,
    )
    assert clf.fit(X, y) is clf
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions.tolist()).issubset({0, 1})
    assert 0.0 <= clf.score(X, y) <= 1.0
    summary = clf.summary()
    assert summary["n_nodes"] == summary["n_leaves"] + summary["n_internal_nodes"]


def test_rust_engine_missing_extension_raises_import_error():
    X, y = tiny_and_data()
    clf = GSNHEngineClassifier(engine="rust", family="ConjUI")
    with pytest.raises(ImportError, match="_rust_gsnh"):
        clf.fit(X, y)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_invalid_engine_name_raises(engine):
    bad = "not-rust-or-python" if engine == "python" else ""
    with pytest.raises(ValueError):
        GSNHEngineClassifier(engine=bad)


@pytest.mark.parametrize("family", ["Any", "BestPerNode", "SquareCNF"])
def test_unsupported_rust_families_raise_before_import(family):
    with pytest.raises(ValueError):
        GSNHEngineClassifier(engine="rust", family=family)


def test_predict_score_and_summary_before_fit_raise():
    X, y = tiny_and_data()
    clf = GSNHEngineClassifier(engine="python")
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.score(X, y)
    with pytest.raises(RuntimeError):
        clf.summary()


def test_wrapper_does_not_change_existing_python_estimator_behavior():
    X, y = tiny_and_data()
    kwargs = dict(
        family="ConjUI",
        max_depth=0,
        min_samples_leaf=1,
        min_samples_split=1,
        use_supervised_binning=False,
        verbose=False,
    )
    wrapped = GSNHEngineClassifier(engine="python", **kwargs).fit(X, y)

    from gsnh_mdt.tree.builder import ExpertGSNHTree
    from gsnh_mdt.tree.stopping import StoppingCriteria
    from gsnh_mdt.types import LanguageFamily

    reference = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=0, min_samples_leaf=1, min_samples_split=1),
        language=LanguageFamily.CONJ_UI,
        use_supervised_binning=False,
        verbose=False,
    ).fit(X, y)
    assert wrapped.predict(X).tolist() == reference.predict(X).astype(int).tolist()


def test_rust_engine_raises_on_extra_options():
    with pytest.raises(ValueError, match="extra options"):
        GSNHEngineClassifier(engine="rust", family="ConjUI", verbose=False)


def test_rust_engine_with_installed_stub_fits_predicts_scores_and_summarizes(monkeypatch):
    class StubRustClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True
            self.y = list(y)
            return self

        def predict(self, X):
            assert self.fitted
            return [0 for _ in X]

        def score(self, X, y):
            preds = self.predict(X)
            return sum(int(a == b) for a, b in zip(preds, y)) / len(y)

        def summary(self):
            return {"n_nodes": 1, "n_leaves": 1, "n_internal_nodes": 0, "max_depth": 0}

    module = types.SimpleNamespace(RustGsnHClassifier=StubRustClassifier)
    monkeypatch.setitem(sys.modules, "_rust_gsnh", module)

    X, y = tiny_and_data()
    clf = GSNHEngineClassifier(engine="rust", family="ConjUI", max_depth=0)
    assert clf.fit(X, y) is clf
    assert clf.predict(X) == [0, 0, 0, 0]
    assert clf.score(X, y) == 0.75
    assert clf.summary() == {"n_nodes": 1, "n_leaves": 1, "n_internal_nodes": 0, "max_depth": 0}
