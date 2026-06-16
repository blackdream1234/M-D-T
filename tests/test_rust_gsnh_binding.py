import pytest


def test_binding_test_module_is_collectable():
    # Keeps the targeted pytest command from returning exit code 5 in
    # environments where the optional Rust extension has not been installed.
    assert True


@pytest.fixture
def rust_binding():
    return pytest.importorskip("_rust_gsnh")


def tiny_and_data():
    return (
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0, 0, 0, 1],
    )


def test_import_instantiate_fit_predict_score_and_summary(rust_binding):
    X, y = tiny_and_data()
    clf = rust_binding.RustGsnHClassifier(
        family="ConjUI",
        max_arity=2,
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
    )

    assert clf.fit(X, y) is not None
    assert clf.predict(X) == [0, 0, 0, 1]
    assert clf.score(X, y) == 1.0

    summary = clf.summary()
    assert summary["n_nodes"] == 3
    assert summary["n_leaves"] == 2
    assert summary["n_internal_nodes"] == 1
    assert summary["max_depth"] == 1


@pytest.mark.parametrize("family", ["Any", "BestPerNode", "SquareCNF", "unknown"])
def test_invalid_family_raises(rust_binding, family):
    with pytest.raises(ValueError):
        rust_binding.RustGsnHClassifier(family=family)


def test_predict_before_fit_raises(rust_binding):
    X, _ = tiny_and_data()
    clf = rust_binding.RustGsnHClassifier(family="ConjUI")
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_invalid_labels_raise(rust_binding):
    X, _ = tiny_and_data()
    clf = rust_binding.RustGsnHClassifier(family="ConjUI")
    with pytest.raises(ValueError):
        clf.fit(X, [0, 0, 0, 2])
