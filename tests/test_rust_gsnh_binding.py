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


def family_dataset(family):
    if family == "ConjUI":
        return tiny_and_data()
    if family == "Horn":
        return (
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
            [1, 0, 1, 1, 0, 1],
        )
    if family == "AntiHorn":
        return (
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
            [1, 1, 1, 0, 1, 1],
        )
    if family == "Affine":
        return (
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            [0, 1, 1, 0],
        )
    if family == "Square2CNF":
        X = []
        y = []
        for a in (0.0, 1.0):
            for b in (0.0, 1.0):
                for c in (0.0, 1.0):
                    for d in (0.0, 1.0):
                        X.append([a, b, c, d])
                        y.append(int((a == 1.0 or b == 1.0) and (c == 1.0 or d == 1.0)))
        return X, y
    raise AssertionError(f"missing fixture for {family}")


@pytest.mark.parametrize("family", ["ConjUI", "Horn", "AntiHorn", "Affine", "Square2CNF"])
def test_supported_family_binding_invariants(rust_binding, family):
    X, y = family_dataset(family)
    clf = rust_binding.RustGsnHClassifier(
        family=family,
        max_arity=2,
        max_depth=1,
        min_samples_leaf=1,
        min_samples_split=2,
    )

    assert clf.fit(X, y) is not None
    predictions = clf.predict(X)
    score = clf.score(X, y)
    summary = clf.summary()

    assert isinstance(predictions, list)
    assert all(isinstance(pred, int) for pred in predictions)
    assert set(predictions).issubset({0, 1})
    assert len(predictions) == len(y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(summary, dict)
    assert {"n_nodes", "n_leaves", "n_internal_nodes", "max_depth"}.issubset(summary)
    assert summary["n_nodes"] == summary["n_leaves"] + summary["n_internal_nodes"]


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


@pytest.mark.parametrize("family", ["ConjUI", "Horn", "AntiHorn", "Affine", "Square2CNF"])
def test_depth_zero_predictions_match_python_reference_majority_leaf(rust_binding, family):
    np = pytest.importorskip("numpy")
    from gsnh_mdt.tree.builder import ExpertGSNHTree
    from gsnh_mdt.tree.stopping import StoppingCriteria
    from gsnh_mdt.types import LanguageFamily

    X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [1, 1, 1, 0, 0]
    language = {
        "ConjUI": LanguageFamily.CONJ_UI,
        "Horn": LanguageFamily.HORN,
        "AntiHorn": LanguageFamily.ANTI_HORN,
        "Affine": LanguageFamily.AFFINE,
        "Square2CNF": LanguageFamily.SQUARE_2CNF,
    }[family]

    stopping = StoppingCriteria(max_depth=0, min_samples_split=1, min_samples_leaf=1)
    py_tree = ExpertGSNHTree(
        stopping_criteria=stopping,
        n_bins=8,
        use_supervised_binning=False,
        search_3d=False,
        mode="journal",
        language=language,
        verbose=False,
    ).fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    py_predictions = py_tree.predict(np.asarray(X, dtype=float)).astype(int).tolist()

    rust_clf = rust_binding.RustGsnHClassifier(
        family=family,
        max_arity=2,
        max_depth=0,
        min_samples_leaf=1,
        min_samples_split=1,
    ).fit(X, y)
    rust_predictions = rust_clf.predict(X)

    assert rust_predictions == py_predictions == [1, 1, 1, 1, 1]
    assert rust_clf.score(X, y) == pytest.approx(sum(int(a == b) for a, b in zip(y, rust_predictions)) / len(y))


@pytest.mark.parametrize("family", ["Any", "BestPerNode", "SquareCNF", "unknown"])
def test_invalid_family_raises(rust_binding, family):
    with pytest.raises(ValueError):
        rust_binding.RustGsnHClassifier(family=family)


@pytest.mark.parametrize("family", ["Any", "BestPerNode", "SquareCNF"])
def test_named_unsupported_families_raise(rust_binding, family):
    with pytest.raises(ValueError):
        rust_binding.RustGsnHClassifier(family=family)


def test_predict_and_score_before_fit_raise(rust_binding):
    X, y = tiny_and_data()
    clf = rust_binding.RustGsnHClassifier(family="ConjUI")
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.score(X, y)


def test_invalid_labels_raise(rust_binding):
    X, _ = tiny_and_data()
    clf = rust_binding.RustGsnHClassifier(family="ConjUI")
    with pytest.raises(ValueError):
        clf.fit(X, [0, 0, 0, 2])


def test_ragged_x_raises(rust_binding):
    clf = rust_binding.RustGsnHClassifier(family="ConjUI")
    with pytest.raises(ValueError):
        clf.fit([[0.0, 1.0], [1.0]], [0, 1])


def test_x_y_length_mismatch_raises(rust_binding):
    X, _ = tiny_and_data()
    clf = rust_binding.RustGsnHClassifier(family="ConjUI")
    with pytest.raises(ValueError):
        clf.fit(X, [0, 1])
