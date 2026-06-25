"""
Method-level regression tests for Phase C builder split.

These tests verify that predict, predict_proba, extract_axp, and
weak_axp_check produce identical outputs after the prediction and
explainer methods are moved to separate modules.
"""

import sys
import os
import json
import numpy as np
import pytest

from data_path import DATA_DIR, HAS_DL8_DATA, RS_ROOT
from golden_utils import find_dataset_file, load_golden

sys.path.insert(0, RS_ROOT)

pytestmark = pytest.mark.skipif(
    not HAS_DL8_DATA,
    reason=f"No .dl8 benchmark data found in {DATA_DIR}",
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_methods.json')
GOLDEN = load_golden(GOLDEN_PATH)
# These goldens correspond to deterministic post-theorem-boundary behavior.

ALL_DATASETS = list(GOLDEN.keys())


def parse_dl8(filepath):
    data = np.loadtxt(filepath, dtype=np.int64)
    return data[:, 1:].astype(np.float64), data[:, 0].astype(np.int32)


def get_tree_class():
    try:
        from gsnh_mdt.tree.builder import ExpertGSNHTree
        from gsnh_mdt.tree.stopping import StoppingCriteria
        from gsnh_mdt.types import LanguageFamily
        return ExpertGSNHTree, StoppingCriteria, LanguageFamily
    except ImportError:
        from gsnh_mdt_v3 import ExpertGSNHTree, StoppingCriteria, LanguageFamily
        return ExpertGSNHTree, StoppingCriteria, LanguageFamily


def train_tree(name, g):
    data_path = find_dataset_file(name)
    X, y = parse_dl8(data_path)

    np.random.seed(g['seed'])
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    classes = np.unique(y)
    if len(classes) > 2:
        majority = classes[np.argmax([np.sum(y == c) for c in classes])]
        y_train = (y_train == majority).astype(np.int32)
        y_test = (y_test == majority).astype(np.int32)

    ExpertGSNHTree, StoppingCriteria, LanguageFamily = get_tree_class()
    tree = ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=g['max_depth'], min_samples_leaf=5),
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )
    tree.fit(X_train, y_train)
    return tree, X_train, y_train, X_test, y_test


# ── predict_proba method-level tests ────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("name", ALL_DATASETS)
class TestPredictProbaMethodLevel:

    def test_predict_proba_10_samples(self, name):
        g = GOLDEN[name]
        tree, _, _, X_test, _ = train_tree(name, g)
        n = min(10, len(X_test))
        probas = tree.predict_proba(X_test[:n])
        for i in range(n):
            for j in range(2):
                assert abs(probas[i][j] - g['predict_proba_10'][i][j]) < 1e-10, \
                    f"predict_proba[{i}][{j}]: {probas[i][j]} vs {g['predict_proba_10'][i][j]}"

    def test_predict_batch(self, name):
        g = GOLDEN[name]
        tree, _, _, X_test, _ = train_tree(name, g)
        n = min(10, len(X_test))
        pred = tree.predict(X_test[:n]).tolist()
        assert pred == g['predict_10'], \
            f"predict mismatch: {pred} vs {g['predict_10']}"


# ── extract_axp method-level tests ──────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("name", ALL_DATASETS)
class TestExtractAxpMethodLevel:

    def test_axp_features_match(self, name):
        g = GOLDEN[name]
        tree, _, _, X_test, _ = train_tree(name, g)
        for i in range(min(5, len(X_test))):
            axp = tree.extract_axp(X_test[i])
            expected = g['axp_5'][i]
            assert sorted(list(axp)) == expected, \
                f"AXp[{i}]: {sorted(list(axp))} vs {expected}"


# ── weak_axp_check method-level tests ───────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("name", ALL_DATASETS)
class TestWeakAxpCheckMethodLevel:

    def test_full_axp_returns_true(self, name):
        """weak_axp_check with the full AXp set must return True."""
        g = GOLDEN[name]
        tree, _, _, X_test, _ = train_tree(name, g)
        for wc in g['weak_check_3']:
            i = wc['instance_idx']
            y_pred = wc['y_pred']
            axp_set = set(wc['axp_features'])
            result = tree.weak_axp_check(X_test[i], y_pred, axp_set)
            assert result == wc['check_full_axp'], \
                f"weak_axp_check(full)[{i}]: {result} vs {wc['check_full_axp']}"

    def test_empty_set_returns_false(self, name):
        """weak_axp_check with empty set should return False for non-trivial trees."""
        g = GOLDEN[name]
        tree, _, _, X_test, _ = train_tree(name, g)
        for wc in g['weak_check_3']:
            i = wc['instance_idx']
            y_pred = wc['y_pred']
            result = tree.weak_axp_check(X_test[i], y_pred, set())
            assert result == wc['check_empty'], \
                f"weak_axp_check(empty)[{i}]: {result} vs {wc['check_empty']}"


# ── score method-level tests ────────────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize("name", ALL_DATASETS)
class TestScoreMethodLevel:

    def test_train_score(self, name):
        g = GOLDEN[name]
        tree, X_train, y_train, _, _ = train_tree(name, g)
        assert abs(float(tree.score(X_train, y_train)) - g['train_score']) < 1e-6

    def test_test_score(self, name):
        g = GOLDEN[name]
        tree, _, _, X_test, y_test = train_tree(name, g)
        assert abs(float(tree.score(X_test, y_test)) - g['test_score']) < 1e-6
