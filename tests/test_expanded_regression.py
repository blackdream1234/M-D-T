"""
Expanded tree regression tests — 9 datasets with comprehensive output coverage.

Tests accuracy, tree size, arity distribution, 5 predict_proba samples,
3 AXp lengths, root split, and reproducibility for each dataset.
"""

import sys
import os
import json
import numpy as np
import pytest

from data_path import DATA_DIR, HAS_DL8_DATA, RS_ROOT

sys.path.insert(0, RS_ROOT)

pytestmark = pytest.mark.skipif(
    not HAS_DL8_DATA,
    reason=f"No .dl8 benchmark data found in {DATA_DIR}",
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_expanded.json')
with open(GOLDEN_PATH) as f:
    GOLDEN = json.load(f)


ALL_DATASETS = list(GOLDEN.keys())


def parse_dl8(filepath):
    data = np.loadtxt(filepath, dtype=np.int64)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float64)
    return X, y


def get_tree_class():
    try:
        from gsnh_mdt.tree.builder import ExpertGSNHTree
        from gsnh_mdt.tree.stopping import StoppingCriteria
        from gsnh_mdt.types import LanguageFamily
        return ExpertGSNHTree, StoppingCriteria, LanguageFamily
    except ImportError:
        from gsnh_mdt_v3 import ExpertGSNHTree, StoppingCriteria, LanguageFamily
        return ExpertGSNHTree, StoppingCriteria, LanguageFamily


def train_golden(name):
    g = GOLDEN[name]
    data_path = os.path.join(DATA_DIR, f'{name}.dl8')
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
    stopping = StoppingCriteria(max_depth=g['max_depth'], min_samples_leaf=5)
    tree = ExpertGSNHTree(
        stopping_criteria=stopping,
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )
    tree.fit(X_train, y_train)
    return tree, X_train, y_train, X_test, y_test, g


@pytest.mark.regression
@pytest.mark.parametrize("name", ALL_DATASETS)
class TestExpandedGolden:

    def test_train_accuracy(self, name):
        tree, X_train, y_train, _, _, g = train_golden(name)
        assert abs(float(tree.score(X_train, y_train)) - g['train_acc']) < 1e-6

    def test_test_accuracy(self, name):
        tree, _, _, X_test, y_test, g = train_golden(name)
        assert abs(float(tree.score(X_test, y_test)) - g['test_acc']) < 1e-6

    def test_tree_size(self, name):
        tree, _, _, _, _, g = train_golden(name)
        assert tree.n_nodes_ == g['n_nodes']
        assert tree.n_leaves_ == g['n_leaves']
        assert tree.max_depth_reached_ == g['max_depth_reached']

    def test_arity_distribution(self, name):
        tree, _, _, _, _, g = train_golden(name)
        for k, v in g['arity_counts'].items():
            assert tree.arity_counts_.get(int(k), 0) == v, \
                f"Arity {k}: got {tree.arity_counts_.get(int(k), 0)}, expected {v}"

    def test_root_split(self, name):
        tree, _, _, _, _, g = train_golden(name)
        root_split = str(tree.root_['predicate']) if tree.root_ and 'predicate' in tree.root_ else None
        assert root_split == g['root_split']

    def test_predict_proba(self, name):
        tree, _, _, X_test, _, g = train_golden(name)
        n = min(5, len(X_test))
        probas = tree.predict_proba(X_test[:n])
        for i in range(n):
            for j in range(2):
                assert abs(probas[i][j] - g['probas_first5'][i][j]) < 1e-10, \
                    f"Proba[{i}][{j}]: {probas[i][j]} vs {g['probas_first5'][i][j]}"

    def test_axp_lengths(self, name):
        tree, _, _, X_test, _, g = train_golden(name)
        for i, expected_len in enumerate(g['axp_lengths']):
            if expected_len is not None and i < len(X_test):
                axp = tree.extract_axp(X_test[i])
                assert len(axp) == expected_len, \
                    f"AXp[{i}]: len={len(axp)}, expected {expected_len}"

    def test_reproducibility(self, name):
        results = []
        for _ in range(2):
            tree, _, _, X_test, y_test, g = train_golden(name)
            results.append({
                'acc': float(tree.score(X_test, y_test)),
                'nodes': tree.n_nodes_,
            })
        assert results[0] == results[1]
