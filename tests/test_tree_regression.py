"""
Tree regression tests — verify ExpertGSNHTree outputs match frozen golden baselines.

These tests are the safety net for the ExpertGSNHTree extraction.
They run against whatever ExpertGSNHTree is importable (monolith or package).
"""

import sys
import os
import json
import numpy as np
import pytest

from data_path import DATA_DIR, HAS_DL8_DATA, RS_ROOT
from golden_utils import find_dataset_file, load_golden

# Make repo root importable (for monolith and data access)
sys.path.insert(0, RS_ROOT)

pytestmark = pytest.mark.skipif(
    not HAS_DL8_DATA,
    reason=f"No .dl8 benchmark data found in {DATA_DIR}",
)

# Load golden baselines
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_baselines.json')
GOLDEN = load_golden(GOLDEN_PATH)
# These goldens correspond to deterministic post-theorem-boundary behavior.



def parse_dl8(filepath):
    """Parse .dl8 file."""
    data = np.loadtxt(filepath, dtype=np.int64)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float64)
    return X, y


def get_tree_class():
    """Import ExpertGSNHTree from whatever source is available."""
    try:
        # Try the package first
        from gsnh_mdt.tree.builder import ExpertGSNHTree
        from gsnh_mdt.tree.stopping import StoppingCriteria
        from gsnh_mdt.types import LanguageFamily
        return ExpertGSNHTree, StoppingCriteria, LanguageFamily
    except ImportError:
        pass
    # Fall back to monolith
    from gsnh_mdt_v3 import ExpertGSNHTree, StoppingCriteria, LanguageFamily
    return ExpertGSNHTree, StoppingCriteria, LanguageFamily


def train_golden(name):
    """Train a tree with the same settings as the golden capture."""
    g = GOLDEN[name]
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


# =============================================================================
# A. Smoke-fit tests — accuracy, tree size, root split
# =============================================================================

@pytest.mark.parametrize("name", ["lymph", "hepatitis", "vote"])
class TestTreeSmokeGolden:

    def test_train_accuracy(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        train_acc = float(tree.score(X_train, y_train))
        assert abs(train_acc - g['train_acc']) < 1e-6, \
            f"Train acc mismatch: {train_acc} vs golden {g['train_acc']}"

    def test_test_accuracy(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        test_acc = float(tree.score(X_test, y_test))
        assert abs(test_acc - g['test_acc']) < 1e-6, \
            f"Test acc mismatch: {test_acc} vs golden {g['test_acc']}"

    def test_tree_size(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        assert tree.n_nodes_ == g['n_nodes'], \
            f"n_nodes mismatch: {tree.n_nodes_} vs golden {g['n_nodes']}"
        assert tree.n_leaves_ == g['n_leaves'], \
            f"n_leaves mismatch: {tree.n_leaves_} vs golden {g['n_leaves']}"

    def test_max_depth(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        assert tree.max_depth_reached_ == g['max_depth_reached'], \
            f"max_depth mismatch: {tree.max_depth_reached_} vs golden {g['max_depth_reached']}"

    def test_root_split(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        root_split = str(tree.root_['predicate']) if tree.root_ and 'predicate' in tree.root_ else None
        assert root_split == g['root_split'], \
            f"Root split mismatch: '{root_split}' vs golden '{g['root_split']}'"

    def test_predict_proba(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        probas = tree.predict_proba(X_test[:3])
        for i in range(3):
            for j in range(2):
                assert abs(probas[i][j] - g['probas_first3'][i][j]) < 1e-10, \
                    f"Proba mismatch at [{i}][{j}]: {probas[i][j]} vs golden {g['probas_first3'][i][j]}"

    def test_axp_length(self, name):
        tree, X_train, y_train, X_test, y_test, g = train_golden(name)
        if g['axp_len'] is not None:
            axp = tree.extract_axp(X_test[0])
            assert len(axp) == g['axp_len'], \
                f"AXp length mismatch: {len(axp)} vs golden {g['axp_len']}"


# =============================================================================
# B. Reproducibility tests — same seed → same output
# =============================================================================

@pytest.mark.parametrize("name", ["lymph", "hepatitis", "vote"])
class TestTreeReproducibility:

    def test_deterministic(self, name):
        """Two fits with same seed must produce identical results."""
        results = []
        for _ in range(2):
            tree, X_train, y_train, X_test, y_test, g = train_golden(name)
            results.append({
                'acc': float(tree.score(X_test, y_test)),
                'n_nodes': tree.n_nodes_,
                'root': str(tree.root_['predicate']) if tree.root_ and 'predicate' in tree.root_ else None,
            })
        assert results[0] == results[1], \
            f"Non-deterministic: run 1={results[0]}, run 2={results[1]}"
