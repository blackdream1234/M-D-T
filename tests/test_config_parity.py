"""
Config parity tests for Phase D.

Verifies that ExpertGSNHTree.from_config(config) produces identical
outputs to ExpertGSNHTree(old_args...) on fixed datasets.

Tests: accuracy, tree size, root split, predict_proba, extract_axp.
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

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_methods.json')
with open(GOLDEN_PATH) as f:
    GOLDEN = json.load(f)

ALL_DATASETS = list(GOLDEN.keys())


def parse_dl8(filepath):
    data = np.loadtxt(filepath, dtype=np.int64)
    return data[:, 1:].astype(np.float64), data[:, 0].astype(np.int32)


def prepare_data(name, g):
    X, y = parse_dl8(os.path.join(DATA_DIR, f'{name}.dl8'))
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
    return X_train, y_train, X_test, y_test


def build_legacy(g):
    """Build tree using legacy 22-param constructor."""
    from gsnh_mdt.tree.builder import ExpertGSNHTree
    from gsnh_mdt.tree.stopping import StoppingCriteria
    from gsnh_mdt.types import LanguageFamily

    return ExpertGSNHTree(
        stopping_criteria=StoppingCriteria(max_depth=g['max_depth'], min_samples_leaf=5),
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )


def build_from_config(g):
    """Build tree using from_config() classmethod."""
    from gsnh_mdt.tree.builder import ExpertGSNHTree
    from gsnh_mdt.tree.stopping import StoppingCriteria
    from gsnh_mdt.config import ModelConfig
    from gsnh_mdt.types import LanguageFamily

    config = ModelConfig(
        stopping=StoppingCriteria(max_depth=g['max_depth'], min_samples_leaf=5),
        n_bins=64,
        language=LanguageFamily.BEST_PER_NODE,
        mode='heuristic',
        verbose=False,
    )
    return ExpertGSNHTree.from_config(config)


@pytest.mark.regression
@pytest.mark.parametrize("name", ALL_DATASETS)
class TestConfigParity:
    """Verify from_config() produces identical results to legacy constructor."""

    def test_accuracy_parity(self, name):
        g = GOLDEN[name]
        X_train, y_train, X_test, y_test = prepare_data(name, g)

        # Legacy
        np.random.seed(g['seed'])
        tree_legacy = build_legacy(g)
        tree_legacy.fit(X_train, y_train)
        acc_legacy = float(tree_legacy.score(X_test, y_test))

        # Config
        np.random.seed(g['seed'])
        tree_config = build_from_config(g)
        tree_config.fit(X_train, y_train)
        acc_config = float(tree_config.score(X_test, y_test))

        assert abs(acc_legacy - acc_config) < 1e-10, \
            f"Accuracy parity: legacy={acc_legacy} vs config={acc_config}"

    def test_tree_size_parity(self, name):
        g = GOLDEN[name]
        X_train, y_train, _, _ = prepare_data(name, g)

        np.random.seed(g['seed'])
        tree_legacy = build_legacy(g)
        tree_legacy.fit(X_train, y_train)

        np.random.seed(g['seed'])
        tree_config = build_from_config(g)
        tree_config.fit(X_train, y_train)

        assert tree_legacy.n_nodes_ == tree_config.n_nodes_, \
            f"n_nodes: legacy={tree_legacy.n_nodes_} vs config={tree_config.n_nodes_}"
        assert tree_legacy.n_leaves_ == tree_config.n_leaves_
        assert tree_legacy.max_depth_reached_ == tree_config.max_depth_reached_

    def test_root_split_parity(self, name):
        g = GOLDEN[name]
        X_train, y_train, _, _ = prepare_data(name, g)

        np.random.seed(g['seed'])
        tree_legacy = build_legacy(g)
        tree_legacy.fit(X_train, y_train)

        np.random.seed(g['seed'])
        tree_config = build_from_config(g)
        tree_config.fit(X_train, y_train)

        root_l = str(tree_legacy.root_['predicate']) if tree_legacy.root_ else None
        root_c = str(tree_config.root_['predicate']) if tree_config.root_ else None
        assert root_l == root_c, f"Root split: legacy={root_l} vs config={root_c}"

    def test_predict_proba_parity(self, name):
        g = GOLDEN[name]
        X_train, y_train, X_test, _ = prepare_data(name, g)

        np.random.seed(g['seed'])
        tree_legacy = build_legacy(g)
        tree_legacy.fit(X_train, y_train)

        np.random.seed(g['seed'])
        tree_config = build_from_config(g)
        tree_config.fit(X_train, y_train)

        n = min(10, len(X_test))
        pp_l = tree_legacy.predict_proba(X_test[:n])
        pp_c = tree_config.predict_proba(X_test[:n])

        for i in range(n):
            for j in range(2):
                assert abs(pp_l[i][j] - pp_c[i][j]) < 1e-10, \
                    f"predict_proba[{i}][{j}]: legacy={pp_l[i][j]} vs config={pp_c[i][j]}"

    def test_extract_axp_parity(self, name):
        g = GOLDEN[name]
        X_train, y_train, X_test, _ = prepare_data(name, g)

        np.random.seed(g['seed'])
        tree_legacy = build_legacy(g)
        tree_legacy.fit(X_train, y_train)

        np.random.seed(g['seed'])
        tree_config = build_from_config(g)
        tree_config.fit(X_train, y_train)

        for i in range(min(3, len(X_test))):
            axp_l = sorted(list(tree_legacy.extract_axp(X_test[i])))
            axp_c = sorted(list(tree_config.extract_axp(X_test[i])))
            assert axp_l == axp_c, f"AXp[{i}]: legacy={axp_l} vs config={axp_c}"


# ── Mapping completeness test ──────────────────────────────────────

class TestConfigMapping:
    """Verify the config mapping covers all constructor parameters."""

    def test_to_constructor_kwargs_covers_all_params(self):
        """Ensure to_constructor_kwargs() produces all 22 constructor params."""
        from gsnh_mdt.config import ModelConfig
        import inspect
        from gsnh_mdt.tree.builder import ExpertGSNHTree

        config = ModelConfig()
        kwargs = config.to_constructor_kwargs()

        sig = inspect.signature(ExpertGSNHTree.__init__)
        init_params = set(sig.parameters.keys()) - {'self'}

        config_keys = set(kwargs.keys())

        assert config_keys == init_params, \
            f"Mismatch:\n  In config but not init: {config_keys - init_params}\n  In init but not config: {init_params - config_keys}"

    def test_default_config_matches_default_constructor(self):
        """ModelConfig() defaults must produce the same tree as ExpertGSNHTree()."""
        from gsnh_mdt.config import ModelConfig
        from gsnh_mdt.tree.builder import ExpertGSNHTree

        config = ModelConfig()
        kwargs = config.to_constructor_kwargs()

        tree_default = ExpertGSNHTree()
        tree_config = ExpertGSNHTree(**kwargs)

        # Compare all attribute values set in __init__
        for attr in ['n_bins', 'binning_strategy', 'top_k', 'use_gain_ratio',
                      'laplace', 'search_1d', 'search_2d', 'search_3d',
                      'use_supervised_binning', 'use_attention', 'use_look_ahead',
                      'look_ahead_gamma', 'look_ahead_top_p', 'verbose', 'mode',
                      'language', 'limit_2d', 'limit_3d',
                      'use_binary_comparisons', 'enable_compare_literals',
                      'prune', 'prune_alpha']:
            val_d = getattr(tree_default, attr)
            val_c = getattr(tree_config, attr)
            assert val_d == val_c, f"Attribute {attr}: default={val_d} vs config={val_c}"
