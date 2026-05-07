"""
Test for the public GSNHClassifier API.

Verifies that the target usage pattern from the plan works:
  from gsnh_mdt.api import GSNHClassifier
  clf = GSNHClassifier(...)
  clf.fit(X_train, y_train)
  pred = clf.predict(X_test)
"""

import sys
import os
import numpy as np
import pytest

RS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR = os.path.join(RS_ROOT, 'data')


def parse_dl8(filepath):
    data = np.loadtxt(filepath, dtype=np.int64)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float64)
    return X, y


class TestGSNHClassifierImport:
    """Verify the public API imports resolve correctly."""

    def test_import_classifier(self):
        from gsnh_mdt.api import GSNHClassifier
        assert GSNHClassifier is not None

    def test_import_config(self):
        from gsnh_mdt.config import ModelConfig, SearchConfig, OptimizationConfig
        config = ModelConfig()
        assert config.n_bins == 64
        assert config.search.search_2d is True
        assert config.optimization.use_attention is True

    def test_import_tree_direct(self):
        from gsnh_mdt.tree.builder import ExpertGSNHTree
        assert ExpertGSNHTree is not None

    def test_import_ensembles(self):
        from gsnh_mdt.ensembles import GSNHRandomForest, GSNHGradientBoosting
        assert GSNHRandomForest is not None
        assert GSNHGradientBoosting is not None


class TestGSNHClassifierSmoke:
    """Smoke test the classifier pipeline on a real dataset."""

    def test_single_tree_fit_predict(self):
        from gsnh_mdt.api import GSNHClassifier
        from gsnh_mdt.types import LanguageFamily

        data_path = os.path.join(DATA_DIR, 'vote.dl8')
        if not os.path.exists(data_path):
            pytest.skip("vote.dl8 not found")

        X, y = parse_dl8(data_path)
        np.random.seed(42)
        idx = np.random.permutation(len(y))
        X_train, y_train = X[idx[:300]], y[idx[:300]]
        X_test, y_test = X[idx[300:]], y[idx[300:]]

        clf = GSNHClassifier(
            model_type='single',
            max_depth=5,
            use_calibration=False,
            use_pruning=False,
            verbose=False,
            language=LanguageFamily.BEST_PER_NODE,
        )
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        acc = float((pred == y_test).mean())
        assert acc > 0.8, f"Expected >80% acc, got {acc:.3f}"

        probas = clf.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
