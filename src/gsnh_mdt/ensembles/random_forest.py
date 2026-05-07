"""
GSNHRandomForest: Random Forest of GSNH trees.

Extracted verbatim from gsnh_mdt_v3.py lines 3310-3438.
"""

import numpy as np
from typing import Optional, Union

from gsnh_mdt.types import LanguageFamily
from gsnh_mdt.tree.builder import ExpertGSNHTree


class GSNHRandomForest:
    """Random Forest of GSNH trees."""

    def __init__(self,
                 n_estimators: int = 50,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 tree_params: Optional[dict] = None,
                 random_state: int = 42,
                 mode: str = 'heuristic',
                 language: LanguageFamily = LanguageFamily.ANY):

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.tree_params = tree_params or {}
        self.random_state = random_state
        self.mode = mode
        self.language = language
        self.tree_params['mode'] = self.mode
        self.tree_params['language'] = self.language

        self.estimators_ = []
        self.oob_score_ = None
        self.feature_importances_ = None

    def _get_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        return n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GSNHRandomForest':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        n_samples, n_features = X.shape
        max_feat = self._get_max_features(n_features)

        np.random.seed(self.random_state)

        oob_preds = np.zeros((n_samples, 2), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int32)

        self.estimators_ = []
        self.feature_importances_ = np.zeros(n_features)

        print(f"Training {self.n_estimators} GSNH trees...")

        for i in range(self.n_estimators):
            # Bootstrap
            if self.bootstrap:
                sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
                unique_sampled = np.unique(sample_idx)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[unique_sampled] = False
            else:
                sample_idx = np.arange(n_samples)
                oob_mask = np.zeros(n_samples, dtype=bool)

            # Feature subsample
            feat_idx = np.sort(
                np.random.choice(n_features, size=max_feat, replace=False)
            )

            X_boot = X[sample_idx][:, feat_idx]
            y_boot = y[sample_idx]

            # Train tree
            tree = ExpertGSNHTree(**self.tree_params)
            tree.fit(X_boot, y_boot)
            tree._feature_indices = feat_idx
            self.estimators_.append(tree)

            # Accumulate feature importances
            if tree.feature_importances_ is not None:
                for local_idx, global_idx in enumerate(feat_idx):
                    if local_idx < len(tree.feature_importances_):
                        self.feature_importances_[global_idx] += (
                            tree.feature_importances_[local_idx]
                        )

            # OOB
            if self.oob_score and oob_mask.sum() > 0:
                X_oob = X[oob_mask][:, feat_idx]
                proba = tree.predict_proba(X_oob)
                oob_preds[oob_mask] += proba
                oob_counts[oob_mask] += 1

            if (i + 1) % 10 == 0:
                print(f"  Trained {i + 1}/{self.n_estimators} trees...")

        # OOB score
        if self.oob_score:
            valid = oob_counts > 0
            if valid.sum() > 0:
                oob_preds[valid] /= oob_counts[valid, np.newaxis]
                oob_pred = (oob_preds[valid, 1] >= 0.5).astype(int)
                self.oob_score_ = float((oob_pred == y[valid]).mean())
                print(f"  OOB Score: {self.oob_score_:.4f}")

        # Normalize importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        print("Training complete!")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        probas = np.zeros((len(X), 2), dtype=np.float64)

        for tree in self.estimators_:
            X_sub = X[:, tree._feature_indices]
            probas += tree.predict_proba(X_sub)

        probas /= len(self.estimators_)
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
