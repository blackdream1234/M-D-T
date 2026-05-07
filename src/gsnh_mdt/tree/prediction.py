"""
Prediction methods for ExpertGSNHTree.

Extracted from tree/builder.py lines 1171-1179, 1414-1435.
All prediction logic: traversal, predict_proba, predict.
"""

import numpy as np


def predict_proba(tree, X):
    """Vectorized batch prediction. Returns (n_samples, 2) array."""
    X = np.asarray(X, dtype=np.float64)
    probas = np.full(len(X), 0.5)
    _batch_traverse(tree.root_, X, np.arange(len(X)), probas)
    return np.column_stack([1 - probas, probas])


def predict(tree, X):
    """Binary class prediction."""
    return (predict_proba(tree, X)[:, 1] >= 0.5).astype(int)


def _batch_traverse(node, X, indices, probas):
    """Traverse tree in batch instead of sample-by-sample."""
    if node is None or len(indices) == 0:
        return
    if node.get('is_leaf', True) or node.get('predicate') is None:
        probas[indices] = node.get('proba', 0.5)
        return
    mask = node['predicate'].evaluate(X[indices])
    left_idx = indices[mask]
    right_idx = indices[~mask]
    _batch_traverse(node.get('left'), X, left_idx, probas)
    _batch_traverse(node.get('right'), X, right_idx, probas)


def _traverse(node, x):
    """Single-sample fallback."""
    if node is None:
        return 0.5
    if node.get('is_leaf', True) or node.get('predicate') is None:
        return node.get('proba', 0.5)
    if node['predicate'].evaluate(x)[0]:
        return _traverse(node.get('left'), x)
    return _traverse(node.get('right'), x)
