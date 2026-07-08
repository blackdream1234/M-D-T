"""
CostComplexityPruner: post-pruning with cost-complexity criterion.

ENHANCED feature — not part of baseline journal logic.
This pruner operates on the current ExpertGSNHTree node schema:
leaves use ``is_leaf=True``, ``predicate=None``, ``left=None``,
``right=None``, and store class information as ``proba``,
``n_positive``, ``n_negative``, and ``n_samples``.
"""

import numpy as np


class CostComplexityPruner:
    """Post-pruning compatible with the current builder node schema."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def prune(self, tree: dict | None, X_val: np.ndarray, y_val: np.ndarray) -> dict | None:
        """Prune tree using a cost-complexity criterion on validation data."""
        if tree is None:
            return None
        if self._is_leaf(tree):
            return tree

        predicate = tree.get("predicate")
        if predicate is None or len(y_val) == 0:
            return self._make_leaf(tree) if predicate is None else tree

        mask = predicate.evaluate(X_val)
        tree["left"] = self.prune(tree.get("left"), X_val[mask], y_val[mask])
        tree["right"] = self.prune(tree.get("right"), X_val[~mask], y_val[~mask])

        subtree_err = self._tree_error(tree, X_val, y_val)
        n_leaves = self._count_leaves(tree)
        leaf_pred = self._leaf_prediction(tree)
        leaf_err = float(np.sum(y_val != leaf_pred)) / max(len(y_val), 1)

        adjusted_subtree = subtree_err + self.alpha * n_leaves
        adjusted_leaf = leaf_err + self.alpha

        if adjusted_leaf <= adjusted_subtree:
            return self._make_leaf(tree)
        return tree

    @staticmethod
    def _is_leaf(node: dict | None) -> bool:
        if node is None:
            return True
        if node.get("is_leaf") is True:
            return True
        if node.get("predicate") is None:
            return True
        return node.get("left") is None and node.get("right") is None

    @staticmethod
    def _leaf_prediction(node: dict) -> int:
        if "proba" in node:
            return int(float(node.get("proba", 0.5)) >= 0.5)
        n_pos = int(node.get("n_positive", 0))
        n_neg = int(node.get("n_negative", 0))
        if n_pos or n_neg:
            return int(n_pos >= n_neg)
        distribution = node.get("distribution")
        if distribution is not None:
            return int(np.argmax(distribution))
        return 1

    @staticmethod
    def _leaf_proba(node: dict) -> float:
        if "proba" in node:
            return float(node.get("proba", 0.5))
        n_pos = int(node.get("n_positive", 0))
        n_neg = int(node.get("n_negative", 0))
        n = n_pos + n_neg
        if n > 0:
            return float(n_pos / n)
        distribution = node.get("distribution")
        if distribution is not None and len(distribution) > 1:
            total = float(np.sum(distribution))
            if total > 0:
                return float(distribution[1] / total)
        return 0.5

    @classmethod
    def _make_leaf(cls, node: dict) -> dict:
        n_pos = int(node.get("n_positive", 0))
        n_neg = int(node.get("n_negative", 0))
        n_samples = int(node.get("n_samples", n_pos + n_neg))
        return {
            "proba": cls._leaf_proba(node),
            "n_samples": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "depth": node.get("depth", 0),
            "predicate": None,
            "left": None,
            "right": None,
            "is_leaf": True,
            "language": node.get("language"),
        }

    def _tree_error(self, node: dict | None, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification error of a subtree."""
        if node is None or len(X) == 0:
            return 0.0
        if self._is_leaf(node):
            pred = self._leaf_prediction(node)
            return float(np.sum(y != pred)) / max(len(y), 1)

        predicate = node.get("predicate")
        if predicate is None:
            pred = self._leaf_prediction(node)
            return float(np.sum(y != pred)) / max(len(y), 1)

        mask = predicate.evaluate(X)
        left_err = self._tree_error(node.get("left"), X[mask], y[mask])
        right_err = self._tree_error(node.get("right"), X[~mask], y[~mask])

        n = max(len(y), 1)
        return (int(mask.sum()) * left_err + int((~mask).sum()) * right_err) / n

    def _count_leaves(self, node: dict | None) -> int:
        if node is None or self._is_leaf(node):
            return 1
        return self._count_leaves(node.get("left")) + self._count_leaves(node.get("right"))
