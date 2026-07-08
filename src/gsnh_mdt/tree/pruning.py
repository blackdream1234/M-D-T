"""
CostComplexityPruner: validation-set cost-complexity pruning.

This pruner is compatible with the current ExpertGSNHTree node schema:
- leaves use predicate=None, left=None, right=None, is_leaf=True;
- nodes store proba, n_positive, n_negative, and n_samples;
- nodes do not require a distribution field.
"""

import numpy as np


class CostComplexityPruner:
    """Post-pruning with the current builder node schema."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)

    def prune(self, tree: dict, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Return a pruned tree using validation-set cost-complexity."""
        if tree is None:
            return None

        if self._is_leaf(tree):
            return tree

        predicate = tree.get("predicate")
        if predicate is None:
            return self._make_leaf(tree)

        if len(X_val) > 0:
            mask = predicate.evaluate(X_val)
            left_X, left_y = X_val[mask], y_val[mask]
            right_X, right_y = X_val[~mask], y_val[~mask]
        else:
            left_X = right_X = X_val
            left_y = right_y = y_val

        tree["left"] = self.prune(tree.get("left"), left_X, left_y)
        tree["right"] = self.prune(tree.get("right"), right_X, right_y)

        if len(y_val) == 0:
            return tree

        subtree_err = self._tree_error(tree, X_val, y_val)
        n_leaves = max(self._count_leaves(tree), 1)
        leaf_err = self._leaf_error(tree, y_val)

        adjusted_subtree = subtree_err + self.alpha * n_leaves
        adjusted_leaf = leaf_err + self.alpha

        if adjusted_leaf <= adjusted_subtree:
            return self._make_leaf(tree)

        return tree

    def _is_leaf(self, node: dict) -> bool:
        if node is None:
            return True
        if node.get("is_leaf") is True:
            return True
        if node.get("predicate") is None:
            return True
        if node.get("left") is None and node.get("right") is None:
            return True
        return False

    def _leaf_prediction(self, node: dict) -> int:
        """Match tree prediction convention: proba >= 0.5 gives class 1."""
        if node is None:
            return 0

        if "proba" in node:
            return int(float(node["proba"]) >= 0.5)

        n_pos = int(node.get("n_positive", 0))
        n_neg = int(node.get("n_negative", 0))
        return int(n_pos >= n_neg)

    def _leaf_error(self, node: dict, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        pred = self._leaf_prediction(node)
        return float(np.mean(y != pred))

    def _tree_error(self, node: dict, X: np.ndarray, y: np.ndarray) -> float:
        if node is None or len(y) == 0:
            return 0.0

        if self._is_leaf(node):
            return self._leaf_error(node, y)

        predicate = node.get("predicate")
        if predicate is None:
            return self._leaf_error(node, y)

        mask = predicate.evaluate(X)
        left_err = self._tree_error(node.get("left"), X[mask], y[mask])
        right_err = self._tree_error(node.get("right"), X[~mask], y[~mask])

        n = max(len(y), 1)
        return (len(y[mask]) * left_err + len(y[~mask]) * right_err) / n

    def _count_leaves(self, node: dict) -> int:
        if node is None:
            return 0
        if self._is_leaf(node):
            return 1
        return self._count_leaves(node.get("left")) + self._count_leaves(node.get("right"))

    def _make_leaf(self, node: dict) -> dict:
        n_pos = int(node.get("n_positive", 0))
        n_neg = int(node.get("n_negative", 0))
        n = int(node.get("n_samples", n_pos + n_neg))

        if "proba" in node:
            proba = float(node["proba"])
        else:
            proba = float((n_pos + 1.0) / (n + 2.0)) if n >= 0 else 0.5

        return {
            "proba": proba,
            "n_samples": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "depth": node.get("depth", 0),
            "predicate": None,
            "left": None,
            "right": None,
            "is_leaf": True,
            "language": node.get("language", ""),
        }
