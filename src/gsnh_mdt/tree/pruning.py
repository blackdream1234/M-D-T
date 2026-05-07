"""
CostComplexityPruner: post-pruning with cost-complexity criterion.

Extracted verbatim from gsnh_mdt_v3.py lines 1732-1809.
ENHANCED feature — not part of baseline journal logic.
"""

import numpy as np


class CostComplexityPruner:
    """Post-pruning with all scalar bugs fixed (Fix #4)."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def prune(self, tree: dict, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Prune tree using cost-complexity criterion on validation data."""
        if 'left' not in tree or 'right' not in tree:
            return tree

        # Recursively prune children first
        tree['left'] = self.prune(tree['left'], X_val, y_val)
        tree['right'] = self.prune(tree['right'], X_val, y_val)

        # Compute error of this subtree vs leaf
        subtree_err = self._tree_error(tree, X_val, y_val)
        n_leaves = self._count_leaves(tree)

        # Leaf error
        if len(y_val) == 0:
            return tree

        pred = tree['distribution']
        leaf_pred = int(np.argmax(pred))
        leaf_err = float(np.sum(y_val != leaf_pred)) / max(len(y_val), 1)

        # Complexity-adjusted comparison
        adjusted_subtree = subtree_err + self.alpha * n_leaves
        adjusted_leaf = leaf_err + self.alpha * 1

        if adjusted_leaf <= adjusted_subtree:
            # Replace subtree with leaf
            return {
                'distribution': pred,
                'n_samples': tree.get('n_samples', len(y_val)),
            }

        return tree

    def _tree_error(self, node: dict, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification error of a subtree."""
        if len(X) == 0:
            return 0.0

        if 'left' not in node or 'right' not in node:
            pred = int(np.argmax(node['distribution']))
            return float(np.sum(y != pred)) / max(len(y), 1)

        predicate = node.get('predicate')
        if predicate is None:
            pred = int(np.argmax(node['distribution']))
            return float(np.sum(y != pred)) / max(len(y), 1)

        mask = predicate.evaluate(X)

        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]

        left_err = self._tree_error(node['left'], left_X, left_y)
        right_err = self._tree_error(node['right'], right_X, right_y)

        n = max(len(y), 1)
        return (len(left_y) * left_err + len(right_y) * right_err) / n

    def _count_leaves(self, node: dict) -> int:
        if 'left' not in node or 'right' not in node:
            return 1
        return self._count_leaves(node['left']) + self._count_leaves(node['right'])
