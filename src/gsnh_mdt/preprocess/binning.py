"""
AdaptiveBinner: discretization with constant-feature safety.

Extracted verbatim from gsnh_mdt_v3.py lines 1578-1680.
Supports: quantile, equal_width, adaptive, supervised (ENHANCED feature).
"""

import numpy as np


class AdaptiveBinner:
    """Adaptive binning with constant-feature safety.

    Strategies:
      - 'quantile': percentile-based (default, baseline)
      - 'equal_width': uniform spacing
      - 'adaptive': variable-width based on density
      - 'supervised': label-aware DT thresholds (ENHANCED, Module 1 from paper)
    """

    def __init__(self, n_bins: int = 64, strategy: str = 'quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_ = {}
        self.bin_indices_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'AdaptiveBinner':
        n_features = X.shape[1]

        for f in range(n_features):
            col = X[:, f]

            if self.strategy == 'supervised' and y is not None:
                edges = self._supervised_bins(col, y, self.n_bins)
            elif self.strategy == 'quantile':
                edges = np.percentile(col, np.linspace(0, 100, self.n_bins + 1))
            elif self.strategy == 'equal_width':
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            elif self.strategy == 'adaptive':
                edges = self._adaptive_bins(col)
            else:
                edges = np.percentile(col, np.linspace(0, 100, self.n_bins + 1))

            edges = np.unique(edges)

            # Fix #3: Handle constant features (1 unique value)
            if len(edges) < 2:
                val = edges[0]
                edges = np.array([val - 1e-10, val, val + 1e-10])
            # Fix: Binary/low-cardinality features (2 unique values -> 1 bin)
            elif len(edges) == 2:
                mid = (edges[0] + edges[1]) / 2.0
                edges = np.array([edges[0], mid, edges[1]])

            self.bin_edges_[f] = edges
            self.bin_indices_[f] = np.searchsorted(
                edges[1:-1], col, side='right'
            )

        return self

    def _supervised_bins(self, col: np.ndarray, y: np.ndarray,
                          max_bins: int) -> np.ndarray:
        """Supervised binning: train a univariate DT to find label-optimal thresholds.

        Algorithm 1 from GSNH_MDT.pdf: uses DecisionTreeClassifier with
        max_leaf_nodes=max_bins to partition feature domain by label.

        ENHANCED feature — not part of baseline journal logic.
        """
        from sklearn.tree import DecisionTreeClassifier

        unique_vals = np.unique(col)
        if len(unique_vals) < 2:
            return np.array([col.min(), col.max()])

        dt = DecisionTreeClassifier(
            max_leaf_nodes=min(max_bins, len(unique_vals)),
            random_state=42
        )
        dt.fit(col.reshape(-1, 1), y)

        thresholds = dt.tree_.threshold[dt.tree_.threshold != -2.0]

        if len(thresholds) == 0:
            return np.array([col.min(), col.max()])

        thresholds = np.sort(np.unique(thresholds))
        edges = np.concatenate([[col.min()], thresholds, [col.max()]])
        return np.unique(edges)

    def _adaptive_bins(self, col: np.ndarray) -> np.ndarray:
        base_edges = np.percentile(
            col, np.linspace(0, 100, self.n_bins // 2 + 1)
        )
        base_edges = np.unique(base_edges)

        if len(base_edges) < 2:
            return base_edges

        hist, _ = np.histogram(col, bins=base_edges)
        median_count = np.median(hist)

        refined = [base_edges[0]]
        for i in range(len(hist)):
            if hist[i] > median_count * 1.5:
                mid = (base_edges[i] + base_edges[i + 1]) / 2
                refined.extend([mid, base_edges[i + 1]])
            else:
                refined.append(base_edges[i + 1])

        return np.unique(refined)

    def get_n_bins(self, feature: int) -> int:
        return len(self.bin_edges_[feature]) - 1
