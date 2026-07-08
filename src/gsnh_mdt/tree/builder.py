"""
ExpertGSNHTree: the core GSNH multivariate decision tree.

Extracted VERBATIM from gsnh_mdt_v3.py lines 1898-3303.
This file is intentionally large (~1400 lines) to preserve exact
behavioral equivalence during extraction. Prediction and explainer
logic will be split out in a later phase after regression parity
is confirmed.

NO methods, defaults, or control flow have been changed.
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple, List, Dict, Any
from itertools import combinations
from collections import defaultdict
from numba import njit

from gsnh_mdt.types import (
    LiteralPolarity, ClauseArity, LanguageFamily,
    CompareOp, GSNHPatternType,
    GSNH_VALID_CONFIGS, GSNH_ANTIHORN_CONFIGS,
    _resolve_language,
)
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.binary import GSNHBinaryLiteral
from gsnh_mdt.literals.compare import CompareLiteral
from gsnh_mdt.literals.predicates import GSNHPredicate, Square2CNFPredicate, Literal
from gsnh_mdt.scoring.entropy import entropy
from gsnh_mdt.scoring.gain import (
    information_gain, gain_ratio, fast_hist_gain, jit_ig_scores,
)
from gsnh_mdt.scoring.penalties import penalized_gain
from gsnh_mdt.search.prefix import (
    build_1d_prefix, build_2d_prefix, build_3d_prefix,
    query_1d, query_2d, query_3d,
    count_2way_union, count_3way_union,
)
from gsnh_mdt.search.tensors import (
    jit_build_tensors_1d, jit_build_tensors_2d, jit_build_tensors_3d,
)
from gsnh_mdt.search.exhaustive_1d import search_1d_exhaustive
from gsnh_mdt.search.exhaustive_2d import search_2d_exhaustive
from gsnh_mdt.search.exhaustive_3d import search_3d_exhaustive
from gsnh_mdt.search.antihorn import search_2d_antihorn, search_3d_antihorn
from gsnh_mdt.search.affine_search import fast_affine_2d, fast_affine_3d
from gsnh_mdt.search.conj_ui import search_2d_conj_ui, search_3d_conj_ui
from gsnh_mdt.search.square_2cnf import search_square_2cnf as search_square_2cnf_candidates
from gsnh_mdt.sat.exact_solver import ExactSATSolver
from gsnh_mdt.preprocess.binning import AdaptiveBinner
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.tree.pruning import CostComplexityPruner
from gsnh_mdt.tree.calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class ExpertGSNHTree:
    """
    Single GSNH tree with all bugs fixed:
    - FIX #1:  Correct 3D inclusion-exclusion
    - FIX #2:  Feature importance only for winning split
    - FIX #3:  Constant feature handling (via AdaptiveBinner)
    - FIX #5:  Proper bin construction
    - FIX #8:  Safe tree traversal
    - FIX #9:  Guard for < 3 features in 3D search
    - FIX #10: Threshold boundary epsilon
    - FIX #12: Consistent gain > 0 checks
    - FIX #13: Input validation
    """

    def __init__(self,
                 stopping_criteria: Optional[StoppingCriteria] = None,
                 n_bins: int = 64,
                 binning_strategy: str = 'quantile',
                 top_k_features: int = 15,
                 use_gain_ratio: bool = False,
                 laplace_smoothing: float = 1.0,
                 search_1d: bool = True,
                 search_2d: bool = True,
                 search_3d: bool = True,
                 # v5 modules
                 use_supervised_binning: bool = True,
                 use_attention: bool = True,
                 use_look_ahead: bool = False,
                 look_ahead_gamma: float = 0.3,
                 look_ahead_top_p: int = 5,
                 verbose: bool = False,
                 mode: str = 'heuristic',
                 language: LanguageFamily = LanguageFamily.ANY,
                 limit_2d: Optional[int] = None,
                 limit_3d: Optional[int] = None,
                 use_binary_comparisons: bool = False,
                 enable_compare_literals: bool = False,
                 prune: bool = False,
                 prune_alpha: float = 0.01,
                 theorem_strict: bool = False,
                 allow_affine_in_bestpn: bool = False,
                 random_state: int = 42):

        self.stopping = stopping_criteria or StoppingCriteria()
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.top_k = top_k_features
        self.use_gain_ratio = use_gain_ratio
        self.laplace = laplace_smoothing
        self.search_1d = search_1d
        self.search_2d = search_2d
        self.search_3d = search_3d
        # v5 module flags
        self.use_supervised_binning = use_supervised_binning
        self.use_attention = use_attention
        self.use_look_ahead = use_look_ahead
        self.look_ahead_gamma = look_ahead_gamma
        self.look_ahead_top_p = look_ahead_top_p
        self.verbose = verbose
        self.mode = mode
        self.language = language
        self.limit_2d = limit_2d
        self.limit_3d = limit_3d
        self.use_binary_comparisons = use_binary_comparisons
        self.enable_compare_literals = enable_compare_literals
        self.prune = prune
        self.prune_alpha = prune_alpha
        self.theorem_strict = theorem_strict
        self.allow_affine_in_bestpn = allow_affine_in_bestpn
        self.random_state = random_state

        self.root_ = None
        self.axp_metadata_ = []  # Backend metadata for AXp path checks
        self.binner_ = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        self.max_depth_reached_ = 0
        self.arity_counts_ = {1: 0, 2: 0, 3: 0}
        self.pattern_counts_ = {}
        self.language_counts_ = {}

    @classmethod
    def from_config(cls, config: 'ModelConfig') -> 'ExpertGSNHTree':
        """Create an ExpertGSNHTree from a ModelConfig dataclass.

        This is the preferred way to create a tree with structured configuration.
        The config is converted into the exact same constructor keyword arguments
        via ``config.to_constructor_kwargs()``. The constructor remains explicit —
        no default reinterpretation, no implicit merging.

        Example::

            from gsnh_mdt.config import ModelConfig, SearchConfig
            from gsnh_mdt.types import LanguageFamily

            config = ModelConfig(
                language=LanguageFamily.BEST_PER_NODE,
                search=SearchConfig(search_3d=False),
            )
            tree = ExpertGSNHTree.from_config(config)
            tree.fit(X, y)

        Args:
            config: A ModelConfig instance.

        Returns:
            An ExpertGSNHTree instance with the same behavior as if
            constructed with the equivalent keyword arguments.
        """
        return cls(**config.to_constructor_kwargs())

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExpertGSNHTree':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        # FIX #13: Input validation
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("X contains Inf values")

        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))  # For MDL penalty
        self.feature_importances_ = np.zeros(self.n_features_)
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        self.max_depth_reached_ = 0
        self._current_depth = 0
        self.arity_counts_ = {1: 0, 2: 0, 3: 0}
        self.pattern_counts_ = {}

        # Binning — supervised (Module 1) or unsupervised
            
        if self.use_supervised_binning:
            self.binner_ = AdaptiveBinner(self.n_bins, 'supervised',
                                          random_state=self.random_state)
            self.binner_.fit(X, y)
        else:
            self.binner_ = AdaptiveBinner(self.n_bins, self.binning_strategy,
                                          random_state=self.random_state)
            self.binner_.fit(X)

        # Feature scores for prioritization (recomputed per-node)
        feature_scores = self._compute_feature_scores(X, y)

        # Build tree with FIXED language
        self.language_counts_ = {}  # Track language family usage
        if self.mode == 'journal':
            if self.language == LanguageFamily.ANY:
                raise ValueError(
                    "Journal mode requires a fixed language (not ANY). "
                    "Use select_language_via_cv() before training, or pass "
                    "language=HORN/ANTI_HORN/AFFINE/CONJ_UI/SQUARE_2CNF/BEST_PER_NODE explicitly."
                )

        start_language = self.language

        # If pruning enabled, hold out validation data
        if self.prune and len(X) >= 500:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
            tr_idx, val_idx = next(sss.split(X, y))
            X_train, X_val = X[tr_idx], X[val_idx]
            y_train, y_val = y[tr_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Compute interaction pairs (synergistic feature combinations)
        if self.search_2d and self.n_features_ > 1:
            self._interaction_pairs_ = self._compute_interaction_pairs(X_train, y_train)
        else:
            self._interaction_pairs_ = []

        self.root_ = self._build_tree(
            X_train, y_train, depth=0, feature_scores=feature_scores,
            language=start_language
        )

        # Post-pruning with held-out validation
        if self.prune and X_val is not None and self.root_ is not None:
            pruner = CostComplexityPruner(alpha=self.prune_alpha)
            self.root_ = pruner.prune(self.root_, X_val, y_val)

        # Normalize importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def _compute_feature_scores(self, X, y):
        """Module 2: Interaction Attention Map.
        
        Uses Mutual Information I(Xj; Y) for feature scoring when
        use_attention=True, otherwise falls back to univariate IG scan.
        """
        if self.use_attention:
            return self._compute_mi_scores(X, y)
        return self._compute_ig_scores(X, y)

    def _compute_mi_scores(self, X, y):
        """Attention Map: Mutual Information scoring (Algorithm 2)."""
        from sklearn.feature_selection import mutual_info_classif
        try:
            scores = mutual_info_classif(
                X, y, discrete_features='auto', random_state=self.random_state, n_neighbors=3
            )
            return scores
        except Exception:
            # Fallback to IG scan if MI fails
            return self._compute_ig_scores(X, y)

    def _compute_ig_scores(self, X, y):
        """Univariate IG scan — delegates to JIT-compiled kernel."""
        nf = self.n_features_
        n = len(y)
        if n == 0:
            return np.zeros(nf)

        # Pre-bin ALL features into a contiguous int64 matrix
        bins_2d = np.zeros((n, nf), dtype=np.int64)
        n_bins_arr = np.ones(nf, dtype=np.int64)
        for f in range(nf):
            edges = self.binner_.bin_edges_[f]
            nb = len(edges) - 1
            n_bins_arr[f] = nb
            if nb >= 2:
                bins_2d[:, f] = np.clip(
                    np.searchsorted(edges[1:-1], X[:, f], side='right'),
                    0, nb - 1
                )

        return jit_ig_scores(
            bins_2d, y.astype(np.int32), nf, n_bins_arr,
            self.stopping.min_samples_leaf,
            self.use_gain_ratio
        )

    def _compute_interaction_pairs(self, X, y, max_pairs=50):
        """Fast O(d²) interaction scan: find feature PAIRS with high joint info.
        
        For each candidate pair, compute joint IG from 4-cell contingency table.
        Return pairs sorted by synergy (joint info beyond individual features).
        """
        n, d = X.shape
        if d <= 1:
            return []
        
        y_bin = (y == 1).astype(np.float64)
        n_pos = y_bin.sum()
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return []
        
        # Limit candidate features for O(d²) scan
        top_k_int = min(d, max(30, int(np.sqrt(d) * 3)))
        
        # Fast individual IG scores via pure numpy (no JIT dependency)
        individual_scores = np.zeros(d)
        H_parent = 0.0
        p_parent = n_pos / n
        if 0 < p_parent < 1:
            H_parent = -(p_parent * np.log2(p_parent)) - ((1-p_parent) * np.log2(1-p_parent))
        
        for f in range(d):
            col = X[:, f]
            med = np.median(col)
            mask_hi = col >= med
            n_hi = mask_hi.sum()
            n_lo = n - n_hi
            if n_hi == 0 or n_lo == 0:
                continue
            pos_hi = y_bin[mask_hi].sum()
            pos_lo = y_bin[~mask_hi].sum()
            e_hi = 0.0
            p_hi = pos_hi / n_hi
            if 0 < p_hi < 1:
                e_hi = -(p_hi * np.log2(p_hi)) - ((1-p_hi) * np.log2(1-p_hi))
            e_lo = 0.0
            p_lo = pos_lo / n_lo
            if 0 < p_lo < 1:
                e_lo = -(p_lo * np.log2(p_lo)) - ((1-p_lo) * np.log2(1-p_lo))
            individual_scores[f] = H_parent - (n_hi * e_hi + n_lo * e_lo) / n
        
        top_idx = np.argsort(-individual_scores)[:top_k_int]
        
        # Also include random features (hidden gems)
        if d > top_k_int:
            rng_interact = np.random.RandomState(self.random_state)
            remaining = np.setdiff1d(np.arange(d), top_idx)
            n_random = min(len(remaining), max(10, top_k_int // 3))
            random_idx = rng_interact.choice(remaining, n_random, replace=False)
            candidate_feats = np.concatenate([top_idx, random_idx])
        else:
            candidate_feats = top_idx
        
        H_parent = 0.0
        p_parent = n_pos / n
        if 0 < p_parent < 1:
            H_parent = -(p_parent * np.log2(p_parent)) - ((1-p_parent) * np.log2(1-p_parent))
        
        pair_scores = []
        for idx_a in range(len(candidate_feats)):
            fi = candidate_feats[idx_a]
            med_i = np.median(X[:, fi])
            bi = (X[:, fi] >= med_i).astype(np.int32)
            
            for idx_b in range(idx_a + 1, len(candidate_feats)):
                fj = candidate_feats[idx_b]
                med_j = np.median(X[:, fj])
                bj = (X[:, fj] >= med_j).astype(np.int32)
                
                joint = bi * 2 + bj
                joint_entropy = 0.0
                for val in range(4):
                    mask_val = (joint == val)
                    n_val = mask_val.sum()
                    if n_val == 0 or n_val == n:
                        continue
                    pos_val = y_bin[mask_val].sum()
                    neg_val = n_val - pos_val
                    if pos_val > 0 and neg_val > 0:
                        p = pos_val / n_val
                        joint_entropy += (n_val / n) * (-(p * np.log2(p)) - ((1-p) * np.log2(1-p)))
                
                interaction_ig = H_parent - joint_entropy
                synergy = interaction_ig - max(individual_scores[fi], individual_scores[fj])
                
                if synergy > 1e-6 or interaction_ig > max(individual_scores[fi], individual_scores[fj]) * 1.1:
                    pair_scores.append((interaction_ig, synergy, int(fi), int(fj)))
        
        pair_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [(ps[2], ps[3]) for ps in pair_scores[:max_pairs]]

    # ── Tensor builders (JIT-delegated) ──────────────────────────────

    def _build_tensors_1d(self, y, bins, n_bins):
        return jit_build_tensors_1d(
            y.astype(np.int32), bins.astype(np.int64), n_bins
        )

    def _build_tensors_2d(self, y, bi, bj, ni, nj):
        return jit_build_tensors_2d(
            y.astype(np.int32), bi.astype(np.int64),
            bj.astype(np.int64), ni, nj
        )

    def _build_tensors_3d(self, y, bi, bj, bk, ni, nj, nk):
        return jit_build_tensors_3d(
            y.astype(np.int32), bi.astype(np.int64),
            bj.astype(np.int64), bk.astype(np.int64),
            ni, nj, nk
        )

    # ── Predicate builders (FIX #10: epsilon) ───────────────────────

    def _make_literal(self, feature: int, anchor: int,
                      lo: int, hi: int) -> GSNHLiteral:
        # Fix #1: Use node-local edges if available (zero quantization error)
        if hasattr(self, '_local_edges_') and feature in self._local_edges_:
            edges = self._local_edges_[feature]
        else:
            edges = self.binner_.bin_edges_[feature]

        if anchor == 0:  # x < t (LT) (NEGATIVE)
            thresh = float(edges[min(hi, len(edges) - 1)])
            pol = LiteralPolarity.LT
        else:  # x >= t (GE) (POSITIVE)
            thresh = float(edges[max(0, lo)])
            pol = LiteralPolarity.GE

        return GSNHLiteral(feature, thresh, pol)

    def _build_pred_1d(self, f, result, gain, language_family=LanguageFamily.HORN):
        lo, hi, anchor = int(result[0]), int(result[1]), int(result[2])
        lit = self._make_literal(f, anchor, lo, hi)
        try:
            return GSNHPredicate((lit,), gain, language_family=language_family)
        except ValueError:
            return None

    def _build_pred_2d(self, features, result, gain, language_family=LanguageFamily.HORN):
        lits = []
        for idx, f in enumerate(features):
            lo = int(result[2 * idx])
            hi = int(result[2 * idx + 1])
            anchor = int(result[4 + idx])
            lits.append(self._make_literal(f, anchor, lo, hi))
        try:
            return GSNHPredicate(tuple(lits), gain, language_family=language_family)
        except ValueError:
            return None

    def _build_pred_3d(self, features, result, gain, language_family=LanguageFamily.HORN):
        lits = []
        for idx, f in enumerate(features):
            lo = int(result[2 * idx])
            hi = int(result[2 * idx + 1])
            anchor = int(result[6 + idx])
            lits.append(self._make_literal(f, anchor, lo, hi))
        try:
            return GSNHPredicate(tuple(lits), gain, language_family=language_family)
        except ValueError:
            return None

    def _build_pred_affine_2d(self, features, result, gain):
        """Build an Affine (XOR) predicate from search_affine_2d result."""
        fi, fj = features
        ti, tj, xnor = int(result[0]), int(result[1]), int(result[2])
        # Fix #1: use local edges for threshold recovery
        edges_i = self._local_edges_.get(fi, self.binner_.bin_edges_[fi])
        edges_j = self._local_edges_.get(fj, self.binner_.bin_edges_[fj])
        thresh_i = float(edges_i[min(ti, len(edges_i) - 1)])
        thresh_j = float(edges_j[min(tj, len(edges_j) - 1)])
        lit_i = GSNHLiteral(fi, thresh_i, LiteralPolarity.LT)
        if xnor == 0:
            lit_j = GSNHLiteral(fj, thresh_j, LiteralPolarity.LT)
        else:
            lit_j = GSNHLiteral(fj, thresh_j, LiteralPolarity.GE)
        try:
            return GSNHPredicate(
                (lit_i, lit_j), gain,
                language_family=LanguageFamily.AFFINE, is_xor=True
            )
        except ValueError:
            return None

    def _build_pred_affine_3d(self, features, result, gain):
        """Fix #3: Build a 3-way XOR predicate from search_affine_3d result."""
        fi, fj, fk = features
        ti, tj, tk, xnor = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        edges_i = self._local_edges_.get(fi, self.binner_.bin_edges_[fi])
        edges_j = self._local_edges_.get(fj, self.binner_.bin_edges_[fj])
        edges_k = self._local_edges_.get(fk, self.binner_.bin_edges_[fk])
        thresh_i = float(edges_i[min(ti, len(edges_i) - 1)])
        thresh_j = float(edges_j[min(tj, len(edges_j) - 1)])
        thresh_k = float(edges_k[min(tk, len(edges_k) - 1)])
        lit_i = GSNHLiteral(fi, thresh_i, LiteralPolarity.LT)
        lit_j = GSNHLiteral(fj, thresh_j, LiteralPolarity.LT)
        if xnor == 0:
            lit_k = GSNHLiteral(fk, thresh_k, LiteralPolarity.LT)
        else:
            lit_k = GSNHLiteral(fk, thresh_k, LiteralPolarity.GE)
        try:
            return GSNHPredicate(
                (lit_i, lit_j, lit_k), gain,
                language_family=LanguageFamily.AFFINE, is_xor=True
            )
        except ValueError:
            return None

    # ── OPTIMIZED LOOK-AHEAD (v6.1 — Zero-Cost Bin Reuse) ─────────────

    def _look_ahead_score(self, y, mask, greedy_gain, curr_bins, top_feats):
        """Fast Look-Ahead with zero-cost discretization.
        
        v6.1: Reuses existing bin indices from parent (curr_bins) instead of
        re-running np.searchsorted. Scans only top_k features.
        
        S_LA(φ) = ΔI(φ) + γ · (|S_L|/|S| · max_ψ ΔI(ψ,S_L) + |S_R|/|S| · max_ψ ΔI(ψ,S_R))
        """
        n = len(y)
        gamma = self.look_ahead_gamma

        # Split labels
        y_l, y_r = y[mask], y[~mask]
        n_l, n_r = len(y_l), len(y_r)

        # Heuristic: If child is too small, gain is 0
        min_n = 2 * self.stopping.min_samples_leaf

        # Left Child Gain — SLICE existing bins (zero-cost discretization)
        g_l = 0.0
        if n_l >= min_n:
            bins_l = {f: curr_bins[f][mask] for f in top_feats}
            g_l = self._fast_1d_scan(y_l, bins_l, top_feats)

        # Right Child Gain
        g_r = 0.0
        if n_r >= min_n:
            bins_r = {f: curr_bins[f][~mask] for f in top_feats}
            g_r = self._fast_1d_scan(y_r, bins_r, top_feats)

        # Weighted average future gain
        return greedy_gain + gamma * ((n_l / n) * g_l + (n_r / n) * g_r)

    def _fast_1d_scan(self, y, bins_dict, features):
        """Scans top features using pre-computed bins + JIT kernel."""
        total_pos = float((y == 1).sum())
        total_neg = float(len(y) - total_pos)

        if total_pos == 0 or total_neg == 0:
            return 0.0

        ml = self.stopping.min_samples_leaf
        best_gain = 0.0

        for f in features:
            b_indices = bins_dict[f]
            n_bins = self.binner_.get_n_bins(f)
            # JIT kernel — pure C-speed histogram scan
            g = fast_hist_gain(
                b_indices, y.astype(np.int32), n_bins,
                total_pos, total_neg, ml
            )
            if g > best_gain:
                best_gain = g

        return best_gain

    # ── Main search (FIX #2: importance only for winner) ────────────

    def _search_best_split(self, X, y, feature_scores,
                            language=LanguageFamily.HORN, bounds=None):
        """Language-aware split search.
        
        When language=ANY: all families (Horn, Anti-Horn, Affine, ConjUI) compete.
        When language=HORN/ANTI_HORN/AFFINE: only that family is searched.
        When language=CONJ_UI: 1D, 2D, and 3D with AND/box constraints.
        When language=SQUARE_2CNF: paper-style (l1∨l2)∧(l3∨l4) search.
        
        Fix #4: bounds is a dict {feature: (lo, hi)} from the tree path.
        Features whose valid range is collapsed are skipped.
        """
        total_pos = float((y == 1).sum())
        total_neg = float(len(y) - total_pos)
        ml = self.stopping.min_samples_leaf

        best_gain = -1.0
        best_pred = None
        best_mask = None
        best_features = []
        best_arity = 0
        best_lang = language  # Default to passed language instead of hardcoded HORN

        # Look-ahead: collect top-P candidates for re-ranking
        candidates = []  # list of (gain, pred, mask, features, arity, lang)

        top_k = min(self.top_k, self.n_features_)
        top_feats = np.argsort(-feature_scores)[:top_k]

        # Fix #1: Node-local adaptive binning.
        # Recompute bin edges using quantiles on THIS node's data
        # instead of global pre-fitted edges → zero quantization error.
        curr_bins = {}
        curr_nbins = {}
        local_edges = {}  # node-local edges for accurate threshold recovery
        if bounds is None:
            bounds = {}
        n_target_bins = self.n_bins
        # EXHAUSTIVE 1D Base Builder: build bins for all features natively
        for f in range(self.n_features_):
            col = X[:, f]
            unique_vals = np.unique(col)
            n_unique = len(unique_vals)
            if n_unique <= 1:
                # Constant feature at this node
                local_edges[f] = np.array([col[0] - 1e-10, col[0], col[0] + 1e-10])
                curr_bins[f] = np.zeros(len(col), dtype=np.int64)
                curr_nbins[f] = 1
            elif n_unique <= n_target_bins:
                # Few unique values: use exact boundaries (like CART)
                midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                edges = np.concatenate([[unique_vals[0] - 1e-10], midpoints, [unique_vals[-1] + 1e-10]])
                local_edges[f] = edges
                curr_bins[f] = np.searchsorted(edges[1:-1], col, side='right')
                curr_nbins[f] = len(edges) - 1
            else:
                # Many unique values: quantile binning on local data
                edges = np.quantile(col, np.linspace(0, 1, n_target_bins + 1))
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([col.min() - 1e-10, col.max() + 1e-10])
                elif len(edges) == 2:
                    mid = (edges[0] + edges[1]) / 2.0
                    edges = np.array([edges[0], mid, edges[1]])
                local_edges[f] = edges
                curr_bins[f] = np.clip(
                    np.searchsorted(edges[1:-1], col, side='right'),
                    0, len(edges) - 2
                )
                curr_nbins[f] = len(edges) - 1
            # Fix #4: skip features whose valid range is empty
            if f in bounds:
                lo_b, hi_b = bounds[f]
                if lo_b >= hi_b:
                    curr_nbins[f] = 1  # disable this feature

        # Store local edges for _make_literal to use (Fix #1)
        self._local_edges_ = local_edges

        n_samples = len(y)

        # Resolve legacy SQUARE_CNF → CONJ_UI
        language = _resolve_language(language)

        # Determine which families to search
        # CRITICAL FIX: each family is searched ONLY when explicitly requested.
        # CONJ_UI does NOT trigger Horn/AntiHorn search (they have different semantics).
        search_horn = language in (LanguageFamily.ANY, LanguageFamily.HORN, LanguageFamily.BEST_PER_NODE)
        search_antihorn = language in (LanguageFamily.ANY, LanguageFamily.ANTI_HORN, LanguageFamily.BEST_PER_NODE)
        search_affine = language in (LanguageFamily.ANY, LanguageFamily.AFFINE) or (
            language == LanguageFamily.BEST_PER_NODE and self.allow_affine_in_bestpn
        )
        search_conj_ui = language in (LanguageFamily.ANY, LanguageFamily.CONJ_UI, LanguageFamily.BEST_PER_NODE)
        do_search_square_2cnf = language in (LanguageFamily.SQUARE_2CNF,)

        # ──── 1D (same for Horn and Anti-Horn) ────
        if self.search_1d and (search_horn or search_antihorn):
            # EXHAUSTIVE 1D SEARCH: Ensure no simple 1-dimensional gains are skipped
            for f in range(self.n_features_):
                nb = curr_nbins[f]
                if nb < 2:
                    continue

                pos_t, neg_t = self._build_tensors_1d(y, curr_bins[f], nb)
                P_pos = build_1d_prefix(pos_t)
                P_neg = build_1d_prefix(neg_t)

                gain, result = search_1d_exhaustive(
                    P_pos, P_neg, total_pos, total_neg, ml
                )

                if gain > 0:
                    gain = penalized_gain(
                        gain, arity=1, n_bins=nb,
                        n_samples=n_samples, n_classes=self.n_classes_
                    )

                if gain > best_gain and gain > 0:
                    if language == LanguageFamily.ANY:
                        lang = LanguageFamily.HORN if search_horn else LanguageFamily.ANTI_HORN
                    else:
                        lang = language
                        
                    pred = self._build_pred_1d(f, result, gain, language_family=lang)
                    if pred is not None:
                        mask = pred.evaluate(X)
                        if mask.sum() >= ml and (~mask).sum() >= ml:
                            best_gain = gain
                            best_pred = pred
                            best_mask = mask
                            best_features = [f]
                            best_arity = 1
                            best_lang = lang
                            if self.use_look_ahead:
                                candidates.append((gain, pred, mask, [f], 1, lang))
                                if len(candidates) > self.look_ahead_top_p * 2:
                                    candidates.sort(key=lambda x: x[0], reverse=True)
                                    candidates = candidates[:self.look_ahead_top_p]

        # ──── 2D Unified Search (Horn + Anti-Horn + Affine) — FAST Perfect XOR ────
        if self.search_2d and len(top_feats) >= 2 and (search_horn or search_antihorn or search_affine or search_conj_ui):
            if self.limit_2d is not None:
                limit = self.limit_2d
            else:
                limit = min(max(10, int(np.sqrt(self.n_features_) * 2.0)), len(top_feats))

            for fi, fj in combinations(top_feats[:limit], 2):
                if self.enable_compare_literals:
                    # Evaluate explicit CompareLiteral splits
                    for op in [CompareOp.LE, CompareOp.GT]:
                        # LE is naturally negative (Horn allows 1 positive), GT is positive
                        lang_cand = LanguageFamily.HORN if op == CompareOp.LE else LanguageFamily.ANTI_HORN
                        if lang_cand == LanguageFamily.HORN and not search_horn:
                            continue
                        if lang_cand == LanguageFamily.ANTI_HORN and not search_antihorn:
                            continue
                            
                        mask = (X[:, fi] <= X[:, fj]) if op == CompareOp.LE else (X[:, fi] > X[:, fj])
                        if mask.sum() >= ml and (~mask).sum() >= ml:
                            n_total = len(y)
                            n1 = int(mask.sum())
                            n2 = n_total - n1
                            pos1 = float(y[mask].sum())
                            pos2 = float(y[~mask].sum())
                            
                            H_parent = entropy(total_pos, total_neg)
                            e1 = entropy(pos1, n1 - pos1)
                            e2 = entropy(pos2, n2 - pos2)
                            gain = H_parent - (n1 * e1 + n2 * e2) / n_total
                            
                            if gain > 0:
                                gain = penalized_gain(gain, arity=2, n_bins=2, n_samples=n_total, n_classes=self.n_classes_)
                                
                            if gain > best_gain and gain > 0:
                                lit = CompareLiteral(fi, fj, op)
                                lang = lang_cand if language == LanguageFamily.ANY else language
                                pred = GSNHPredicate((lit,), gain, lang)
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj]
                                best_arity = 2 
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj], 2, lang))
                ni, nj = curr_nbins[fi], curr_nbins[fj]
                if ni < 2 or nj < 2:
                    continue

                # 1. Build Tensors ONCE
                pos_t, neg_t = self._build_tensors_2d(
                    y, curr_bins[fi], curr_bins[fj], ni, nj
                )
                
                # 2. Build Prefix Sums ONCE
                P_pos = build_2d_prefix(pos_t)
                P_neg = build_2d_prefix(neg_t)
                
                eff_bins = int(np.sqrt(float(ni) * float(nj)))
                pair_step = 1  # Exhaustive search for perfection

                if search_horn:
                    gain, result = search_2d_exhaustive(
                        P_pos, P_neg, total_pos, total_neg, ml, pair_step
                    )
                    if gain > 0:
                        gain = penalized_gain(
                            gain, arity=2, n_bins=max(eff_bins, 2),
                            n_samples=n_samples, n_classes=self.n_classes_
                        )
                    if gain > best_gain and gain > 0:
                        lang = LanguageFamily.HORN if language == LanguageFamily.ANY else language
                        pred = self._build_pred_2d(
                            (fi, fj), result, gain, lang
                        )
                        if pred is not None:
                            mask = pred.evaluate(X)
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj]
                                best_arity = 2
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj], 2, lang))

                if search_antihorn:
                    gain, result = search_2d_antihorn(
                        P_pos, P_neg, total_pos, total_neg, ml, pair_step
                    )
                    if gain > 0:
                        gain = penalized_gain(
                            gain, arity=2, n_bins=max(eff_bins, 2),
                            n_samples=n_samples, n_classes=self.n_classes_
                        )
                    if gain > best_gain and gain > 0:
                        lang = LanguageFamily.ANTI_HORN if language == LanguageFamily.ANY else language
                        pred = self._build_pred_2d(
                            (fi, fj), result, gain, lang
                        )
                        if pred is not None:
                            mask = pred.evaluate(X)
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj]
                                best_arity = 2
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj], 2, lang))

                # ── ConjUI 2D (AND of interval literals / box) ──
                if search_conj_ui:
                    gain, result = search_2d_conj_ui(
                        P_pos, P_neg, total_pos, total_neg, ml, pair_step
                    )
                    if gain > 0:
                        gain = penalized_gain(
                            gain, arity=2, n_bins=max(eff_bins, 2),
                            n_samples=n_samples, n_classes=self.n_classes_
                        )
                    if gain > best_gain and gain > 0:
                        lang = LanguageFamily.CONJ_UI
                        pred = self._build_pred_2d(
                            (fi, fj), result, gain, lang
                        )
                        if pred is not None:
                            mask = pred.evaluate(X)
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj]
                                best_arity = 2
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj], 2, lang))

                if search_affine:
                    if self.mode == 'journal' and (ni > 2 or nj > 2):
                        pass
                    else:
                        gain, result = fast_affine_2d(
                            P_pos, P_neg, total_pos, total_neg, ml
                        )
                        if gain > 0:
                            gain = penalized_gain(
                                gain, arity=2, n_bins=max(eff_bins, 2),
                                n_samples=n_samples, n_classes=self.n_classes_
                            )
                        if gain > best_gain and gain > 0:
                            lang = LanguageFamily.AFFINE if language == LanguageFamily.ANY else language
                            pred = self._build_pred_affine_2d(
                                (fi, fj), result, gain
                            )
                            if pred is not None:
                                mask = pred.evaluate(X)
                                if mask.sum() >= ml and (~mask).sum() >= ml:
                                    best_gain = gain
                                    best_pred = pred
                                    best_mask = mask
                                    best_features = [fi, fj]
                                    best_arity = 2
                                    best_lang = lang
                                    if self.use_look_ahead:
                                        candidates.append((gain, pred, mask, [fi, fj], 2, lang))
                
                if self.use_look_ahead and len(candidates) > self.look_ahead_top_p * 2:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    candidates = candidates[:self.look_ahead_top_p]

            # ──── Interaction-Guided Pair Search ────
            # Search additional pairs identified by joint information scan.
            # These pairs may combine features OUTSIDE top-k that are synergistic.
            if hasattr(self, '_interaction_pairs_') and self._interaction_pairs_:
                seen_pairs = set()
                for fi, fj in combinations(top_feats[:limit], 2):
                    seen_pairs.add((min(fi,fj), max(fi,fj)))
                
                for fi, fj in self._interaction_pairs_:
                    pair_key = (min(fi,fj), max(fi,fj))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    
                    ni, nj = curr_nbins.get(fi, 0), curr_nbins.get(fj, 0)
                    if ni < 2 or nj < 2:
                        continue

                    if self.enable_compare_literals:
                        for op in [CompareOp.LE, CompareOp.GT]:
                            lang_cand = LanguageFamily.HORN if op == CompareOp.LE else LanguageFamily.ANTI_HORN
                            if lang_cand == LanguageFamily.HORN and not search_horn:
                                continue
                            if lang_cand == LanguageFamily.ANTI_HORN and not search_antihorn:
                                continue
                            mask = (X[:, fi] <= X[:, fj]) if op == CompareOp.LE else (X[:, fi] > X[:, fj])
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                n_total = len(y)
                                n1 = int(mask.sum())
                                n2 = n_total - n1
                                pos1 = float(y[mask].sum())
                                pos2 = float(y[~mask].sum())
                                H_p = entropy(total_pos, total_neg)
                                e1 = entropy(pos1, n1 - pos1)
                                e2 = entropy(pos2, n2 - pos2)
                                gain = H_p - (n1 * e1 + n2 * e2) / n_total
                                if gain > 0:
                                    gain = penalized_gain(gain, arity=2, n_bins=2,
                                        n_samples=n_total, n_classes=self.n_classes_)
                                if gain > best_gain and gain > 0:
                                    lit = CompareLiteral(fi, fj, op)
                                    lang = lang_cand if language in (LanguageFamily.ANY, LanguageFamily.BEST_PER_NODE) else language
                                    pred = GSNHPredicate((lit,), gain, lang)
                                    best_gain = gain
                                    best_pred = pred
                                    best_mask = mask
                                    best_features = [fi, fj]
                                    best_arity = 2
                                    best_lang = lang

                    pos_t, neg_t = self._build_tensors_2d(
                        y, curr_bins[fi], curr_bins[fj], ni, nj)
                    P_pos = build_2d_prefix(pos_t)
                    P_neg = build_2d_prefix(neg_t)
                    eff_bins = int(np.sqrt(float(ni) * float(nj)))

                    if search_horn:
                        gain, result = search_2d_exhaustive(
                            P_pos, P_neg, total_pos, total_neg, ml, 1)
                        if gain > 0:
                            gain = penalized_gain(gain, arity=2, n_bins=max(eff_bins, 2),
                                n_samples=n_samples, n_classes=self.n_classes_)
                        if gain > best_gain and gain > 0:
                            lang = LanguageFamily.HORN if language in (LanguageFamily.ANY, LanguageFamily.BEST_PER_NODE) else language
                            pred = self._build_pred_2d((fi, fj), result, gain, lang)
                            if pred is not None:
                                mask = pred.evaluate(X)
                                if mask.sum() >= ml and (~mask).sum() >= ml:
                                    best_gain = gain
                                    best_pred = pred
                                    best_mask = mask
                                    best_features = [fi, fj]
                                    best_arity = 2
                                    best_lang = lang

                    if search_antihorn:
                        gain, result = search_2d_antihorn(
                            P_pos, P_neg, total_pos, total_neg, ml, 1)
                        if gain > 0:
                            gain = penalized_gain(gain, arity=2, n_bins=max(eff_bins, 2),
                                n_samples=n_samples, n_classes=self.n_classes_)
                        if gain > best_gain and gain > 0:
                            lang = LanguageFamily.ANTI_HORN if language in (LanguageFamily.ANY, LanguageFamily.BEST_PER_NODE) else language
                            pred = self._build_pred_2d((fi, fj), result, gain, lang)
                            if pred is not None:
                                mask = pred.evaluate(X)
                                if mask.sum() >= ml and (~mask).sum() >= ml:
                                    best_gain = gain
                                    best_pred = pred
                                    best_mask = mask
                                    best_features = [fi, fj]
                                    best_arity = 2
                                    best_lang = lang

        # ──── 3D Unified Search (Horn + Anti-Horn + Affine) — FAST Perfect XOR ────
        if (self.search_3d and self.n_features_ >= 3 and len(top_feats) >= 3
                and (search_horn or search_antihorn or search_affine or search_conj_ui)):

            if self.limit_3d is not None:
                limit = self.limit_3d
            else:
                limit = min(max(6, int(np.cbrt(self.n_features_) * 1.5)), len(top_feats))

            for fi, fj, fk in combinations(top_feats[:limit], 3):
                ni = curr_nbins[fi]
                nj = curr_nbins[fj]
                nk = curr_nbins[fk]
                if ni < 2 or nj < 2 or nk < 2:
                    continue

                pos_t, neg_t = self._build_tensors_3d(
                    y, curr_bins[fi], curr_bins[fj], curr_bins[fk],
                    ni, nj, nk
                )
                P_pos = build_3d_prefix(pos_t)
                P_neg = build_3d_prefix(neg_t)
                
                eff_bins = int(np.cbrt(float(ni) * float(nj) * float(nk)))
                
                # Dynamic Step to avoid O(B^3) explosion if bins are large (e.g. 64 or 128)
                # Keep effective resolution around 32 bins for expensive Horn/Anti-Horn
                max_b = max(ni, nj, nk)
                if max_b > 32:
                    trip_step = max(1, max_b // 32)
                else:
                    trip_step = 1

                if search_horn:
                    gain, result = search_3d_exhaustive(
                        P_pos, P_neg, total_pos, total_neg, ml, trip_step
                    )
                    if gain > 0:
                        gain = penalized_gain(
                            gain, arity=3, n_bins=max(eff_bins, 2),
                            n_samples=n_samples, n_classes=self.n_classes_
                        )
                    if gain > best_gain and gain > 0:
                        lang = LanguageFamily.HORN if language == LanguageFamily.ANY else language
                        pred = self._build_pred_3d(
                            (fi, fj, fk), result, gain, lang
                        )
                        if pred is not None:
                            mask = pred.evaluate(X)
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj, fk]
                                best_arity = 3
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj, fk], 3, lang))
                
                if search_antihorn:
                    gain, result = search_3d_antihorn(
                        P_pos, P_neg, total_pos, total_neg, ml, trip_step
                    )
                    if gain > 0:
                        gain = penalized_gain(
                            gain, arity=3, n_bins=max(eff_bins, 2),
                            n_samples=n_samples, n_classes=self.n_classes_
                        )
                    if gain > best_gain and gain > 0:
                        lang = LanguageFamily.ANTI_HORN if language == LanguageFamily.ANY else language
                        pred = self._build_pred_3d(
                            (fi, fj, fk), result, gain, lang
                        )
                        if pred is not None:
                            mask = pred.evaluate(X)
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj, fk]
                                best_arity = 3
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj, fk], 3, lang))

                # ── ConjUI 3D (AND of interval literals / box) ──
                if search_conj_ui:
                    gain, result = search_3d_conj_ui(
                        P_pos, P_neg, total_pos, total_neg, ml, trip_step
                    )
                    if gain > 0:
                        gain = penalized_gain(
                            gain, arity=3, n_bins=max(eff_bins, 2),
                            n_samples=n_samples, n_classes=self.n_classes_
                        )
                    if gain > best_gain and gain > 0:
                        lang = LanguageFamily.CONJ_UI
                        pred = self._build_pred_3d(
                            (fi, fj, fk), result, gain, lang
                        )
                        if pred is not None:
                            mask = pred.evaluate(X)
                            if mask.sum() >= ml and (~mask).sum() >= ml:
                                best_gain = gain
                                best_pred = pred
                                best_mask = mask
                                best_features = [fi, fj, fk]
                                best_arity = 3
                                best_lang = lang
                                if self.use_look_ahead:
                                    candidates.append((gain, pred, mask, [fi, fj, fk], 3, lang))

                if search_affine:
                    # AFFINE (O(1) Integral Image) — ALWAYS EXHAUSTIVE!
                    # Because it's fast enough and provides 100% XOR perfection
                    if self.mode == 'journal' and (ni > 2 or nj > 2 or nk > 2):
                        pass
                    else:
                        gain, result = fast_affine_3d(
                            P_pos, P_neg, total_pos, total_neg, ml, trip_step
                        )
                        if gain > 0:
                            gain = penalized_gain(
                                gain, arity=3, n_bins=max(eff_bins, 2),
                                n_samples=n_samples, n_classes=self.n_classes_
                            )
                        if gain > best_gain and gain > 0:
                            lang = LanguageFamily.AFFINE if language == LanguageFamily.ANY else language
                            pred = self._build_pred_affine_3d(
                                (fi, fj, fk), result, gain
                            )
                            if pred is not None:
                                mask = pred.evaluate(X)
                                if mask.sum() >= ml and (~mask).sum() >= ml:
                                    best_gain = gain
                                    best_pred = pred
                                    best_mask = mask
                                    best_features = [fi, fj, fk]
                                    best_arity = 3
                                    best_lang = lang
                                    if self.use_look_ahead:
                                        candidates.append((gain, pred, mask, [fi, fj, fk], 3, lang))
                
                if self.use_look_ahead and len(candidates) > self.look_ahead_top_p * 2:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    candidates = candidates[:self.look_ahead_top_p]
        # FIX #2: Accumulate importance ONLY for winning split

        # ──── Square 2CNF search (paper-style) ────
        if do_search_square_2cnf and len(top_feats) >= 2:
            sq2_gain, sq2_pred = search_square_2cnf_candidates(
                X, y, top_feats[:min(10, len(top_feats))],
                local_edges, curr_nbins, ml,
                max_candidates=500,
                n_classes=self.n_classes_,
                penalty=True
            )
            if sq2_gain > best_gain and sq2_gain > 0 and sq2_pred is not None:
                mask = sq2_pred.evaluate(X)
                if mask.sum() >= ml and (~mask).sum() >= ml:
                    best_gain = sq2_gain
                    best_pred = sq2_pred
                    best_mask = mask
                    best_features = list(sq2_pred.features_used())
                    best_arity = 2
                    best_lang = LanguageFamily.SQUARE_2CNF
                    if self.use_look_ahead:
                        candidates.append((sq2_gain, sq2_pred, mask,
                                           best_features, 2, best_lang))

        # ──── Module 3: Look-Ahead re-ranking (v6.1 — zero-cost bins) ────
        if self.use_look_ahead and len(candidates) > 1:
            scored = []
            for (g, p, m, feats, ar, lang) in candidates:
                # PASS curr_bins and top_feats — zero np.searchsorted cost
                la_score = self._look_ahead_score(
                    y, m, g, curr_bins, top_feats
                )
                scored.append((la_score, g, p, m, feats, ar, lang))
            scored.sort(key=lambda x: x[0], reverse=True)
            # Winner by look-ahead
            _, best_gain, best_pred, best_mask, best_features, best_arity, best_lang = scored[0]

        if best_features and best_gain > 0:
            for f in best_features:
                self.feature_importances_[f] += best_gain / best_arity

        return best_gain, best_pred, best_mask, best_lang

    # ── Tree building ───────────────────────────────────────────────

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int,
                    feature_scores: np.ndarray,
                    language=LanguageFamily.ANY, bounds=None):
        self.n_nodes_ += 1
        self.max_depth_reached_ = max(self.max_depth_reached_, depth)
        self._current_depth = depth
        if bounds is None:
            bounds = {}

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n = len(y)

        proba = float((n_pos + self.laplace) / (n + 2 * self.laplace))

        node = {
            'proba': proba,
            'n_samples': n,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'depth': depth,
            'predicate': None,
            'left': None,
            'right': None,
            'is_leaf': True,
            'language': language.value,
        }

        # v6.0: Cache MI at root, fast IG fallback at depth 1-2,
        #        reuse parent scores at depth 3+
        if depth == 0:
            feature_scores = self._compute_feature_scores(X, y)
        elif depth <= 2 and n >= 100:
            feature_scores = self._compute_ig_scores(X, y)  # Fast fallback
        # else: reuse inherited feature_scores from parent

        # Language-aware search
        best_gain, best_pred, best_mask, best_lang = self._search_best_split(
            X, y, feature_scores, language, bounds=bounds
        )

        # JOURNAL MODE: Strict language enforcement
        if self.mode == 'journal' and language != LanguageFamily.BEST_PER_NODE:
            # Must use the fixed language, cannot switch
            if best_lang != language:
                # No valid split found in required language
                self.n_leaves_ += 1
                return node

        # Stopping
        should_stop, reason = self.stopping.should_stop(
            n, n_pos, n_neg, depth, best_gain
        )

        if should_stop or best_pred is None or best_mask is None:
            self.n_leaves_ += 1
            return node

        # Validate
        left_n = int(best_mask.sum())
        right_n = int((~best_mask).sum())

        if left_n < self.stopping.min_samples_leaf:
            self.n_leaves_ += 1
            return node
        if right_n < self.stopping.min_samples_leaf:
            self.n_leaves_ += 1
            return node

        # Record pattern and language
        pk = best_pred.pattern_type.value
        self.pattern_counts_[pk] = self.pattern_counts_.get(pk, 0) + 1
        self.arity_counts_[best_pred.arity.value] += 1
        lk = best_lang.value
        self.language_counts_[lk] = self.language_counts_.get(lk, 0) + 1

        node['predicate'] = best_pred
        node['is_leaf'] = False
        node['language'] = best_lang.value

        if self.verbose and depth <= 3:
            print(f"{'  ' * depth}SPLIT[{pk}|{best_pred.arity.value}L|{lk}]: "
                  f"{best_pred} (gain={best_gain:.4f}, "
                  f"n={n}, left={left_n}, right={right_n})")

        # Journal mode: propagate language. BEST_PER_NODE lets children choose independently.
        child_language = language if language != LanguageFamily.BEST_PER_NODE else LanguageFamily.BEST_PER_NODE
        # Fix #4: Propagate path bounding box.
        #   - TRUE branch (disjunction satisfied): at least one literal true,
        #     bounds are weaker → pass parent bounds unchanged.
        #   - FALSE branch (all literals negated): precise per-feature bounds.
        #     For each literal:
        #       POSITIVE x[f] <= t → negated = x[f] > t → lo[f] = max(lo[f], t)
        #       NEGATIVE x[f] > t  → negated = x[f] <= t → hi[f] = min(hi[f], t)
        bounds_left = dict(bounds)  # shallow copy for TRUE branch

        bounds_right = dict(bounds)  # start from parent bounds
        if not best_pred.is_xor:  # XOR predicates don't give simple conjunctive bounds
            if hasattr(best_pred, "iter_literals"):
                pred_lits = list(best_pred.iter_literals())
            else:
                pred_lits = getattr(best_pred, "literals", [])
            for lit in pred_lits:
                if not hasattr(lit, "feature") or not hasattr(lit, "polarity") or not hasattr(lit, "threshold"):
                    continue
                f = lit.feature
                lo_b, hi_b = bounds_right.get(f, (-np.inf, np.inf))
                if lit.polarity == LiteralPolarity.GE:
                    # Literal: x[f] >= t → negation: x[f] < t
                    hi_b = min(hi_b, lit.threshold)
                else:
                    # Literal: x[f] < t → negation: x[f] >= t
                    lo_b = max(lo_b, lit.threshold)
                bounds_right[f] = (lo_b, hi_b)

        node['left'] = self._build_tree(
            X[best_mask], y[best_mask], depth + 1, feature_scores,
            language=child_language, bounds=bounds_left
        )
        node['right'] = self._build_tree(
            X[~best_mask], y[~best_mask], depth + 1, feature_scores,
            language=child_language, bounds=bounds_right
        )

        return node

    # ── Prediction (FIX #8: safe traversal) ─────────────────────────
    # Delegates to gsnh_mdt.tree.prediction module.

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from gsnh_mdt.tree.prediction import predict_proba as _predict_proba
        return _predict_proba(self, X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        from gsnh_mdt.tree.prediction import predict as _predict
        return _predict(self, X)

    # ── Explainability delegates ──────────────────────────────────────
    # Delegates to gsnh_mdt.tree.explainer module.

    def weak_axp_check(self, x: np.ndarray, y: int, S: set) -> bool:
        """Check if partial instance x_S guarantees prediction y using CSP."""
        from gsnh_mdt.tree.explainer import weak_axp_check as _weak_axp_check
        return _weak_axp_check(self, x, y, S)

    def _is_sat_path(self, path_edges, x, S):
        """Route to correct SAT checker based on language family."""
        from gsnh_mdt.tree.explainer import _is_sat_path
        return _is_sat_path(self, path_edges, x, S)

    def _path_sat_numeric(self, path_edges, x, S) -> bool:
        """Delegate to explainer module."""
        from gsnh_mdt.tree.explainer import _path_sat_numeric
        return _path_sat_numeric(path_edges, x, S)

    def _solve_or_clauses_dfs(self, or_clauses, intervals, x, S, idx) -> bool:
        """Delegate to explainer module."""
        from gsnh_mdt.tree.explainer import _solve_or_clauses_dfs
        return _solve_or_clauses_dfs(or_clauses, intervals, x, S, idx)

    def _affine_path_sat(self, path_edges, x, S) -> bool:
        """Delegate to explainer module."""
        from gsnh_mdt.tree.explainer import _affine_path_sat
        return _affine_path_sat(path_edges, x, S)

    def extract_axp(self, x: np.ndarray) -> set:
        """Extract a single minimal AXp for an instance. Returns set of features."""
        from gsnh_mdt.tree.explainer import extract_axp as _extract_axp
        return _extract_axp(self, x)

    def _batch_traverse(self, node, X, indices, probas):
        """Delegate to prediction module."""
        from gsnh_mdt.tree.prediction import _batch_traverse
        return _batch_traverse(node, X, indices, probas)

    def _traverse(self, node, x):
        """Delegate to prediction module."""
        from gsnh_mdt.tree.prediction import _traverse
        return _traverse(node, x)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def print_tree(self, node=None, indent="", prefix="Root"):
        if node is None:
            node = self.root_
        if node is None:
            print(f"{indent}{prefix}: EMPTY")
            return

        if node.get('is_leaf', True) or node.get('predicate') is None:
            print(f"{indent}{prefix}: LEAF (n={node['n_samples']}, "
                  f"p={node['proba']:.3f})")
        else:
            pred = node['predicate']
            print(f"{indent}{prefix}: [{pred.pattern_type.value}|"
                  f"{pred.arity.value}L] {pred} "
                  f"(gain={pred.information_gain:.4f}, n={node['n_samples']})")
            self.print_tree(node.get('left'), indent + "  ", "T")
            self.print_tree(node.get('right'), indent + "  ", "F")

    def get_summary(self) -> str:
        return (f"Nodes={self.n_nodes_}, Leaves={self.n_leaves_}, "
                f"Depth={self.max_depth_reached_}, "
                f"Arities={self.arity_counts_}, "
                f"Patterns={self.pattern_counts_}")