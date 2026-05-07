"""
GSNHClassifier: complete GSNH classifier pipeline.

Extracted verbatim from gsnh_mdt_v3.py lines 3739-3919.
"""

import warnings
import numpy as np
from typing import Optional

from gsnh_mdt.types import LanguageFamily
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.tree.pruning import CostComplexityPruner
from gsnh_mdt.tree.calibration import ProbabilityCalibrator
from gsnh_mdt.ensembles.random_forest import GSNHRandomForest
from gsnh_mdt.ensembles.gradient_boosting import GSNHGradientBoosting


class GSNHClassifier:
    """
    Complete GSNH classifier pipeline:
    - Single tree / Random Forest / Gradient Boosting
    - Automatic model selection
    - Probability calibration
    - Post-pruning for single trees
    """

    def __init__(self,
                 model_type: str = 'auto',
                 n_bins: int = 64,
                 max_depth: int = 15,
                 min_samples_leaf: int = 5,
                 n_estimators: int = 50,
                 learning_rate: float = 0.05,
                 use_calibration: bool = True,
                 calibration_method: str = 'platt',
                 use_pruning: bool = True,
                 pruning_alpha: float = 0.01,
                 random_state: int = 42,
                 verbose: bool = True,
                 mode: str = 'heuristic',
                 language: LanguageFamily = LanguageFamily.ANY,
                 theorem_strict: bool = False):

        self.model_type = model_type
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.use_pruning = use_pruning
        self.pruning_alpha = pruning_alpha
        self.random_state = random_state
        self.verbose = verbose
        self.mode = mode
        self.language = language
        self.theorem_strict = theorem_strict

        self.model_ = None
        self.calibrator_ = None
        self.selected_model_type_ = None

    def _select_model_type(self, X, y):
        n_samples = len(y)
        if self.model_type != 'auto':
            return self.model_type
        if n_samples < 500:
            return 'single'
        elif n_samples < 2000:
            return 'forest'
        return 'boosting'

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        if self.mode == 'journal' and self.language == LanguageFamily.ANY:
            raise ValueError(
                "Journal mode requires an explicit fixed language or certified mixed mode; "
                "language=ANY is not allowed."
            )

        np.random.seed(self.random_state)

        # Split for calibration/pruning
        n = len(y)
        indices = np.random.permutation(n)

        if self.use_calibration or self.use_pruning:
            cal_size = int(n * 0.15)
            cal_idx = indices[:cal_size]
            train_idx = indices[cal_size:]
            X_train, X_cal = X[train_idx], X[cal_idx]
            y_train, y_cal = y[train_idx], y[cal_idx]
        else:
            X_train, y_train = X, y
            X_cal, y_cal = None, None

        # Select model type
        self.selected_model_type_ = self._select_model_type(X_train, y_train)

        if self.verbose:
            print(f"Training {self.selected_model_type_} model...")

        # Create model
        self.model_ = self._create_model()

        # Train
        self.model_.fit(X_train, y_train)

        # Post-pruning for single trees
        if (self.use_pruning
                and self.selected_model_type_ == 'single'
                and X_cal is not None
                and len(X_cal) > 0):
            if self.verbose:
                print("Applying cost-complexity pruning...")
            pruner = CostComplexityPruner(alpha=self.pruning_alpha)
            self.model_.root_ = pruner.prune(
                self.model_.root_, X_cal, y_cal
            )

        # Probability calibration
        if self.use_calibration and X_cal is not None and len(X_cal) > 0:
            if self.verbose:
                print(f"Calibrating probabilities ({self.calibration_method})...")
            probas = self.model_.predict_proba(X_cal)[:, 1]
            self.calibrator_ = ProbabilityCalibrator(
                method=self.calibration_method
            )
            self.calibrator_.fit(probas, y_cal)

        if self.verbose:
            print("Training complete!")

        return self

    def _create_model(self):
        if self.selected_model_type_ == 'single':
            stopping = StoppingCriteria(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            return ExpertGSNHTree(
                stopping_criteria=stopping,
                n_bins=self.n_bins,
                mode=self.mode,
                language=self.language,
                theorem_strict=self.theorem_strict
            )

        elif self.selected_model_type_ == 'forest':
            tree_stopping = StoppingCriteria(
                max_depth=min(self.max_depth, 10),
                min_samples_leaf=self.min_samples_leaf,
            )
            return GSNHRandomForest(
                n_estimators=self.n_estimators,
                tree_params={
                    'stopping_criteria': tree_stopping,
                    'n_bins': min(self.n_bins, 40),
                },
                random_state=self.random_state,
                mode=self.mode,
                language=self.language
            )

        elif self.selected_model_type_ == 'boosting':
            return GSNHGradientBoosting(
                n_estimators=self.n_estimators * 2,
                learning_rate=self.learning_rate,
                max_depth=min(self.max_depth, 5),
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                mode=self.mode,
                language=self.language
            )

        raise ValueError(f"Unknown model type: {self.selected_model_type_}")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probas = self.model_.predict_proba(X)

        if self.calibrator_ is not None:
            calibrated = self.calibrator_.calibrate(probas[:, 1])
            probas = np.column_stack([1 - calibrated, calibrated])

        return probas

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def extract_axp(self, x: np.ndarray) -> set:
        """Extract a single minimal AXp (only supported for single trees)."""
        if self.selected_model_type_ == 'single':
            return self.model_.extract_axp(x)
        else:
            raise NotImplementedError("AXp extraction is currently only supported for single trees.")

    def score(self, X, y):
        return float((self.predict(X) == y).mean())
