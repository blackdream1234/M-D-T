"""
GSNHGradientBoosting: Gradient Boosting with GSNH-aware weak learners.

Extracted verbatim from gsnh_mdt_v3.py lines 3445-3732.
"""

import numpy as np
from typing import Optional

from gsnh_mdt.types import LanguageFamily


class GSNHGradientBoosting:
    """
    Gradient Boosting with GSNH-aware weak learners.
    FIX #6: Uses regression stumps instead of converting residuals to binary.
    FIX #7: Enforces minimum number of iterations.
    """

    def __init__(self,
                 n_estimators: int = 200,
                 learning_rate: float = 0.05,
                 max_depth: int = 5,
                 subsample: float = 0.8,
                 colsample: float = 0.8,
                 min_samples_leaf: int = 10,
                 early_stopping_rounds: int = 20,
                 validation_fraction: float = 0.15,
                 min_iterations: int = 10,
                 l2_reg: float = 0.1,
                 lr_decay: float = 0.995,
                 random_state: int = 42,
                 mode: str = 'heuristic',
                 language: LanguageFamily = LanguageFamily.ANY):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample = colsample
        self.min_samples_leaf = min_samples_leaf
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.min_iterations = min_iterations
        self.l2_reg = l2_reg
        self.lr_decay = lr_decay
        self.random_state = random_state
        self.mode = mode
        self.language = language

        self.stumps_ = []
        self.stump_weights_ = []
        self.init_pred_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        self.best_iteration_ = None
        self.feature_importances_ = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _log_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred)
                        + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.feature_importances_ = np.zeros(n_features)

        # Validation split
        val_size = int(n_samples * self.validation_fraction)
        indices = np.random.permutation(n_samples)

        if val_size > 0:
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Initialize with log-odds
        pos_rate = np.clip(y_train.mean(), 1e-7, 1 - 1e-7)
        self.init_pred_ = np.log(pos_rate / (1 - pos_rate))

        F_train = np.full(len(y_train), self.init_pred_)
        if X_val is not None:
            F_val = np.full(len(y_val), self.init_pred_)

        best_val_loss = float('inf')
        rounds_no_improve = 0
        current_lr = self.learning_rate

        print(f"Training GSNH Gradient Boosting "
              f"(n={self.n_estimators}, lr={self.learning_rate})...")

        for i in range(self.n_estimators):
            # Compute pseudo-residuals
            p_train = self._sigmoid(F_train)
            residuals = y_train - p_train

            # L2 regularization
            residuals = residuals / (1 + self.l2_reg)

            # Row subsample
            if self.subsample < 1.0:
                n_sub = int(len(y_train) * self.subsample)
                row_idx = np.random.choice(
                    len(y_train), size=n_sub, replace=False
                )
            else:
                row_idx = np.arange(len(y_train))

            # Column subsample
            if self.colsample < 1.0:
                n_cols = max(1, int(n_features * self.colsample))
                col_idx = np.random.choice(
                    n_features, size=n_cols, replace=False
                )
            else:
                col_idx = np.arange(n_features)

            X_sub = X_train[row_idx][:, col_idx]
            r_sub = residuals[row_idx]

            # FIX #6: Fit regression stump directly on residuals
            stump = self._fit_stump(X_sub, r_sub, col_idx)

            if stump is None:
                continue

            self.stumps_.append(stump)
            self.stump_weights_.append(current_lr)

            # Update predictions
            pred_train = self._predict_stump(stump, X_train)
            F_train += current_lr * pred_train

            # Training loss
            train_loss = self._log_loss(y_train, self._sigmoid(F_train))
            self.train_losses_.append(train_loss)

            # Validation
            if X_val is not None:
                pred_val = self._predict_stump(stump, X_val)
                F_val += current_lr * pred_val
                val_loss = self._log_loss(y_val, self._sigmoid(F_val))
                self.val_losses_.append(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    self.best_iteration_ = i
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                # FIX #7: Enforce minimum iterations before early stopping
                if (rounds_no_improve >= self.early_stopping_rounds
                        and i >= self.min_iterations):
                    print(f"  Early stopping at round {i + 1}")
                    break

            # Learning rate decay
            current_lr *= self.lr_decay

            if (i + 1) % 25 == 0:
                msg = f"  Round {i + 1}: train_loss={train_loss:.4f}"
                if X_val is not None:
                    msg += f", val_loss={val_loss:.4f}, lr={current_lr:.5f}"
                print(msg)

        # FIX #7: Ensure best_iteration is reasonable
        if self.best_iteration_ is None:
            self.best_iteration_ = len(self.stumps_) - 1
        else:
            self.best_iteration_ = max(
                self.min_iterations - 1,
                self.best_iteration_
            )
            self.best_iteration_ = min(
                self.best_iteration_,
                len(self.stumps_) - 1
            )

        # Normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        print(f"Training complete! Best iteration: {self.best_iteration_ + 1}")
        return self

    def _fit_stump(self, X, residuals, col_idx):
        """
        FIX #6: Fit a regression stump directly to continuous residuals.
        No binary conversion — preserves residual magnitude.
        """
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_leaf * 2:
            return None

        best_reduction = -float('inf')
        best_feature = None
        best_threshold = None

        total_sum = residuals.sum()
        total_n = len(residuals)

        for f in range(n_features):
            sorted_idx = np.argsort(X[:, f])
            sorted_r = residuals[sorted_idx]
            sorted_x = X[sorted_idx, f]

            left_sum = 0.0
            left_n = 0

            for j in range(self.min_samples_leaf - 1,
                           n_samples - self.min_samples_leaf):
                left_sum += sorted_r[j]
                left_n += 1

                # Skip identical feature values
                if j < n_samples - 1 and sorted_x[j] == sorted_x[j + 1]:
                    continue

                right_sum = total_sum - left_sum
                right_n = total_n - left_n

                if right_n < self.min_samples_leaf:
                    break

                # Variance reduction
                left_mean = left_sum / left_n
                right_mean = right_sum / right_n
                overall_mean = total_sum / total_n

                reduction = (
                    left_n * (left_mean - overall_mean) ** 2
                    + right_n * (right_mean - overall_mean) ** 2
                )

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = f
                    if j < n_samples - 1:
                        best_threshold = (sorted_x[j] + sorted_x[j + 1]) / 2
                    else:
                        best_threshold = sorted_x[j]

        if best_feature is None:
            return None

        # Compute leaf values
        mask = X[:, best_feature] <= best_threshold
        left_val = residuals[mask].mean() if mask.sum() > 0 else 0.0
        right_val = residuals[~mask].mean() if (~mask).sum() > 0 else 0.0

        # Track feature importance
        global_feat = col_idx[best_feature]
        self.feature_importances_[global_feat] += best_reduction

        return {
            'feature': global_feat,
            'threshold': best_threshold,
            'left_value': left_val,
            'right_value': right_val,
        }

    def _predict_stump(self, stump, X):
        mask = X[:, stump['feature']] <= stump['threshold']
        return np.where(mask, stump['left_value'], stump['right_value'])

    def predict_proba(self, X, use_best=True):
        X = np.asarray(X, dtype=np.float64)

        n_stumps = (self.best_iteration_ + 1
                    if use_best
                    else len(self.stumps_))

        F = np.full(len(X), self.init_pred_)

        for i in range(min(n_stumps, len(self.stumps_))):
            F += self.stump_weights_[i] * self._predict_stump(
                self.stumps_[i], X
            )

        probas = self._sigmoid(F)
        return np.column_stack([1 - probas, probas])

    def predict(self, X, use_best=True):
        return (self.predict_proba(X, use_best)[:, 1] >= 0.5).astype(int)
