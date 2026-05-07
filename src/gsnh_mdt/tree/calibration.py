"""
ProbabilityCalibrator: Platt scaling and isotonic calibration.

Extracted verbatim from gsnh_mdt_v3.py lines 1816-1891.
ENHANCED feature — not part of baseline journal logic.
"""

import numpy as np


class ProbabilityCalibrator:
    """Platt scaling or binned isotonic calibration."""

    def __init__(self, method: str = 'platt'):
        self.method = method
        self.calibrator_ = None

    def fit(self, probas: np.ndarray, y_true: np.ndarray):
        """Fit the calibrator on held-out data."""
        if self.method == 'platt':
            self._fit_platt(probas, y_true)
        elif self.method == 'isotonic':
            self._fit_isotonic(probas, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def _fit_platt(self, probas: np.ndarray, y_true: np.ndarray):
        """Platt scaling: fit sigmoid a*x + b."""
        # Clip to avoid log(0)
        p = np.clip(probas, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1 - p))

        def nll(params):
            a, b = params
            z = a * logits + b
            z = np.clip(z, -30, 30)
            sigma = 1 / (1 + np.exp(-z))
            sigma = np.clip(sigma, 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(sigma) + (1 - y_true) * np.log(1 - sigma))

        from scipy.optimize import minimize
        result = minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead')
        self.calibrator_ = ('platt', result.x[0], result.x[1])

    def _fit_isotonic(self, probas: np.ndarray, y_true: np.ndarray):
        """Isotonic regression via binning."""
        n_bins = min(20, len(probas) // 10)
        if n_bins < 2:
            self.calibrator_ = ('identity',)
            return

        sorted_idx = np.argsort(probas)
        bin_size = len(probas) // n_bins

        bin_means = []
        bin_true_rates = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(probas)
            idx = sorted_idx[start:end]
            bin_means.append(np.mean(probas[idx]))
            bin_true_rates.append(np.mean(y_true[idx]))

        self.calibrator_ = ('isotonic', np.array(bin_means), np.array(bin_true_rates))

    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Apply calibration to predicted probabilities."""
        if self.calibrator_ is None:
            return probas

        kind = self.calibrator_[0]

        if kind == 'identity':
            return probas

        if kind == 'platt':
            _, a, b = self.calibrator_
            p = np.clip(probas, 1e-7, 1 - 1e-7)
            logits = np.log(p / (1 - p))
            z = a * logits + b
            z = np.clip(z, -30, 30)
            return 1 / (1 + np.exp(-z))

        if kind == 'isotonic':
            _, bin_means, bin_true_rates = self.calibrator_
            return np.interp(probas, bin_means, bin_true_rates)

        return probas
