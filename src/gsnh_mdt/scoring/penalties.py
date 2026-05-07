"""
BIC-based penalized gain for anti-overfitting (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 569-599.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def penalized_gain(raw_gain: float, arity: int, n_bins: int,
                    n_samples: int, n_classes: int) -> float:
    """BIC-based penalized gain.

    Uses Bayesian Information Criterion:
      penalty = k * log(N) / (2 * N)

    where k = degrees of freedom (arity for threshold splits).

    Advantages over ad-hoc MDL heuristic:
      - Naturally scales with N (no over-penalization for small nodes)
      - Handles class imbalance through the entropy terms in raw_gain
      - Asymptotically selects the true model
      - O(1) computation
    """
    if raw_gain <= 0 or n_samples <= 0:
        return -1.0

    # Degrees of freedom: arity thresholds + pattern selection
    k = float(arity) + 1.0

    # BIC penalty: k * ln(N) / (2N)
    bic_penalty = k * np.log(max(float(n_samples), 2.0)) / (2.0 * float(n_samples))

    penalized = raw_gain - bic_penalty

    if penalized <= 0.0:
        return -1.0

    return penalized
