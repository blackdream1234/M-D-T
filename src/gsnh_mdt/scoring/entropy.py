"""
Binary entropy function (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 509-517.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def entropy(pos: float, neg: float) -> float:
    """Binary entropy H(pos, neg).

    Convention: 0 * log2(0) = 0.
    """
    total = pos + neg
    if total <= 0:
        return 0.0
    p = pos / total
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
