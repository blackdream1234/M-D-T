"""
Exhaustive 1D GSNH search (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 877-911.
"""


import numpy as np
from numba import njit

from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.search.prefix import query_1d


@njit(cache=True)
def search_1d_exhaustive(P_pos: np.ndarray, P_neg: np.ndarray,
                          total_pos: float, total_neg: float,
                          min_leaf: int) -> tuple[float, np.ndarray]:
    """Exhaustive 1D GSNH search: 2 patterns (NEG, POS)."""
    n = P_pos.shape[0] - 1
    total = total_pos + total_neg

    best_gain = -1.0
    best_result = np.array([0, n, 1], dtype=np.int64)

    for anchor in range(2):
        for t in range(1, n + 1):
            if anchor == 0:
                lo, hi = 0, t
            else:
                lo, hi = t, n

            in_pos = query_1d(P_pos, lo, hi)
            in_neg = query_1d(P_neg, lo, hi)
            in_total = in_pos + in_neg
            out_total = total - in_total

            if in_total < min_leaf or out_total < min_leaf:
                continue

            gain = information_gain(total_pos, total_neg, in_pos, in_neg)

            if gain > best_gain and gain > 0:
                best_gain = gain
                best_result[0] = lo
                best_result[1] = hi
                best_result[2] = anchor

    return best_gain, best_result
