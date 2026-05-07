"""
Exhaustive 2D GSNH search (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 914-961.
"""


import numpy as np
from numba import njit

from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.search.prefix import count_2way_union


@njit(cache=True)
def search_2d_exhaustive(P_pos: np.ndarray, P_neg: np.ndarray,
                          total_pos: float, total_neg: float,
                          min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """Exhaustive 2D GSNH search: 3 Horn patterns."""
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    total = total_pos + total_neg

    best_gain = -1.0
    best_result = np.array([0, ni, 0, nj, 1, 1], dtype=np.int64)

    for config in range(3):
        if config == 0:
            ai, aj = 0, 0
        elif config == 1:
            ai, aj = 0, 1
        else:
            ai, aj = 1, 0

        for ti in range(step, ni + 1, step):
            i1 = 0 if ai == 0 else ti
            i2 = ti if ai == 0 else ni

            for tj in range(step, nj + 1, step):
                j1 = 0 if aj == 0 else tj
                j2 = tj if aj == 0 else nj

                in_pos = count_2way_union(P_pos, i1, i2, j1, j2)
                in_neg = count_2way_union(P_neg, i1, i2, j1, j2)
                in_total = in_pos + in_neg
                out_total = total - in_total

                if in_total < min_leaf or out_total < min_leaf:
                    continue

                gain = information_gain(total_pos, total_neg, in_pos, in_neg)

                if gain > best_gain and gain > 0:
                    best_gain = gain
                    best_result[0] = i1
                    best_result[1] = i2
                    best_result[2] = j1
                    best_result[3] = j2
                    best_result[4] = ai
                    best_result[5] = aj

    return best_gain, best_result
