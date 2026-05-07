"""
Anti-Horn exhaustive search kernels (JIT-compiled).

Anti-Horn constraint: at most 1 negative literal per clause.
Extracted verbatim from gsnh_mdt_v3.py lines 1049-1185.
"""


import numpy as np
from numba import njit, prange

from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.search.prefix import count_2way_union, count_3way_union


@njit(cache=True)
def search_2d_antihorn(P_pos: np.ndarray, P_neg: np.ndarray,
                        total_pos: float, total_neg: float,
                        min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """Exhaustive 2D Anti-Horn search: 3 patterns with at most 1 negative literal."""
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    total = total_pos + total_neg

    best_gain = -1.0
    best_result = np.array([0, ni, 0, nj, 0, 0], dtype=np.int64)

    for config in range(3):
        if config == 0:
            ai, aj = 1, 1  # All positive
        elif config == 1:
            ai, aj = 1, 0  # First POS, second NEG
        else:
            ai, aj = 0, 1  # First NEG, second POS

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


@njit(cache=True, parallel=True)
def search_3d_antihorn(P_pos: np.ndarray, P_neg: np.ndarray,
                        total_pos: float, total_neg: float,
                        min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """Exhaustive 3D Anti-Horn search: 4 patterns with at most 1 negative literal."""
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    nk = P_pos.shape[2] - 1
    total = total_pos + total_neg

    gains = np.full((ni + 1, nj + 1, nk + 1, 4), -1.0, dtype=np.float32)

    iters_i = ni // step

    for ii in prange(iters_i):
        ti = step + ii * step
        for tj in range(step, nj + 1, step):
            for tk in range(step, nk + 1, step):
                for config in range(4):
                    if config == 0:
                        ai, aj, ak = 1, 1, 1  # All positive
                    elif config == 1:
                        ai, aj, ak = 0, 1, 1  # First NEG only
                    elif config == 2:
                        ai, aj, ak = 1, 0, 1  # Second NEG only
                    else:
                        ai, aj, ak = 1, 1, 0  # Third NEG only

                    i1 = 0 if ai == 0 else ti
                    i2 = ti if ai == 0 else ni
                    j1 = 0 if aj == 0 else tj
                    j2 = tj if aj == 0 else nj
                    k1 = 0 if ak == 0 else tk
                    k2 = tk if ak == 0 else nk

                    in_pos = count_3way_union(P_pos, i1, i2, j1, j2, k1, k2)
                    in_neg = count_3way_union(P_neg, i1, i2, j1, j2, k1, k2)
                    in_total = in_pos + in_neg
                    out_total = total - in_total

                    if in_total >= min_leaf and out_total >= min_leaf:
                        gain = information_gain(total_pos, total_neg, in_pos, in_neg)
                        gains[ti, tj, tk, config] = gain

    flat_idx = np.argmax(gains)
    best_gain = gains.ravel()[flat_idx]

    if best_gain <= 0:
        return -1.0, np.zeros(9, dtype=np.int64)

    idx_config = flat_idx % 4
    rem = flat_idx // 4
    idx_k = rem % (nk + 1)
    rem //= (nk + 1)
    idx_j = rem % (nj + 1)
    idx_i = rem // (nj + 1)

    ti, tj, tk = idx_i, idx_j, idx_k
    config = idx_config

    if config == 0:
        ai, aj, ak = 1, 1, 1
    elif config == 1:
        ai, aj, ak = 0, 1, 1
    elif config == 2:
        ai, aj, ak = 1, 0, 1
    else:
        ai, aj, ak = 1, 1, 0

    i1 = 0 if ai == 0 else ti
    i2 = ti if ai == 0 else ni
    j1 = 0 if aj == 0 else tj
    j2 = tj if aj == 0 else nj
    k1 = 0 if ak == 0 else tk
    k2 = tk if ak == 0 else nk

    res = np.array([i1, i2, j1, j2, k1, k2, ai, aj, ak], dtype=np.int64)
    return float(best_gain), res
