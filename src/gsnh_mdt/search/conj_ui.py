"""
ConjUI (Conjunction of Unary Intervals) exhaustive search kernels.

Pure AND / box / intersection search — formerly called "SquareCNF" in
early experiments.  This is NOT the paper's square 2CNF language.

ConjUI predicates have AND semantics:
    l₁ ∧ l₂ ∧ l₃
meaning the intersection (box) of half-spaces, not their union.

The search uses prefix sums with direct box queries (query_2d / query_3d)
instead of inclusion-exclusion union counts.  All 2^d polarity combinations
are valid since there is no Horn/AntiHorn restriction.
"""

import numpy as np
from numba import njit, prange

from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.search.prefix import query_2d, query_3d


@njit(cache=True)
def search_2d_conj_ui(P_pos: np.ndarray, P_neg: np.ndarray,
                      total_pos: float, total_neg: float,
                      min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """Exhaustive 2D ConjUI search: all 4 polarity configurations.

    Unlike Horn (3 configs, OR/union), ConjUI searches all 4 polarity
    combos with AND/intersection semantics:
        (0,0): x[i] < t_i  AND  x[j] < t_j    (box: low-low)
        (0,1): x[i] < t_i  AND  x[j] >= t_j   (box: low-high)
        (1,0): x[i] >= t_i AND  x[j] < t_j     (box: high-low)
        (1,1): x[i] >= t_i AND  x[j] >= t_j    (box: high-high)

    The in-count is computed via direct box query (intersection),
    not via inclusion-exclusion (union).
    """
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    total = total_pos + total_neg

    best_gain = -1.0
    best_result = np.array([0, ni, 0, nj, 1, 1], dtype=np.int64)

    for config in range(4):
        if config == 0:
            ai, aj = 0, 0
        elif config == 1:
            ai, aj = 0, 1
        elif config == 2:
            ai, aj = 1, 0
        else:
            ai, aj = 1, 1

        for ti in range(step, ni + 1, step):
            i1 = 0 if ai == 0 else ti
            i2 = ti if ai == 0 else ni

            for tj in range(step, nj + 1, step):
                j1 = 0 if aj == 0 else tj
                j2 = tj if aj == 0 else nj

                # AND semantics: intersection = direct box query
                in_pos = query_2d(P_pos, i1, i2, j1, j2)
                in_neg = query_2d(P_neg, i1, i2, j1, j2)
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
def search_3d_conj_ui(P_pos: np.ndarray, P_neg: np.ndarray,
                      total_pos: float, total_neg: float,
                      min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """Exhaustive 3D ConjUI search: all 8 polarity configurations.

    AND/intersection semantics across all 2^3 = 8 polarity combos.
    Uses direct box query_3d instead of count_3way_union.
    """
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    nk = P_pos.shape[2] - 1
    total = total_pos + total_neg

    gains = np.full((ni + 1, nj + 1, nk + 1, 8), -1.0, dtype=np.float32)

    iters_i = ni // step

    for ii in prange(iters_i):
        ti = step + ii * step
        for tj in range(step, nj + 1, step):
            for tk in range(step, nk + 1, step):
                for config in range(8):
                    ai = (config >> 2) & 1
                    aj = (config >> 1) & 1
                    ak = config & 1

                    i1 = 0 if ai == 0 else ti
                    i2 = ti if ai == 0 else ni
                    j1 = 0 if aj == 0 else tj
                    j2 = tj if aj == 0 else nj
                    k1 = 0 if ak == 0 else tk
                    k2 = tk if ak == 0 else nk

                    # AND semantics: direct box intersection
                    in_pos = query_3d(P_pos, i1, i2, j1, j2, k1, k2)
                    in_neg = query_3d(P_neg, i1, i2, j1, j2, k1, k2)
                    in_total = in_pos + in_neg
                    out_total = total - in_total

                    if in_total >= min_leaf and out_total >= min_leaf:
                        gain = information_gain(total_pos, total_neg,
                                                in_pos, in_neg)
                        gains[ti, tj, tk, config] = gain

    flat_idx = np.argmax(gains)
    best_gain = gains.ravel()[flat_idx]

    if best_gain <= 0:
        return -1.0, np.zeros(9, dtype=np.int64)

    idx_config = flat_idx % 8
    rem = flat_idx // 8
    idx_k = rem % (nk + 1)
    rem //= (nk + 1)
    idx_j = rem % (nj + 1)
    idx_i = rem // (nj + 1)

    ti, tj, tk = idx_i, idx_j, idx_k
    config = idx_config

    ai = (config >> 2) & 1
    aj = (config >> 1) & 1
    ak = config & 1

    i1 = 0 if ai == 0 else ti
    i2 = ti if ai == 0 else ni
    j1 = 0 if aj == 0 else tj
    j2 = tj if aj == 0 else nj
    k1 = 0 if ak == 0 else tk
    k2 = tk if ak == 0 else nk

    res = np.array([i1, i2, j1, j2, k1, k2, ai, aj, ak], dtype=np.int64)
    return float(best_gain), res
