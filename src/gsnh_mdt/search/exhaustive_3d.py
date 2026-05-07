"""
Exhaustive 3D GSNH search (JIT-compiled, parallel).

Extracted verbatim from gsnh_mdt_v3.py lines 964-1042.
"""


import numpy as np
from numba import njit, prange

from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.search.prefix import count_3way_union


@njit(cache=True, parallel=True)
def search_3d_exhaustive(P_pos: np.ndarray, P_neg: np.ndarray,
                          total_pos: float, total_neg: float,
                          min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """Exhaustive 3D Unified Search (Horn: ONE NEGATIVE literal in body, head is POS)."""
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
                        ai, aj, ak = 0, 0, 0
                    elif config == 1:
                        ai, aj, ak = 1, 0, 0
                    elif config == 2:
                        ai, aj, ak = 0, 1, 0
                    else:
                        ai, aj, ak = 0, 0, 1

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
        ai, aj, ak = 0, 0, 0
    elif config == 1:
        ai, aj, ak = 1, 0, 0
    elif config == 2:
        ai, aj, ak = 0, 1, 0
    else:
        ai, aj, ak = 0, 0, 1

    i1 = 0 if ai == 0 else ti
    i2 = ti if ai == 0 else ni
    j1 = 0 if aj == 0 else tj
    j2 = tj if aj == 0 else nj
    k1 = 0 if ak == 0 else tk
    k2 = tk if ak == 0 else nk

    res = np.array([i1, i2, j1, j2, k1, k2, ai, aj, ak], dtype=np.int64)
    return float(best_gain), res
