"""
Affine (XOR) search via integral images (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 1192-1323.
O(1) per threshold evaluation via prefix-sum queries.
"""


import numpy as np
from numba import njit, prange

from gsnh_mdt.scoring.gain import information_gain
from gsnh_mdt.search.prefix import query_2d, query_3d


@njit(cache=True)
def fast_affine_2d(P_pos: np.ndarray, P_neg: np.ndarray,
                    total_pos: float, total_neg: float,
                    min_leaf: int) -> tuple[float, np.ndarray]:
    """O(1) 2D Affine Search using Integral Images.

    Checks ALL thresholds. Complexity: O(ni * nj) instead of O(ni * nj * N).
    """
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    total = total_pos + total_neg
    best_gain = -1.0
    best_result = np.array([0, 0, 0], dtype=np.int64)

    for ti in range(1, ni):
        for tj in range(1, nj):
            pos_1 = query_2d(P_pos, 0, ti, tj, nj)
            pos_2 = query_2d(P_pos, ti, ni, 0, tj)
            xor_pos = pos_1 + pos_2

            neg_1 = query_2d(P_neg, 0, ti, tj, nj)
            neg_2 = query_2d(P_neg, ti, ni, 0, tj)
            xor_neg = neg_1 + neg_2

            # XOR=True as positive pattern
            in_pos = xor_pos
            in_neg = xor_neg
            in_total = in_pos + in_neg
            out_total = total - in_total

            if in_total >= min_leaf and out_total >= min_leaf:
                gain = information_gain(total_pos, total_neg, in_pos, in_neg)
                if gain > best_gain and gain > 0:
                    best_gain = gain
                    best_result[0] = ti
                    best_result[1] = tj
                    best_result[2] = 0

            # XOR=False (XNOR) as positive pattern
            inv_pos = total_pos - xor_pos
            inv_neg = total_neg - xor_neg
            inv_total = inv_pos + inv_neg
            out_inv = total - inv_total

            if inv_total >= min_leaf and out_inv >= min_leaf:
                gain = information_gain(total_pos, total_neg, inv_pos, inv_neg)
                if gain > best_gain and gain > 0:
                    best_gain = gain
                    best_result[0] = ti
                    best_result[1] = tj
                    best_result[2] = 1

    return best_gain, best_result


@njit(cache=True, parallel=True)
def fast_affine_3d(P_pos: np.ndarray, P_neg: np.ndarray,
                    total_pos: float, total_neg: float,
                    min_leaf: int, step: int) -> tuple[float, np.ndarray]:
    """O(1) 3D Affine Search using Integral Images.

    Checks ALL thresholds. Complexity: O(ni * nj * nk) instead of O(ni * nj * nk * N).
    """
    ni = P_pos.shape[0] - 1
    nj = P_pos.shape[1] - 1
    nk = P_pos.shape[2] - 1
    total = total_pos + total_neg

    gains = np.full((ni + 1, nj + 1, nk + 1, 2), -1.0, dtype=np.float32)

    iters_i = ni // step

    for ii in prange(iters_i):
        ti = step + ii * step
        for tj in range(step, nj + 1, step):
            for tk in range(step, nk + 1, step):
                p1 = query_3d(P_pos, ti, ni, tj, nj, 0, tk)
                p2 = query_3d(P_pos, ti, ni, 0, tj, tk, nk)
                p3 = query_3d(P_pos, 0, ti, tj, nj, tk, nk)
                p4 = query_3d(P_pos, 0, ti, 0, tj, 0, tk)
                xor_pos = p1 + p2 + p3 + p4

                n1 = query_3d(P_neg, ti, ni, tj, nj, 0, tk)
                n2 = query_3d(P_neg, ti, ni, 0, tj, tk, nk)
                n3 = query_3d(P_neg, 0, ti, tj, nj, tk, nk)
                n4 = query_3d(P_neg, 0, ti, 0, tj, 0, tk)
                xor_neg = n1 + n2 + n3 + n4

                # XOR=True
                in_total = xor_pos + xor_neg
                out_total = total - in_total
                if in_total >= min_leaf and out_total >= min_leaf:
                    gain = information_gain(total_pos, total_neg, xor_pos, xor_neg)
                    gains[ti, tj, tk, 0] = gain

                # XOR=False (XNOR)
                inv_pos = total_pos - xor_pos
                inv_neg = total_neg - xor_neg
                inv_total = inv_pos + inv_neg
                out_inv = total - inv_total
                if inv_total >= min_leaf and out_inv >= min_leaf:
                    gain = information_gain(total_pos, total_neg, inv_pos, inv_neg)
                    gains[ti, tj, tk, 1] = gain

    flat_idx = np.argmax(gains)
    best_gain = gains.ravel()[flat_idx]

    if best_gain <= 0:
        return -1.0, np.zeros(4, dtype=np.int64)

    idx_type = flat_idx % 2
    rem = flat_idx // 2
    idx_k = rem % (nk + 1)
    rem //= (nk + 1)
    idx_j = rem % (nj + 1)
    idx_i = rem // (nj + 1)

    ti, tj, tk = idx_i, idx_j, idx_k

    res = np.array([ti, tj, tk, idx_type], dtype=np.int64)
    return float(best_gain), res
