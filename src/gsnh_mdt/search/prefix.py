"""
Prefix-sum builders, queries, and inclusion-exclusion union counts (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 760-870.
These are the core O(1) query primitives for exhaustive search.
"""

import numpy as np
from numba import njit

# =============================================================================
# PREFIX SUM BUILDERS
# =============================================================================

@njit(cache=True)
def build_1d_prefix(T: np.ndarray) -> np.ndarray:
    n = T.shape[0]
    P = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        P[i + 1] = P[i] + T[i]
    return P


@njit(cache=True)
def build_2d_prefix(T: np.ndarray) -> np.ndarray:
    ni, nj = T.shape
    P = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    for i in range(ni):
        row_sum = 0.0
        for j in range(nj):
            row_sum += T[i, j]
            P[i + 1, j + 1] = P[i, j + 1] + row_sum
    return P


@njit(cache=True)
def build_3d_prefix(T: np.ndarray) -> np.ndarray:
    ni, nj, nk = T.shape
    P = np.zeros((ni + 1, nj + 1, nk + 1), dtype=np.float64)
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                P[i + 1, j + 1, k + 1] = (
                    T[i, j, k]
                    + P[i, j + 1, k + 1] + P[i + 1, j, k + 1] + P[i + 1, j + 1, k]
                    - P[i + 1, j, k] - P[i, j + 1, k] - P[i, j, k + 1]
                    + P[i, j, k]
                )
    return P


# =============================================================================
# PREFIX SUM QUERIES
# =============================================================================

@njit(cache=True)
def query_1d(P: np.ndarray, lo: int, hi: int) -> float:
    return P[hi] - P[lo]


@njit(cache=True)
def query_2d(P: np.ndarray, i1: int, i2: int, j1: int, j2: int) -> float:
    return P[i2, j2] - P[i1, j2] - P[i2, j1] + P[i1, j1]


@njit(cache=True)
def query_3d(P: np.ndarray, i1: int, i2: int, j1: int, j2: int,
             k1: int, k2: int) -> float:
    return (P[i2, j2, k2] - P[i1, j2, k2] - P[i2, j1, k2] - P[i2, j2, k1]
            + P[i1, j1, k2] + P[i1, j2, k1] + P[i2, j1, k1]
            - P[i1, j1, k1])


# =============================================================================
# INCLUSION-EXCLUSION UNION COUNTS
# =============================================================================

@njit(cache=True)
def count_2way_union(P: np.ndarray,
                     i1: int, i2: int,
                     j1: int, j2: int) -> float:
    """
    |A union B| = |A| + |B| - |A intersect B|
    A = rows with feature_i in [i1, i2)
    B = rows with feature_j in [j1, j2)
    """
    ni = P.shape[0] - 1
    nj = P.shape[1] - 1
    A = query_2d(P, i1, i2, 0, nj)
    B = query_2d(P, 0, ni, j1, j2)
    AB = query_2d(P, i1, i2, j1, j2)
    return A + B - AB


@njit(cache=True)
def count_3way_union(P: np.ndarray,
                     i1: int, i2: int,
                     j1: int, j2: int,
                     k1: int, k2: int) -> float:
    """
    |A union B union C| = |A|+|B|+|C| - |AB| - |AC| - |BC| + |ABC|

    Fix #1: Each set sums over FULL range of OTHER dimensions.
    A = feature_i in [i1,i2) -> sum over ALL j (0..nj), ALL k (0..nk)
    B = feature_j in [j1,j2) -> sum over ALL i (0..ni), ALL k (0..nk)
    C = feature_k in [k1,k2) -> sum over ALL i (0..ni), ALL j (0..nj)
    """
    ni = P.shape[0] - 1
    nj = P.shape[1] - 1
    nk = P.shape[2] - 1
    A = query_3d(P, i1, i2, 0, nj, 0, nk)
    B = query_3d(P, 0, ni, j1, j2, 0, nk)
    C = query_3d(P, 0, ni, 0, nj, k1, k2)
    AB = query_3d(P, i1, i2, j1, j2, 0, nk)
    AC = query_3d(P, i1, i2, 0, nj, k1, k2)
    BC = query_3d(P, 0, ni, j1, j2, k1, k2)
    ABC = query_3d(P, i1, i2, j1, j2, k1, k2)
    return A + B + C - AB - AC - BC + ABC
