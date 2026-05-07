"""
JIT-compiled histogram tensor builders for 1D, 2D, and 3D searches.

Extracted verbatim from gsnh_mdt_v3.py lines 640-702.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def jit_build_tensors_1d(y, bins, n_bins):
    """Build 1D pos/neg histograms from binned feature indices."""
    pos = np.zeros(n_bins, dtype=np.float64)
    neg = np.zeros(n_bins, dtype=np.float64)
    for idx in range(len(y)):
        b = bins[idx]
        if b >= n_bins:
            b = n_bins - 1
        if y[idx] == 1:
            pos[b] += 1.0
        else:
            neg[b] += 1.0
    return pos, neg


@njit(cache=True)
def jit_build_tensors_2d(y, bi, bj, ni, nj):
    """Build 2D joint pos/neg histograms."""
    pos = np.zeros((ni, nj), dtype=np.float64)
    neg = np.zeros((ni, nj), dtype=np.float64)
    for idx in range(len(y)):
        i = bi[idx]
        if i >= ni:
            i = ni - 1
        j = bj[idx]
        if j >= nj:
            j = nj - 1
        if y[idx] == 1:
            pos[i, j] += 1.0
        else:
            neg[i, j] += 1.0
    return pos, neg


@njit(cache=True)
def jit_build_tensors_3d(y, bi, bj, bk, ni, nj, nk):
    """Build 3D joint pos/neg histograms — FLATTENED for cache locality.

    Uses 1D array with manual index arithmetic: flat[i*nj*nk + j*nk + k].
    CPU reads contiguous memory sequentially, avoiding cache misses.
    Final reshape is a zero-cost view operation.
    """
    total = ni * nj * nk
    pos_flat = np.zeros(total, dtype=np.float64)
    neg_flat = np.zeros(total, dtype=np.float64)
    njk = nj * nk  # pre-compute stride
    for idx in range(len(y)):
        i = bi[idx]
        if i >= ni:
            i = ni - 1
        j = bj[idx]
        if j >= nj:
            j = nj - 1
        k = bk[idx]
        if k >= nk:
            k = nk - 1
        flat_idx = i * njk + j * nk + k
        if y[idx] == 1:
            pos_flat[flat_idx] += 1.0
        else:
            neg_flat[flat_idx] += 1.0
    return pos_flat.reshape((ni, nj, nk)), neg_flat.reshape((ni, nj, nk))
