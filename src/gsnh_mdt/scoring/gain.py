"""
Information gain, gain ratio, and related scoring kernels (JIT-compiled).

Extracted verbatim from gsnh_mdt_v3.py lines 520-562 and 602-753.

Note: fast_hist_gain and jit_ig_scores are ENHANCED features used by the
look-ahead module, but are self-contained scoring computations.
"""


import numpy as np
from numba import njit

from gsnh_mdt.scoring.entropy import entropy


@njit(cache=True)
def information_gain(total_pos: float, total_neg: float,
                     in_pos: float, in_neg: float) -> float:
    """Information gain for a binary split.

    IG = H(parent) - (n_in/n)*H(in) - (n_out/n)*H(out)
    """
    total = total_pos + total_neg
    in_total = in_pos + in_neg
    out_total = total - in_total

    if in_total <= 0 or out_total <= 0 or total <= 0:
        return -1.0

    out_pos = total_pos - in_pos
    out_neg = total_neg - in_neg

    H_parent = entropy(total_pos, total_neg)
    H_in = entropy(in_pos, in_neg)
    H_out = entropy(out_pos, out_neg)

    gain = H_parent - (in_total / total) * H_in - (out_total / total) * H_out
    return max(0.0, gain)


@njit(cache=True)
def gain_ratio(total_pos: float, total_neg: float,
               in_pos: float, in_neg: float) -> float:
    """Gain ratio: IG normalized by split information."""
    ig = information_gain(total_pos, total_neg, in_pos, in_neg)
    if ig <= 0:
        return -1.0

    total = total_pos + total_neg
    in_total = in_pos + in_neg
    out_total = total - in_total

    if total <= 0 or in_total <= 0 or out_total <= 0:
        return -1.0

    p_in = in_total / total
    p_out = out_total / total
    split_info = -(p_in * np.log2(p_in) + p_out * np.log2(p_out))

    if split_info <= 1e-10:
        return ig

    return ig / split_info


@njit(cache=True)
def fast_hist_gain(bins, y, n_bins, total_pos, total_neg, min_leaf):
    """Best 1D gain for a single feature from pre-computed bin indices.

    Enhanced feature: used by look-ahead to scan children using parent's
    bin arrays without re-binning. Pure JIT, zero Python overhead.
    """
    n = len(y)
    pos_hist = np.zeros(n_bins, dtype=np.float64)
    neg_hist = np.zeros(n_bins, dtype=np.float64)

    for i in range(n):
        b = bins[i]
        if b >= n_bins:
            b = n_bins - 1
        if y[i] == 1:
            pos_hist[b] += 1.0
        else:
            neg_hist[b] += 1.0

    cum_pos = 0.0
    cum_neg = 0.0
    best_g = 0.0

    for i in range(n_bins - 1):
        cum_pos += pos_hist[i]
        cum_neg += neg_hist[i]
        left_n = cum_pos + cum_neg
        right_n = float(n) - left_n
        if left_n < min_leaf or right_n < min_leaf:
            continue
        g = information_gain(total_pos, total_neg, cum_pos, cum_neg)
        if g > best_g:
            best_g = g

    return best_g


@njit(cache=True)
def jit_ig_scores(bins_2d, y, n_features, n_bins_arr, min_leaf, use_gain_ratio):
    """JIT-compiled univariate IG scoring for all features.

    Args:
        bins_2d: (n_samples, n_features) int64 - pre-binned indices
        y: (n_samples,) int32
        n_bins_arr: (n_features,) int64
        min_leaf: minimum samples per leaf
        use_gain_ratio: if True, use gain ratio instead of IG
    """
    n = len(y)
    scores = np.zeros(n_features, dtype=np.float64)
    total_pos = 0.0
    for i in range(n):
        if y[i] == 1:
            total_pos += 1.0
    total_neg = float(n) - total_pos
    if total_pos == 0.0 or total_neg == 0.0:
        return scores

    for f in range(n_features):
        nb = n_bins_arr[f]
        if nb < 2:
            continue
        pos_hist = np.zeros(nb, dtype=np.float64)
        neg_hist = np.zeros(nb, dtype=np.float64)
        for i in range(n):
            b = bins_2d[i, f]
            if b >= nb:
                b = nb - 1
            if y[i] == 1:
                pos_hist[b] += 1.0
            else:
                neg_hist[b] += 1.0

        cum_pos = 0.0
        cum_neg = 0.0
        for s in range(nb - 1):
            cum_pos += pos_hist[s]
            cum_neg += neg_hist[s]
            left_n = cum_pos + cum_neg
            right_n = float(n) - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            if use_gain_ratio:
                g = gain_ratio(total_pos, total_neg, cum_pos, cum_neg)
            else:
                g = information_gain(total_pos, total_neg, cum_pos, cum_neg)
            if g > scores[f]:
                scores[f] = g
    return scores
