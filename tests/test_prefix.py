"""
Tests for prefix-sum builders and queries.

Validates that prefix sum construction and O(1) queries produce
correct results on small known examples.
"""

import numpy as np
import pytest

from gsnh_mdt.search.prefix import (
    build_1d_prefix, build_2d_prefix, build_3d_prefix,
    query_1d, query_2d, query_3d,
    count_2way_union, count_3way_union,
)


class TestPrefixSum1D:
    def test_simple(self):
        T = np.array([1.0, 2.0, 3.0, 4.0])
        P = build_1d_prefix(T)
        assert query_1d(P, 0, 4) == 10.0
        assert query_1d(P, 1, 3) == 5.0  # 2+3
        assert query_1d(P, 0, 1) == 1.0

    def test_single_element(self):
        T = np.array([5.0])
        P = build_1d_prefix(T)
        assert query_1d(P, 0, 1) == 5.0


class TestPrefixSum2D:
    def test_simple(self):
        T = np.array([[1.0, 2.0], [3.0, 4.0]])
        P = build_2d_prefix(T)
        assert query_2d(P, 0, 2, 0, 2) == 10.0
        assert query_2d(P, 0, 1, 0, 1) == 1.0
        assert query_2d(P, 1, 2, 1, 2) == 4.0


class TestPrefixSum3D:
    def test_simple(self):
        T = np.ones((2, 2, 2))
        P = build_3d_prefix(T)
        assert query_3d(P, 0, 2, 0, 2, 0, 2) == 8.0
        assert query_3d(P, 0, 1, 0, 1, 0, 1) == 1.0


class TestUnionCounts:
    def test_2way(self):
        T = np.array([[1.0, 0.0], [0.0, 1.0]])
        P = build_2d_prefix(T)
        # A=[0,1), B=[0,1): |A| + |B| - |A∩B| = 1 + 1 - 1 = 1
        result = count_2way_union(P, 0, 1, 0, 1)
        assert result == 1.0

    def test_3way(self):
        T = np.ones((2, 2, 2))
        P = build_3d_prefix(T)
        result = count_3way_union(P, 0, 1, 0, 1, 0, 1)
        # Each set covers half the cube (4 elements)
        # |A∪B∪C| by inclusion-exclusion = 4+4+4-2-2-2+1 = 7
        assert result == 7.0
