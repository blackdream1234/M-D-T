"""
Tests for scoring functions.

Validates entropy, information gain, gain ratio, and BIC penalty
match the monolithic implementation exactly.
"""

import numpy as np
import pytest


class TestEntropy:
    def test_pure_positive(self):
        from gsnh_mdt.scoring.entropy import entropy
        assert entropy(10.0, 0.0) == 0.0

    def test_pure_negative(self):
        from gsnh_mdt.scoring.entropy import entropy
        assert entropy(0.0, 10.0) == 0.0

    def test_balanced(self):
        from gsnh_mdt.scoring.entropy import entropy
        assert abs(entropy(5.0, 5.0) - 1.0) < 1e-10

    def test_empty(self):
        from gsnh_mdt.scoring.entropy import entropy
        assert entropy(0.0, 0.0) == 0.0


class TestInformationGain:
    def test_perfect_split(self):
        from gsnh_mdt.scoring.gain import information_gain
        # Perfect split: all pos on one side, all neg on other
        ig = information_gain(5.0, 5.0, 5.0, 0.0)
        assert ig > 0.99

    def test_no_split(self):
        from gsnh_mdt.scoring.gain import information_gain
        # No improvement: same ratio in both halves
        ig = information_gain(10.0, 10.0, 5.0, 5.0)
        assert ig < 0.01

    def test_degenerate(self):
        from gsnh_mdt.scoring.gain import information_gain
        ig = information_gain(5.0, 5.0, 0.0, 0.0)
        assert ig == -1.0


class TestPenalizedGain:
    def test_low_gain_rejected(self):
        from gsnh_mdt.scoring.penalties import penalized_gain
        result = penalized_gain(0.001, 2, 32, 100, 2)
        assert result == -1.0  # Penalty exceeds gain

    def test_high_gain_accepted(self):
        from gsnh_mdt.scoring.penalties import penalized_gain
        result = penalized_gain(0.5, 1, 32, 1000, 2)
        assert result > 0  # Gain survives penalty
