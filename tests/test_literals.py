"""
Tests for extracted literal types.

Validates that literals evaluate, negate, and convert to intervals
identically to the monolithic implementation.
"""

import numpy as np
import pytest

from gsnh_mdt.types import LiteralPolarity, CompareOp, LanguageFamily
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.binary import GSNHBinaryLiteral
from gsnh_mdt.literals.compare import CompareLiteral
from gsnh_mdt.literals.predicates import GSNHPredicate


class TestGSNHLiteral:
    def test_ge_evaluate(self):
        lit = GSNHLiteral(0, 0.5, LiteralPolarity.GE)
        X = np.array([[0.3], [0.5], [0.7]])
        result = lit.evaluate(X)
        assert list(result) == [False, True, True]

    def test_lt_evaluate(self):
        lit = GSNHLiteral(0, 0.5, LiteralPolarity.LT)
        X = np.array([[0.3], [0.5], [0.7]])
        result = lit.evaluate(X)
        assert list(result) == [True, False, False]

    def test_negate(self):
        lit = GSNHLiteral(0, 0.5, LiteralPolarity.GE)
        neg = lit.negate()
        assert neg.polarity == LiteralPolarity.LT
        assert neg.threshold == 0.5
        assert neg.feature == 0

    def test_to_interval_ge(self):
        lit = GSNHLiteral(0, 0.3, LiteralPolarity.GE)
        lo, hi = lit.to_interval()
        assert lo == 0.3
        assert hi == np.inf

    def test_to_interval_lt(self):
        lit = GSNHLiteral(0, 0.7, LiteralPolarity.LT)
        lo, hi = lit.to_interval()
        assert lo == -np.inf
        assert hi == 0.7


class TestGSNHBinaryLiteral:
    def test_evaluate(self):
        lit = GSNHBinaryLiteral(0, 1, LiteralPolarity.GE)
        X = np.array([[0.3, 0.5], [0.7, 0.5], [0.5, 0.5]])
        result = lit.evaluate(X)
        assert list(result) == [False, True, True]

    def test_negate(self):
        lit = GSNHBinaryLiteral(0, 1, LiteralPolarity.GE)
        neg = lit.negate()
        assert neg.polarity == LiteralPolarity.LT


class TestCompareLiteral:
    def test_le_evaluate(self):
        lit = CompareLiteral(0, 1, CompareOp.LE)
        X = np.array([[0.3, 0.5], [0.8, 0.5]])
        result = lit.evaluate(X)
        assert list(result) == [True, False]

    def test_negate_le_gt(self):
        lit = CompareLiteral(0, 1, CompareOp.LE)
        neg = lit.negate()
        assert neg.op == CompareOp.GT


class TestGSNHPredicate:
    def test_horn_valid(self):
        """Horn clause with 1 positive literal should be accepted."""
        pred = GSNHPredicate(
            literals=(
                GSNHLiteral(0, 0.5, LiteralPolarity.GE),  # positive
                GSNHLiteral(1, 0.3, LiteralPolarity.LT),  # negative
            ),
            information_gain=0.1,
            language_family=LanguageFamily.HORN,
        )
        assert pred.arity.value == 2

    def test_horn_violation(self):
        """Horn clause with 2 positive literals must be rejected."""
        with pytest.raises(ValueError, match="Horn violation"):
            GSNHPredicate(
                literals=(
                    GSNHLiteral(0, 0.5, LiteralPolarity.GE),
                    GSNHLiteral(1, 0.5, LiteralPolarity.GE),
                ),
                information_gain=0.1,
                language_family=LanguageFamily.HORN,
            )

    def test_antihorn_violation(self):
        """Anti-Horn clause with 2 negative literals must be rejected."""
        with pytest.raises(ValueError, match="Anti-Horn violation"):
            GSNHPredicate(
                literals=(
                    GSNHLiteral(0, 0.5, LiteralPolarity.LT),
                    GSNHLiteral(1, 0.5, LiteralPolarity.LT),
                ),
                information_gain=0.1,
                language_family=LanguageFamily.ANTI_HORN,
            )

    def test_evaluate_disjunction(self):
        """Test OR semantics for split evaluation."""
        pred = GSNHPredicate(
            literals=(
                GSNHLiteral(0, 0.5, LiteralPolarity.GE),
                GSNHLiteral(1, 0.3, LiteralPolarity.LT),
            ),
            information_gain=0.1,
            language_family=LanguageFamily.HORN,
        )
        X = np.array([[0.6, 0.4], [0.4, 0.2], [0.4, 0.4]])
        result = pred.evaluate(X)
        # x[0]>=0.5 OR x[1]<0.3
        assert list(result) == [True, True, False]
