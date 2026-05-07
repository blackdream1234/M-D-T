"""
GSNHBinaryLiteral: relational literal comparing two features.

Extracted verbatim from gsnh_mdt_v3.py lines 163-191.
"""

from dataclasses import dataclass

import numpy as np

from gsnh_mdt.types import LiteralPolarity


@dataclass(frozen=True)
class GSNHBinaryLiteral:
    """Relational literal: x[feature_i] >= x[feature_j] or x[feature_i] < x[feature_j]."""
    feature_i: int
    feature_j: int
    polarity: LiteralPolarity

    @property
    def feature(self):
        """Compatibility property. Prefer using feature_i directly."""
        return self.feature_i

    def is_positive(self) -> bool:
        return self.polarity == LiteralPolarity.GE

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.polarity == LiteralPolarity.GE:
            return X[:, self.feature_i] >= X[:, self.feature_j]
        return X[:, self.feature_i] < X[:, self.feature_j]

    def negate(self) -> 'GSNHBinaryLiteral':
        new_pol = LiteralPolarity.LT if self.polarity == LiteralPolarity.GE else LiteralPolarity.GE
        return GSNHBinaryLiteral(self.feature_i, self.feature_j, new_pol)

    def __str__(self) -> str:
        op = "≥" if self.is_positive() else "<"
        return f"(x[{self.feature_i}] {op} x[{self.feature_j}])"

    def __repr__(self) -> str:
        return self.__str__()
