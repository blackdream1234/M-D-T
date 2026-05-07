"""
GSNHLiteral: threshold literal for GSNH predicates.

Extracted verbatim from gsnh_mdt_v3.py lines 127-161.
"""

from dataclasses import dataclass

import numpy as np

from gsnh_mdt.types import LiteralPolarity


@dataclass(frozen=True)
class GSNHLiteral:
    """A threshold literal: x[feature] >= threshold or x[feature] < threshold."""
    feature: int
    threshold: float
    polarity: LiteralPolarity

    def is_positive(self) -> bool:
        return self.polarity == LiteralPolarity.GE

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.polarity == LiteralPolarity.GE:
            return X[:, self.feature] >= self.threshold
        return X[:, self.feature] < self.threshold

    def negate(self) -> 'GSNHLiteral':
        new_pol = (LiteralPolarity.LT if self.polarity == LiteralPolarity.GE
                   else LiteralPolarity.GE)
        return GSNHLiteral(self.feature, self.threshold, new_pol)

    def to_interval(self) -> tuple:
        """Return (lo, hi) interval for this literal's constraint.
        GE: x[f] >= t  ->  [t, +inf)
        LT: x[f] < t   ->  (-inf, t)
        """
        if self.polarity == LiteralPolarity.GE:
            return (self.threshold, np.inf)
        else:
            return (-np.inf, self.threshold)

    def __str__(self) -> str:
        op = "≥" if self.is_positive() else "<"
        return f"(x[{self.feature}] {op} {self.threshold:.4f})"

    def __repr__(self) -> str:
        return self.__str__()
