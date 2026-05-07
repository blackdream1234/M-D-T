"""
CompareLiteral: binary comparison literal for ordered discrete domains.

Extracted verbatim from gsnh_mdt_v3.py lines 194-238.
"""

from dataclasses import dataclass

import numpy as np

from gsnh_mdt.types import CompareOp


@dataclass(frozen=True)
class CompareLiteral:
    """Binary comparison literal for ordered discrete domains.

    Represents constraints like x[i] <= x[j] directly.

    In journal terms, this is a binary relation over UI-literals:
    x_i <= x_j is equivalent to not(x_i >= a) or (x_j >= a) for all relevant a,
    but implemented directly for efficiency.
    """
    feature_i: int
    feature_j: int
    op: CompareOp

    def is_positive(self) -> bool:
        """For Horn/Anti-Horn: GE/GT are positive, LE/LT are negative."""
        return self.op in (CompareOp.GE, CompareOp.GT)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.op == CompareOp.LE:
            return X[:, self.feature_i] <= X[:, self.feature_j]
        elif self.op == CompareOp.LT:
            return X[:, self.feature_i] < X[:, self.feature_j]
        elif self.op == CompareOp.GE:
            return X[:, self.feature_i] >= X[:, self.feature_j]
        else:  # GT
            return X[:, self.feature_i] > X[:, self.feature_j]

    def negate(self) -> 'CompareLiteral':
        """Negation: not(x<=y) = x>y, not(x<y) = x>=y, etc."""
        neg_map = {
            CompareOp.LE: CompareOp.GT,
            CompareOp.LT: CompareOp.GE,
            CompareOp.GE: CompareOp.LT,
            CompareOp.GT: CompareOp.LE,
        }
        return CompareLiteral(self.feature_i, self.feature_j, neg_map[self.op])

    def __str__(self) -> str:
        op_str = {CompareOp.LE: "≤", CompareOp.LT: "<",
                  CompareOp.GE: "≥", CompareOp.GT: ">"}
        return f"(x[{self.feature_i}] {op_str[self.op]} x[{self.feature_j}])"

    def __repr__(self) -> str:
        return self.__str__()
