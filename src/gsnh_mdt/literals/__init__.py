"""
Literal types for GSNH-MDT predicates.

Provides three literal classes:
- GSNHLiteral: threshold literal x[j] >= t or x[j] < t
- GSNHBinaryLiteral: relational literal x[i] >= x[j] or x[i] < x[j]
- CompareLiteral: ordered comparison x[i] op x[j]

And two predicate classes:
- GSNHPredicate: Horn / AntiHorn / ConjUI / Affine predicate
- Square2CNFPredicate: paper-style (l1∨l2)∧(l3∨l4) predicate
"""

from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.binary import GSNHBinaryLiteral
from gsnh_mdt.literals.compare import CompareLiteral
from gsnh_mdt.literals.predicates import GSNHPredicate, Square2CNFPredicate

__all__ = [
    "GSNHLiteral", "GSNHBinaryLiteral", "CompareLiteral",
    "GSNHPredicate", "Square2CNFPredicate",
]
