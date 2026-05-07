"""
GSNHPredicate: a constraint from a tractable language family.

Extracted verbatim from gsnh_mdt_v3.py lines 330-502.
This is BASELINE JOURNAL LOGIC — the Horn/Anti-Horn validation
in __post_init__ must not be modified.

Square2CNFPredicate: real paper-style square 2CNF predicate.
Conjunction of 2-literal disjunctive clauses: (l1 ∨ l2) ∧ (l3 ∨ l4).
"""

from dataclasses import dataclass, field

# Type alias for any literal type
from typing import Optional, Union

import numpy as np

from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.binary import GSNHBinaryLiteral
from gsnh_mdt.literals.compare import CompareLiteral
from gsnh_mdt.types import (
    ClauseArity,
    GSNHPatternType,
    LanguageFamily,
)

Literal = Union[GSNHLiteral, GSNHBinaryLiteral, CompareLiteral]


@dataclass
class GSNHPredicate:
    """A predicate represents a constraint from language L.

    In journal mode, this is a single constraint (not a formula).
    For Horn: a Horn clause (at most 1 positive literal) — OR semantics
    For Anti-Horn: at most 1 negative literal — OR semantics
    For Affine: XOR semantics
    For ConjUI: pure AND of unary interval literals (box constraint)
    """
    literals: tuple[Literal, ...]
    information_gain: float = 0.0
    language_family: LanguageFamily = LanguageFamily.HORN
    is_xor: bool = False
    is_compare: bool = False
    pattern_type: GSNHPatternType = field(init=False)
    arity: ClauseArity = field(init=False)

    def __post_init__(self):
        n_lits = len(self.literals)
        if n_lits < 1 or n_lits > 3:
            raise ValueError(f"Arity must be 1-3, got {n_lits}")

        # Normalize legacy SQUARE_CNF → CONJ_UI
        if self.language_family == LanguageFamily.SQUARE_CNF:
            object.__setattr__(self, 'language_family', LanguageFamily.CONJ_UI)

        # Detect comparison literals
        has_compare = any(
            isinstance(l, (CompareLiteral, GSNHBinaryLiteral))
            for l in self.literals
        )
        object.__setattr__(self, 'is_compare', has_compare)

        n_positive = sum(1 for lit in self.literals if lit.is_positive())
        n_negative = n_lits - n_positive

        # Validate language family constraints (BASELINE JOURNAL LOGIC)
        if self.language_family == LanguageFamily.HORN:
            if n_positive > 1:
                raise ValueError(
                    f"Horn violation: {n_positive} positive literals. "
                    f"Max allowed: 1. Literals: {[str(l) for l in self.literals]}"
                )
        elif self.language_family == LanguageFamily.ANTI_HORN:
            if n_negative > 1:
                raise ValueError(
                    f"Anti-Horn violation: {n_negative} negative literals. "
                    f"Max allowed: 1. Literals: {[str(l) for l in self.literals]}"
                )
        elif self.language_family == LanguageFamily.CONJ_UI:
            pass  # All polarities valid — no restriction for AND semantics

        object.__setattr__(self, 'arity', ClauseArity(n_lits))
        object.__setattr__(self, 'pattern_type', self._classify())

    def _classify(self) -> GSNHPatternType:
        """Classify this predicate into a specific pattern type."""
        if self.is_compare:
            if len(self.literals) == 1:
                return GSNHPatternType.COMPARE_LE
            return GSNHPatternType.COMPARE_BINARY

        n = len(self.literals)
        n_pos = sum(1 for l in self.literals if l.is_positive())

        if self.is_xor:
            return GSNHPatternType.AFFINE_2D if n == 2 else GSNHPatternType.AFFINE_3D

        # ConjUI patterns — must be checked BEFORE the Horn/AntiHorn fallthrough
        if self.language_family == LanguageFamily.CONJ_UI:
            if n == 1:
                return GSNHPatternType.CONJ_UI_1L
            elif n == 2:
                all_same = (n_pos == 0 or n_pos == 2)
                return GSNHPatternType.CONJ_UI_2D_ALL if all_same else GSNHPatternType.CONJ_UI_2D_MIXED
            else:
                all_same = (n_pos == 0 or n_pos == 3)
                return GSNHPatternType.CONJ_UI_3D_ALL if all_same else GSNHPatternType.CONJ_UI_3D_MIXED

        if self.language_family == LanguageFamily.ANTI_HORN:
            if n == 1:
                return GSNHPatternType.AH_UNARY_POS if n_pos else GSNHPatternType.AH_UNARY_NEG
            elif n == 2:
                if n_pos == 2:
                    return GSNHPatternType.AH_BINARY_ALL_POS
                return (GSNHPatternType.AH_BINARY_NEG_FIRST
                        if not self.literals[0].is_positive()
                        else GSNHPatternType.AH_BINARY_NEG_SECOND)
            else:
                if n_pos == 3:
                    return GSNHPatternType.AH_TERNARY_ALL_POS
                if not self.literals[0].is_positive():
                    return GSNHPatternType.AH_TERNARY_NEG_FIRST
                if not self.literals[1].is_positive():
                    return GSNHPatternType.AH_TERNARY_NEG_SECOND
                return GSNHPatternType.AH_TERNARY_NEG_THIRD

        # Horn patterns (default fallthrough)
        if n == 1:
            return GSNHPatternType.UNARY_POS if n_pos else GSNHPatternType.UNARY_NEG
        elif n == 2:
            if n_pos == 0:
                return GSNHPatternType.BINARY_ALL_NEG
            return (GSNHPatternType.BINARY_POS_FIRST
                    if self.literals[0].is_positive()
                    else GSNHPatternType.BINARY_POS_SECOND)
        else:
            if n_pos == 0:
                return GSNHPatternType.TERNARY_ALL_NEG
            if self.literals[0].is_positive():
                return GSNHPatternType.TERNARY_POS_FIRST
            if self.literals[1].is_positive():
                return GSNHPatternType.TERNARY_POS_SECOND
            return GSNHPatternType.TERNARY_POS_THIRD

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate predicate on data matrix. True = positive branch of split."""
        if self.is_xor:
            result = self.literals[0].evaluate(X)
            for lit in self.literals[1:]:
                result = result ^ lit.evaluate(X)
            return result
        elif self.language_family == LanguageFamily.CONJ_UI:
            # AND semantics: intersection of half-spaces
            result = self.literals[0].evaluate(X)
            for lit in self.literals[1:]:
                result = result & lit.evaluate(X)
            return result
        else:
            # OR semantics (Horn, Anti-Horn)
            result = self.literals[0].evaluate(X)
            for lit in self.literals[1:]:
                result = result | lit.evaluate(X)
            return result

    def evaluate_partial(self, x: np.ndarray, S: set) -> Optional[bool]:
        """Partial evaluation for AXp engine.

        Returns True, False, or None (unknown due to unassigned features).
        """
        if self.is_xor:
            for lit in self.literals:
                feat = lit.feature_i if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)) else lit.feature
                if feat not in S:
                    if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                        other = lit.feature_j
                        if other not in S:
                            return None
                    else:
                        return None
            return bool(self.evaluate(x.reshape(1, -1))[0])
        elif self.language_family == LanguageFamily.CONJ_UI:
            # AND short-circuit: any False → False, all True → True, else None
            all_true = True
            for lit in self.literals:
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    feat_i, feat_j = lit.feature_i, lit.feature_j
                    if feat_i not in S or feat_j not in S:
                        all_true = False
                        continue
                    val = lit.evaluate(x.reshape(1, -1))[0]
                else:
                    if lit.feature not in S:
                        all_true = False
                        continue
                    val = lit.evaluate(x.reshape(1, -1))[0]
                if not val:
                    return False  # One False literal → whole AND is False
            if all_true:
                return True
            return None
        else:
            # OR short-circuit: any True → True, all False → False, else None
            all_false = True
            for lit in self.literals:
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    feat_i, feat_j = lit.feature_i, lit.feature_j
                    if feat_i not in S or feat_j not in S:
                        all_false = False
                        continue
                    val = lit.evaluate(x.reshape(1, -1))[0]
                else:
                    if lit.feature not in S:
                        all_false = False
                        continue
                    val = lit.evaluate(x.reshape(1, -1))[0]

                if val:
                    return True

            if all_false:
                return False
            return None

    def to_horn_clause(self) -> str:
        """Convert to logical notation for debugging."""
        if self.is_xor:
            lits = " ⊕ ".join(str(l) for l in self.literals)
            return f"{lits} [Affine/XOR]"

        if self.language_family == LanguageFamily.CONJ_UI:
            lits = " ∧ ".join(str(l) for l in self.literals)
            return f"{lits} [ConjUI/Box]"

        positives = [lit for lit in self.literals if lit.is_positive()]
        negatives = [lit for lit in self.literals if not lit.is_positive()]

        if not positives:
            neg_strs = " ∨ ".join(str(l) for l in negatives)
            return f"{neg_strs} [no head]"

        head = str(positives[0])
        if not negatives:
            return f"⊤ → {head}"

        body = " ∧ ".join(str(l.negate()) for l in negatives)
        return f"{body} → {head}"

    def __str__(self) -> str:
        if self.language_family == LanguageFamily.CONJ_UI:
            return " ∧ ".join(str(lit) for lit in self.literals)
        return " ∨ ".join(str(lit) for lit in self.literals)


# =============================================================================
# SQUARE 2CNF PREDICATE — Real paper-style (Carbonnel 2025)
# =============================================================================

@dataclass
class Square2CNFPredicate:
    """Paper-style square 2CNF predicate.

    Represents a conjunction of 2-literal disjunctive clauses:
        (l₁ ∨ l₂) ∧ (l₃ ∨ l₄)

    Each clause is a tuple of exactly 2 GSNHLiteral objects.
    Initially supports exactly 1-3 clauses of 2 literals each.

    Attributes
    ----------
    clauses : tuple of 2-tuples of GSNHLiteral
        Each element is (lit_a, lit_b) representing (lit_a ∨ lit_b).
    information_gain : float
        BIC-penalized information gain of this split.
    language_family : LanguageFamily
        Always SQUARE_2CNF.
    pattern_type : GSNHPatternType
        Computed in __post_init__.
    arity : ClauseArity
        Number of clauses.
    """
    clauses: tuple  # tuple[tuple[GSNHLiteral, GSNHLiteral], ...]
    information_gain: float = 0.0
    language_family: LanguageFamily = field(default=LanguageFamily.SQUARE_2CNF, init=False)
    pattern_type: GSNHPatternType = field(init=False)
    arity: ClauseArity = field(init=False)

    # These match GSNHPredicate interface for builder/explainer compatibility
    is_xor: bool = field(default=False, init=False)
    is_compare: bool = field(default=False, init=False)

    def __post_init__(self):
        if len(self.clauses) < 1 or len(self.clauses) > 3:
            raise ValueError(f"Square2CNF: need 1-3 clauses, got {len(self.clauses)}")
        for i, clause in enumerate(self.clauses):
            if len(clause) != 2:
                raise ValueError(f"Clause {i}: must have exactly 2 literals, got {len(clause)}")
            for lit in clause:
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    raise ValueError(
                        f"Square2CNF does not support Compare/Binary literals yet: {lit}")

        # Classify
        n_clauses = len(self.clauses)
        object.__setattr__(self, 'arity', ClauseArity(n_clauses))

        # Check for degenerate cases (clause where both lits are identical)
        is_degenerate = any(
            c[0].feature == c[1].feature and c[0].threshold == c[1].threshold
            and c[0].polarity == c[1].polarity
            for c in self.clauses
        )
        if is_degenerate:
            object.__setattr__(self, 'pattern_type', GSNHPatternType.SQUARE_2CNF_DEGENERATE)
        else:
            object.__setattr__(self, 'pattern_type', GSNHPatternType.SQUARE_2CNF_2C2L)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate predicate: conjunction of disjunctive clauses.

        P(x) = ∧_c (l_{c,1}(x) ∨ l_{c,2}(x))
        """
        result = np.ones(X.shape[0], dtype=bool)
        for a, b in self.clauses:
            result &= (a.evaluate(X) | b.evaluate(X))
        return result

    def evaluate_partial(self, x: np.ndarray, S: set) -> Optional[bool]:
        """Partial evaluation for AXp engine.

        For each clause (a ∨ b):
          - True if at least one literal is known True
          - False if both literals are known False
          - None if indeterminate

        Overall AND:
          - False if any clause is False (short-circuit)
          - True if all clauses are True
          - None otherwise
        """
        all_determined = True
        for a, b in self.clauses:
            a_known = a.feature in S
            b_known = b.feature in S

            if a_known and b_known:
                a_val = bool(a.evaluate(x.reshape(1, -1))[0])
                b_val = bool(b.evaluate(x.reshape(1, -1))[0])
                if not (a_val or b_val):
                    return False  # Both False → clause False → AND False
                # clause is True, continue
            elif a_known:
                a_val = bool(a.evaluate(x.reshape(1, -1))[0])
                if a_val:
                    continue  # clause satisfied
                else:
                    all_determined = False  # b unknown, a=False → indeterminate
            elif b_known:
                b_val = bool(b.evaluate(x.reshape(1, -1))[0])
                if b_val:
                    continue  # clause satisfied
                else:
                    all_determined = False  # a unknown, b=False → indeterminate
            else:
                all_determined = False  # both unknown

        if all_determined:
            return True
        return None

    def iter_literals(self):
        """Yield all literals across all clauses."""
        for clause in self.clauses:
            for lit in clause:
                yield lit

    def features_used(self) -> set:
        """Return set of feature indices used by this predicate."""
        return {lit.feature for lit in self.iter_literals()}

    @property
    def literals(self):
        """Flat tuple of all literals — compatibility with explainer code
        that iterates pred.literals."""
        return tuple(lit for clause in self.clauses for lit in clause)

    def __str__(self) -> str:
        return " ∧ ".join(f"({a} ∨ {b})" for a, b in self.clauses)

    def to_horn_clause(self) -> str:
        """Convert to logical notation for debugging."""
        return f"{self} [Square2CNF]"
