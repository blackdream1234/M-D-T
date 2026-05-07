"""
Core enumerations, type aliases, and configuration constants.

These are the foundational types used throughout the GSNH-MDT package.
Extracted verbatim from gsnh_mdt_v3.py lines 48–120.
"""

import warnings
from enum import Enum, auto

# =============================================================================
# ENUMERATIONS
# =============================================================================

class LiteralPolarity(Enum):
    """Polarity of a threshold literal."""
    GE = auto()  # x >= t (POSITIVE in journal)
    LT = auto()  # x < t  (NEGATIVE in journal)


class ClauseArity(Enum):
    """Arity of a GSNH clause (number of literals)."""
    UNARY = 1
    BINARY = 2
    TERNARY = 3


class LanguageFamily(Enum):
    """The tractable GSNH language families.

    - HORN: at most 1 positive literal per clause (OR semantics)
    - ANTI_HORN: at most 1 negative literal per clause (OR semantics)
    - AFFINE: XOR constraint x1 XOR x2 XOR ... = c
    - CONJ_UI: pure AND of unary interval literals / box constraints.
               Formerly called SQUARE_CNF in early experiments.
               This is NOT the paper's square 2CNF language.
    - SQUARE_2CNF: real paper-style square 2CNF (Carbonnel 2025).
                   Conjunction of 2-literal disjunctive clauses:
                   (l1 ∨ l2) ∧ (l3 ∨ l4)
    - ANY: all families compete (unconstrained, rejected in journal mode)
    - BEST_PER_NODE: heuristic per-node language selection via topological
      profiler.  Selects the single most promising family at each node
      based on O(N) bounding-box analysis instead of exhaustive competition
      across all families.  Affine is excluded to preserve uniform SAT
      fragments for tractable explanation extraction.
    """
    HORN = "Horn"
    ANTI_HORN = "AntiHorn"
    AFFINE = "Affine"
    CONJ_UI = "ConjUI"          # New canonical name (was SQUARE_CNF)
    SQUARE_2CNF = "Square2CNF"  # Real paper square 2CNF
    ANY = "Any"
    BEST_PER_NODE = "BestPerNode"

    # Legacy alias — deprecated, maps to CONJ_UI
    SQUARE_CNF = "SquareCNF"


def _resolve_language(lang):
    """Resolve legacy language names to canonical ones.

    Returns the canonical LanguageFamily, emitting a deprecation warning
    if SQUARE_CNF is used.
    """
    if lang == LanguageFamily.SQUARE_CNF:
        warnings.warn(
            "LanguageFamily.SQUARE_CNF is deprecated. "
            "Use LanguageFamily.CONJ_UI (pure AND / box) or "
            "LanguageFamily.SQUARE_2CNF (paper-style (l1∨l2)∧(l3∨l4)). "
            "SQUARE_CNF is now treated as CONJ_UI.",
            DeprecationWarning,
            stacklevel=3,
        )
        return LanguageFamily.CONJ_UI
    return lang


class CompareOp(Enum):
    """Comparison operators for relational literals."""
    LE = "LE"  # x[i] <= x[j]
    LT = "LT"  # x[i] < x[j]
    GE = "GE"  # x[i] >= x[j]
    GT = "GT"  # x[i] > x[j]


class GSNHPatternType(Enum):
    """Split pattern identifiers for tracking which pattern was selected."""
    # Standard Horn patterns
    UNARY_NEG = "1L_NEG"
    UNARY_POS = "1L_POS"
    BINARY_ALL_NEG = "2L_ALL_NEG"
    BINARY_POS_FIRST = "2L_POS_0"
    BINARY_POS_SECOND = "2L_POS_1"
    TERNARY_ALL_NEG = "3L_ALL_NEG"
    TERNARY_POS_FIRST = "3L_POS_0"
    TERNARY_POS_SECOND = "3L_POS_1"
    TERNARY_POS_THIRD = "3L_POS_2"
    # Anti-Horn patterns
    AH_UNARY_NEG = "AH_1L_NEG"
    AH_UNARY_POS = "AH_1L_POS"
    AH_BINARY_ALL_POS = "AH_2L_ALL_POS"
    AH_BINARY_NEG_FIRST = "AH_2L_NEG_0"
    AH_BINARY_NEG_SECOND = "AH_2L_NEG_1"
    AH_TERNARY_ALL_POS = "AH_3L_ALL_POS"
    AH_TERNARY_NEG_FIRST = "AH_3L_NEG_0"
    AH_TERNARY_NEG_SECOND = "AH_3L_NEG_1"
    AH_TERNARY_NEG_THIRD = "AH_3L_NEG_2"
    # Affine patterns
    AFFINE_2D = "AFF_2D"
    AFFINE_3D = "AFF_3D"
    # ConjUI patterns (pure AND — formerly SQ_CNF)
    CONJ_UI_1L = "CONJ_UI_1L"
    CONJ_UI_2D_ALL = "CONJ_UI_2D_ALL"      # 2D same polarity
    CONJ_UI_2D_MIXED = "CONJ_UI_2D_MIX"    # 2D mixed polarities
    CONJ_UI_3D_ALL = "CONJ_UI_3D_ALL"      # 3D same polarity
    CONJ_UI_3D_MIXED = "CONJ_UI_3D_MIX"    # 3D mixed polarities
    # Legacy aliases (deprecated — kept for backward compat with golden files)
    SQ_CNF_1L = "SQ_CNF_1L"
    SQ_2CNF_ALL = "SQ_2CNF_ALL"
    SQ_2CNF_MIXED = "SQ_2CNF_MIX"
    SQ_3CNF_ALL = "SQ_3CNF_ALL"
    SQ_3CNF_MIXED = "SQ_3CNF_MIX"
    # Square 2CNF patterns (paper-style)
    SQUARE_2CNF_2C2L = "SQ2CNF_2C2L"           # 2 clauses × 2 literals
    SQUARE_2CNF_DEGENERATE = "SQ2CNF_DEGEN"     # degenerate (clause reduces to single lit)
    # Comparison literal patterns
    COMPARE_LE = "COMPARE_LE"
    COMPARE_BINARY = "COMPARE_BINARY"


# =============================================================================
# POLARITY CONFIGURATION TABLES
# =============================================================================

# Horn configs: at most 1 positive literal (True = GE = POSITIVE)
GSNH_VALID_CONFIGS = {
    1: [(False,), (True,)],
    2: [(False, False), (True, False), (False, True)],
    3: [(False, False, False), (True, False, False),
        (False, True, False), (False, False, True)],
}

# Anti-Horn configs: at most 1 negative literal (mirror of Horn)
# False = LT = NEGATIVE, so at most 1 False entry
GSNH_ANTIHORN_CONFIGS = {
    1: [(False,), (True,)],
    2: [(True, True), (False, True), (True, False)],
    3: [(True, True, True), (False, True, True),
        (True, False, True), (True, True, False)],
}

# ConjUI configs: all polarities valid (AND semantics, no restriction)
# Renamed from GSNH_SQUARE_CNF_CONFIGS
GSNH_CONJ_UI_CONFIGS = {
    1: [(False,), (True,)],
    2: [(False, False), (False, True), (True, False), (True, True)],
    3: [(False, False, False), (False, False, True), (False, True, False),
        (False, True, True), (True, False, False), (True, False, True),
        (True, True, False), (True, True, True)],
}

# Legacy alias — kept for backward compatibility
GSNH_SQUARE_CNF_CONFIGS = GSNH_CONJ_UI_CONFIGS
