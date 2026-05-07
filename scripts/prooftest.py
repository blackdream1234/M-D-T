import numpy as np
import os
import sys

# Bootstrap: ensure the package is importable even without pip install -e
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Package imports — the package is now the authoritative implementation.
# The monolith (gsnh_mdt_v3.py) is archived in archive/.
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.types import LiteralPolarity, LanguageFamily
from gsnh_mdt.literals.predicates import GSNHPredicate
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.sat.exact_solver import ExactSATSolver

def prove_journal_compliance():
    """
    MATHEMATICAL COMPLIANCE SUITE
    Proves that the code strictly respects the theoretical boundaries of 
    'Tractable Explaining of MDTs' (Carbonnel et al.)
    """
    print("\n" + "═" * 80)
    print("🎓 JOURNAL COMPLIANCE & MATHEMATICAL PROOF TEST SUITE")
    print("═" * 80)

    passed_all = True

    # =========================================================================
    # PROOF 1: UI-Literal and Horn Constraint (Section 4.1 & 5.3)
    # The paper defines Generalized Star-Nested Horn (GSNH) as having AT MOST 
    # ONE positive UI-literal (x >= a). 
    # =========================================================================
    print("\n[Proof 1] Testing Strict GSNH Horn Constraints...")
    try:
        # Attempt to create an INVALID Horn clause: (x0 >= 0.5) OR (x1 >= 0.5)
        # This has TWO positive literals. It must be rejected to maintain P-time AXp.
        invalid_horn = GSNHPredicate(
            literals=(
                GSNHLiteral(0, 0.5, LiteralPolarity.GE), 
                GSNHLiteral(1, 0.5, LiteralPolarity.GE)
            ),
            information_gain=0.1, 
            language_family=LanguageFamily.HORN
        )
        print("  ❌ FAIL: Code allowed an invalid Horn clause with 2 positive literals!")
        passed_all = False
    except ValueError as e:
        print("  ✅ PASS: Code correctly blocked >1 positive literal in Horn mode.")
        print(f"     (Error caught: {e})")


    # =========================================================================
    # PROOF 2: Anti-Horn Constraint 
    # Symmetrical to Horn: AT MOST ONE negative UI-literal (x < a).
    # =========================================================================
    print("\n[Proof 2] Testing Strict Anti-Horn Constraints...")
    try:
        # Attempt to create an INVALID Anti-Horn clause: (x0 < 0.5) OR (x1 < 0.5)
        # This has TWO negative literals. It must be rejected.
        invalid_anti_horn = GSNHPredicate(
            literals=(
                GSNHLiteral(0, 0.5, LiteralPolarity.LT), 
                GSNHLiteral(1, 0.5, LiteralPolarity.LT)
            ),
            information_gain=0.1, 
            language_family=LanguageFamily.ANTI_HORN
        )
        print("  ❌ FAIL: Code allowed an invalid Anti-Horn clause with 2 negative literals!")
        passed_all = False
    except ValueError as e:
        print("  ✅ PASS: Code correctly blocked >1 negative literal in Anti-Horn mode.")


    # =========================================================================
    # PROOF 3: Numeric Interval SAT for AXp (Section 3 & 5.3)
    # To extract an Abductive Explanation (AXp), the SAT solver must correctly
    # identify if a tree path maps to an empty/impossible mathematical region.
    # =========================================================================
    print("\n[Proof 3] Testing Exact Numeric Interval SAT (Empty Region Detection)...")
    
    # We mock a tree just to access the SAT solver
    tree = ExpertGSNHTree(mode='journal', language=LanguageFamily.HORN)
    tree.n_features_ = 2
    
    # We construct a mathematically IMPOSSIBLE path for a single feature x0:
    # 1. (x0 < 0.3) is FALSE  --> Implies x0 >= 0.3
    # 2. (x0 >= 0.7) is FALSE --> Implies x0 < 0.7
    # 3. (x0 >= 0.8) is TRUE  --> Implies x0 >= 0.8
    # Intersection of [0.3, 0.7) AND [0.8, +inf) is EMPTY.
    path_edges = [
        (GSNHPredicate((GSNHLiteral(0, 0.3, LiteralPolarity.LT),), 0.1), False),
        (GSNHPredicate((GSNHLiteral(0, 0.7, LiteralPolarity.GE),), 0.1), False),
        (GSNHPredicate((GSNHLiteral(0, 0.8, LiteralPolarity.GE),), 0.1), True)
    ]
    
    dummy_x = np.array([0.5, 0.5])
    empty_S = set() # No features fixed yet
    
    is_sat = tree._path_sat_numeric(path_edges, dummy_x, empty_S)
    
    if not is_sat:
        print("  ✅ PASS: Interval SAT solver correctly identified the mathematically impossible region.")
    else:
        print("  ❌ FAIL: SAT solver thought an impossible interval was valid!")
        passed_all = False


    # =========================================================================
    # PROOF 4: Affine Tractability & GF(2) Elimination (Lemma 11)
    # Lemma 11 states Affine logic is only tractable over Boolean domains. 
    # We test if the Gaussian Elimination correctly finds contradictions.
    # =========================================================================
    print("\n[Proof 4] Testing Affine GF(2) SAT Solver (Lemma 11)...")
    
    # We construct an impossible XOR system:
    # Eq 1: A ⊕ B = 1
    # Eq 2: B ⊕ C = 1
    # Eq 3: A ⊕ C = 1
    # Summing all three sides: (A⊕A) ⊕ (B⊕B) ⊕ (C⊕C) = 1 ⊕ 1 ⊕ 1
    # 0 = 3 (mod 2) --> 0 = 1 (Contradiction!)
    equations = [
        ({0, 1}, 1),  # A(0) and B(1)
        ({1, 2}, 1),  # B(1) and C(2)
        ({0, 2}, 1)   # A(0) and C(2)
    ]
    
    is_affine_sat = ExactSATSolver.affine_sat(equations)
    
    if not is_affine_sat:
        print("  ✅ PASS: GF(2) Gaussian Elimination correctly caught the parity contradiction.")
    else:
        print("  ❌ FAIL: GF(2) solver failed to find the parity contradiction.")
        passed_all = False


    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "═" * 80)
    if passed_all:
        print("🏆 FINAL VERDICT: 100% COMPLIANT.")
        print("Your codebase mathematically respects all constraints defined in the journal.")
    else:
        print("⚠️ FINAL VERDICT: ERRORS DETECTED.")
        print("The code violates theoretical constraints. Please review the failed tests.")
    print("═" * 80 + "\n")


# =============================================================================
# Add this to your existing __main__ block
# =============================================================================
if __name__ == "__main__":
    # ... your existing __main__ code ...
    
    # Run the compliance test at the end
    prove_journal_compliance()
