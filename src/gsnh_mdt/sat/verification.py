"""
Brute-force SAT verification for testing exact solvers.

Extracted verbatim from gsnh_mdt_v3.py lines 1555-1572.
"""

from itertools import product


def brute_force_sat_verify(clauses: list[list[tuple[int, bool]]], n_vars: int) -> bool:
    """O(2^n) brute force to verify ExactSATSolver completeness in tests."""
    for assignment in product([False, True], repeat=n_vars):
        satisfied = True
        for clause in clauses:
            clause_sat = False
            for var, is_positive in clause:
                if var < n_vars:
                    val = assignment[var]
                    if val == is_positive:
                        clause_sat = True
                        break
            if not clause_sat:
                satisfied = False
                break
        if satisfied:
            return True
    return False
