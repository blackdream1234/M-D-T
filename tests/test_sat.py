"""
Tests for SAT solvers.

Validates Horn-SAT, Anti-Horn-SAT, 2-SAT, and Affine-SAT
against brute-force verification on small instances.
"""

import pytest

from gsnh_mdt.sat.exact_solver import ExactSATSolver
from gsnh_mdt.sat.verification import brute_force_sat_verify


class TestHornSAT:
    def test_satisfiable(self):
        # (x0 -> x1), unit: x0
        clauses = [
            [(0, False), (1, True)],  # ¬x0 ∨ x1
            [(0, True)],              # x0 (unit)
        ]
        assert ExactSATSolver.horn_sat(clauses) is True

    def test_unsatisfiable(self):
        # x0, x1, ¬x0 ∨ ¬x1 (all-negative constraint)
        clauses = [
            [(0, True)],              # x0
            [(1, True)],              # x1
            [(0, False), (1, False)], # ¬x0 ∨ ¬x1
        ]
        # Forward chaining: x0=T, then ¬x0∨x1 fires x1=T, then ¬x0∨¬x1 fails
        # Since we need (0, False), (1, True) as Horn: x0→x1
        # But the above has unit x0, unit x1, then all-negative ¬x0∨¬x1
        # Forward chaining sets x0=True (from unit), then all-negative {x0,x1}:
        # Need both false. But x0 is true. Wait, it's ¬x0∨¬x1 = not all true.
        # x1 is set true by unit clause. Both true -> all-negative fails.
        assert ExactSATSolver.horn_sat(clauses) is False


class TestAntiHornSAT:
    def test_via_reduction(self):
        # Same as horn but flipped
        clauses = [
            [(0, True), (1, False)],  # x0 ∨ ¬x1
            [(0, False)],             # ¬x0 (unit)
        ]
        assert ExactSATSolver.antihorn_sat(clauses) is True


class TestTwoSAT:
    def test_satisfiable(self):
        clauses = [
            [(0, True), (1, True)],    # x0 ∨ x1
            [(0, False), (1, False)],  # ¬x0 ∨ ¬x1
        ]
        assert ExactSATSolver.two_sat(clauses) is True

    def test_unsatisfiable(self):
        # x0, ¬x0
        clauses = [
            [(0, True)],
            [(0, False)],
        ]
        assert ExactSATSolver.two_sat(clauses) is False


class TestAffineSAT:
    def test_satisfiable(self):
        # x0 XOR x1 = 1 — satisfiable (x0=0,x1=1 or x0=1,x1=0)
        equations = [({0, 1}, 1)]
        assert ExactSATSolver.affine_sat(equations) is True

    def test_unsatisfiable_parity(self):
        # A⊕B=1, B⊕C=1, A⊕C=1 → contradiction (0=1 mod 2)
        equations = [
            ({0, 1}, 1),
            ({1, 2}, 1),
            ({0, 2}, 1),
        ]
        assert ExactSATSolver.affine_sat(equations) is False


class TestBruteForceConsistency:
    """Property test: exact solver and brute force agree on random instances."""

    def test_horn_vs_brute_force(self):
        import random
        random.seed(42)
        for _ in range(50):
            n_vars = 4
            n_clauses = random.randint(1, 5)
            clauses = []
            for _ in range(n_clauses):
                clause_len = random.randint(1, 3)
                clause = []
                for j in range(clause_len):
                    v = random.randint(0, n_vars - 1)
                    # Horn: at most 1 positive
                    s = True if j == 0 and random.random() < 0.5 else False
                    clause.append((v, s))
                clauses.append(clause)

            exact = ExactSATSolver.horn_sat(clauses)
            brute = brute_force_sat_verify(clauses, n_vars)
            assert exact == brute, f"Mismatch on {clauses}"


def test_two_sat_large_chain_no_recursion_error():
    clauses = [[(i, False), (i + 1, True)] for i in range(1500)]

    assert ExactSATSolver.two_sat(clauses) is True


def test_two_sat_large_unsat_no_recursion_error():
    clauses = [[(i, False), (i + 1, True)] for i in range(1500)]
    clauses.extend([[(0, True)], [(0, False)]])

    assert ExactSATSolver.two_sat(clauses) is False
