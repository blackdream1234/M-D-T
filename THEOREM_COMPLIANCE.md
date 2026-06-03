# Theorem Compliance Protocol

This document outlines the strict boundary between the theorem-certified mathematical core of the GSNH-MDT framework and its empirical, heuristic, or auxiliary components.

## 1. Verified Mathematical Core
The Coq development proves the following formal properties:
- **Numeric threshold atom**: `B(f,t) := x_f >= t`
- **Literal encoding**: `GE(f,t) := B(f,t)` and `LT(f,t) := not B(f,t)`
- **Structural threshold order clauses**: If `t1 <= t2`, then `B(f,t2) -> B(f,t1)`, encoded as `not B(f,t2) OR B(f,t1)`
- **Ordered selected-path CNF**: Formed by selected feature unit clauses + structural threshold-order clauses + path branch clauses
- **Weak AXp reflection**: Checker result is `true` iff all opposite paths are blocked under selected features
- **Deletion-based AXp extraction**: Returns a numeric subset-minimal weak AXp via deterministic feature removal

## 2. Fixed Horn Theorem Mode
- Theorem-compliant if:
  - Structural threshold encoding is used.
  - Backend is `structural_horn`.
  - Unsupported literals (like `CompareLiteral`, `GSNHBinaryLiteral`) are absent.
  - No DFS fallback is used.

## 3. Fixed AntiHorn Theorem Mode
- Theorem-compliant if:
  - Structural threshold encoding is used.
  - Backend is `structural_antihorn`.
  - Unsupported literals are absent.
  - No DFS fallback is used.

## 4. Square2CNF Theorem Mode
- Certified **only** through explicit 2-CNF + `two_sat` (Kosaraju's SCC algorithm).
- The `False` branch requires an auxiliary switch variable to convert the DNF `(not l1 AND not l2) OR (not l3 AND not l4)` into an equisatisfiable 2-CNF over `s`: `(not s OR not l1)`, `(not s OR not l2)`, `(s OR not l3)`, `(s OR not l4)`.

## 5. BEST_PER_NODE
- Empirical by default.
- Theorem-certified **only** with a path-level certificate (`horn`, `antihorn`, or `2cnf`) where every explanation path receives a tractability certificate.
- **Unrestricted mixed BEST_PER_NODE is not claimed to be polynomial.** Mixed paths that do not satisfy any of the certificate conditions are rejected in `theorem_strict` mode and marked as non-theorem.

## 6. Auxiliary/Prototype Methods
- Methods like ConjUI, Affine, fallback/case split solvers are **not** theorem-certified unless separately proven or they fall under a polynomial classification (e.g., Affine with Gaussian Elimination over GF(2) is polynomial but not via a CNF certificate in our Coq development).
- For benchmark purposes, they are routed to auxiliary tables.

## 7. What is NOT Verified
- The Python implementation is thoroughly tested against the Coq specification, but it is **not** extracted directly from Coq.
- Benchmark accuracy and size reductions are empirical findings, not formally verified.
- Unrestricted mixed-family SAT across heterogeneous nodes is **not** claimed to be in P.
- Training search optimality (the greedy split selection) is heuristic and not formally verified.
- Candidate caps in the Square2CNF search phase are empirical heuristics.

## 8. theorem_strict Usage Warning

`theorem_strict` must be configured intentionally during evaluation/training. Do not manually toggle `theorem_strict=True` after training an empirical or `BEST_PER_NODE` tree unless all checked paths are certified. Otherwise, `NonTheoremPathError` is the expected behavior.
