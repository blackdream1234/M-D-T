# Theorem Compliance Protocol

This document outlines the strict boundary between the theorem-certified mathematical core of the GSNH-MDT framework and its empirical, heuristic, or auxiliary components.

## 1. Verified Mathematical Core

The Coq development proves the following formal properties:
- **Numeric threshold atom**: `B(f,t) := x_f >= t`.
- **Literal encoding**: `GE(f,t) := B(f,t)` and `LT(f,t) := not B(f,t)`.
- **Structural threshold order clauses**: If `t1 <= t2`, then `B(f,t2) -> B(f,t1)`, encoded as `not B(f,t2) OR B(f,t1)`.
- **Ordered selected-path CNF**: Formed by selected feature unit clauses + structural threshold-order clauses + path branch clauses.
- **Weak AXp reflection**: Checker result is `true` iff all opposite paths are blocked under selected features.
- **Deletion-based AXp extraction**: Returns a numeric subset-minimal weak AXp via deterministic feature removal.

## 2. Fixed Horn Theorem Mode

- Certified only for the implemented **structural Horn SAT fragment** over supported threshold literals.
- The Python checker verifies ordinary Horn SAT-fragment shape and routes to `structural_horn`.
- Unsupported literals such as `CompareLiteral`, `GSNHBinaryLiteral`, and Affine/XOR predicates are rejected in `theorem_strict` mode.
- Do **not** claim the Python implementation checks the stronger star-nested Horn language unless a separate star-nested validator is enabled.
- No DFS fallback may be used for theorem-certified Horn rows.

## 3. Fixed AntiHorn Theorem Mode

- Certified only for the implemented **structural AntiHorn SAT fragment** over supported threshold literals.
- The Python checker verifies ordinary AntiHorn SAT-fragment shape and routes to `structural_antihorn`.
- Unsupported literals are rejected in `theorem_strict` mode.
- Do **not** claim the Python implementation checks the stronger star-nested AntiHorn language unless a separate star-nested validator is enabled.
- No DFS fallback may be used for theorem-certified AntiHorn rows.

## 4. Square2CNF Theorem Mode

- Certified only when the path is encoded as explicit 2-CNF and solved by the `two_sat` backend.
- A Square2CNF row may enter theorem-certified benchmark tables only when `axp_backend="two_sat"`, `path_certificate="2cnf"`, `theorem_certified=True`, and `theorem_mode_used=True`.
- The false branch of `(l1 OR l2) AND (l3 OR l4)` is encoded by the exact 4-clause 2-CNF complement over the original literals:
  `(not l1 OR not l3) AND (not l1 OR not l4) AND (not l2 OR not l3) AND (not l2 OR not l4)`.
- Candidate caps in the Square2CNF search phase are empirical heuristics and are not theorem claims about global training optimality.

## 5. Affine / GF(2)

- The Coq proof now covers GF(2) normalization, RHS-flip complement, row-operation preservation, and contradiction certificates.
- Python benchmark certification remains disabled for Affine until a verified GF(2) certificate checker is implemented in Python.
- `ExactSATSolver.affine_sat` may be used as an auxiliary solver, but its successful return value alone must not set `theorem_certified=True`.
- In `theorem_strict` mode, Affine paths are recorded as `axp_backend="affine"`, `path_certificate="affine_unverified"`, and `theorem_certified=False`; non-Boolean-compatible affine threshold structure is rejected.

## 6. BEST_PER_NODE

- `BEST_PER_NODE` / BestPN is empirical by default.
- BestPN does not enable Affine search by default.
- BestPN is theorem-certified only path-by-path when every checked path receives an accepted Horn, AntiHorn, or 2-CNF certificate.
- Unrestricted mixed BEST_PER_NODE is not claimed to be polynomial or theorem-certified.
- Mixed paths that do not satisfy certificate conditions are rejected in `theorem_strict` mode or kept out of theorem-certified benchmark tables.

## 7. Auxiliary/Prototype Methods

- ConjUI, Affine, fallback interval DFS, prototype case-split solvers, and unrestricted mixed-family paths are auxiliary/empirical unless separately certified by an accepted path certificate.
- Benchmark theorem tables must exclude `interval_dfs_fallback`, `prototype_case_split`, `rejected_non_theorem`, `affine`, and `none` backends.

## 8. What is NOT Verified

- The Python implementation is tested and theorem-aligned, but it is **not** fully formally verified and is **not** extracted directly from Coq.
- Benchmark accuracy and size reductions are empirical findings, not formally verified.
- Unrestricted mixed-family SAT across heterogeneous nodes is not claimed to be in P.
- Training search optimality and greedy split selection are heuristic and not formally verified.

## 9. theorem_strict Usage Warning

`theorem_strict` must be configured intentionally during evaluation/training. Do not manually toggle `theorem_strict=True` after training an empirical or `BEST_PER_NODE` tree unless all checked paths are certified. Otherwise, `NonTheoremPathError` is the expected behavior.

## Affine Coq proof progress

Affine/XOR predicates are tractable over Boolean domains because they correspond to linear equations over GF(2). The file `coq/GSNH_Affine_AXp.v` now formalizes a separate Affine proof layer over Boolean threshold atoms.

Current Affine Coq coverage:

- affine equation semantics over signed Boolean threshold atoms;
- false-branch encoding by flipping the right-hand side;
- affine path encoding correctness;
- affine weak-AXp encoding reflection;
- finite exhaustive affine reference checker;
- solver soundness and candidate-level completeness for the exhaustive checker;
- path-level and all-opposite-path UNSAT soundness;
- selected-feature constrained UNSAT soundness;
- finite assignment representation for affine atoms;
- assignment-agreement preservation;
- completeness bridge from equality-compatible assignments to finite candidates;
- compatible-assignment AXp blocking.

Affine is still not included in the theorem-certified benchmark table because the current development does not yet include:

- a verified polynomial GF(2) Gaussian-elimination solver;
- integration with structural threshold-order clauses;
- numeric AXp equivalence for Affine;
- subset-minimal extraction theorem specialized to Affine;
- Python theorem-metadata admission for an `affine` certificate.

Therefore, Affine remains reported as an auxiliary family in the current benchmark, although its Boolean tractability is theoretically supported.
