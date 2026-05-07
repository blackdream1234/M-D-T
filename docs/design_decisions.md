# Design Decisions

This document records key design decisions made during the monolith-to-package refactoring, with rationale.

## 1. Verbatim Extraction Over Incremental Refactoring

**Decision**: Extract code verbatim from the monolith via `sed`, preserving every method, default value, and control flow path.

**Rationale**: The monolith contains algorithmically complex, numerically sensitive code (JIT-compiled search kernels, SAT solvers, entropy calculations). Any accidental change — even whitespace in `@njit` functions — can alter Numba compilation behavior. Verbatim extraction with golden baseline regression tests is the only safe approach.

**Trade-off**: `tree/builder.py` is temporarily 1,462 lines. This is acceptable because behavior preservation matters more than code elegance during Phase 2.

## 2. `sed` Extraction vs Manual Copying

**Decision**: Use `sed -n 'start,end'p` to extract class bodies, then prepend import headers.

**Rationale**: Eliminates transcription errors. Any hand-copied code risks typos in thresholds, loop bounds, or JIT decorators that would silently change behavior. `sed` guarantees byte-level fidelity of the class body.

## 3. StoppingCriteria Returns `Tuple[bool, str]`

**Decision**: Preserve the `(should_stop, reason_string)` return type from the monolith.

**Rationale**: The initial extraction simplified this to `bool`, which broke `_build_tree()` at line 1097 where the result is unpacked as `should_stop, reason = ...`. The reason string is used for logging and debugging during tree construction.

**Lesson**: Even "simple" dataclasses cannot be simplified during extraction.

## 4. Baseline vs Enhanced Feature Labels

**Decision**: Every module docstring explicitly labels features as **baseline** (journal-compliant) or **enhanced** (research/optimization).

**Rationale**: The thesis requires demonstrating that the baseline algorithm matches the journal submission. Enhanced features (supervised binning, attention, look-ahead) must be independently togglable and clearly documented.

| Category | Examples |
|----------|---------|
| Baseline | Horn/Anti-Horn/Affine search, exact SAT solvers, `_search_best_split` core loop |
| Enhanced | Supervised binning, interaction attention, look-ahead re-ranking, pruning, calibration |

## 5. Config Dataclasses Are Defined But Not Wired

**Decision**: `config.py` contains `ModelConfig`, `SearchConfig`, `OptimizationConfig`, but these are not yet used by `ExpertGSNHTree.__init__()`.

**Rationale**: Wiring them would change the constructor signature, which is called by `GSNHClassifier`, `GSNHRandomForest`, benchmark scripts, and test suites. This is a Phase 3 task that requires careful shim construction.

## 6. `@njit` Functions Remain Module-Level

**Decision**: All JIT-compiled functions stay as module-level functions, never as class methods.

**Rationale**: Numba's `@njit` cannot compile methods that reference Python class instances. The existing design keeps JIT functions stateless and composable via imports. This pattern is preserved exactly.

## 7. Golden Baseline Regression Tests

**Decision**: Capture and freeze exact outputs (accuracy, tree size, root split string, `predict_proba` values, AXp lengths) from the monolith before any extraction.

**Rationale**: Standard unit tests check individual functions. Regression tests check end-to-end behavior: that the full pipeline (binning → search → build → predict → explain) produces identical results after extraction. Any deviation is detected before it can propagate.

## 8. No Circular Import Risk

**Decision**: Strict layered dependency: `api → ensembles → tree → {search, scoring, sat, literals, preprocess}`.

**Rationale**: The monolith had all code in one file — no import order issues. The package must enforce a DAG. Each subpackage's `__init__.py` re-exports only from its own modules. No upward imports are permitted.

## 9. Scripts Copied, Not Moved

**Decision**: `benchmark_dl8.py` and `prooftest` are copied to `scripts/`, not moved from the project root.

**Rationale**: The original scripts may still be used independently against the monolith. Once the package is fully validated and the monolith is retired, the originals can be removed.

## 10. `tree/builder.py` Split Is Deferred

**Decision**: Prediction (`predict`, `predict_proba`, `_traverse`, `_batch_traverse`), explainability (`weak_axp_check`, `extract_axp`, `_is_sat_path`, `_path_sat_numeric`, etc.), and build logic remain co-located in `builder.py`.

**Rationale**: Splitting requires:
1. Expanded regression coverage (more datasets, more outputs frozen)
2. Careful method-by-method extraction with the same `sed`-then-verify pattern
3. Compatibility shims so that `tree.predict_proba(...)` still works from `builder.py`

This is planned for Phase 4 (builder split phase), not Phase 2 (safe extraction).
