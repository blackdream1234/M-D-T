# Migration Plan

## Current State

The GSNH-MDT codebase has been extracted from a 4,465-line monolith (`gsnh_mdt_v3.py`) into an installable Python package (`gsnh_mdt/`). This is a **safe extracted baseline under current regression coverage**.

### Completed Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 — Audit | ✅ | Full inventory of 177 items, baseline/enhanced boundary map |
| Phase 2 — Safe Extractions | ✅ (15/16) | All modules extracted verbatim; 2.14 (explainer split) deferred |
| Phase 4 — Packaging (partial) | ✅ | `pyproject.toml`, `.gitignore`, pip-installable |

### Test Coverage

65 tests passing: literals (13), scoring (9), prefix sums (6), SAT solvers (9), tree regression (24), API smoke (5).

---

## Remaining Phases

### Phase A — Documentation & Tooling (current)

1. ~~`README.md`~~ ✅
2. ~~`docs/architecture.md`~~ ✅
3. ~~`docs/design_decisions.md`~~ ✅
4. ~~`docs/migration_plan.md`~~ ✅ (this file)
5. `ruff` + `black` + `pre-commit` configuration

### Phase B — Expanded Regression Coverage

**Prerequisite for any builder.py split.**

Add 4–6 more datasets to the golden baselines:

| Dataset | Size | Features | Why |
|---------|------|----------|-----|
| `tic-tac-toe.dl8` | 958 | 27 | binary, moderate interaction |
| `ionosphere.dl8` | 351 | 780 | high-dimensional |
| `kr-vs-kp.dl8` | 3,196 | 73 | large, interaction-heavy |
| `mushroom.dl8` | 8,124 | 117 | large binary classification |
| `anneal.dl8` | 798 | 93 | multi-class edge case |
| `heart-cleveland.dl8` | 303 | 52 | small clinical |

Additional outputs to freeze per dataset:
- `predict_proba` for 5 test samples (not just 3)
- `get_summary()` fields: `n_nodes`, `n_leaves`, `max_depth`, `arity_counts`, `pattern_counts`
- AXp length for 3 test samples
- Root split string
- Reproducibility (2 identical fits)

### Phase C — Builder Split

**Prerequisites**: Phase B regression suite passes.

Split `tree/builder.py` (1,462 lines) into:

| Target | Methods | Lines (est.) |
|--------|---------|-------------|
| `tree/builder.py` | `__init__`, `fit`, `_build_tree`, `_search_best_split`, `_build_pred_*`, `_make_literal`, `_build_tensors_*`, `_compute_*`, `_look_ahead_score`, `_fast_1d_scan`, `score`, `print_tree`, `get_summary` | ~1,100 |
| `tree/prediction.py` | `predict_proba`, `predict`, `_batch_traverse`, `_traverse` | ~40 |
| `tree/explainer.py` | `weak_axp_check`, `_is_sat_path`, `_path_sat_numeric`, `_solve_or_clauses_dfs`, `_affine_path_sat`, `extract_axp` | ~220 |

**Strategy**: Use mixin classes or move methods with backward-compatible delegates on `ExpertGSNHTree`.

### Phase D — Config Dataclass Wiring

**Prerequisites**: Phase C complete, all tests pass.

Replace `ExpertGSNHTree.__init__(22 params)` with:
```python
ExpertGSNHTree(config: ModelConfig)
```

Steps:
1. Add `ModelConfig` as optional first argument with backward-compatible kwargs
2. Deprecation warnings on old kwargs
3. Update `GSNHClassifier._create_model()`, `GSNHRandomForest`, benchmark scripts
4. Remove old kwargs after one version cycle

### Phase E — Final Polish

- `pre-commit` hooks enforced
- `ruff` clean on all source
- Full 46-dataset benchmark regression
- `prooftest` integration test
- CI configuration (if applicable)
- Retire `gsnh_mdt_v3.py` monolith

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Builder split changes behavior | High | Expanded regression suite (Phase B) |
| Config wiring breaks downstream | Medium | Backward-compatible kwargs + deprecation |
| JIT recompilation differences | Low | `@njit(cache=True)` preserves compilation |
| Floating-point tolerance | Low | Tests use 10⁻⁶ tolerance, not exact equality |

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase A (docs + tooling) | 1 session |
| Phase B (expanded regression) | 1 session |
| Phase C (builder split) | 1–2 sessions |
| Phase D (config wiring) | 1 session |
| Phase E (polish) | 1 session |
