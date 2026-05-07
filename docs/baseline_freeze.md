# Baseline Freeze Record

> **Date**: 2026-03-04
> **Phase**: F — Final Baseline Validation
> **Test suite**: 199 tests passing

## Authoritative Implementation

The `gsnh_mdt` package (`src/gsnh_mdt/`) is the **sole maintained
implementation** of GSNH-MDT. The original monolith (`gsnh_mdt_v3.py`) is
archived in `archive/` and must not be modified.

## Full Benchmark Confirmation

The complete benchmark ran successfully from the package implementation.

| Metric | Value |
|--------|-------|
| **Datasets** | 46 |
| **Splits per dataset** | 10 (StratifiedShuffleSplit, seed=42) |
| **Models per split** | 3 (sklearn DT d=7, GSNH d=5, GSNH d=7) |
| **Total model fits** | 1,380 |
| **Runtime** | 94 minutes |
| **Skipped** | 0 |
| **Failed** | 0 |
| **Import source** | `gsnh_mdt.tree.builder.ExpertGSNHTree` (package) |

### Benchmark Results Summary

| Model | Avg Accuracy | Avg Tree Size | Avg AXp Length |
|-------|-------------|---------------|----------------|
| sklearn DT (d=7) | 0.8604 ± 0.1222 | 69.7 | 5.21 |
| GSNH-MDT (d=5) | 0.8617 ± 0.1171 | 19.5 | 2.83 |
| GSNH-MDT (d=7) | 0.8647 ± 0.1188 | 27.5 | — |

### Win/Loss vs sklearn DT (d=7)

| Model | Wins | Losses | Ties |
|-------|------|--------|------|
| GSNH (d=5) | 13 | 10 | 23 |
| GSNH (d=7) | 16 | 6 | 24 |

### GSNH vs Blossom MDT (d=5) — 46 datasets

| | GSNH | Blossom |
|--|------|---------|
| Avg accuracy | 0.8617 | 0.8608 |
| Avg tree size | 19.5 | 17.2 |
| Wins | 16 | 12 |
| Ties | 18 | — |

### Numerical Deviations

**None observed.** The benchmark ran from the package implementation with
zero runtime errors and produced results consistent with the established
baseline. No behavior-level deviations were detected.

The package implementation is confirmed as the execution path for all
future experiments.

## Regression Coverage

| Suite | Count | Datasets | What's Frozen |
|-------|-------|----------|---------------|
| Leaf modules | 36 | — | Literals, scoring, prefix sums, SAT solvers |
| Original regression | 24 | vote, hepatitis, lymph | Golden baselines |
| Expanded regression | 72 | 9 datasets | Accuracy, tree size, arity, predict_proba, AXp |
| Method-level regression | 35 | 5 datasets | predict_proba, predict, extract_axp, weak_axp_check, score |
| Config parity | 27 | 5 datasets | from_config() vs legacy constructor |
| API smoke | 5 | — | Import, fit, predict cycle |

### Numerical Tolerance

- `predict_proba`: exact match within 1e-10
- `score` (accuracy): exact match within 1e-6
- `extract_axp`: exact feature set match

## Known Limitations

1. **Coverage is representative, not exhaustive.** Behavioral equivalence is
   verified on regression datasets but not mathematically proven for all inputs.

2. **Floating-point sensitivity.** Small changes to NumPy/Python version may
   cause bit-level divergence in probability outputs.

3. **`_traverse()` on 1D arrays.** Single-sample fallback method expects 2D
   input. `_batch_traverse()` is the primary code path.

4. **Config parity** is verified for `BEST_PER_NODE` on 5 datasets. Other
   combinations share the same constructor path but lack dedicated parity tests.

5. **`CalibrationConfig`** is defined but not yet wired through `from_config()`.

## Claims That Must Not Be Overstated

- Behavior is preserved **under current regression coverage**, not universally.
- Config parity is verified on **representative datasets**, not all.
- The package is the **maintained** implementation; the monolith is **archived**.
- Tests are strong but **do not constitute a mathematical proof** of total
  equivalence between package and monolith.

## Settings Classification

| Classification | Count | Fields |
|---------------|-------|--------|
| **BASELINE** | 14 | stopping, n_bins, binning_strategy, top_k_features, use_gain_ratio, laplace_smoothing, search_1d/2d/3d, mode, language, verbose, limit_2d, limit_3d |
| **ENHANCED** | 7 | use_supervised_binning, use_attention, use_look_ahead, look_ahead_gamma, look_ahead_top_p, prune, prune_alpha |
| **EXPERIMENTAL** | 2 | use_binary_comparisons, enable_compare_literals |
