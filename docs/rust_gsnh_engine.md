# Rust GSNH Engine Plan

## Boundary

The existing Python GSNH implementation remains the reference/oracle.  Rust is
being added beside it for speed, memory layout, and future candidate-search
optimizations.  Rust must not become the default engine until Python-vs-Rust
agreement tests pass for the relevant layer.

## Repository inspection summary

- Current Python tree implementation: `src/gsnh_mdt/tree/builder.py` contains
  `ExpertGSNHTree`, split search orchestration, recursive tree construction,
  and public prediction/score methods.
- Predicate and literal representation: `src/gsnh_mdt/literals/base.py`,
  `src/gsnh_mdt/literals/binary.py`, `src/gsnh_mdt/literals/compare.py`, and
  `src/gsnh_mdt/literals/predicates.py` define threshold, comparison, Horn,
  AntiHorn, ConjUI, Square2CNF, and Affine/XOR predicate objects.
- Language-family enums and allowed polarity patterns live in
  `src/gsnh_mdt/types.py`.
- Candidate split generation is split across `src/gsnh_mdt/search/`, including
  `exhaustive_1d.py`, `exhaustive_2d.py`, `exhaustive_3d.py`, `antihorn.py`,
  `conj_ui.py`, `square_2cnf.py`, and `affine_search.py`.
- Scoring is in `src/gsnh_mdt/scoring/entropy.py`, `gain.py`, and
  `penalties.py`; information gain is the core split score.
- Prediction traversal is exposed by `ExpertGSNHTree.predict()` and delegated to
  `src/gsnh_mdt/tree/prediction.py`.
- Benchmark orchestration and CSV/LaTeX/plot/report generation are in
  `scripts/benchmark_dl8_languages_updated.py`; this stays Python-owned.

## Target split of responsibilities

Python keeps dataset loading, experiment orchestration, sklearn comparison,
CSV/plot/LaTeX/report output, theorem-boundary filtering, and correctness
comparison.  Rust receives compact data structures, bitsets, predicate
execution, scoring, deterministic candidate search, tree prediction, caching,
and later optional parallel search.

## Phase 1 implemented now

The new crate `rust_gsnh/` has been added.  The currently implemented safe
layers are:

- `Dataset`: dense row-major `f64` features plus binary `u8` labels.
- `FeatureSummary`: per-feature min/max/constant/binary metadata.
- `.dl8` text parser matching the benchmark convention: first column is label,
  remaining columns are features.
- `BitSet`: compact deterministic `Vec<u64>` masks for future sample-set and
  predicate-mask operations.
- `ThresholdPredicate`: deterministic per-row threshold mask evaluation from a
  `Dataset` into a `BitSet`.
- Label masks and basic scoring parity: positive/negative label masks, class
  counts, entropy, information gain, gain ratio, and BIC-style penalized gain.
- Deterministic 1D threshold candidate generation and best-split selection for
  scalar threshold predicates.
- Rust unit/integration tests covering shape validation, label validation,
  row-major layout, feature summaries, `.dl8` parsing contract, bitset mask
  behavior, predicate mask behavior, scoring formulas, and 1D threshold search.

## Bitset module status

`rust_gsnh/src/bitset.rs` now supports checked `set`, `unset`, and `contains`,
construction from indices, all-set construction, sorted index extraction,
cardinality, union, intersection, difference, complement, subset checks, and
intersection checks.  All binary operations validate equal lengths, and
out-of-range indices return errors instead of panicking.  Complements mask the
unused padding bits in the final `u64`, so lengths such as 70 do not expose
extra bits.

Bitsets are needed for future GSNH speed because split predicates naturally
produce sample masks.  A compact `Vec<u64>` representation will allow fast
class-counting, branch partitioning, candidate reuse, and cached predicate masks
without repeatedly allocating Python/NumPy boolean arrays in the Rust core.

## Predicate module status

`rust_gsnh/src/predicates.rs` now implements threshold predicate mask evaluation
for `LessEqual`, `LessThan`, `GreaterEqual`, and `GreaterThan` using ordinary
finite `f64` comparison semantics.  The Python reference for threshold literals
is `GSNHLiteral.evaluate()`: `LiteralPolarity.GE` evaluates `X[:, feature] >=
threshold`, and `LiteralPolarity.LT` evaluates `X[:, feature] < threshold`.
Python's `CompareLiteral` also defines `<=`, `<`, `>=`, and `>` semantics for
feature-to-feature comparisons, so the Rust threshold layer supports the same
ordered comparison operators against scalar thresholds.

`ComparisonOp::Equal` is present in the API but intentionally returns an error
because the Python GSNH threshold literal family does not define equality
predicates.  Ordered comparisons involving NaN follow Python/NumPy behavior for
ordinary comparisons: they evaluate to false.  No PyO3 binding exists yet, so
the current equivalence-style tests mirror the documented Python semantics in
Rust; automated Python-vs-Rust predicate tests are deferred until bindings are
introduced.

## Label-mask and scoring module status

`rust_gsnh/src/scoring.rs` now provides `positive_label_mask`,
`negative_label_mask`, count helpers, and `ClassCounts` for masks built from a
`Dataset`.  Length mismatches return errors through the underlying `BitSet`
operations instead of panicking.

The implemented scoring formulas match the inspected Python references in
`src/gsnh_mdt/scoring/`:

- `entropy(pos, neg)`: binary entropy in base 2, returning `0.0` for empty or
  pure counts.
- `information_gain(total_pos, total_neg, in_pos, in_neg)`: parent entropy minus
  weighted child entropies, returning `-1.0` for empty parent/inside/outside
  splits and clamping valid negative roundoff to `0.0`.
- `gain_ratio(...)`: information gain divided by split information, returning
  `-1.0` when raw information gain is not positive, and returning raw IG if
  split information is below `1e-10`.
- `penalized_gain(raw_gain, arity, n_samples)`: BIC-style penalty
  `(arity + 1) * ln(max(n_samples, 2)) / (2 * n_samples)`, returning `-1.0`
  when raw or penalized gain is nonpositive.

No PyO3 binding exists yet, so scoring parity is covered by deterministic Rust
tests whose constants are computed from these Python formulas.  Automated
Python-vs-Rust scoring calls remain a TODO for the binding phase.

## Deterministic 1D candidate-generation status

`rust_gsnh/src/search.rs` now implements the smallest safe search layer: scalar
1D threshold candidate generation and best-split selection.  The threshold
convention matches Python's exact low-cardinality node-local behavior in
`ExpertGSNHTree._search_best_split`: sorted unique feature values are converted
to midpoints between adjacent unique values; constant features produce no
threshold candidates.  Quantile-binned high-cardinality behavior is not moved to
Rust yet.

For each threshold, Rust evaluates `LessThan` first and `GreaterEqual` second,
matching Python's low-anchor (`x < t`) before high-anchor (`x >= t`) exhaustive
1D scan.  Candidate masks use `inside = predicate true` and
`outside = complement(inside)`.  The score is Python's default tree-search 1D
objective: raw information gain followed by BIC-style `penalized_gain` with
arity 1.

Tie-breaking is deterministic: higher score wins; then smaller feature index;
then smaller threshold; then fixed operator order (`LessThan` before
`GreaterEqual`).  This preserves Python's ascending feature scan and first-best
behavior for the Rust subset implemented here.

No PyO3 binding exists yet, so 1D candidate parity is covered by deterministic
Rust tests whose expected thresholds, masks, class counts, and scores are
computed from the inspected Python conventions.  Automated Python-vs-Rust
candidate equivalence remains a TODO for the binding phase.

## Build and test

```bash
cd rust_gsnh
cargo test
```

From the repository root:

```bash
cargo test --manifest-path rust_gsnh/Cargo.toml
```

## Future safe implementation order

1. Language-family predicate composition one family at a time, starting with the simplest Python-matched family.
2. Small-tree prediction equivalence.
3. PyO3/maturin wrapper exposing `engine="python"`, `engine="rust"`, and
   `engine="compare"`.
4. Benchmarks with speedup ratios only after correctness parity is stable.

## Known limitations

- No PyO3 binding yet.
- No Rust predicate formulas beyond single-threshold masks yet.
- No full Rust split search beyond deterministic 1D threshold candidates yet.
- The Rust 1D API does not yet take `min_samples_leaf`; invalid empty splits
  match Python scoring by returning `-1.0`, while caller-level leaf-size pruning
  is deferred until the search/config layer is added.
- High-cardinality quantile binning remains Python-only for now; Rust currently
  mirrors the exact-value midpoint threshold convention used for low-cardinality
  node-local 1D candidates.
- No Rust tree construction yet.
- No theorem certification is moved to Rust in this phase.
- Python remains the only production engine and oracle.

## Next safe optimization step

Implement language-family predicate composition one family at a time, starting
with the simplest Python-matched family.  Do not implement tree recursion until
family-level predicate masks and scores are stable.
