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
- Rust unit/integration tests covering shape validation, label validation,
  row-major layout, feature summaries, `.dl8` parsing contract, and bitset mask
  behavior.

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

1. Threshold predicate evaluation with Python-vs-Rust mask equivalence.
2. Entropy/information-gain scoring with numeric parity tests.
3. Deterministic 1D candidate generation and chosen-split equivalence.
4. Horn/AntiHorn/ConjUI/Square2CNF/Affine predicate families, one at a time.
5. Small-tree prediction equivalence.
6. PyO3/maturin wrapper exposing `engine="python"`, `engine="rust"`, and
   `engine="compare"`.
7. Benchmarks with speedup ratios only after correctness parity is stable.

## Known limitations

- No PyO3 binding yet.
- No Rust predicate evaluation yet.
- No Rust split search yet.
- No Rust tree construction yet.
- No theorem certification is moved to Rust in this phase.
- Python remains the only production engine and oracle.

## Next safe optimization step

Implement threshold predicate mask evaluation using `Dataset` and `BitSet`, then
add Python-vs-Rust equivalence tests showing that Rust masks match Python
`GSNHLiteral.evaluate()` on the same deterministic inputs.
