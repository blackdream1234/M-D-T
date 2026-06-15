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
- `ComposedPredicate`: generic Python-matched mask composition for AND, OR,
  and XOR over threshold predicates.
- Label masks and basic scoring parity: positive/negative label masks, class
  counts, entropy, information gain, gain ratio, and BIC-style penalized gain.
- Deterministic 1D threshold candidate generation and best-split selection for
  scalar threshold predicates.
- Fixed ConjUI composed-candidate evaluation with class counts, min-leaf
  filtering, information gain, and BIC-style penalized gain.
- Fixed Affine/XOR composed-candidate evaluation for supplied XOR predicates,
  without search enumeration or theorem certificates.
- Fixed Horn and AntiHorn OR-clause candidate evaluation for supplied predicates,
  including Python-matched polarity validation.
- Rust unit/integration tests covering shape validation, label validation,
  row-major layout, feature summaries, `.dl8` parsing contract, bitset mask
  behavior, predicate mask behavior, composed mask behavior, scoring formulas,
  ConjUI/Affine/Horn/AntiHorn fixed-candidate evaluation, and 1D threshold search.

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


## Predicate-composition mask status

`rust_gsnh/src/predicates.rs` now includes a generic `ComposedPredicate` mask
layer over threshold literals.  This mirrors the inspected Python
`GSNHPredicate.evaluate()` semantics: Horn and AntiHorn clauses use OR over
literal masks, ConjUI / box predicates use AND, and Affine's current auxiliary
Python representation uses XOR / odd parity.  Empty composed predicates return
an error, matching Python's `GSNHPredicate` arity check that rejects zero
literals.

This generic mask-composition layer intentionally remains separate from
family-specific validation and theorem certification.  Horn and AntiHorn
polarity validation is now enforced by the fixed-candidate evaluators in
`search.rs`, while `ComposedPredicate` itself still just composes masks.
Square2CNF's conjunction of binary disjunctive clauses and Affine/GF(2)
certificate rules remain outside this generic layer.  The tests manually compute
Python-equivalent AND, OR, and XOR masks for two and three literals, invalid
literal handling, deterministic sorted indices, class counts, and
min-leaf-style branch-size checks.  Automated Python-vs-Rust composed-predicate
equivalence remains a TODO for the PyO3 binding phase.

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


## ConjUI fixed-candidate evaluation status

`rust_gsnh/src/search.rs` now includes
`evaluate_composed_candidate_with_min_leaf` for one fixed ConjUI predicate.  The
function accepts only `MaskOp::And`, evaluates the composed predicate mask,
forms the outside mask as the complement, enforces `min_samples_leaf` on both
branches, computes inside/outside `ClassCounts`, computes Python-matched
information gain, and then applies the existing BIC-style `penalized_gain` with
the caller-provided arity.  Candidates whose branch sizes are invalid or whose
raw/penalized gains are nonpositive return `Ok(None)`, matching Python's
candidate-rejection behavior.

This is not a search enumerator: callers must provide a concrete ConjUI
predicate.  Square2CNF and Affine certificate/search support remain unsupported
in Rust.  In particular, this step does not implement Square2CNF clause
structure, Affine/GF(2) certificates, tree recursion, PyO3 bindings, or
benchmark integration.


## Affine/XOR fixed-candidate evaluation status

`rust_gsnh/src/search.rs` now also includes
`evaluate_affine_candidate_with_min_leaf` for one supplied fixed Affine/XOR
predicate.  The function accepts only `MaskOp::Xor`, evaluates odd-parity masks
through `ComposedPredicate`, forms the outside mask as the complement, enforces
`min_samples_leaf` on both branches, computes inside/outside `ClassCounts`,
computes Python-matched information gain, and applies the existing BIC-style
`penalized_gain` with the caller-provided arity.  Invalid branch sizes and
nonpositive raw/penalized gains return `Ok(None)`.

This is only fixed XOR mask evaluation and scoring.  Rust still does not
enumerate Affine predicates, construct GF(2) bases, check Affine certificates,
validate theorem conditions, or integrate Affine with tree search or benchmarks.

## Horn and AntiHorn fixed-candidate evaluation status

`rust_gsnh/src/search.rs` now includes
`evaluate_horn_candidate_with_min_leaf` and
`evaluate_antihorn_candidate_with_min_leaf` for supplied fixed OR-clause
predicates.  Both functions accept only `MaskOp::Or`, evaluate OR masks through
`ComposedPredicate`, form the outside mask as the complement, enforce
`min_samples_leaf` on both branches, compute inside/outside `ClassCounts`,
compute Python-matched information gain, and apply the existing BIC-style
`penalized_gain` with the caller-provided arity.  Invalid branch sizes and
nonpositive raw/penalized gains return `Ok(None)`.

The polarity rules match Python `GSNHPredicate.__post_init__` and literal
polarity conventions: `>=` and `>` are positive directions, while `<` and `<=`
are negative directions.  Horn accepts at most one positive literal in the OR
clause; AntiHorn accepts at most one negative literal.  Violations return clear
`Err` values before scoring.  This step does not enumerate Horn/AntiHorn
candidates, validate theorem certificates, or integrate either family with Rust
tree search.

## Square2CNF fixed-candidate evaluation status

`rust_gsnh/src/square_cnf.rs` now implements a Square2CNF-specific fixed
predicate structure instead of forcing Square2CNF through `ComposedPredicate`.
The public structures are `Square2Clause`, `Square2CNFPredicate`,
`Square2CNFCandidate`, and `EvaluatedSquare2CNFPredicate`.  A clause evaluates
as the OR/union of two `ThresholdPredicate` masks, and the full predicate
evaluates as the AND/intersection of all clause masks.

This mirrors Python `Square2CNFPredicate`: the current Python name is
`Square2CNF` / `LanguageFamily.SQUARE_2CNF`, while legacy `SquareCNF` refers to
the older ConjUI/box-style code path.  Python accepts one to three clauses and
requires each clause to contain exactly two supported threshold literals; it
rejects zero clauses, more than three clauses, non-binary clauses, and
Compare/Binary literals.  The Rust structure has fixed two-literal clauses by
type and returns an error for zero or more than three clauses at evaluation
time.

`evaluate_square2cnf_candidate_with_min_leaf` evaluates one supplied predicate,
forms the outside mask as the complement, enforces `min_samples_leaf` on both
branches (`0` disables branch-size rejection, matching the existing Rust/Python
count convention), computes inside/outside `ClassCounts`, computes
Python-matched information gain, and applies the existing BIC-style
`penalized_gain` with the caller-provided arity.  Invalid branch sizes and
nonpositive raw/penalized gains return `Ok(None)`.

This step does not enumerate Square2CNF candidates, implement the Python
`search_square_2cnf` candidate cap or feature-pair pruning, construct theorem
certificates, integrate with benchmarks, or perform tree recursion.

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
arity 1.  Python-compatible `min_samples_leaf` filtering is exposed through the
`*_with_min_leaf` APIs; a candidate is skipped unless both inside and outside
row counts are at least `min_samples_leaf`.

Tie-breaking is deterministic: higher score wins; then smaller feature index;
then smaller threshold; then fixed operator order (`LessThan` before
`GreaterEqual`).  This preserves Python's ascending feature scan and first-best
behavior for the Rust subset implemented here.

No PyO3 binding exists yet, so 1D candidate parity is covered by deterministic
Rust tests whose expected thresholds, masks, class counts, leaf-size filtering,
and scores are computed from the inspected Python conventions.  Automated
Python-vs-Rust candidate equivalence remains a TODO for the binding phase.

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

1. Add a unified fixed-predicate evaluation facade that dispatches by family name.
2. Language-family predicate validation/search one family at a time.
3. Small-tree prediction equivalence.
4. PyO3/maturin wrapper exposing `engine="python"`, `engine="rust"`, and
   `engine="compare"`.
5. Benchmarks with speedup ratios only after correctness parity is stable.

## Known limitations

- No PyO3 binding yet.
- No composed-predicate enumeration/search yet; Rust only evaluates supplied ConjUI, Affine/XOR, Horn, and AntiHorn predicates.
- Horn and AntiHorn are fixed-predicate evaluators only; there is no Rust Horn/AntiHorn search, theorem certificate checker, or benchmark integration.
- Square2CNF fixed-candidate evaluation exists, but Square2CNF candidate enumeration/search and theorem certificates are still not implemented in Rust.
- Affine/XOR evaluation is mask/scoring only; there is no Affine enumeration,
  GF(2) basis construction, theorem certificate checker, or benchmark integration.
- No full Rust split search beyond deterministic 1D threshold candidates yet.
- The non-suffixed Rust 1D helpers use `min_samples_leaf = 1` for backward
  compatibility; callers that need Python tree parity should call the
  `*_with_min_leaf` APIs with the tree's configured value.
- High-cardinality quantile binning remains Python-only for now; Rust currently
  mirrors the exact-value midpoint threshold convention used for low-cardinality
  node-local 1D candidates.
- No Rust tree construction yet.
- No theorem certification is moved to Rust in this phase.
- Python remains the only production engine and oracle.

## Next safe optimization step

Implement a unified fixed-predicate evaluation facade that dispatches by family name now that ConjUI, Affine/XOR, Horn, AntiHorn, and Square2CNF fixed evaluators are present.  Do not implement search or tree recursion until family-level predicate masks and scores remain stable behind that facade.
