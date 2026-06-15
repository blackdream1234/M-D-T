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

Candidate enumeration is documented separately below. Rust still does not
implement the Python `search_square_2cnf` candidate cap, theorem certificates,
benchmark integration, or tree recursion.

## Unified fixed-predicate family facade status

`rust_gsnh/src/family.rs` now provides the conservative dispatch layer for
already-constructed fixed predicates.  The Rust `LanguageFamily` enum supports
only the families that have fixed evaluators today: `Horn`, `AntiHorn`,
`ConjUI`, `Affine`, and `Square2CNF`.  Python families/names outside this
subset remain unsupported in Rust for now: `Any`, `BestPerNode`, and legacy
`SquareCNF`.  Legacy Python `SquareCNF` means the old ConjUI/box-style path,
whereas `Square2CNF` is the paper-style conjunction of two-literal OR clauses.

The facade uses `FixedPredicate::Composed` for Python `GSNHPredicate`-style
families: ConjUI (`MaskOp::And`), Affine/XOR (`MaskOp::Xor`), Horn
(`MaskOp::Or` plus Horn polarity validation), and AntiHorn (`MaskOp::Or` plus
AntiHorn polarity validation).  It uses `FixedPredicate::Square2CNF` only for
Square2CNF clause structures.

`evaluate_fixed_predicate_with_min_leaf` rejects mismatched shapes early:
Square2CNF with a composed predicate returns `Err`, and ConjUI/Affine/Horn/
AntiHorn with a Square2CNF predicate returns `Err`.  Wrong mask operators,
invalid Horn/AntiHorn polarity, invalid literal feature indices, empty
predicates, min-leaf rejection, and nonpositive-gain rejection are delegated to
the existing fixed-family evaluators, so no scoring logic is duplicated.

This is still fixed-predicate evaluation only.  It does not enumerate
candidates, recurse trees, bind to Python, use parallelism, integrate with
benchmarks, or construct theorem certificates.

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

## Deterministic ConjUI search status

`rust_gsnh/src/conjui.rs` now implements the first composed-family search
layer: deterministic ConjUI enumeration up to arity 2.  This is still not tree
recursion and does not search Horn, AntiHorn, Affine, or Square2CNF.

The matched Python reference is `src/gsnh_mdt/search/conj_ui.py` plus the
ConjUI branch in `ExpertGSNHTree._search_best_split`.  Python's optimized
ConjUI kernels support 2D and 3D box/intersection search with AND semantics,
all polarity configurations, direct prefix-sum box counts, `min_leaf` filtering
on both branches before gain computation, raw information gain, and then
BIC-style `penalized_gain` in the builder.  Legacy `SquareCNF` means this
ConjUI/box family, not the newer `Square2CNF` clause family.

The Rust subset deliberately starts smaller: arity 1 and arity 2 only.  Arity 1
enumerates one-literal `MaskOp::And` predicates and matches the existing Rust
1D threshold search.  Arity 2 enumerates canonical feature pairs `i < j`,
midpoint thresholds from `generate_1d_thresholds`, and the four Python ConjUI
polarity configurations in deterministic order: LT/LT, LT/GE, GE/LT, GE/GE.
This exhaustive mask-based implementation is simpler than Python's prefix-sum
kernel but represents the same small deterministic candidate set for the Rust
midpoint-threshold subset.

Candidates are scored only through the existing fixed ConjUI evaluator, so the
score remains raw information gain followed by BIC-style `penalized_gain`, and
`min_samples_leaf` uses the same branch sample-count rule as the fixed-family
layer.  `min_samples_leaf = 0` disables branch-size rejection, matching the
existing Rust convention.  If no positive penalized-gain candidate survives,
the function returns `Ok(None)`.

Tie-breaking is deterministic: higher score wins; then smaller arity; then the
lexicographically smaller literal sequence by feature index, threshold, and
operator order (`LessThan` before `GreaterEqual`).  This mirrors Python's
first-best deterministic scan where feasible while making ties explicit for the
standalone Rust subset.

Automated Python-vs-Rust ConjUI search equivalence remains a TODO until PyO3
bindings exist.  Current tests use small deterministic datasets with manually
computed masks, counts, and scores from the inspected Python semantics.

## Deterministic Horn search status

`rust_gsnh/src/horn.rs` now implements deterministic Horn enumeration up to
arity 2.  This is still not tree recursion and does not search AntiHorn,
Affine, Square2CNF, or benchmark-level method combinations.

The matched Python reference is `src/gsnh_mdt/search/exhaustive_2d.py`, the
Horn branch in `ExpertGSNHTree._search_best_split`, and the polarity tables in
`src/gsnh_mdt/types.py`.  Python's 2D Horn kernel searches three OR/union
polarity configurations: LT/LT, GE/LT, and LT/GE.  The omitted GE/GE pattern
would contain two positive literals and violates the Horn rule.  Python applies
`min_leaf` to both inside and outside branches before information gain, and the
builder then applies BIC-style `penalized_gain`.

The Rust subset deliberately supports arity 1 and arity 2 only.  Arity 1
enumerates one-literal `MaskOp::Or` predicates and matches the existing Rust 1D
threshold search.  Arity 2 enumerates canonical feature pairs `i < j`, midpoint
thresholds from `generate_1d_thresholds`, and the three Horn polarity
configurations in deterministic order: LT/LT, GE/LT, LT/GE.  Every candidate is
passed through the fixed Horn evaluator, so the at-most-one-positive-literal
rule is checked again before scoring.

Scoring and validity are shared with the fixed-family layer: raw information
gain, BIC-style `penalized_gain`, and branch sample-count filtering via
`min_samples_leaf`.  `min_samples_leaf = 0` disables branch-size rejection,
matching the existing Rust convention.  If no positive penalized-gain candidate
survives, the function returns `Ok(None)`.

Tie-breaking is deterministic: higher score wins; then smaller arity; then the
lexicographically smaller literal sequence by feature index, threshold, and
operator order (`LessThan` before `GreaterEqual`).  Automated Python-vs-Rust
Horn search equivalence remains a TODO until PyO3 bindings exist.

## Deterministic AntiHorn search status

`rust_gsnh/src/antihorn.rs` now implements deterministic AntiHorn enumeration
up to arity 2.  This is still not tree recursion and does not search Affine,
Square2CNF, or benchmark-level method combinations.

The matched Python reference is `src/gsnh_mdt/search/antihorn.py`, the AntiHorn
branch in `ExpertGSNHTree._search_best_split`, and the polarity tables in
`src/gsnh_mdt/types.py`.  Python's 2D AntiHorn kernel searches three OR/union
polarity configurations: GE/GE, GE/LT, and LT/GE.  The omitted LT/LT pattern
would contain two negative literals and violates the AntiHorn rule.  Python
applies `min_leaf` to both inside and outside branches before information gain,
and the builder then applies BIC-style `penalized_gain`.

The Rust subset deliberately supports arity 1 and arity 2 only.  Arity 1
enumerates one-literal `MaskOp::Or` predicates and matches the existing Rust 1D
threshold search.  Arity 2 enumerates canonical feature pairs `i < j`, midpoint
thresholds from `generate_1d_thresholds`, and the three AntiHorn polarity
configurations in deterministic order: GE/GE, LT/GE, GE/LT.  Every candidate is
passed through the fixed AntiHorn evaluator, so the at-most-one-negative-literal
rule is checked again before scoring.

Scoring and validity are shared with the fixed-family layer: raw information
gain, BIC-style `penalized_gain`, and branch sample-count filtering via
`min_samples_leaf`.  `min_samples_leaf = 0` disables branch-size rejection,
matching the existing Rust convention.  If no positive penalized-gain candidate
survives, the function returns `Ok(None)`.

Tie-breaking is deterministic: higher score wins; then smaller arity; then the
lexicographically smaller literal sequence by feature index, threshold, and
operator order (`LessThan` before `GreaterEqual`).  Automated Python-vs-Rust
AntiHorn search equivalence remains a TODO until PyO3 bindings exist.


## Deterministic Affine/XOR search status

`rust_gsnh/src/affine.rs` now implements deterministic Affine/XOR
enumeration up to arity 2.  This is still not tree recursion and does not
search Square2CNF or benchmark-level method combinations.

The matched Python reference is `src/gsnh_mdt/search/affine_search.py`, the
Affine branch in `ExpertGSNHTree._search_best_split`, and the Affine predicate
construction helpers in `src/gsnh_mdt/tree/builder.py`.  Python's optimized 2D
Affine kernel uses prefix-sum box counts and evaluates both XOR-true and
XOR-false/XNOR regions.  The builder represents the selected split as a
`GSNHPredicate` with `language_family=LanguageFamily.AFFINE` and `is_xor=True`.
Rust keeps the same public mask family, `MaskOp::Xor`; XNOR-style regions are
covered by enumerating XOR over opposite literal polarities rather than by
adding a separate XNOR operator.

The Rust subset deliberately supports arity 1 and arity 2 only.  Arity 1
enumerates one-literal `MaskOp::Xor` predicates and matches the existing Rust
1D threshold search.  Arity 2 enumerates canonical feature pairs `i < j`,
midpoint thresholds from `generate_1d_thresholds`, and the four deterministic
literal-polarity combinations LT/LT, LT/GE, GE/LT, and GE/GE.  This exhaustive
mask-based implementation is simpler than Python's integral-image search but
represents the same small deterministic threshold-literal XOR subset used by
the Rust engine.

Scoring and validity are shared with the fixed-family layer: raw information
gain, BIC-style `penalized_gain`, and branch sample-count filtering via
`min_samples_leaf`.  `min_samples_leaf = 0` disables branch-size rejection,
matching the existing Rust convention.  If no positive penalized-gain candidate
survives, the function returns `Ok(None)`.

Tie-breaking is deterministic: higher score wins; then smaller arity; then the
lexicographically smaller literal sequence by feature index, threshold, and
operator order (`LessThan` before `GreaterEqual`).  Automated Python-vs-Rust
Affine search equivalence remains a TODO until PyO3 bindings exist.

This search does not implement 3D Affine, GF(2) basis construction, theorem
certificate checking, benchmark integration, or tree recursion.


## Deterministic Square2CNF search status

`rust_gsnh/src/square_cnf.rs` now implements deterministic Square2CNF
candidate enumeration for `max_clauses = 1` and `max_clauses = 2`. This is
still candidate search only: it does not build trees, construct theorem
certificates, bind through PyO3, run in parallel, or integrate with benchmarks.

The inspected Python reference is `src/gsnh_mdt/search/square_2cnf.py` plus
`Square2CNFPredicate` in `src/gsnh_mdt/literals/predicates.py`. The active
family name is `Square2CNF` / `LanguageFamily.SQUARE_2CNF`; legacy
`SquareCNF` is the older ConjUI/box name. Python's active search generates
threshold literals in feature order with GE before LT, builds binary OR
clauses from literal pairs `(i, j)` with `i < j`, precomputes clause masks, and
combines clause pairs as `(clause_a AND clause_b)` with `a < b`. It skips
identical literals and skips two-clause formulas whose clauses use the exact
same feature set.

The Rust subset is intentionally exhaustive and deterministic for the midpoint
threshold literals already used by the Rust engine. It supports one-clause
predicates because Python `Square2CNFPredicate` accepts 1--3 clauses, and it
supports two-clause predicates because that is the active paper-style search
shape. Each clause is exactly two threshold literals and evaluates as OR over
its literal masks; the predicate evaluates as AND over clause masks.

Scoring and validity are delegated to `evaluate_square2cnf_candidate_with_min_leaf`:
inside/outside masks are computed from the predicate, both branches must satisfy
`min_samples_leaf` (`0` disables branch-size rejection), raw information gain is
computed, and BIC-style `penalized_gain` is applied using the number of clauses
as the arity. Nonpositive raw or penalized gain returns `Ok(None)`.

Tie-breaking is deterministic: higher score wins; then fewer clauses; then the
lexicographically smaller clause sequence; inside a clause, literals are ordered
by feature index, threshold, and the inspected Square2CNF operator order
(`GreaterEqual` before `LessThan`). Automated Python-vs-Rust Square2CNF search
equivalence remains a TODO until PyO3 bindings exist.

This search does not implement three-clause enumeration, theorem-certificate
checking, benchmark integration, full-family search, or tree recursion.


## Unified family-search facade status

`rust_gsnh/src/family.rs` now provides `FamilySearchConfig`,
`BestFamilySplit`, and `best_family_split` for searching exactly one selected
language family. This is separate from the fixed-predicate facade: callers pass
a family choice, a `max_arity`, and `min_samples_leaf`, and the facade dispatches
to one existing deterministic family search function. It does not compare across
families.

Supported families are the Rust `LanguageFamily` variants already used by the
fixed facade: `ConjUI`, `Horn`, `AntiHorn`, `Affine`, and `Square2CNF`. For
ConjUI, Horn, AntiHorn, and Affine, `max_arity` means maximum number of threshold
literals. For Square2CNF, `max_arity` means maximum number of two-literal OR
clauses.

`BestFamilySplit::Composed` wraps the `EvaluatedComposedPredicate` result used
by ConjUI, Horn, AntiHorn, and Affine/XOR. `BestFamilySplit::Square2CNF` wraps
the `EvaluatedSquare2CNFPredicate` result used by Square2CNF. Errors and
`Ok(None)` are propagated directly from the selected family search, including
invalid arity, NaN threshold-generation errors, min-leaf rejection, and
nonpositive-gain rejection.

Unsupported Python family names remain intentionally unsupported in Rust:
`Any`, `BestPerNode`, and legacy `SquareCNF`. The facade is not all-family
search, not BestPerNode, not benchmark integration, and not tree recursion.

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

1. Implement a shallow one-node Rust tree/stump builder that calls `best_family_split` for one selected family.
2. Add small-tree prediction equivalence only after the stump builder is stable.
3. Add PyO3/maturin wrapper exposing `engine="python"`, `engine="rust"`, and
   `engine="compare"` after Rust search parity is established.
4. Benchmarks with speedup ratios only after correctness parity is stable.

## Known limitations

- No PyO3 binding yet.
- ConjUI enumeration/search exists only for arity 1 and arity 2; there is no 3D ConjUI Rust search yet.
- Horn enumeration/search exists only for arity 1 and arity 2; there is no 3D Horn Rust search yet.
- AntiHorn enumeration/search exists only for arity 1 and arity 2; there is no 3D AntiHorn Rust search yet.
- Affine/XOR enumeration/search exists only for arity 1 and arity 2; there is no 3D Affine Rust search yet.
- Square2CNF enumeration/search exists only for one- and two-clause predicates; there is no 3-clause Square2CNF Rust search yet.
- Horn and AntiHorn have no Rust theorem certificate checker or benchmark integration.
- Square2CNF theorem certificates are still not implemented in Rust.
- Affine/XOR search is threshold-literal XOR search only; there is no GF(2)
  basis construction, theorem certificate checker, or benchmark integration.
- No recursive Rust tree learning yet; family search is available only through one selected-family facade and the underlying deterministic family searches.
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

Implement a shallow one-node Rust tree/stump builder that calls `best_family_split` for one selected family and returns either a leaf or a root split with two leaves. Keep theorem certificates out of Rust and do not implement recursive tree learning yet.
