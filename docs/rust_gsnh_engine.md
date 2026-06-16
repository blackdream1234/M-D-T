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

## One-node stump builder status

`rust_gsnh/src/tree.rs` now implements a minimal depth-1 stump API on top of
`best_family_split`. The builder either returns a single majority leaf when the
selected family has no valid positive split, or returns one root split with a
true/inside leaf and a false/outside leaf. It does not recurse and does not add
a depth parameter.

The inspected Python tree code stores leaves as dictionaries with class counts,
`proba`, `predicate = None`, and `is_leaf = True`. Python predicts class 1 when
the positive probability is at least `0.5`; with its default Laplace smoothing,
exact count ties therefore predict the positive class. The Rust stump mirrors
that prediction convention directly as `PredictionLabel::Positive` when
`positive >= negative`, and `PredictionLabel::Negative` otherwise.

Branch convention also mirrors Python: predicate true / inside-mask rows go to
the true leaf (Python's left child), while predicate false / outside-mask rows
go to the false leaf (Python's right child). Prediction re-evaluates the stored
predicate against the supplied dataset, checks the requested row index, and
returns the selected leaf's majority label.

The supported split payloads are exactly the current `BestFamilySplit` variants:
`Composed` for ConjUI, Horn, AntiHorn, and Affine/XOR, and `Square2CNF` for
Square2CNF. The public stump API is `majority_leaf_from_mask`,
`build_stump_with_family`, `predict_stump_row`, and `predict_stump`.
Automated Python-vs-Rust stump equivalence remains a TODO until PyO3 bindings
exist.

Current limitations: no recursive tree learning, no pruning, no BestPerNode or
all-family search, no theorem certificates, no benchmark integration, and no
accuracy helper yet.

## Stump prediction and accuracy helper status

`rust_gsnh/src/tree.rs` also provides lightweight prediction-quality helpers for
the one-node stump. `PredictionLabel::Negative` maps to binary label `0`, and
`PredictionLabel::Positive` maps to binary label `1`, matching Python's
`predict(...).astype(int)` output. `prediction_label_to_u8` converts one label,
and `prediction_labels_to_u8` converts a prediction vector.

Accuracy mirrors the Python benchmark and estimator formula
`float((pred == y).mean())`: the helper counts exact binary-label matches and
divides by `dataset.n_samples()`. `accuracy_from_predictions` accepts Rust
`PredictionLabel` values, `accuracy_from_u8_predictions` accepts binary `u8`
predictions, and `stump_accuracy` reuses `predict_stump` before computing the
score.

Error behavior is intentionally strict. Prediction length mismatches return
`Err`, and `accuracy_from_u8_predictions` rejects labels other than `0` and `1`
because Rust `Dataset` labels are binary. These helpers do not modify tree
structure, search for new splits, recurse, bind through PyO3, or integrate with
benchmarks. Automated Python-vs-Rust stump prediction equivalence remains a TODO
until PyO3 bindings exist.

## Depth-limited recursive tree skeleton status

`rust_gsnh/src/tree.rs` now includes a minimal recursive `DecisionTree` skeleton
for exactly one selected `LanguageFamily`. `TreeBuildConfig` wraps the existing
`FamilySearchConfig` plus `max_depth` and `min_samples_split`; each internal
node stores one `BestFamilySplit`, class counts for the node's active rows, its
depth, and true/false child trees.

Depth starts at `0`, matching Python's `_build_tree(..., depth=0)` root call.
The Rust skeleton stops when `depth >= max_depth`, when active samples are fewer
than `min_samples_split`, when the active node is pure, when selected-family
search returns `Ok(None)`, or when a selected split would make an empty branch.
`min_samples_leaf` remains delegated to the selected family search through
`FamilySearchConfig`, as in the earlier stump/family-search layers.

Recursive search is active-subset aware. For each node, Rust copies only active
rows into a small node-local `Dataset`, calls `best_family_split` on that local
view, then evaluates the selected predicate against the original dataset and
intersects it with the active mask to form true/false child masks. This avoids
the unsafe shortcut of repeatedly searching the full dataset at every node.

Branch convention remains unchanged: split true / inside rows go to
`true_child`, and split false / outside rows go to `false_child`. Prediction uses
`predict_tree_row` / `predict_tree`, and `tree_accuracy` reuses the existing
accuracy helpers. This is still a skeleton only: it does not implement
BestPerNode, cross-family comparison, pruning, theorem certificates, benchmark
integration, PyO3, or parallelism. Automated Python-vs-Rust recursive tree
equivalence remains a TODO until PyO3 bindings exist.

## Active-subset recursive parity tests

The recursive tree tests now exercise the active-subset invariant directly.
`dataset_from_mask` preserves `active_mask.indices()` order when copying rows
and labels into the node-local `Dataset`; tests check row order, feature values,
and labels. `active_split_masks_from_predicate` evaluates the selected split on
the original dataset, intersects the true mask with the active mask, and derives
the false mask as active minus true. Tests assert both branch masks are subsets
of the active mask, disjoint, and have a union exactly equal to the active mask.

ConjUI-only recursive stabilization tests use explicit `FamilySearchConfig`
values (`family = ConjUI`, `max_arity = 2` or test-specific arity, and explicit
`min_samples_leaf`) to avoid broadening recursive behavior for Horn, AntiHorn,
Affine, or Square2CNF. The depth-2 deterministic tree test now verifies that the
second-level split occurs inside only one active branch, that rows outside that
branch are not reused, and that the depth-2 tree can improve over the matching
depth-1 stump when a valid second-level split exists.

The local/global row-index invariant is therefore: local search may choose a
predicate using a compact node-local dataset, but child masks are always formed
in global row-index space by re-evaluating that predicate on the original
dataset and intersecting with the current active mask. Automated Python-vs-Rust
recursive tree equivalence remains a TODO until PyO3 bindings exist.

## Tree summary and introspection status

`rust_gsnh/src/tree.rs` now exposes read-only introspection helpers for
`DecisionTree`. `TreeSummary` reports `n_nodes`, `n_leaves`,
`n_internal_nodes`, and `max_depth`. A leaf contributes one node and one leaf; a
split contributes one internal node plus both child subtrees; total nodes are
therefore `n_internal_nodes + n_leaves`.

`observed_tree_depth` is structural: a single leaf has depth `0`, a root split
with two leaf children has depth `1`, and deeper trees add one level per split
edge. `summarize_tree` combines the count helpers, and `training_accuracy` is a
convenience wrapper around `tree_accuracy` for already-built trees. These helpers
are read-only and do not alter tree construction, prediction, search, scoring,
active-subset handling, or family semantics.

## Stable Rust train/predict API status

`rust_gsnh/src/api.rs` now defines a stable Rust-facing wrapper around the
existing selected-family tree builder. This API is shaped so a future PyO3 layer
can wrap a small number of public structs/functions, but this phase does not add
PyO3, maturin, benchmark integration, theorem certificates, pruning, BestPerNode,
or new learning semantics.

`RustGsnHConfig` contains:

- `family`: the single `LanguageFamily` to search at every node.
- `max_arity`: literal arity for ConjUI, Horn, AntiHorn, and Affine/XOR; max
  clause count for Square2CNF.
- `max_depth`: depth limit for the existing recursive tree skeleton. `0` builds
  a majority leaf.
- `min_samples_leaf`: forwarded to family search. The existing Rust convention
  allows `0` to disable branch-size rejection.
- `min_samples_split`: minimum active rows required before attempting a split.

`fit_rust_gsnh` validates `max_arity >= 1` and `min_samples_split >= 1`, converts
`RustGsnHConfig` into the existing `FamilySearchConfig` and `TreeBuildConfig`,
calls `build_tree_with_family`, stores `summarize_tree(&tree)` in the model, and
returns `training_accuracy(&tree, dataset)` in `RustGsnHFitResult`.

`RustGsnHModel` stores the trained `DecisionTree`, the immutable training
configuration, and the `TreeSummary` captured at fit time. `RustGsnHFitResult`
stores the model plus its training-set accuracy.

`predict_rust_gsnh` reuses `predict_tree` and converts `PredictionLabel` values
to Python-friendly binary labels (`Negative -> 0`, `Positive -> 1`).
`score_rust_gsnh` reuses `tree_accuracy`, and `summarize_rust_gsnh` returns the
stored summary. Unsupported family/arity combinations still fail through the
existing selected-family search functions; the stable API does not add
cross-family comparison or alter scoring.

This wrapper is intentionally Rust-only for now. Automated Python-vs-Rust
fit/predict parity remains a TODO until PyO3 exists.

## PyO3 binding skeleton status

`rust_gsnh/src/python.rs` now contains a minimal PyO3-facing binding skeleton
for the stable Rust API. The crate keeps normal Rust library usage intact and
adds a `cdylib` crate type plus a crate-local `pyproject.toml` for maturin. The
root Python package is not replaced, and the binding does not add benchmark
integration, theorem certificates, pruning, BestPerNode, rayon, or new learning
semantics.

The intended Python extension module name is `_rust_gsnh`, exposing a
`RustGsnHClassifier` class:

```python
from _rust_gsnh import RustGsnHClassifier

classifier = RustGsnHClassifier(
    family="ConjUI",
    max_arity=2,
    max_depth=2,
    min_samples_leaf=1,
    min_samples_split=2,
)

classifier.fit(X, y)
predictions = classifier.predict(X)
accuracy = classifier.score(X, y)
summary = classifier.summary()
```

The first binding accepts only plain Python lists:

- `X`: `list[list[float]]`
- `y`: `list[int]` with binary values `0` or `1`

Supported family strings are `"ConjUI"`, `"Horn"`, `"AntiHorn"`, `"Affine"`,
and `"Square2CNF"`. Unsupported names such as `"Any"`, `"BestPerNode"`, and
legacy `"SquareCNF"` are rejected at construction time.

`fit(X, y)` builds a Rust `Dataset` and calls `fit_rust_gsnh`. `predict(X)`
builds a prediction-only Rust `Dataset` with dummy binary labels and calls
`predict_rust_gsnh`. `score(X, y)` builds a labelled Rust `Dataset` and calls
`score_rust_gsnh`. `summary()` returns a Python dictionary with `n_nodes`,
`n_leaves`, `n_internal_nodes`, and `max_depth`. Calling `predict`, `score`, or
`summary` before `fit` raises a Python exception.

The binding skeleton is gated behind the Rust `python` feature so ordinary
`cargo test --manifest-path rust_gsnh/Cargo.toml` remains a pure Rust check.
`cargo test --manifest-path rust_gsnh/Cargo.toml --features python` now compiles
the binding helper layer and its Rust-side tests. The actual PyO3 extension code
is isolated behind a narrower `pyo3-extension` gate so the repository can still
verify the `python` feature in offline environments where PyO3 cannot be
downloaded.

In a network-enabled development environment with maturin and PyO3 available,
the intended build/test commands are:

```bash
maturin develop --manifest-path rust_gsnh/Cargo.toml --features python
pytest tests/test_rust_gsnh_binding.py -q
```

The Python binding test module skips binding-dependent checks cleanly when
`_rust_gsnh` is not installed, so the existing Python suite does not depend on
the extension by default.

Tiny equivalence coverage is now present for the Rust-supported list-only subset
when `_rust_gsnh` is installed: each supported family is exercised through
fit/predict/score/summary invariants, and a depth-0 majority-leaf case compares
Rust binding predictions with `ExpertGSNHTree` predictions exactly. Deeper exact
parity tests remain deferred because the Python reference includes additional
binning, stopping, pruning, and search behavior outside the current Rust subset.

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

1. Add an optional Python wrapper mode such as `engine="rust"` in a separate
   wrapper file without replacing the default Python GSNH engine.
2. Extend Python-vs-Rust equivalence tests after the wrapper is stable.
3. Benchmarks with speedup ratios only after correctness parity is stable.

## Known limitations

- A minimal PyO3 binding skeleton exists, but it is optional, list-only, and not
  integrated into the production Python estimator or benchmarks.
- In offline environments, `_rust_gsnh` may be unavailable; binding-dependent
  tests skip in that case. The default cargo test path and the `python` feature
  helper path both remain testable without installing the extension.
- ConjUI enumeration/search exists only for arity 1 and arity 2; there is no 3D ConjUI Rust search yet.
- Horn enumeration/search exists only for arity 1 and arity 2; there is no 3D Horn Rust search yet.
- AntiHorn enumeration/search exists only for arity 1 and arity 2; there is no 3D AntiHorn Rust search yet.
- Affine/XOR enumeration/search exists only for arity 1 and arity 2; there is no 3D Affine Rust search yet.
- Square2CNF enumeration/search exists only for one- and two-clause predicates; there is no 3-clause Square2CNF Rust search yet.
- Horn and AntiHorn have no Rust theorem certificate checker or benchmark integration.
- Square2CNF theorem certificates are still not implemented in Rust.
- Affine/XOR search is threshold-literal XOR search only; there is no GF(2)
  basis construction, theorem certificate checker, or benchmark integration.
- Recursive Rust tree learning is limited to a minimal depth-limited skeleton for one selected family; active-subset parity is tested for ConjUI only, and there is no pruning, BestPerNode, or benchmark integration.
- The non-suffixed Rust 1D helpers use `min_samples_leaf = 1` for backward
  compatibility; callers that need Python tree parity should call the
  `*_with_min_leaf` APIs with the tree's configured value.
- High-cardinality quantile binning remains Python-only for now; Rust currently
  mirrors the exact-value midpoint threshold convention used for low-cardinality
  node-local 1D candidates.
- Rust tree construction now includes the stump plus a minimal active-subset-aware recursive skeleton and read-only summary helpers.
- Stable Rust train/predict wrappers exist, but there is still no PyO3-facing
  benchmark integration, pruning, BestPerNode, or theorem certificate path in
  Rust.
- No theorem certification is moved to Rust in this phase.
- Python remains the only production engine and oracle.

## Next safe optimization step

Add an optional Python wrapper mode such as `engine="rust"` in a separate wrapper
file. Keep it opt-in, keep the default Python GSNH engine unchanged, and do not
connect benchmarks, theorem certificates, pruning, parallelism, or BestPerNode
yet.
