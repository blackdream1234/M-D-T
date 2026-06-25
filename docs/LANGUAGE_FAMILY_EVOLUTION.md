# Evolution of GSNH-MDT Language Families: From Smart 3D MDT to Certified Logical Predicates

## 1. Executive summary

The project began with the **Smart 3D MDT** terminology: split candidates were described by their geometric dimension, namely **1D**, **2D**, and **3D** threshold predicates. In that older formulation, the central 3D pattern was a disjunctive corner predicate

```text
φ(x) = ψ_i(x_i) OR ψ_j(x_j) OR ψ_k(x_k),
```

where each one-dimensional condition `ψ_f` was anchored to either the low side or high side of a threshold:

```text
x_f <= t    for a low anchor
x_f >= t    for a high anchor
```

The current code reorganizes this geometric terminology into explicit **logical language families**: `HORN`, `ANTI_HORN`, `AFFINE`, `CONJ_UI`, `SQUARE_2CNF`, `ANY`, and `BEST_PER_NODE` in `LanguageFamily` (`src/gsnh_mdt/types.py:28-56`). The benchmark layer then presents the user-facing families `1D`, `Horn`, `AntiHorn`, `ConjUI`, `Square2CNF`, `Affine`, and `BestPN` (`scripts/benchmark_dl8_languages_updated.py:269-353`).

This new naming is more precise than the old 1D/2D/3D naming because it separates:

1. **logical semantics**: OR clauses, AND/box predicates, 2-CNF formulas, XOR/parity constraints;
2. **tractability/certification status**: theorem-aligned Horn, AntiHorn, and certified 2-CNF paths versus auxiliary or empirical families;
3. **search dimensionality**: whether a candidate uses one, two, or three features.

The short mapping is: old 1D remains the univariate baseline; old 2D/3D disjunctive corner predicates are now better described as bounded multivariate disjunctive threshold predicates and can fall into **Horn** or **AntiHorn** depending on polarity; old conjunction/box-style behavior is now **ConjUI**; **Square2CNF**, **Affine**, and **BestPN** are later additions rather than direct renamings of old Smart 3D MDT.

> Note on source history: I did not find a standalone old “Smart 3D MDT proof document” in the current repository tree. This report therefore uses the old formulation supplied in the prompt, together with surviving code evidence for prefix sums, inclusion-exclusion, 1D/2D/3D search, and the current language-family implementation.

---

## 2. Old Smart 3D MDT formulation

### 2.1 Original 1D meaning

In the old Smart 3D MDT terminology, **1D** meant a split using one feature and one threshold condition. The current surviving 1D exhaustive search still reflects this: `search_1d_exhaustive` loops over two anchors, where anchor `0` uses the lower interval `[0,t)` and anchor `1` uses the upper interval `[t,n)` (`src/gsnh_mdt/search/exhaustive_1d.py:15-48`). Interpreted in literal notation, these are the two unary threshold directions:

```text
x_f < t
x_f >= t
```

The current benchmark explicitly preserves this as the method label `1D`, implemented using `HORN` as the backing enum with `search_1d=True`, `search_2d=False`, and `search_3d=False` (`scripts/benchmark_dl8_languages_updated.py:269-285`). The benchmark comment explains that unary Horn and unary AntiHorn both represent ordinary threshold splits (`scripts/benchmark_dl8_languages_updated.py:269-274`).

### 2.2 Original 2D meaning

In the old terminology, **2D** meant a split candidate involving two feature-threshold conditions. For the disjunctive corner formulation, the predicate was conceptually

```text
φ(x) = ψ_i(x_i) OR ψ_j(x_j),
```

with each `ψ` being either a low-side or high-side threshold anchor. The surviving prefix-sum code has a 2D inclusion-exclusion union counter:

```text
|A union B| = |A| + |B| - |A intersect B|.
```

This is implemented by `count_2way_union`, which queries a 2D prefix table for `A`, `B`, and `AB` (`src/gsnh_mdt/search/prefix.py:78-92`). The 2D prefix builder and O(1) rectangle query are implemented by `build_2d_prefix` and `query_2d` (`src/gsnh_mdt/search/prefix.py:24-33`, `src/gsnh_mdt/search/prefix.py:61-64`).

In the current logical terminology, such a two-literal disjunction is not “just 2D”; it is a logical clause whose tractable family depends on polarity. If it has at most one positive literal, it is Horn. If it has at most one negative literal, it is AntiHorn (`src/gsnh_mdt/types.py:31-32`).

### 2.3 Original 3D meaning

In the old terminology, **3D** meant a split candidate involving three feature-threshold conditions, especially the disjunctive corner pattern:

```text
φ(x) = ψ_i(x_i) OR ψ_j(x_j) OR ψ_k(x_k).
```

The surviving 3D search imports and uses 3D prefix and union primitives. The tree builder imports `build_3d_prefix`, `query_3d`, and `count_3way_union` from the prefix module (`src/gsnh_mdt/tree/builder.py:36-40`) and imports the 3D Horn, AntiHorn, Affine, and ConjUI searches (`src/gsnh_mdt/tree/builder.py:44-50`). The current pattern enum still records ternary Horn patterns such as `TERNARY_ALL_NEG`, `TERNARY_POS_FIRST`, `TERNARY_POS_SECOND`, and `TERNARY_POS_THIRD`, plus corresponding AntiHorn ternary patterns (`src/gsnh_mdt/types.py:88-107`).

### 2.4 Low/high anchors

The old low/high anchor terminology corresponds to choosing one side of a threshold. In current literal terminology:

- high anchor is represented by `LiteralPolarity.GE`, i.e. `x >= t`;
- low anchor is represented by `LiteralPolarity.LT`, i.e. `x < t` (`src/gsnh_mdt/types.py:14-16`).

The prompt’s old notation used `x_f <= t` for low anchors; the current implementation uses strict `<` for the low-side literal. This is a bin/threshold convention difference that should be stated carefully in the thesis.

### 2.5 3D corner pattern

A 3D corner pattern was a union of three anchored half-space/interval events. If `A`, `B`, and `C` are the three feature-side events, the in-branch count is

```text
|A ∪ B ∪ C|.
```

The surviving `count_3way_union` function computes this using inclusion-exclusion:

```text
|A union B union C| = |A| + |B| + |C| - |AB| - |AC| - |BC| + |ABC|.
```

The function explicitly sums each one-feature set over the full range of the other dimensions (`src/gsnh_mdt/search/prefix.py:95-118`). The unit test `TestUnionCounts.test_3way` checks the same inclusion-exclusion identity on a small cube (`tests/test_prefix.py:57-63`).

### 2.6 Why prefix sums and inclusion-exclusion were used

The old Smart 3D MDT implementation needed to evaluate many threshold combinations. Prefix sums made each box query O(1) after building the prefix tensor. The prefix module describes these as “core O(1) query primitives for exhaustive search” (`src/gsnh_mdt/search/prefix.py:1-6`). It implements 1D, 2D, and 3D prefix builders (`src/gsnh_mdt/search/prefix.py:15-49`) and O(1) 1D/2D/3D queries (`src/gsnh_mdt/search/prefix.py:56-71`). Inclusion-exclusion then converts O(1) box queries into O(1) OR/union counts for 2D and 3D disjunctive corner predicates (`src/gsnh_mdt/search/prefix.py:74-118`).

### 2.7 What was proven or argued mathematically

The old Smart 3D MDT argument can be summarized as follows:

1. **Prefix-sum correctness**: the prefix arrays correctly accumulate bin counts in 1D/2D/3D. This is implemented by `build_1d_prefix`, `build_2d_prefix`, and `build_3d_prefix` (`src/gsnh_mdt/search/prefix.py:15-49`) and tested on small examples (`tests/test_prefix.py:18-47`).
2. **O(1) box queries**: once the prefix table is built, `query_1d`, `query_2d`, and `query_3d` recover interval/box counts by constant-size arithmetic formulas (`src/gsnh_mdt/search/prefix.py:56-71`).
3. **Inclusion-exclusion for OR**: 2-way and 3-way OR counts are computed from box queries using inclusion-exclusion (`src/gsnh_mdt/search/prefix.py:78-118`).
4. **Information gain selection**: 1D/2D/3D searches compute in/out positive/negative counts and select the candidate with the best positive information gain, as shown for 1D in `search_1d_exhaustive` (`src/gsnh_mdt/search/exhaustive_1d.py:33-48`) and for Square2CNF in the later direct-mask search (`src/gsnh_mdt/search/square_2cnf.py:147-171`).
5. **Finite search space**: candidate thresholds are generated from discrete bins/edges, and the loops range over finite bin indices; for Square2CNF, the implementation explicitly limits candidate combinations with `max_candidates` (`src/gsnh_mdt/search/square_2cnf.py:52-74`, `src/gsnh_mdt/search/square_2cnf.py:128-134`).
6. **Approximate binning**: the current builder uses adaptive/node-local binning and discrete edges before searching candidate predicates; the search routines operate over those finite bin edges and feature subsets (`src/gsnh_mdt/search/square_2cnf.py:32-49`).

---

## 3. Current language families

### A. 1D

**Definition.** 1D is one threshold literal only:

```text
x_f >= t
x_f < t
```

The current code represents these directions by `LiteralPolarity.GE` and `LiteralPolarity.LT` (`src/gsnh_mdt/types.py:14-16`). The 1D exhaustive search checks two anchor directions over one prefix array (`src/gsnh_mdt/search/exhaustive_1d.py:15-48`).

**Role.** 1D is the internal univariate baseline. The benchmark method `1D` uses `HORN` as the enum, enables `search_1d`, disables `search_2d` and `search_3d`, and labels it “Univariate threshold baseline” (`scripts/benchmark_dl8_languages_updated.py:269-285`).

**Theorem status.** A unary threshold literal is compatible with structural threshold encoding. The theorem protocol defines the numeric atom `B(f,t) := x_f >= t`, literal encoding `GE := B` and `LT := not B`, and structural threshold-order clauses (`THEOREM_COMPLIANCE.md:5-12`). In practice, 1D is usually Horn-compatible because unary clauses are Horn and AntiHorn.

### B. Horn

**Definition.** Horn is an OR clause with at most one positive literal. The enum doc states: `HORN: at most 1 positive literal per clause (OR semantics)` (`src/gsnh_mdt/types.py:31`). `GSNHPredicate` likewise describes Horn as a clause with at most one positive literal and OR semantics (`src/gsnh_mdt/literals/predicates.py:31-40`).

**Example.** In threshold-atom notation:

```text
¬B1 OR ¬B2 OR B3
```

where only `B3` is positive.

**Relation to old 2D/3D.** Many old 2D/3D OR corner predicates map to Horn when their polarity pattern has at most one high/positive `GE` literal. The current pattern enum includes binary and ternary Horn patterns such as all-negative and one-positive cases (`src/gsnh_mdt/types.py:88-97`). The builder imports and calls the exhaustive 2D/3D Horn searches (`src/gsnh_mdt/tree/builder.py:44-46`).

**Theorem status.** Horn is theorem-compliant when structural threshold encoding is used, the backend is `structural_horn`, unsupported literals are absent, and no DFS fallback is used (`THEOREM_COMPLIANCE.md:14-19`). The explainer routes pure Horn paths to `_path_sat_structural_horn` and records `structural_horn` metadata (`src/gsnh_mdt/tree/explainer.py:220-226`).

### C. AntiHorn

**Definition.** AntiHorn is an OR clause with at most one negative literal. The enum doc states: `ANTI_HORN: at most 1 negative literal per clause (OR semantics)` (`src/gsnh_mdt/types.py:32`). `GSNHPredicate` also records AntiHorn as at most one negative literal with OR semantics (`src/gsnh_mdt/literals/predicates.py:31-40`).

**Example.** In threshold-atom notation:

```text
B1 OR B2 OR ¬B3
```

where only `¬B3` is negative.

**Relation to old 2D/3D.** Many old 2D/3D OR corner predicates map to AntiHorn when their polarity pattern has at most one low/negative `LT` literal. The current pattern enum includes binary and ternary AntiHorn patterns such as all-positive and one-negative cases (`src/gsnh_mdt/types.py:98-107`). The builder imports and calls the 2D/3D AntiHorn search routines (`src/gsnh_mdt/tree/builder.py:47`).

**Theorem status.** AntiHorn is theorem-compliant when structural threshold encoding is used, the backend is `structural_antihorn`, unsupported literals are absent, and no DFS fallback is used (`THEOREM_COMPLIANCE.md:21-26`). The explainer routes pure AntiHorn paths to `_path_sat_structural_horn(..., LanguageFamily.ANTI_HORN)` and records `structural_antihorn` metadata (`src/gsnh_mdt/tree/explainer.py:228-234`).

### D. ConjUI

**Definition.** ConjUI is a legacy conjunction/box-style family: pure AND of unary interval literals. The enum doc says `CONJ_UI` is a pure AND of unary interval literals / box constraints and was formerly called `SQUARE_CNF`; it also states this is not the paper's square 2CNF language (`src/gsnh_mdt/types.py:34-39`). `GSNHPredicate.__str__` prints ConjUI predicates using conjunction rather than disjunction (`src/gsnh_mdt/literals/predicates.py:120-210`).

**Important distinction.** ConjUI is **not** the same as all old 2D/3D Smart MDT. It corresponds only to conjunction/box-style predicates, not to the old disjunctive corner OR predicates. The ConjUI search file explicitly says it uses AND/intersection semantics, direct box queries, and not inclusion-exclusion union counts (`src/gsnh_mdt/search/conj_ui.py:1-13`). Its 2D search docstring lists the four box combinations and says the in-count is a direct box query, not inclusion-exclusion (`src/gsnh_mdt/search/conj_ui.py:23-38`).

**Theorem status.** ConjUI is auxiliary/empirical unless separately certified. The theorem-compliance document lists ConjUI among auxiliary/prototype methods that are not theorem-certified unless separately proven or otherwise classified (`THEOREM_COMPLIANCE.md:36-39`). The benchmark config marks ConjUI as `category="auxiliary"` and describes it as old SquareCNF behavior: conjunction/box of interval literals (`scripts/benchmark_dl8_languages_updated.py:306-315`).

### E. Square2CNF

**Definition.** Square2CNF is the later paper-style 2-CNF family:

```text
(l1 OR l2) AND (l3 OR l4)
```

The enum doc identifies `SQUARE_2CNF` as the real paper-style square 2CNF and describes it as a conjunction of 2-literal disjunctive clauses (`src/gsnh_mdt/types.py:37-39`). `Square2CNFPredicate` represents a conjunction of 2-literal disjunctive clauses and evaluates it as an AND over OR clauses (`src/gsnh_mdt/literals/predicates.py:261-330`). The search routine generates candidate `(l1 OR l2) AND (l3 OR l4)` predicates and evaluates them by direct masks (`src/gsnh_mdt/search/square_2cnf.py:52-81`, `src/gsnh_mdt/search/square_2cnf.py:122-171`).

**Relation to old formulation.** Square2CNF is **not** the old 3D corner OR. The old 3D corner OR is one disjunctive clause over up to three anchored threshold literals. Square2CNF is a conjunction of two binary disjunctions, hence a structured 2-CNF formula.

**Theorem status.** Square2CNF is certified only under strict conditions: explicit 2-CNF encoding plus the `two_sat` backend (`THEOREM_COMPLIANCE.md:28-30`). The path-certificate encoder requires exactly two clauses and exactly two literals per clause before treating Square2CNF as theorem-certified (`src/gsnh_mdt/sat/path_certificate.py:117-135`). The false branch is encoded with an auxiliary switch variable into four 2-CNF clauses (`src/gsnh_mdt/sat/path_certificate.py:137-166`). The benchmark config labels Square2CNF as main and describes it as certified via explicit 2-CNF encoding and the `two_sat` backend (`scripts/benchmark_dl8_languages_updated.py:338-351`). For thesis claims, Square2CNF must not be called theorem-compliant unless `backend=two_sat` and `certificate=2cnf` (`REPORT_CLAIMS.md:7-12`).

### F. Affine

**Definition.** Affine represents XOR/parity constraints. The enum doc says `AFFINE` is an XOR constraint (`src/gsnh_mdt/types.py:31-33`), and pattern identifiers include `AFFINE_2D` and `AFFINE_3D` (`src/gsnh_mdt/types.py:108-110`).

**Relation to old formulation.** Affine was added later and is not part of the original Smart 3D MDT disjunctive corner formulation.

**Theorem status.** Affine SAT is polynomial in principle by Gaussian elimination over GF(2), and the exact solver includes an `affine_sat` Gaussian-elimination routine (`src/gsnh_mdt/sat/exact_solver.py:163-220`). However, the current Coq/CNF certificate pipeline treats Affine as auxiliary unless separately proven or separately certified (`THEOREM_COMPLIANCE.md:36-39`). The benchmark config marks Affine as `category="auxiliary"` and describes it as XOR/Affine splits (`scripts/benchmark_dl8_languages_updated.py:316-325`).

### G. BestPN

**Definition.** BestPN corresponds to `BEST_PER_NODE`, an adaptive per-node selection strategy. The enum doc says it is heuristic per-node language selection, choosing the most promising family at each node based on topological/bounding-box analysis (`src/gsnh_mdt/types.py:40-45`). The benchmark config uses `enum_name="BEST_PER_NODE"`, enables 1D/2D/3D search, runs in heuristic mode, and categorizes it as empirical (`scripts/benchmark_dl8_languages_updated.py:326-335`).

**Relation to old formulation.** BestPN was added later as an empirical adaptive strategy; it is not the old Smart 3D MDT itself.

**Theorem status.** BestPN is empirical by default. It is theorem-certified only if every checked path receives a Horn, AntiHorn, or 2-CNF certificate (`THEOREM_COMPLIANCE.md:32-35`). The theorem-strict explainer path uses path-level certificate checking for mixed/BestPN-like paths and routes certificates to Horn, AntiHorn, or 2-SAT backends (`src/gsnh_mdt/tree/explainer.py:246-299`). The report-claims file explicitly says BEST_PER_NODE is empirical by default and becomes theorem-certified only when every explanation path receives a Horn, AntiHorn, or 2-CNF certificate (`REPORT_CLAIMS.md:3-5`).

---

## 4. Mapping table

| Old name / concept | Current implementation name | Correct interpretation | Theorem status |
|---|---|---|---|
| old 1D threshold split | `1D` benchmark method backed by `LanguageFamily.HORN` | One unary threshold literal, either `x_f < t` or `x_f >= t`; preserved as the univariate baseline (`scripts/benchmark_dl8_languages_updated.py:269-285`) | Certified through structural threshold encoding; unary clauses are Horn-compatible (`THEOREM_COMPLIANCE.md:5-19`) |
| old 2D OR corner split | Horn or AntiHorn 2-literal OR clause, depending on polarity | Bounded two-feature disjunctive threshold predicate; not automatically ConjUI (`src/gsnh_mdt/types.py:88-107`) | Certified if the clause/path classifies as Horn or AntiHorn and uses structural backend (`THEOREM_COMPLIANCE.md:14-26`) |
| old 3D OR corner split | Horn or AntiHorn ternary OR clause, depending on polarity | Bounded three-feature disjunctive threshold predicate `ψ_i OR ψ_j OR ψ_k`; current enum has ternary Horn and AntiHorn pattern identifiers (`src/gsnh_mdt/types.py:94-107`) | Certified only when polarity satisfies Horn or AntiHorn restrictions and path routing stays in certified structural mode |
| old low/high anchored threshold | `LiteralPolarity.LT` / `LiteralPolarity.GE` | Low side is current `x < t`; high side is current `x >= t` (`src/gsnh_mdt/types.py:14-16`) | Certified as threshold atom `B(f,t)` or its negation under structural encoding (`THEOREM_COMPLIANCE.md:5-12`) |
| old prefix-sum counting | `build_1d_prefix`, `build_2d_prefix`, `build_3d_prefix`, `query_*`, `count_*_union` | Efficient counting machinery for finite binned threshold search and inclusion-exclusion OR counts (`src/gsnh_mdt/search/prefix.py:15-118`) | Computational primitive, not itself a theorem-certified language family |
| old conjunction/box behavior | `ConjUI` / former `SQUARE_CNF` | Pure AND/intersection of unary interval literals; direct box queries, not OR corner inclusion-exclusion (`src/gsnh_mdt/search/conj_ui.py:1-13`) | Auxiliary/empirical unless separately certified (`THEOREM_COMPLIANCE.md:36-39`) |
| new Square2CNF | `SQUARE_2CNF`, `Square2CNFPredicate` | Later structured 2-CNF family `(l1 OR l2) AND (l3 OR l4)`, not old 3D corner OR (`src/gsnh_mdt/types.py:37-39`; `src/gsnh_mdt/literals/predicates.py:261-330`) | Certified only with exactly two binary clauses, explicit 2-CNF path certificate, `two_sat` backend, and theorem mode (`src/gsnh_mdt/sat/path_certificate.py:117-166`) |
| new Affine | `AFFINE` | Later XOR/parity split family (`src/gsnh_mdt/types.py:31-33`, `src/gsnh_mdt/types.py:108-110`) | Polynomial via GF(2) in principle, but auxiliary in the current CNF/Coq certificate pipeline unless separately proven (`THEOREM_COMPLIANCE.md:36-39`) |
| new BestPN | `BEST_PER_NODE` / `BestPN` | Later empirical per-node adaptive selection among families (`src/gsnh_mdt/types.py:40-45`; `scripts/benchmark_dl8_languages_updated.py:326-335`) | Empirical by default; certified only if every checked path has Horn, AntiHorn, or 2-CNF certificate (`THEOREM_COMPLIANCE.md:32-35`) |

---

## 5. Misunderstandings to avoid

### Wrong sentence 1

> “The old 1D/2D/3D is now ConjUI.”

Correct replacement:

> “The old 1D/2D/3D Smart MDT search was refactored into explicit logical language families.”

Explanation: ConjUI is only the pure AND/box family formerly called SquareCNF; it is not the old disjunctive OR corner family (`src/gsnh_mdt/types.py:34-39`; `src/gsnh_mdt/search/conj_ui.py:1-13`).

### Wrong sentence 2

> “Every 3D predicate is Horn.”

Correct replacement:

> “Disjunctive threshold predicates are represented by Horn or AntiHorn depending on polarity.”

Explanation: Horn permits at most one positive literal, while AntiHorn permits at most one negative literal (`src/gsnh_mdt/types.py:31-32`). A 3D OR clause can violate both restrictions if it has mixed polarity outside those limits.

### Wrong sentence 3

> “Every 3D predicate is AntiHorn.”

Correct replacement:

> “Disjunctive threshold predicates are represented by Horn or AntiHorn depending on polarity.”

Explanation: The enum separates ternary Horn patterns and ternary AntiHorn patterns (`src/gsnh_mdt/types.py:94-107`). Neither family contains all possible 3D OR clauses.

### Wrong sentence 4

> “Square2CNF is the same as old 3D MDT.”

Correct replacement:

> “Square2CNF is a later 2-CNF language family.”

Explanation: Square2CNF is `(l1 OR l2) AND (l3 OR l4)`, whereas old 3D corner OR was one OR over three anchored conditions (`src/gsnh_mdt/types.py:37-39`; `src/gsnh_mdt/literals/predicates.py:261-330`).

### Wrong sentence 5

> “BestPN is theorem-certified by default.”

Correct replacement:

> “BestPN is empirical unless path-level certificates prove tractability.”

Explanation: The theorem protocol says BEST_PER_NODE is empirical by default and theorem-certified only with path-level Horn, AntiHorn, or 2-CNF certificates (`THEOREM_COMPLIANCE.md:32-35`).

### Wrong sentence 6

> “The whole Python learner is formally verified.”

Correct replacement:

> “The theorem guarantee applies to the certified explanation checker/extractor, not to global training optimality.”

Explanation: The theorem-compliance document explicitly says Python is not extracted directly from Coq, benchmark results are empirical, unrestricted mixed-family SAT is not claimed in P, and greedy split selection is heuristic (`THEOREM_COMPLIANCE.md:41-46`).

---

## 6. Thesis-ready paragraph

The initial Smart 3D MDT formulation described multivariate decision-tree splits by geometric search dimension, using 1D, 2D, and 3D threshold predicates over low- or high-anchored threshold conditions. The implementation was later reorganized into explicit logical language families: 1D, Horn, AntiHorn, ConjUI, Square2CNF, Affine, and BestPN. This reorganization improves theoretical clarity because it separates the dimensionality of a searched predicate from the logical form that determines tractability and explanation certification. In the current implementation, fixed Horn and AntiHorn modes are certified through structural threshold encoding and polynomial SAT backends, while Square2CNF is certified only when paths are explicitly encoded as 2-CNF and solved by the `two_sat` backend under the required theorem-mode metadata. ConjUI, Affine, and BestPN should be presented as auxiliary or empirical unless they receive separate path-level certificates or separate formal treatment. Thus, the old Smart 3D MDT terminology is best understood as the historical search description, whereas the final thesis classification should use the newer logical language-family terminology.

---

## 7. Professor explanation

“Before, I called the search 1D, 2D, and 3D Smart MDT. Now I reorganized it into logical families. 1D is still the univariate baseline. The old 2D/3D disjunctive threshold predicates are separated into Horn and AntiHorn depending on polarity. The old box/conjunction behavior is now called ConjUI. Square2CNF and Affine were added later. BestPN is an empirical adaptive strategy.”

A slightly more formal oral version:

“Dimension tells us how many features a split candidate uses; logical family tells us whether the path satisfiability problem is Horn, AntiHorn, 2-CNF, XOR/Affine, box-style, or empirical mixed. The thesis should use the logical family names for theory claims and keep 1D/2D/3D as historical/search-dimensional terminology.”

---

## 8. Code evidence

### Enum names and family meanings

- `LanguageFamily` defines `HORN`, `ANTI_HORN`, `AFFINE`, `CONJ_UI`, `SQUARE_2CNF`, `ANY`, `BEST_PER_NODE`, and legacy alias `SQUARE_CNF` (`src/gsnh_mdt/types.py:28-56`).
- The enum doc defines Horn, AntiHorn, Affine, ConjUI, Square2CNF, and BestPN semantics (`src/gsnh_mdt/types.py:31-45`).
- `GSNHPatternType` records unary, binary, ternary Horn/AntiHorn patterns, Affine patterns, ConjUI patterns, legacy SquareCNF aliases, and Square2CNF patterns (`src/gsnh_mdt/types.py:86-126`).

### Predicate classes

- `GSNHPredicate` documents Horn as OR with at most one positive literal, AntiHorn as OR with at most one negative literal, Affine as XOR, and ConjUI as pure AND/box semantics (`src/gsnh_mdt/literals/predicates.py:31-40`).
- `Square2CNFPredicate` documents and implements a conjunction of 2-literal disjunctive clauses (`src/gsnh_mdt/literals/predicates.py:261-330`).

### Search flags and search routing in the builder

- The builder imports the prefix-sum builders and union-count primitives (`src/gsnh_mdt/tree/builder.py:36-40`).
- The builder imports 1D, 2D, 3D, AntiHorn, Affine, ConjUI, and Square2CNF search routines (`src/gsnh_mdt/tree/builder.py:44-50`).
- The benchmark method configuration shows how current user-facing methods enable 1D/2D/3D search flags and assign categories (`scripts/benchmark_dl8_languages_updated.py:255-353`).

### Prefix sums and old Smart 3D counting logic

- `build_1d_prefix`, `build_2d_prefix`, and `build_3d_prefix` implement prefix construction (`src/gsnh_mdt/search/prefix.py:15-49`).
- `query_1d`, `query_2d`, and `query_3d` implement constant-size interval/box query formulas (`src/gsnh_mdt/search/prefix.py:56-71`).
- `count_2way_union` and `count_3way_union` implement inclusion-exclusion union counts for OR-style corner predicates (`src/gsnh_mdt/search/prefix.py:78-118`).

### Square2CNF search

- `search_square_2cnf` generates `(l1 OR l2) AND (l3 OR l4)` candidates and evaluates direct masks (`src/gsnh_mdt/search/square_2cnf.py:52-81`, `src/gsnh_mdt/search/square_2cnf.py:122-171`).

### Theorem routing in the explainer

- The explainer routes pure Horn paths to the structural Horn backend and pure AntiHorn paths to the structural AntiHorn backend (`src/gsnh_mdt/tree/explainer.py:220-234`).
- In theorem-strict mixed-path handling, it builds a path CNF, classifies the certificate, and routes `horn`, `antihorn`, and `2cnf` certificates to their corresponding SAT procedures (`src/gsnh_mdt/tree/explainer.py:246-299`).
- Non-strict empirical paths can fall back to numeric interval/case-splitting logic for ConjUI, Square2CNF, and mixed families (`src/gsnh_mdt/tree/explainer.py:300-310`).

### Path certificate logic

- `build_ordered_selected_path_cnf` constructs selected-feature clauses, structural order clauses, and path clauses (`src/gsnh_mdt/sat/path_certificate.py:45-102`).
- `_encode_square2cnf_path_clauses` requires exactly two clauses with exactly two literals per clause for theorem-certified Square2CNF (`src/gsnh_mdt/sat/path_certificate.py:117-135`).
- The Square2CNF false branch is encoded with an auxiliary switch variable into 2-CNF clauses (`src/gsnh_mdt/sat/path_certificate.py:137-166`).
- `classify_cnf_fragment` classifies CNFs as Horn, AntiHorn, 2CNF, or none, and `is_polynomial_safe_path` marks only Horn, AntiHorn, and 2CNF certificates as safe (`src/gsnh_mdt/sat/path_certificate.py:173-234`).

### Benchmark method definitions

- `default_method_configs` defines user-facing `1D`, `Horn`, `AntiHorn`, `ConjUI`, `Affine`, `BestPN`, and optional `Square2CNF` rows with categories and descriptions (`scripts/benchmark_dl8_languages_updated.py:269-353`).
- Benchmark metadata treats Horn and AntiHorn as certified defaults in conservative cases and records theorem-mode metadata such as `theorem_mode_used` (`scripts/benchmark_dl8_languages_updated.py:680-713`).

### Claim-boundary documents

- `THEOREM_COMPLIANCE.md` defines the verified core, Horn/AntiHorn/Square2CNF theorem modes, BEST_PER_NODE constraints, auxiliary methods, and what is not verified (`THEOREM_COMPLIANCE.md:1-50`).
- `REPORT_CLAIMS.md` states the thesis claim boundary and lists claims not to make, including that unrestricted mixed BEST_PER_NODE is not claimed non-NP-hard, not all rows are theorem-certified, Square2CNF requires `backend=two_sat` and `certificate=2cnf`, and Python is not fully formally verified (`REPORT_CLAIMS.md:1-12`).

---

## 9. Final claim boundary

GSNH-MDT has a theorem-aligned explanation checker/extractor for certified Horn, AntiHorn, and 2-CNF path encodings. The learned tree construction and adaptive split search remain empirical/heuristic. The old Smart 3D MDT terminology should be presented as the historical starting point, not as the final theoretical classification.

---

## Unclear code points requiring confirmation

1. **Old proof document location.** I did not find a dedicated old Smart 3D MDT proof document in the current repository tree. If it exists outside this checkout, its exact terminology should be cross-checked against this report.
2. **Low anchor inequality convention.** The prompt says old low anchors were written as `x_f <= t`; the current literal type is `LT`, represented as `x < t`. This is likely a bin-boundary convention, but the thesis should choose one notation and explain the implementation convention.
3. **ConjUI historical naming.** The code clearly says ConjUI was formerly called SquareCNF, but it should not be described as the whole old Smart 3D MDT system. It is only the conjunction/box-style part.
4. **Square2CNF certification metadata.** The code-level theorem requirement is exact two-clause/two-literal path encoding plus 2-CNF classification and `two_sat`; benchmark-level statements should continue to require `theorem_mode_used=True` and safe backend/certificate metadata.
