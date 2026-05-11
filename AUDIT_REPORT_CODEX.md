# Formal Audit Report (Codex)

## 1) Executive verdict

**almost ready** — theorem-strict routing is largely implemented, but benchmark theorem-row filtering and Square2CNF arity handling still permit misleading certification states.

## 2) Claim verdict table

| Claim | Verdict |
|---|---|
| Fixed Horn theorem-compliant | conditional |
| Fixed AntiHorn theorem-compliant | conditional |
| Square2CNF theorem-compliant | conditional |
| BEST_PER_NODE theorem-compliant | conditional |
| unrestricted mixed BEST_PER_NODE not NP-hard | no (must not be claimed) |
| AXp subset-minimal | conditional |
| Python formally verified | no |

## 3) Critical issues

1. **Theorem-table admission is not gated on theorem mode usage, so non-strict runs can appear theorem-certified if metadata is favorable.**  
   - file path: `scripts/benchmark_dl8_languages_updated.py`  
   - function/class: `LanguageComparisonBenchmark._is_theorem_row`  
   - explanation: `_is_theorem_row` checks `theorem_certified`, backend, and certificate, but does **not** require `theorem_mode_used=True`. This allows rows from non-theorem-strict execution (especially mixed/BestPN) to enter theorem outputs if metadata is structurally safe.  
   - correction plan: require `theorem_mode_used` to be truthy for theorem tables, or split “certifiable fragment” from “theorem-run” tables explicitly.  
   - test to add: construct synthetic rows with `theorem_mode_used=False` and safe certificate; assert `_is_theorem_row` returns False.

## 4) High-priority issues

1. **Square2CNF theorem path allows 1- or 3-clause true branches to be certified as `2cnf`, while false branches reject non-2-clause forms.**  
   - file path: `src/gsnh_mdt/literals/predicates.py`, `src/gsnh_mdt/sat/path_certificate.py`, `src/gsnh_mdt/tree/explainer.py`  
   - function/class: `Square2CNFPredicate.__post_init__`, `_encode_square2cnf_path_clauses`, `_is_sat_path`  
   - explanation: predicate currently allows 1..3 clauses; theorem strict route marks Square2CNF certified when CNF remains 2CNF, but false-branch encoder hard-rejects `len!=2`. This creates asymmetric certification semantics by branch/path.  
   - correction plan: either (A) restrict theorem-certified Square2CNF to exactly 2 clauses at predicate validation for theorem mode, or (B) implement theorem-justified false-branch encoding for generalized k-clause forms and update proof/claims.  
   - test to add: theorem_strict path with 1-clause and 3-clause Square2CNF on both branch polarities; assert deterministic reject-or-certify policy.

2. **`weak_axp_check` propagates theorem rejection exceptions instead of distinguishing “non-theorem path” from SAT/UNSAT outcomes in audit metadata aggregation.**  
   - file path: `src/gsnh_mdt/tree/explainer.py`  
   - function/class: `weak_axp_check`  
   - explanation: any `NonTheoremPathError` from `_is_sat_path` aborts the check entirely; this is strict, but downstream benchmark metrics can interpret raised exceptions as method failures rather than certification rejections unless carefully separated.  
   - correction plan: keep raising for theorem_strict correctness, but add explicit exception handling in benchmark collection to classify as rejected/censored, not algorithm crash.  
   - test to add: theorem_strict mixed path causing `NonTheoremPathError`; assert benchmark metadata records rejected reason and does not count as generic run failure.

## 5) Medium-priority issues

1. **Float atom-key canonicalization risk remains unresolved.**  
   - file path: `src/gsnh_mdt/sat/threshold_encoder.py`  
   - function/class: `atom_id`  
   - explanation: exact tuple key `(int(feature), float(threshold))` can split logically equal thresholds if produced by numerically unstable pipelines (or collapse intended distinctions if upstream rounds).  
   - correction plan: define a single threshold quantization/canonicalization policy at binning/training boundary and enforce in encoder/tests.  
   - test to add: near-equal thresholds (e.g., `0.30000000000000004` vs `0.3`) across same feature; assert expected key behavior.

2. **Documentation/implementation mismatch risk on structural-order clause strategy.**  
   - file path: `src/gsnh_mdt/sat/threshold_encoder.py`  
   - function/class: `add_structural_order_clauses`  
   - explanation: implementation emits all-pairs implication clauses; ensure theorem docs/tests explicitly reflect all-pairs (not adjacent-chain-only wording).  
   - correction plan: lock terminology in docs and include targeted test asserting all-pairs generation count.

## 6) Low-priority issues

1. Duplicate import line in threshold encoder (`typing` imported twice).  
2. `threshold_encoder.build_ordered_selected_path_cnf` returns `[]` on unsupported pred shapes; prefer typed failure to avoid silent misuse.

## 7) Missing tests

- `test_theorem_row_requires_theorem_mode_used_true`: theorem table filter must reject non-strict rows.
- `test_square2cnf_theorem_mode_arity_policy_true_branch`: 1/3-clause policy deterministic in theorem mode.
- `test_square2cnf_theorem_mode_arity_policy_false_branch`: same for false branches.
- `test_benchmark_rejection_not_failure_for_non_theorem_path`: rejection metadata vs run failure accounting.
- `test_threshold_encoder_float_key_canonicalization`: near-equal thresholds behavior.
- `test_structural_order_all_pairs_count`: confirms all-pairs generation and docs consistency.

## 8) Exact patches recommended

1. `scripts/benchmark_dl8_languages_updated.py` / `_is_theorem_row`: add `if not row.get("theorem_mode_used", False): return False`.
2. `src/gsnh_mdt/literals/predicates.py` / `Square2CNFPredicate.__post_init__`: enforce theorem-profile arity (exactly 2) when theorem mode is required, or label non-2-clause as auxiliary.
3. `src/gsnh_mdt/sat/path_certificate.py` / `_encode_square2cnf_path_clauses`: make arity policy explicit in error reason (proof-boundary wording).
4. `scripts/benchmark_dl8_languages_updated.py` / run loop exception handling: classify `NonTheoremPathError` separately from hard failures.

## 9) Final wording for thesis (safe)

“GSNH-MDT provides theorem-aligned explainability guarantees **conditionally**: Horn and AntiHorn are certified when routed through structural threshold encoding with their corresponding polynomial SAT backends; Square2CNF is certified only for explicitly 2-CNF-certified paths solved with `two_sat`; and mixed Best-per-Node behavior remains empirical unless each checked path receives a valid Horn/AntiHorn/2CNF certificate. The Python implementation is extensively tested against the Coq-level specification but is not itself formally verified or extracted from Coq.”
