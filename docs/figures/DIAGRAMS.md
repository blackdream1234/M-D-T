# GSNH-MDT Diagram Pack

This folder contains thesis/presentation diagrams for the GSNH-MDT workflow.

## Files

1. `training_pipeline.mmd` / `.dot`
   - End-to-end training flow: dataset to learned MDT.
2. `language_families.mmd` / `.dot`
   - Predicate-language comparison (1D, Horn, AntiHorn, Square2CNF, ConjUI, Affine, BestPN).
3. `axp_extraction_pipeline.mmd` / `.dot`
   - Weak AXp checking and deletion-based subset-minimal extraction.
4. `theorem_strict_checker.mmd` / `.dot`
   - Ordered selected-path CNF, fragment classification, solver routing, metadata output.
5. `square2cnf_two_sat_encoding.mmd` / `.dot`
   - Theorem-certified Square2CNF true/false branch encodings and `two_sat` routing.
6. `benchmark_reporting_pipeline.mmd` / `.dot`
   - Benchmark aggregation and theorem-table filtering rules.

## Claim boundaries reflected in diagrams

- Theorem compliance is **conditional**.
- Do **not** interpret unrestricted mixed BestPN as polynomial.
- Affine and ConjUI are shown as auxiliary empirical families.
- Square2CNF is theorem-certified only with:
  - `theorem_strict=True`
  - `backend=two_sat`
  - `path_certificate=2cnf`
- Python implementation is tested against spec but not formally extracted/verified from Coq.

## Rendering

### Mermaid
- Render manually with Mermaid CLI:
  - `mmdc -i docs/figures/training_pipeline.mmd -o docs/figures/training_pipeline.svg`

### Graphviz
- Render manually with Graphviz:
  - `dot -Tsvg docs/figures/training_pipeline.dot -o docs/figures/training_pipeline.dot.svg`

### Helper script
- Use `scripts/generate_diagrams.py` to attempt automatic rendering when tools are available.
- The script never fails hard if `mmdc` or `dot` are unavailable.
