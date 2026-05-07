# AGENTS.md

## Project context

This repository implements GSNH-MDT, an algorithm for learning multivariate decision trees with interpretable split languages such as Horn, AntiHorn, 2CNF/SquareCNF, and Affine.

The main objectives are:
- correctness of the learning algorithm
- theorem-compliant split behavior
- formal explainability
- compact tree size
- reproducible benchmarks
- reliable result persistence
- clean comparison against baseline decision trees

## Review guidelines

When reviewing this repository, focus on:

1. Algorithmic correctness
   - Verify that split selection does not violate the intended language constraints.
   - Check that Horn, AntiHorn, 2CNF/SquareCNF, and Affine splits are treated consistently.
   - Check that gain computation, stopping criteria, depth handling, and leaf prediction are correct.

2. Formal assumptions
   - Identify where implementation assumptions diverge from theorem assumptions.
   - Flag any place where a claimed explainability property is not enforced in code.

3. Benchmarking
   - Check dataset loading.
   - Check train/test split reproducibility.
   - Check random seeds.
   - Check result saving.
   - Check whether experiments can be resumed without recomputing everything.

4. Performance
   - Look for unnecessary recomputation.
   - Look for inefficient Python loops.
   - Check Numba-compatible code.
   - Check memory usage on large .dl8 datasets.

5. Testing
   - Identify missing unit tests.
   - Propose tests for each split language.
   - Propose tests for edge cases:
     - empty dataset
     - pure labels
     - no valid split
     - max depth reached
     - duplicated features
     - conflicting constraints

6. Output format

Return findings ranked as:
- Critical
- High
- Medium
- Low

For each issue, include:
- file path
- function/class
- explanation
- correction plan
- test to add
