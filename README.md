# gsnh-mdt

**GSNH Multivariate Decision Trees** — a research-grade Python implementation of the Generalized Schaefer-Normalized Horn (GSNH) framework for building multivariate decision trees with formally verifiable split structures.

> **Status**: This package is the **authoritative implementation**. The original
> monolith (`gsnh_mdt_v3.py`) is archived in `archive/` and must not be modified.
> See [docs/baseline_freeze.md](docs/baseline_freeze.md) for the freeze record.

GSNH-MDT constructs decision trees whose internal predicates belong to tractable SAT language families (Horn, Anti-Horn, 2-SAT, Affine/XOR). This guarantees that the satisfiability of any root-to-leaf path can be decided in polynomial time, enabling exact explainability ([abductive explanations](https://doi.org/10.48550/arXiv.2309.12345)) without heuristics.

### Key Features

- **Multivariate splits** — 1D, 2D, and 3D clauses with exhaustive threshold search
- **Language-constrained** — splits are guaranteed to belong to a tractable SAT family
- **Exact explainability** — polynomial-time weak Abductive Explanations (AXp)
- **Integral-image search** — O(1) per-threshold evaluation via prefix sums
- **Ensemble methods** — Random Forest and Gradient Boosting wrappers
- **JIT-compiled kernels** — critical search loops compiled via Numba

## Installation

```bash
# From source (editable)
git clone <repo> && cd gsnh_mdt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
from gsnh_mdt.api import GSNHClassifier
from gsnh_mdt.types import LanguageFamily

clf = GSNHClassifier(
    model_type='single',
    max_depth=7,
    language=LanguageFamily.BEST_PER_NODE,
)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")

# Extract a minimal explanation for a single prediction
axp = clf.extract_axp(X_test[0])
print(f"Explanation uses {len(axp)} features: {axp}")
```

### Direct Tree Access

```python
from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria

tree = ExpertGSNHTree(
    stopping_criteria=StoppingCriteria(max_depth=5),
    n_bins=64,
    language=LanguageFamily.HORN,
    mode='journal',
)
tree.fit(X_train, y_train)
tree.print_tree()
```

## Package Structure

```
src/gsnh_mdt/
  types.py          — enums (LanguageFamily, LiteralPolarity, etc.) + config tables
  config.py         — typed dataclass configs (ModelConfig, SearchConfig, etc.)
  literals/         — GSNHLiteral, GSNHBinaryLiteral, CompareLiteral, GSNHPredicate
  scoring/          — entropy, information_gain, gain_ratio, BIC penalty (@njit)
  search/           — prefix sums, tensor builders, exhaustive 1D/2D/3D, antihorn, affine
  sat/              — ExactSATSolver (Horn, Anti-Horn, 2-SAT, Affine), verification
  preprocess/       — AdaptiveBinner (quantile, supervised, adaptive strategies)
  tree/
    builder.py      — ExpertGSNHTree (training + split search)
    prediction.py   — predict, predict_proba, batch/single traversal
    explainer.py    — AXp extraction, path SAT (numeric + affine)
    stopping.py     — StoppingCriteria
    pruning.py      — CostComplexityPruner
    calibration.py  — ProbabilityCalibrator
  ensembles/        — GSNHRandomForest, GSNHGradientBoosting
  api/              — GSNHClassifier (pipeline: model selection + calibration + pruning)
```

See [docs/architecture.md](docs/architecture.md) for the dependency graph and module design.

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Current coverage: **199 tests** across 9 test files — literals, scoring, prefix sums, SAT solvers, tree regression (golden baselines on 9 datasets), method-level prediction/explainer regression, config parity, and API smoke tests.

## Language Families

| Family | Clause Structure | SAT Complexity | Solver |
|--------|-----------------|----------------|--------|
| Horn | ≤1 positive literal (arity ≤3) | P (forward chaining) | `ExactSATSolver.horn_sat` |
| Anti-Horn | ≤1 negative literal (arity ≤3) | P (reduce to Horn) | `ExactSATSolver.antihorn_sat` |
| ConjUI / Box | Pure AND of interval literals | P (intersection) | `_path_sat_numeric` |
| Square2CNF | Paper-style `(l1∨l2)∧(l3∨l4)` | P (De Morgan splitting) | `_path_sat_numeric` |
| Affine | XOR equations over GF(2) | P (Gaussian elim.) | `ExactSATSolver.affine_sat` |
| `BEST_PER_NODE` | empirical adaptive heuristic | P (tries all) | adaptive selection |

### Language Definitions

- **Horn**: Single Horn clauses, arity up to 3.
- **AntiHorn**: Single dual-Horn clauses, arity up to 3.
- **ConjUI / Box**: Conjunctions of unary interval literals. This was formerly called `SquareCNF` in early experiments. It is tractable but is not the generalized square 2CNF language from the paper.
- **Square2CNF**: Paper-style square 2CNF variant. Implemented as conjunction of two 2-literal disjunctive clauses: `(l1 ∨ l2) ∧ (l3 ∨ l4)`.
- **BestPerNode**: Empirical adaptive heuristic. Not used for formal tractability claims unless separately proven.

## Development

```bash
# Lint
ruff check src/ tests/

# Format
black src/ tests/

# Run only regression tests
python -m pytest tests/test_tree_regression.py -v
```

## License

Research use. See LICENSE for details.
