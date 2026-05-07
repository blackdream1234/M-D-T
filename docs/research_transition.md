# Research Transition

> **Date**: 2026-03-04
> **Status**: Refactor closed. Package baseline frozen and validated.

## 1. Refactor is closed

Phases A through F are complete. The `gsnh_mdt` package is the sole maintained
implementation of GSNH-MDT. The original monolith is archived in `archive/`
and must not receive further modifications.

## 2. Package baseline is frozen and validated

- **199 regression tests** pass across 9 datasets
- **Full benchmark** executed from package on 46 datasets × 10 splits (94 min)
- **Config parity** verified — `from_config()` matches legacy constructor on
  representative datasets within documented floating-point tolerance
- **Proof script** passes 4/4 from package imports

## 3. Future work should focus on algorithmic improvements

The following research directions can now proceed safely:

### Search strategy improvements
- Improved 2D/3D search pruning and ordering
- Adaptive arity selection per node
- Feature interaction detection beyond top-k

### Scoring and objective functions
- Alternative information-theoretic criteria
- Multi-objective scoring (accuracy + explainability)
- Label-aware gain refinements

### Pruning and bounds
- Tighter BIC penalties
- Cross-validation-based pruning
- Depth-adaptive stopping criteria

### Language synthesis and adaptation
- Per-node language family optimization
- Hybrid language strategies
- Language complexity budget constraints

### Ensemble strategy improvements
- Gradient boosting refinements
- Feature subsampling strategies
- OOB-based early stopping

## 4. Structural changes should be minimized

Unless justified by new research requirements:

- Do not re-split `builder.py` further
- Do not merge `prediction.py` / `explainer.py` back
- Do not remove the 22-param constructor
- Do not alter benchmark seeds or protocol
- Do not mix baseline and experimental settings

New modules may be added (e.g., `search/new_strategy.py`) without
disrupting existing structure. Extend, don't restructure.

## 5. Research workflow

```bash
# Run tests before and after any change
python -m pytest tests/ -q

# Run benchmark to validate algorithmic changes
python scripts/benchmark_dl8.py

# Run proof script to verify mathematical compliance
python scripts/prooftest.py
```

All scripts import from the package. No monolith dependency remains.
