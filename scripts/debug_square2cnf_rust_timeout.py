#!/usr/bin/env python3
"""Diagnose Rust Square2CNF smoke timeout without changing the benchmark.

This helper reproduces the HTRU_2-bin sampling configuration used by the
experimental Rust smoke benchmark, then compares the active Python Square2CNF
search-space cap with the Rust exhaustive Square2CNF search-space estimate.
It does not optimize, alter production defaults, or claim Square2CNF speedups.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_rust_gsnh_smoke import (  # noqa: E402
    PYTHON_ENGINE_KWARGS,
    deterministic_train_test_split,
    load_dataset,
    log,
)
from gsnh_mdt.engine import GSNHEngineClassifier  # noqa: E402
from gsnh_mdt.literals.base import GSNHLiteral  # noqa: E402
from gsnh_mdt.scoring.gain import information_gain  # noqa: E402
from gsnh_mdt.scoring.penalties import penalized_gain  # noqa: E402
from gsnh_mdt.types import LiteralPolarity  # noqa: E402


@dataclass(frozen=True)
class PythonSquare2Stats:
    n_features_considered: int
    n_literals: int
    n_clauses: int
    potential_two_clause_pairs_after_skip: int
    evaluated_cap: int
    evaluated: int
    best_updates: int
    best_gain: float
    elapsed_s: float


@dataclass(frozen=True)
class RustSquare2Estimate:
    n_features_considered: int
    n_threshold_atoms: int
    n_literals: int
    n_clauses: int
    n_one_clause_candidates: int
    n_two_clause_candidates_after_skip: int
    n_total_candidates: int
    n_feature_set_skip_pairs: int
    threshold_phase_s: float
    counting_phase_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug Rust Square2CNF timeout on sampled HTRU_2-bin data."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing .dl8 files.")
    parser.add_argument(
        "--dataset", default="HTRU_2-bin", help="Dataset stem to locate recursively."
    )
    parser.add_argument("--max-rows", type=int, default=100, help="Deterministic row cap.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling and split seed.")
    parser.add_argument("--depth", type=int, default=1, help="Tree depth for Python wrapper fit.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split fraction.")
    return parser.parse_args()


def find_dataset(data_dir: str, dataset_stem: str) -> Path:
    root = Path(data_dir).expanduser().resolve()
    candidates = sorted(root.rglob("*.dl8"))
    for path in candidates:
        if path.stem == dataset_stem:
            return path
    visible = ", ".join(path.stem for path in candidates[:20])
    raise FileNotFoundError(
        f"Could not find {dataset_stem!r} under {root}; first visible .dl8 stems: {visible}"
    )


def choose_python_square2_model(depth: int, X_train: np.ndarray, y_train: np.ndarray):
    model = GSNHEngineClassifier(
        engine="python",
        family="Square2CNF",
        max_arity=2,
        max_depth=depth,
        min_samples_leaf=1,
        min_samples_split=2,
        **PYTHON_ENGINE_KWARGS,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def rust_threshold_counts(X: np.ndarray) -> tuple[list[int], float]:
    start = time.perf_counter()
    counts = []
    for feature_index in range(X.shape[1]):
        unique_values = np.unique(X[:, feature_index])
        counts.append(max(0, len(unique_values) - 1))
    return counts, time.perf_counter() - start


def estimate_rust_square2_search_space(X: np.ndarray) -> RustSquare2Estimate:
    threshold_counts, threshold_phase_s = rust_threshold_counts(X)
    count_start = time.perf_counter()
    literal_counts = [2 * count for count in threshold_counts]
    n_literals = int(sum(literal_counts))
    n_threshold_atoms = int(sum(threshold_counts))
    n_clauses = comb2(n_literals)

    feature_set_clause_counts: dict[tuple[int, int | None], int] = {}
    for feature, literal_count in enumerate(literal_counts):
        if literal_count:
            feature_set_clause_counts[(feature, None)] = comb2(literal_count)
    for left_feature in range(len(literal_counts)):
        left_count = literal_counts[left_feature]
        if left_count == 0:
            continue
        for right_feature in range(left_feature + 1, len(literal_counts)):
            right_count = literal_counts[right_feature]
            if right_count:
                feature_set_clause_counts[(left_feature, right_feature)] = left_count * right_count

    n_feature_set_skip_pairs = sum(comb2(count) for count in feature_set_clause_counts.values())
    n_two_clause_candidates = comb2(n_clauses) - n_feature_set_skip_pairs
    counting_phase_s = time.perf_counter() - count_start
    return RustSquare2Estimate(
        n_features_considered=X.shape[1],
        n_threshold_atoms=n_threshold_atoms,
        n_literals=n_literals,
        n_clauses=n_clauses,
        n_one_clause_candidates=n_clauses,
        n_two_clause_candidates_after_skip=n_two_clause_candidates,
        n_total_candidates=n_clauses + n_two_clause_candidates,
        n_feature_set_skip_pairs=n_feature_set_skip_pairs,
        threshold_phase_s=threshold_phase_s,
        counting_phase_s=counting_phase_s,
    )


def build_python_local_edges(tree, X: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Mirror root node-local edge construction used by ExpertGSNHTree."""
    local_edges: dict[int, np.ndarray] = {}
    n_bins: dict[int, int] = {}
    n_target_bins = tree.n_bins
    for feature in range(tree.n_features_):
        column = X[:, feature]
        unique_values = np.unique(column)
        if len(unique_values) <= 1:
            edges = np.array([column[0] - 1e-10, column[0], column[0] + 1e-10])
        elif len(unique_values) <= n_target_bins:
            midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
            edges = np.concatenate(
                [[unique_values[0] - 1e-10], midpoints, [unique_values[-1] + 1e-10]]
            )
        else:
            edges = np.quantile(column, np.linspace(0, 1, n_target_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([column.min() - 1e-10, column.max() + 1e-10])
            elif len(edges) == 2:
                mid = (edges[0] + edges[1]) / 2.0
                edges = np.array([edges[0], mid, edges[1]])
        local_edges[feature] = edges
        n_bins[feature] = len(edges) - 1
    return local_edges, n_bins


def generate_python_square2_literals(features, edges, n_bins) -> list[GSNHLiteral]:
    literals = []
    for feature in features:
        feature_edges = edges[feature]
        nb = n_bins[feature]
        step = max(1, nb // 8)
        for bin_index in range(step, nb, step):
            if bin_index < len(feature_edges):
                threshold = float(feature_edges[bin_index])
                literals.append(GSNHLiteral(feature, threshold, LiteralPolarity.GE))
                literals.append(GSNHLiteral(feature, threshold, LiteralPolarity.LT))
    return literals


def python_square2_diagnostic(
    model: GSNHEngineClassifier, X_train: np.ndarray, y_train: np.ndarray
) -> PythonSquare2Stats:
    tree = model.model_
    feature_scores = tree._compute_feature_scores(X_train, y_train)
    top_k = min(tree.top_k, tree.n_features_)
    top_features = np.argsort(-feature_scores)[:top_k]
    features = top_features[: min(10, len(top_features))]
    edges, n_bins = build_python_local_edges(tree, X_train)
    start = time.perf_counter()
    literals = generate_python_square2_literals(features, edges, n_bins)
    literal_masks = np.array([literal.evaluate(X_train) for literal in literals], dtype=bool)

    clause_indices = []
    for left_idx in range(len(literals)):
        for right_idx in range(left_idx + 1, len(literals)):
            left = literals[left_idx]
            right = literals[right_idx]
            if (
                left.feature == right.feature
                and left.polarity == right.polarity
                and left.threshold == right.threshold
            ):
                continue
            clause_indices.append((left_idx, right_idx))

    clause_feature_sets = []
    for left_idx, right_idx in clause_indices:
        clause_feature_sets.append({literals[left_idx].feature, literals[right_idx].feature})
    potential_pairs = 0
    for left_clause in range(len(clause_indices)):
        for right_clause in range(left_clause + 1, len(clause_indices)):
            if clause_feature_sets[left_clause] != clause_feature_sets[right_clause]:
                potential_pairs += 1

    if clause_indices:
        clause_masks = np.array(
            [literal_masks[left] | literal_masks[right] for left, right in clause_indices],
            dtype=bool,
        )
    else:
        clause_masks = np.empty((0, len(y_train)), dtype=bool)

    total_pos = float(y_train.sum())
    total_neg = float(len(y_train) - total_pos)
    max_candidates = 500
    evaluated = 0
    best_updates = 0
    best_gain = -1.0
    for left_clause in range(len(clause_indices)):
        if evaluated >= max_candidates:
            break
        for right_clause in range(left_clause + 1, len(clause_indices)):
            if evaluated >= max_candidates:
                break
            if clause_feature_sets[left_clause] == clause_feature_sets[right_clause]:
                continue
            mask = clause_masks[left_clause] & clause_masks[right_clause]
            in_total = int(mask.sum())
            out_total = len(y_train) - in_total
            evaluated += 1
            if in_total < 1 or out_total < 1:
                continue
            in_pos = float(y_train[mask].sum())
            in_neg = float(in_total - in_pos)
            gain = information_gain(total_pos, total_neg, in_pos, in_neg)
            if gain > 0:
                gain = penalized_gain(gain, arity=2, n_bins=4, n_samples=len(y_train), n_classes=2)
            if gain > best_gain and gain > 0:
                best_gain = float(gain)
                best_updates += 1
    elapsed_s = time.perf_counter() - start
    return PythonSquare2Stats(
        n_features_considered=int(len(features)),
        n_literals=int(len(literals)),
        n_clauses=int(len(clause_indices)),
        potential_two_clause_pairs_after_skip=int(potential_pairs),
        evaluated_cap=max_candidates,
        evaluated=int(evaluated),
        best_updates=int(best_updates),
        best_gain=float(best_gain),
        elapsed_s=elapsed_s,
    )


def print_kv(title: str, value: Any) -> None:
    print(f"\n## {title}", flush=True)
    print(value, flush=True)


def main() -> int:
    args = parse_args()
    log("Square2CNF Rust timeout debug startup")
    try:
        dataset_path = find_dataset(args.data_dir, args.dataset)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    log(f"using dataset: {dataset_path}")
    load_start = time.perf_counter()
    X, y = load_dataset(dataset_path, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = deterministic_train_test_split(
        X, y, args.seed, args.test_size
    )
    print_kv("load/sample/split time seconds", time.perf_counter() - load_start)
    print_kv("train/test shape", {"X_train": X_train.shape, "X_test": X_test.shape})

    log("Python Square2CNF fit start")
    python_model, python_fit_s = choose_python_square2_model(args.depth, X_train, y_train)
    log(f"Python Square2CNF fit end: elapsed={python_fit_s:.6f}s")
    python_stats = python_square2_diagnostic(python_model, X_train, y_train)
    rust_estimate = estimate_rust_square2_search_space(X_train)

    print_kv("Python fit seconds", python_fit_s)
    print_kv("Python active Square2CNF search-space", python_stats)
    print_kv("Rust exhaustive Square2CNF search-space estimate", rust_estimate)
    print_kv(
        "Diagnosis",
        {
            "candidate_explosion": rust_estimate.n_total_candidates > python_stats.evaluated,
            "missing_pruning_or_cap": python_stats.evaluated_cap,
            "different_arity_candidate_limits": "Python caps two-clause candidates at 500 over top-10 binned features; Rust exhaustively evaluates one-clause and two-clause candidates over all midpoint literals.",
            "inefficient_mask_construction_risk": "Rust evaluates literal/clause masks per candidate in the current fixed evaluator; the candidate count makes repeated recomputation dominant before prediction parity can be checked.",
            "tie_breaking_excessive": "Unlikely primary cause: tie-breaking occurs after scoring candidates; enumeration size dominates first.",
            "incorrect_loop_bounds": "No off-by-one evidence from static inspection; bounds intentionally cover exhaustive clause pairs.",
        },
    )
    print_kv(
        "Safety note",
        "Do not make Square2CNF Rust performance claims until the Rust fit finishes and predictions_equal=True on this sampled real-data case.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
