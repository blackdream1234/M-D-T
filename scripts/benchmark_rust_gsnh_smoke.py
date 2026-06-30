#!/usr/bin/env python3
"""Experimental smoke benchmark for the optional Rust GSNH engine.

This script compares the opt-in ``engine="rust"`` wrapper against the default
Python wrapper on a small recursive .dl8 sample. It is intentionally separate
from the thesis/professor benchmark scripts: Python remains the correctness
oracle, Rust is never selected by default, and no theorem-certification claims
are made for these smoke results.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gsnh_mdt.engine import GSNHEngineClassifier

FAMILIES = ("ConjUI", "Horn", "AntiHorn", "Affine", "Square2CNF")
CSV_COLUMNS = [
    "dataset",
    "family",
    "depth",
    "n_train",
    "n_test",
    "python_accuracy",
    "rust_accuracy",
    "predictions_equal",
    "python_time_s",
    "rust_time_s",
    "speedup_rust_vs_python",
    "rust_n_nodes",
    "rust_max_depth",
    "error",
]
PYTHON_ENGINE_KWARGS = {
    "use_supervised_binning": False,
    "search_3d": False,
    "mode": "journal",
}


def verify_rust_extension_installed() -> None:
    """Fail clearly when the optional PyO3 extension is unavailable."""
    try:
        importlib.import_module("_rust_gsnh")
    except ImportError as exc:
        raise SystemExit(
            "The optional _rust_gsnh extension is not installed. Build it first with:\n"
            "  maturin develop --manifest-path rust_gsnh/Cargo.toml "
            "--features python,pyo3-extension"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Python-vs-Rust GSNH wrapper smoke benchmark."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing .dl8 files.")
    parser.add_argument("--max-datasets", type=int, default=2, help="Maximum .dl8 files to load.")
    parser.add_argument("--depth", type=int, default=1, help="Maximum tree depth for both engines.")
    parser.add_argument(
        "--output-dir",
        default="experiment_artifacts/rust_gsnh_smoke",
        help="Directory where CSV/JSON/Markdown smoke outputs are written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    return parser.parse_args()


def discover_dl8_files(data_dir: str, max_datasets: int) -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    files = sorted(root.rglob("*.dl8"))
    if not files:
        raise FileNotFoundError(f"No .dl8 files found recursively under {root}")
    if max_datasets > 0:
        files = files[:max_datasets]
    return files


def parse_dl8_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load label-first .dl8 data into (X, y_raw)."""
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([int(token) for token in stripped.split()])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: non-integer token") from exc

    if not rows:
        raise ValueError(f"{path}: empty .dl8 file")
    width = len(rows[0])
    if width < 2:
        raise ValueError(f"{path}: expected label plus at least one feature")
    if any(len(row) != width for row in rows):
        raise ValueError(f"{path}: inconsistent number of columns")

    data = np.asarray(rows, dtype=np.float64)
    return data[:, 1:], data[:, 0].astype(np.int32)


def binarize_labels(y: np.ndarray) -> np.ndarray:
    """Mirror the benchmark label-first convention with deterministic binary labels."""
    labels = np.unique(y)
    if len(labels) < 2:
        return np.zeros_like(y, dtype=np.int32)
    if len(labels) > 2:
        counts = np.bincount(y.astype(int))
        majority = int(np.argmax(counts))
        return (y == majority).astype(np.int32)
    return (y == labels[1]).astype(np.int32)


def remove_constant_columns(X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return X
    return X[:, np.var(X, axis=0) > 1e-12]


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    X, y_raw = parse_dl8_file(path)
    X = remove_constant_columns(X)
    if X.shape[1] == 0:
        raise ValueError(f"{path}: no non-constant features")
    y = binarize_labels(y_raw)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{path}: one-class target after binarization")
    return X.astype(np.float64), y.astype(np.int32)


def deterministic_train_test_split(
    X: np.ndarray, y: np.ndarray, seed: int, test_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic class-aware split without depending on scikit-learn."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        shuffled = rng.permutation(class_indices)
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue
        n_test = int(round(len(shuffled) * test_fraction))
        n_test = min(max(1, n_test), len(shuffled) - 1)
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_indices = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    if len(test_indices) == 0 and len(train_indices) > 1:
        shuffled = rng.permutation(train_indices)
        test_indices = shuffled[:1]
        train_indices = shuffled[1:]
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("unable to create non-empty deterministic train/test split")

    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if len(predictions) != len(labels):
        raise ValueError(f"prediction length mismatch: {len(predictions)} vs {len(labels)}")
    return float(np.mean(predictions == labels))


def blank_result(
    dataset: str, family: str, depth: int, n_train: int, n_test: int
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "family": family,
        "depth": depth,
        "n_train": n_train,
        "n_test": n_test,
        "python_accuracy": "",
        "rust_accuracy": "",
        "max_abs_score_diff": "",
        "predictions_equal": "",
        "python_time_s": "",
        "rust_time_s": "",
        "speedup_rust_vs_python": "",
        "rust_n_nodes": "",
        "rust_max_depth": "",
        "error": "",
    }


def run_family_smoke(
    dataset_name: str,
    family: str,
    depth: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    row = blank_result(dataset_name, family, depth, len(y_train), len(y_test))
    try:
        python_model = GSNHEngineClassifier(
            engine="python",
            family=family,
            max_arity=2,
            max_depth=depth,
            min_samples_leaf=1,
            min_samples_split=2,
            **PYTHON_ENGINE_KWARGS,
        )
        python_start = time.perf_counter()
        python_model.fit(X_train, y_train)
        python_time = time.perf_counter() - python_start

        rust_model = GSNHEngineClassifier(
            engine="rust",
            family=family,
            max_arity=2,
            max_depth=depth,
            min_samples_leaf=1,
            min_samples_split=2,
        )
        rust_start = time.perf_counter()
        rust_model.fit(X_train.tolist(), y_train.tolist())
        rust_time = time.perf_counter() - rust_start

        python_predictions = np.asarray(python_model.predict(X_test), dtype=np.int32)
        rust_predictions = np.asarray(rust_model.predict(X_test.tolist()), dtype=np.int32)
        python_accuracy = accuracy(python_predictions, y_test)
        rust_accuracy = accuracy(rust_predictions, y_test)
        summary = rust_model.summary()

        row.update(
            {
                "python_accuracy": python_accuracy,
                "rust_accuracy": rust_accuracy,
                "max_abs_score_diff": abs(python_accuracy - rust_accuracy),
                "predictions_equal": bool(np.array_equal(python_predictions, rust_predictions)),
                "python_time_s": python_time,
                "rust_time_s": rust_time,
                "speedup_rust_vs_python": python_time / rust_time if rust_time > 0 else "",
                "rust_n_nodes": summary.get("n_nodes", ""),
                "rust_max_depth": summary.get("max_depth", ""),
            }
        )
    except Exception as exc:  # noqa: BLE001 - smoke benchmark records failures and continues.
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rust_smoke_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "rust_smoke_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    summary_path = output_dir / "rust_smoke_summary.md"
    successful = [row for row in rows if not row["error"]]
    equal = [row for row in successful if row["predictions_equal"] is True]
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Rust GSNH smoke benchmark\n\n")
        handle.write("Experimental validation for the optional `_rust_gsnh` engine. ")
        handle.write("Python remains the correctness oracle and Rust remains opt-in. ")
        handle.write(
            "Affine rows are empirical/auxiliary and are not theorem-certification claims.\n\n"
        )
        handle.write(f"- Total rows: {len(rows)}\n")
        handle.write(f"- Successful rows: {len(successful)}\n")
        handle.write(f"- Prediction-identical rows: {len(equal)}\n\n")
        handle.write(
            "| dataset | family | python_accuracy | rust_accuracy | predictions_equal | error |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row['dataset']} | {row['family']} | {row['python_accuracy']} | "
                f"{row['rust_accuracy']} | {row['predictions_equal']} | {row['error']} |\n"
            )


def main() -> int:
    args = parse_args()
    verify_rust_extension_installed()

    rows: list[dict[str, Any]] = []
    for path in discover_dl8_files(args.data_dir, args.max_datasets):
        try:
            X, y = load_dataset(path)
            X_train, X_test, y_train, y_test = deterministic_train_test_split(X, y, args.seed)
        except Exception as exc:  # noqa: BLE001 - dataset smoke loading should continue.
            rows.append(
                {
                    **blank_result(path.stem, "LOAD", args.depth, 0, 0),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for family in FAMILIES:
            rows.append(
                run_family_smoke(path.stem, family, args.depth, X_train, X_test, y_train, y_test)
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    write_outputs(rows, output_dir)
    print(f"Wrote Rust GSNH smoke outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
