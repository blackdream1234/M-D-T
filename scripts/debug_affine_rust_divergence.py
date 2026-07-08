#!/usr/bin/env python3
"""Focused debug helper for Affine Python-vs-Rust wrapper divergence.

This is intentionally not a benchmark entry point. It reproduces the HTRU_2-bin
sample used by ``benchmark_rust_gsnh_smoke.py`` and prints the diagnostics needed
to compare the Python oracle wrapper with the opt-in Rust wrapper.
"""

from __future__ import annotations

import argparse
import importlib
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug HTRU_2-bin depth-1 Affine Python-vs-Rust wrapper divergence."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing .dl8 files.")
    parser.add_argument(
        "--dataset", default="HTRU_2-bin", help="Dataset stem to locate recursively."
    )
    parser.add_argument("--max-rows", type=int, default=300, help="Deterministic row cap.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling and split seed.")
    parser.add_argument("--depth", type=int, default=1, help="Tree depth for both wrappers.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split fraction.")
    return parser.parse_args()


def require_rust_extension() -> None:
    try:
        importlib.import_module("_rust_gsnh")
    except ImportError as exc:
        raise SystemExit(
            "The optional _rust_gsnh extension is required for this debug helper. "
            "Install it with: maturin develop --manifest-path rust_gsnh/Cargo.toml "
            "--features python,pyo3-extension"
        ) from exc
    log("_rust_gsnh import succeeded")


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


def fit_wrapper(engine: str, depth: int, X_train: np.ndarray, y_train: np.ndarray):
    kwargs: dict[str, Any] = {
        "engine": engine,
        "family": "Affine",
        "max_arity": 2,
        "max_depth": depth,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
    }
    if engine == "python":
        kwargs.update(PYTHON_ENGINE_KWARGS)
    model = GSNHEngineClassifier(**kwargs)
    if engine == "rust":
        model.fit(X_train.tolist(), y_train.tolist())
    else:
        model.fit(X_train, y_train)
    return model


def describe_python_root(model: GSNHEngineClassifier) -> dict[str, Any]:
    root = getattr(model.model_, "root_", None)
    if not isinstance(root, dict):
        return {"available": False, "reason": "Python root_ dict unavailable"}
    predicate = root.get("predicate")
    left = root.get("left") or {}
    right = root.get("right") or {}
    return {
        "available": True,
        "is_leaf": root.get("is_leaf"),
        "language": root.get("language"),
        "predicate": str(predicate) if predicate is not None else None,
        "predicate_repr": repr(predicate) if predicate is not None else None,
        "left_samples": left.get("n_samples"),
        "left_positive": left.get("n_positive"),
        "left_negative": left.get("n_negative"),
        "left_proba": left.get("proba"),
        "right_samples": right.get("n_samples"),
        "right_positive": right.get("n_positive"),
        "right_negative": right.get("n_negative"),
        "right_proba": right.get("proba"),
    }


def print_kv(title: str, value: Any) -> None:
    print(f"\n## {title}", flush=True)
    print(value, flush=True)


def main() -> int:
    args = parse_args()
    log("Affine divergence debug startup")
    require_rust_extension()
    dataset_path = find_dataset(args.data_dir, args.dataset)
    log(f"using dataset: {dataset_path}")

    X, y = load_dataset(dataset_path, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = deterministic_train_test_split(
        X, y, args.seed, args.test_size
    )

    log("Python Affine fit start")
    python_model = fit_wrapper("python", args.depth, X_train, y_train)
    log("Python Affine fit end")
    log("Rust Affine fit start")
    rust_model = fit_wrapper("rust", args.depth, X_train, y_train)
    log("Rust Affine fit end")

    python_predictions = np.asarray(python_model.predict(X_test), dtype=np.int32)
    rust_predictions = np.asarray(rust_model.predict(X_test.tolist()), dtype=np.int32)
    mismatches = np.flatnonzero(python_predictions != rust_predictions)
    python_score = float(python_model.score(X_test, y_test))
    rust_score = float(rust_model.score(X_test.tolist(), y_test.tolist()))

    print_kv("Python predictions", python_predictions.tolist())
    print_kv("Rust predictions", rust_predictions.tolist())
    print_kv("Mismatch count", int(len(mismatches)))
    print_kv("Mismatch indices in test order", mismatches.tolist())
    print_kv("Python score", python_score)
    print_kv("Rust score", rust_score)
    print_kv("Python selected split/root", describe_python_root(python_model))
    print_kv("Rust summary", rust_model.summary())
    print_kv(
        "Rust split information",
        "Not exposed by the current _rust_gsnh binding; add a read-only debug accessor before "
        "asserting split-object parity.",
    )
    print_kv(
        "Likely divergence class to check first",
        "Threshold/search-space preprocessing: the Python oracle uses quantile/bin-index Affine "
        "search over selected top feature pairs, while the Rust engine enumerates raw midpoint "
        "thresholds for feature pairs. Tiny binary XOR parity can pass even when real-valued "
        "threshold generation diverges.",
    )
    print_kv(
        "Affine theorem status",
        "Affine remains empirical/auxiliary in this path; do not claim Rust speedup for Affine "
        "until prediction parity is restored.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
