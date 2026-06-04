#!/usr/bin/env python3
"""
GSNH-MDT Language Family Benchmark — Professor's .dl8 Data
==========================================================

This benchmark compares sklearn DT(depth=7) with GSNH-MDT variants:

Main / theorem-oriented:
    - GSNH-1D
    - GSNH-Horn
    - GSNH-AntiHorn
    - GSNH-Square2CNF

Auxiliary / ablation:
    - GSNH-ConjUI  (old SquareCNF/Box behavior: AND of interval literals)
    - GSNH-Affine

Empirical only:
    - GSNH-BestPN

Key corrections vs older script:
    1. 1D no longer looks for a non-existing LanguageFamily.ONE_D.
       It uses LanguageFamily.HORN with search_2d=False and search_3d=False.
    2. Old "SqCNF" labels are replaced by "ConjUI".
    3. Paper-style square CNF is reported separately as "Square2CNF".
    4. BEST_PER_NODE is run in heuristic mode and labelled empirical.
    5. Results are printed dynamically, so adding/removing families does not
       break table widths or LaTeX column counts.
    6. Per-method failures do not kill the whole dataset.
    7. Optional AXp validity/minimality audit can be enabled.
    8. JSON, CSV, and LaTeX outputs are saved.

.dl8 format expected:
    label f1 f2 f3 ... fn
    space-separated integers, first column = class, rest = features

Typical usage:
    python scripts/benchmark_dl8_languages.py
    python scripts/benchmark_dl8_languages.py --quick
    python scripts/benchmark_dl8_languages.py --runs 10 --depths 5 7
    python scripts/benchmark_dl8_languages.py --include-square2cnf --audit-axp
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import zlib
import math
import os
import sys
from pathlib import Path
import time
import traceback
import warnings
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# IMPORT GSNH
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def import_gsnh():
    """Import the local gsnh_mdt package."""
    try:
        import types

        gsnh = types.SimpleNamespace()
        from gsnh_mdt.tree.builder import ExpertGSNHTree
        from gsnh_mdt.tree.stopping import StoppingCriteria
        from gsnh_mdt.types import LanguageFamily

        gsnh.ExpertGSNHTree = ExpertGSNHTree
        gsnh.StoppingCriteria = StoppingCriteria
        gsnh.LanguageFamily = LanguageFamily

        print("✓ Imported from gsnh_mdt package")
        print(f"  Available LanguageFamily values: {[e.name for e in LanguageFamily]}")
        return gsnh
    except Exception as e:
        print(f"✗ Cannot import GSNH package: {e}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_dl8(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a .dl8 file: first column is label, remaining columns are features."""
    rows: List[List[int]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([int(x) for x in line.split()])
            except ValueError as exc:
                raise ValueError(f"{filepath}:{lineno}: non-integer token") from exc

    if not rows:
        raise ValueError("empty .dl8 file")

    width = len(rows[0])
    if width < 2:
        raise ValueError("expected at least label + one feature")
    if any(len(r) != width for r in rows):
        raise ValueError("inconsistent number of columns")

    data = np.asarray(rows, dtype=np.float64)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float64)
    return X, y


def binarize_labels(y: np.ndarray) -> np.ndarray:
    """Make labels binary in a deterministic way."""
    labels = np.unique(y)
    if len(labels) < 2:
        return np.zeros_like(y, dtype=np.int32)
    if len(labels) > 2:
        # One-vs-rest majority class for multi-class files.
        counts = np.bincount(y.astype(int))
        majority = int(np.argmax(counts))
        return (y == majority).astype(np.int32)
    # Map smaller/first label to 0 and larger/second label to 1.
    return (y == labels[1]).astype(np.int32)


def remove_constant_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove zero-variance columns and return (X_filtered, mask)."""
    if X.shape[1] == 0:
        return X, np.zeros(0, dtype=bool)
    valid = np.var(X, axis=0) > 1e-12
    return X[:, valid], valid


def resolve_data_dir(cli_data_dir: Optional[str] = None) -> Path:
    """Resolve benchmark data directory by CLI, env, then repo-local data/."""
    candidates: List[Path] = []
    if cli_data_dir:
        candidates.append(Path(cli_data_dir))
    if os.environ.get("GSNH_MDT_DATA_DIR"):
        candidates.append(Path(os.environ["GSNH_MDT_DATA_DIR"]))
    if os.environ.get("DATA_DIR"):
        candidates.append(Path(os.environ["DATA_DIR"]))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "data")

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    return candidates[-1].expanduser().resolve()


def discover_dl8_files(data_dir: os.PathLike[str] | str) -> List[Path]:
    """Discover .dl8 files recursively and report visible files if none exist."""
    data_path = Path(data_dir).expanduser().resolve()
    files = sorted(data_path.rglob("*.dl8"))
    if not files:
        visible: List[str] = []
        if data_path.exists():
            visible = sorted(
                str(path.relative_to(data_path))
                for path in data_path.rglob("*")
                if path.is_file()
            )[:50]
        raise FileNotFoundError(
            f"No .dl8 files found recursively under {data_path}. "
            f"First files visible under data dir: {visible}"
        )
    return files


def load_all_dl8(
    data_dir: os.PathLike[str] | str,
    max_datasets: Optional[int] = None,
    dataset_filter: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, Dict[str, Any]]":
    """Load all .dl8 files recursively from data_dir."""
    dl8_files = discover_dl8_files(data_dir)
    if dataset_filter:
        wanted = set(dataset_filter)
        dl8_files = [p for p in dl8_files if p.stem in wanted]

    if max_datasets is not None:
        dl8_files = dl8_files[:max_datasets]

    datasets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for filepath in dl8_files:
        name = filepath.stem
        try:
            X, y_raw = parse_dl8(str(filepath))
            n_unary_original = X.shape[1]
            y = binarize_labels(y_raw)

            X, valid_mask = remove_constant_columns(X)
            if X.shape[1] == 0:
                print(f"  ⚠ {name:<35} skipped: no non-constant features")
                continue

            unique_x = np.unique(X)
            is_binary = bool(np.all(np.isin(unique_x, [0, 1])))

            # Need at least 2 classes for stratified splitting.
            if len(np.unique(y)) < 2:
                print(f"  ⚠ {name:<35} skipped: one-class target")
                continue

            datasets[name] = {
                "X": X,
                "y": y,
                "n_unary_original": n_unary_original,
                "n_features": X.shape[1],
                "n_samples": len(y),
                "is_binary": is_binary,
                "pos_rate": float(y.mean()),
                "source": filepath,
            }

            print(
                f"  ✓ {name:<35} n={len(y):>6}  d={X.shape[1]:>4}  "
                f"pos={y.mean():.2%}  binary={is_binary}"
            )
        except Exception as e:
            print(f"  ✗ {name:<35} {str(e)[:100]}")

    return datasets



def make_quick_synthetic_datasets() -> "OrderedDict[str, Dict[str, Any]]":
    """Create a tiny deterministic fallback dataset for quick smoke runs.

    This is used only when --quick is requested and no .dl8 files are present,
    so CI can still verify that the evidence package writer produces all CSV,
    LaTeX, PNG, and report artifacts without requiring private benchmark data.
    """
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, size=(96, 8)).astype(np.float64)
    y = ((X[:, 0] + X[:, 1] + (X[:, 2] * X[:, 3])) >= 2).astype(np.int32)
    datasets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    datasets["quick_synthetic"] = {
        "X": X,
        "y": y,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_unary_original": int(X.shape[1]),
        "pos_rate": float(y.mean()),
        "source": "synthetic quick-mode fallback",
    }
    return datasets


# =============================================================================
# METHODS / METRICS
# =============================================================================

@dataclass(frozen=True)
class MethodConfig:
    label: str
    enum_name: str
    search_1d: bool
    search_2d: bool
    search_3d: bool
    mode: str
    category: str
    description: str
    enabled_by_default: bool = True


THEOREM_MODE_METHODS = {"1D", "Horn", "AntiHorn", "Square2CNF"}

REQUIRED_RESULT_COLUMNS = [
    "dataset", "n_samples", "n_features_original", "n_features_encoded", "positive_rate",
    "method_key", "method_label", "category", "depth",
    "accuracy_mean", "accuracy_std", "acc", "acc_std",
    "tree_nodes_mean", "tree_nodes_std", "size", "leaves_mean", "leaves_std",
    "max_depth_reached", "avg_leaf_depth", "avg_predicate_arity", "max_predicate_arity",
    "total_literals", "compression_vs_sklearn_dt7", "mean_axp_length", "expl",
    "axp_time_mean", "axp_time_std", "train_time_mean", "train_time_std",
    "predict_time_mean", "predict_time_std", "weak_axp_checks_mean",
    "opposite_paths_checked_mean", "sat_vars_mean", "sat_clauses_mean",
    "axp_backend", "path_certificate", "theorem_mode_used", "theorem_certified",
    "rejected_reason", "n_success", "n_fail", "random_state", "n_runs",
    "train_test_split_protocol",
]


def default_method_configs(include_square2cnf: bool = True) -> List[MethodConfig]:
    """Language configurations.

    `1D` intentionally uses HORN as the backing enum because unary Horn and
    unary AntiHorn both represent ordinary threshold splits.
    """
    methods = [
        MethodConfig(
            label="1D",
            enum_name="HORN",
            search_1d=True,
            search_2d=False,
            search_3d=False,
            mode="journal",
            category="main",
            description="Univariate threshold baseline",
        ),
        MethodConfig(
            label="Horn",
            enum_name="HORN",
            search_1d=True,
            search_2d=True,
            search_3d=False,
            mode="journal",
            category="main",
            description="Horn clauses, arity up to 2 by default",
        ),
        MethodConfig(
            label="AntiHorn",
            enum_name="ANTI_HORN",
            search_1d=True,
            search_2d=True,
            search_3d=False,
            mode="journal",
            category="main",
            description="Dual-Horn/Anti-Horn clauses, arity up to 2 by default",
        ),
        MethodConfig(
            label="ConjUI",
            enum_name="CONJ_UI",
            search_1d=False,
            search_2d=True,
            search_3d=False,
            mode="journal",
            category="auxiliary",
            description="Old SquareCNF behavior: conjunction/box of interval literals",
        ),
        MethodConfig(
            label="Affine",
            enum_name="AFFINE",
            search_1d=False,
            search_2d=True,
            search_3d=False,
            mode="journal",
            category="auxiliary",
            description="XOR/Affine splits",
        ),
        MethodConfig(
            label="BestPN",
            enum_name="BEST_PER_NODE",
            search_1d=True,
            search_2d=True,
            search_3d=True,
            mode="heuristic",
            category="empirical",
            description="Adaptive per-node heuristic; not used for theorem claim",
        ),
    ]

    if include_square2cnf:
        # Put Square2CNF after ConjUI in reports.
        methods.insert(
            4,
            MethodConfig(
                label="Square2CNF",
                enum_name="SQUARE_2CNF",
                search_1d=False,
                search_2d=True,
                search_3d=False,
                mode="journal",
                category="main",
                description="Certified Square2CNF via explicit 2-CNF encoding and two_sat backend",
            ),
        )
    return methods


@dataclass
class MethodResult:
    acc: float = math.nan
    acc_std: float = math.nan
    size: float = math.nan
    expl: float = math.nan
    train_time: float = math.nan
    train_time_std: float = math.nan
    axp_time: float = math.nan
    axp_time_std: float = math.nan
    predict_time: float = math.nan
    predict_time_std: float = math.nan
    leaves: float = math.nan
    leaves_std: float = math.nan
    max_depth_reached: float = math.nan
    avg_leaf_depth: float = math.nan
    avg_predicate_arity: float = math.nan
    max_predicate_arity: float = math.nan
    total_literals: float = math.nan
    weak_axp_checks_mean: float = math.nan
    opposite_paths_checked_mean: float = math.nan
    sat_vars_mean: float = math.nan
    sat_clauses_mean: float = math.nan
    axp_valid_rate: float = math.nan
    axp_minimal_rate: float = math.nan
    n_success: int = 0
    n_fail: int = 0
    accs: List[float] = None
    errors: List[str] = None

    # Theorem-compliance / AXp-backend metadata.
    # These fields are scalar summaries so each benchmark row can be
    # classified into theorem-certified vs auxiliary/prototype outputs.
    axp_backend: str = "none"
    theorem_certified: bool = False
    path_certificate: str = "none"
    rejected_reason: str = ""
    theorem_mode_used: bool = False

    def __post_init__(self):
        if self.accs is None:
            self.accs = []
        if self.errors is None:
            self.errors = []


def safe_mean(xs: Sequence[float]) -> float:
    arr = np.asarray([x for x in xs if x is not None and not np.isnan(x)], dtype=float)
    return float(np.mean(arr)) if len(arr) else math.nan


def safe_std(xs: Sequence[float]) -> float:
    arr = np.asarray([x for x in xs if x is not None and not np.isnan(x)], dtype=float)
    return float(np.std(arr)) if len(arr) else math.nan


def _clean_metadata_value(value: Any, default: str = "none") -> str:
    """Normalize metadata values for CSV/JSON rows."""
    if value is None:
        return default
    if isinstance(value, str):
        return value if value else default
    if isinstance(value, (list, tuple, set)):
        vals = [_clean_metadata_value(v, default=default) for v in value]
        vals = [v for v in vals if v and v != default]
        if not vals:
            return default
        counts = Counter(vals)
        if len(counts) == 1:
            return next(iter(counts))
        return "mixed:" + ",".join(f"{k}={counts[k]}" for k in sorted(counts))
    return str(value)


def _summarize_metadata_values(values: Sequence[Any], default: str = "none") -> str:
    vals = [_clean_metadata_value(v, default=default) for v in values]
    vals = [v for v in vals if v and v != default]
    if not vals:
        return default
    counts = Counter(vals)
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed:" + ",".join(f"{k}={counts[k]}" for k in sorted(counts))


def _summarize_bool_values(values: Sequence[Any]) -> bool:
    vals = [bool(v) for v in values if v is not None]
    return bool(vals) and all(vals)


def count_gsnh_nodes(node: Optional[dict]) -> int:
    if node is None:
        return 0
    if node.get("is_leaf", True) or node.get("predicate") is None:
        return 1
    return 1 + count_gsnh_nodes(node.get("left")) + count_gsnh_nodes(node.get("right"))



def analyze_gsnh_tree(node: Optional[dict]) -> Dict[str, float]:
    """Compute structural complexity metrics for a fitted GSNH tree."""
    leaf_depths: List[int] = []
    arities: List[int] = []
    total_literals = 0

    def walk(n: Optional[dict], depth: int) -> None:
        nonlocal total_literals
        if n is None:
            return
        pred = n.get("predicate") if isinstance(n, dict) else None
        if n.get("is_leaf", True) or pred is None:
            leaf_depths.append(depth)
            return
        if hasattr(pred, "clauses"):
            clause_lens = [len(c) for c in getattr(pred, "clauses", [])]
            arity = len(clause_lens)
            lit_count = int(sum(clause_lens))
        else:
            lits = getattr(pred, "literals", [])
            arity = len(lits)
            lit_count = len(lits)
        arities.append(int(arity))
        total_literals += int(lit_count)
        walk(n.get("left"), depth + 1)
        walk(n.get("right"), depth + 1)

    walk(node, 0)
    return {
        "leaves": float(len(leaf_depths)),
        "max_depth_reached": float(max(leaf_depths) if leaf_depths else 0),
        "avg_leaf_depth": safe_mean(leaf_depths),
        "avg_predicate_arity": safe_mean(arities),
        "max_predicate_arity": float(max(arities) if arities else 0),
        "total_literals": float(total_literals),
    }


def analyze_sklearn_tree(tree: DecisionTreeClassifier) -> Dict[str, float]:
    """Compute structural complexity metrics for sklearn's fitted CART tree."""
    t = tree.tree_
    leaf_depths: List[int] = []

    def walk(node_id: int, depth: int) -> None:
        if t.children_left[node_id] == t.children_right[node_id]:
            leaf_depths.append(depth)
            return
        walk(t.children_left[node_id], depth + 1)
        walk(t.children_right[node_id], depth + 1)

    walk(0, 0)
    internal = int(t.node_count - len(leaf_depths))
    return {
        "leaves": float(len(leaf_depths)),
        "max_depth_reached": float(max(leaf_depths) if leaf_depths else 0),
        "avg_leaf_depth": safe_mean(leaf_depths),
        "avg_predicate_arity": 1.0 if internal else 0.0,
        "max_predicate_arity": 1.0 if internal else 0.0,
        "total_literals": float(internal),
    }


def _metadata_numeric_mean(items: Sequence[Any], name: str) -> float:
    vals: List[float] = []
    for item in items or []:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if isinstance(value, (int, float, np.integer, np.floating)):
            vals.append(float(value))
    return safe_mean(vals)

def sklearn_path_length(tree: DecisionTreeClassifier, x: np.ndarray) -> int:
    t = tree.tree_
    nid = 0
    depth = 0
    while t.children_left[nid] != t.children_right[nid]:
        depth += 1
        if x[t.feature[nid]] <= t.threshold[nid]:
            nid = t.children_left[nid]
        else:
            nid = t.children_right[nid]
    return depth


def avg_sklearn_expl(tree: DecisionTreeClassifier, X: np.ndarray) -> float:
    if len(X) == 0:
        return math.nan
    return float(np.mean([sklearn_path_length(tree, X[i]) for i in range(len(X))]))


def avg_gsnh_axp_length(tree: Any, X: np.ndarray, n_samples: int, seed: int) -> float:
    if getattr(tree, "root_", None) is None or len(X) == 0:
        return math.nan
    rng = np.random.default_rng(seed)
    k = min(n_samples, len(X))
    idx = rng.choice(len(X), size=k, replace=False)
    vals = []
    for i in idx:
        try:
            vals.append(len(tree.extract_axp(X[i])))
        except Exception:
            vals.append(math.nan)
    return safe_mean(vals)


def audit_axp_minimality(tree: Any, X: np.ndarray, n_samples: int, seed: int) -> Tuple[float, float]:
    """Return (weak_valid_rate, minimal_rate) for sampled test instances.

    This is intentionally optional because it may be expensive.
    """
    if getattr(tree, "root_", None) is None or len(X) == 0:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    k = min(n_samples, len(X))
    idx = rng.choice(len(X), size=k, replace=False)

    weak_ok = 0
    minimal_ok = 0
    tested = 0

    for i in idx:
        x = X[i]
        try:
            y_pred = int(tree.predict(x.reshape(1, -1))[0])
            S = set(tree.extract_axp(x))
            if not tree.weak_axp_check(x, y_pred, set(S)):
                tested += 1
                continue
            weak_ok += 1

            is_minimal = True
            for f in list(S):
                S2 = set(S)
                S2.remove(f)
                if tree.weak_axp_check(x, y_pred, S2):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_ok += 1
            tested += 1
        except Exception:
            tested += 1

    if tested == 0:
        return math.nan, math.nan
    return weak_ok / tested, minimal_ok / tested


def audit_language_counts(tree: Any, method: MethodConfig, expected_value: str) -> Optional[str]:
    """Return None if OK, otherwise a warning string."""
    if method.label == "BestPN":
        return None
    counts = getattr(tree, "language_counts_", {}) or {}
    unexpected = [k for k in counts.keys() if k != expected_value]
    if unexpected:
        return f"unexpected language counts for {method.label}: {counts}, expected only {expected_value}"
    return None


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, MethodConfig):
        return asdict(obj)
    if isinstance(obj, MethodResult):
        return asdict(obj)
    return str(obj)


# =============================================================================
# BENCHMARK
# =============================================================================

class LanguageComparisonBenchmark:
    """Compare language families at multiple depths vs sklearn DT depth 7."""

    def __init__(
        self,
        gsnh_module: Any,
        methods: List[MethodConfig],
        depths: Sequence[int],
        n_runs: int = 10,
        random_state: int = 42,
        axp_samples: int = 50,
        audit_axp: bool = False,
        audit_samples: int = 20,
        enable_3d: bool = False,
    ):
        self.gsnh = gsnh_module
        self.LF = self.gsnh.LanguageFamily
        self.methods = methods
        self.depths = list(depths)
        self.n_runs = int(n_runs)
        self.rs = int(random_state)
        self.axp_samples = int(axp_samples)
        self.audit_axp = bool(audit_axp)
        self.audit_samples = int(audit_samples)
        self.enable_3d = bool(enable_3d)

        self.results_: "OrderedDict[str, Dict[str, MethodResult]]" = OrderedDict()
        self.dataset_meta_: Dict[str, Dict[str, Any]] = {}
        self.failures_: List[Dict[str, Any]] = []
        self.lang_map = self._build_language_mapping()

    def _build_language_mapping(self) -> Dict[str, Any]:
        available = {e.name: e for e in self.LF}
        out = {}
        for method in self.methods:
            if method.enum_name in available:
                out[method.label] = available[method.enum_name]
            else:
                print(f"⚠ Skipping {method.label}: enum {method.enum_name} not found")
        return out

    def _adaptive_settings(self, X: np.ndarray) -> Tuple[int, bool, int]:
        d = X.shape[1]
        if d > 500:
            n_bins, use_3d, top_k = 16, False, 10
        elif d > 100:
            n_bins, use_3d, top_k = 32, False, 12
        else:
            n_bins, use_3d, top_k = 64, d <= 50, 15

        unique_x = np.unique(X)
        if len(unique_x) <= 3:
            n_bins = min(n_bins, 8)

        if not self.enable_3d:
            use_3d = False

        return n_bins, use_3d, top_k

    @staticmethod
    def _metadata_from_obj(obj: Any) -> Dict[str, Any]:
        """Best-effort extraction of theorem metadata from tree metadata.

        Accepts dicts, dataclasses/objects, lists of per-path metadata, and
        legacy attributes such as ``explainer_backend_``.
        """
        defaults = {
            "axp_backend": "none",
            "theorem_certified": False,
            "path_certificate": "none",
            "rejected_reason": "",
            "theorem_mode_used": False,
        }
        if obj is None:
            return dict(defaults)

        if isinstance(obj, list):
            if not obj:
                return dict(defaults)
            items = [LanguageComparisonBenchmark._metadata_from_obj(x) for x in obj]
            return {
                "axp_backend": _summarize_metadata_values([x["axp_backend"] for x in items]),
                "theorem_certified": _summarize_bool_values([x["theorem_certified"] for x in items]),
                "path_certificate": _summarize_metadata_values([x["path_certificate"] for x in items]),
                "rejected_reason": _summarize_metadata_values([x["rejected_reason"] for x in items], default=""),
                "theorem_mode_used": _summarize_bool_values([x["theorem_mode_used"] for x in items]),
            }

        def get(name: str, *alts: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                for key in (name, *alts):
                    if key in obj:
                        return obj[key]
            for key in (name, *alts):
                if hasattr(obj, key):
                    return getattr(obj, key)
            return default

        out = {
            "axp_backend": get("axp_backend", "backend", "backend_name", default=defaults["axp_backend"]),
            "theorem_certified": get("theorem_certified", "is_theorem_certified", default=defaults["theorem_certified"]),
            "path_certificate": get("path_certificate", "certificate", "fragment", default=defaults["path_certificate"]),
            "rejected_reason": get("rejected_reason", "reason", default=defaults["rejected_reason"]),
            "theorem_mode_used": get("theorem_mode_used", "theorem_strict", default=defaults["theorem_mode_used"]),
        }
        out["axp_backend"] = _clean_metadata_value(out["axp_backend"], "none")
        out["path_certificate"] = _clean_metadata_value(out["path_certificate"], "none")
        out["rejected_reason"] = _clean_metadata_value(out["rejected_reason"], "")
        out["theorem_certified"] = bool(out["theorem_certified"])
        out["theorem_mode_used"] = bool(out["theorem_mode_used"])
        return out

    @staticmethod
    def _extract_tree_theorem_metadata(tree: Any, method: Optional[MethodConfig] = None) -> Dict[str, Any]:
        """Extract benchmark-row theorem metadata from a fitted GSNH tree."""
        for attr in (
            "axp_metadata_",
            "last_axp_metadata_",
            "theorem_metadata_",
            "path_certificate_metadata_",
        ):
            if hasattr(tree, attr):
                meta = getattr(tree, attr)
                if isinstance(meta, list) and not meta:
                    continue
                if meta is not None:
                    return LanguageComparisonBenchmark._metadata_from_obj(meta)

        backend = getattr(tree, "explainer_backend_", None)
        if backend is not None:
            return {
                "axp_backend": _clean_metadata_value(backend),
                "theorem_certified": backend in {"structural_horn", "structural_antihorn", "two_sat"},
                "path_certificate": (
                    "horn" if backend == "structural_horn"
                    else "antihorn" if backend == "structural_antihorn"
                    else "2cnf" if backend == "two_sat"
                    else "none"
                ),
                "rejected_reason": "",
                "theorem_mode_used": bool(getattr(tree, "theorem_strict", False)),
            }

        if method is not None:
            # Conservative default for old trees without explicit metadata.
            # Empirical methods stay non-certified unless the explainer records
            # a certificate.
            if method.label in {"1D", "Horn"}:
                return {
                    "axp_backend": "structural_horn",
                    "theorem_certified": True,
                    "path_certificate": "horn",
                    "rejected_reason": "",
                    "theorem_mode_used": True,
                }
            if method.label == "AntiHorn":
                return {
                    "axp_backend": "structural_antihorn",
                    "theorem_certified": True,
                    "path_certificate": "antihorn",
                    "rejected_reason": "",
                    "theorem_mode_used": True,
                }

        return {
            "axp_backend": "none",
            "theorem_certified": False,
            "path_certificate": "none",
            "rejected_reason": "",
            "theorem_mode_used": False,
        }

    @staticmethod
    def _certificates_are_safe(cert: str) -> bool:
        cert = _clean_metadata_value(cert, "none")
        allowed = {"horn", "antihorn", "2cnf"}
        if cert in allowed:
            return True
        if not cert.startswith("mixed:"):
            return False
        body = cert[len("mixed:"):]
        if not body:
            return False
        labels = []
        for part in body.split(","):
            label = part.split("=", 1)[0].strip()
            if label:
                labels.append(label)
        return bool(labels) and all(label in allowed for label in labels)

    @staticmethod
    def _is_theorem_row(row: Dict[str, Any]) -> bool:
        """Return True iff a flattened result row is theorem-certified.

        Fallback/prototype backends are excluded even if a row is accidentally
        marked theorem_certified=True.
        """
        if not bool(row.get("theorem_certified", False)):
            return False
        if not bool(row.get("theorem_mode_used", False)):
            return False

        backend = _clean_metadata_value(row.get("axp_backend"), "none")
        cert = _clean_metadata_value(row.get("path_certificate"), "none")
        label = _clean_metadata_value(row.get("method_label"), "")

        forbidden = {"interval_dfs_fallback", "prototype_case_split", "rejected_non_theorem", "affine", "none"}
        if backend in forbidden:
            return False
        if any(x in backend for x in forbidden):
            return False

        if label == "Square2CNF":
            return backend == "two_sat" and cert == "2cnf"

        # This benchmark keeps BestPN in the empirical table even if a future
        # path-level checker records safe-looking mixed certificates.  Add a
        # separate explicitly-certified BestPN method before including it in
        # theorem tables.
        if label == "BestPN":
            return False

        return cert in {"horn", "antihorn", "2cnf"} or backend in {"structural_horn", "structural_antihorn", "two_sat"}

    def run_all(self, datasets: "OrderedDict[str, Dict[str, Any]]", skip_large: Optional[int] = 2000):
        total = len(datasets)
        for idx, (name, info) in enumerate(datasets.items(), start=1):
            X, y = info["X"], info["y"]
            self.dataset_meta_[name] = {k: v for k, v in info.items() if k not in ("X", "y")}

            if skip_large and X.shape[1] > skip_large:
                print(f"\n[{idx}/{total}] {name} — SKIPPED (d={X.shape[1]} > {skip_large})")
                continue

            print(f"\n[{idx}/{total}] {name}: n={len(y)}, d={X.shape[1]}, pos={y.mean():.2%}")
            t0 = time.time()
            self._evaluate_one(name, X, y)
            print(f"    ⏱ dataset elapsed: {time.time() - t0:.1f}s")

    def _new_metric_store(self) -> Dict[str, Dict[str, List[float]]]:
        metrics = {
            "sklearn_dt7": {
                "acc": [],
                "size": [],
                "expl": [],
                "time": [],
                "axp_valid": [],
                "axp_minimal": [],
                "axp_backend": [],
                "theorem_certified": [],
                "path_certificate": [],
                "rejected_reason": [],
                "theorem_mode_used": [],
                "leaves": [],
                "max_depth_reached": [],
                "avg_leaf_depth": [],
                "avg_predicate_arity": [],
                "max_predicate_arity": [],
                "total_literals": [],
                "predict_time": [],
                "axp_time": [],
                "weak_axp_checks": [],
                "opposite_paths_checked": [],
                "sat_vars": [],
                "sat_clauses": [],
                "errors": [],
            }
        }
        for method in self.methods:
            if method.label not in self.lang_map:
                continue
            for depth in self.depths:
                metrics[f"gsnh_{method.label}_d{depth}"] = {
                    "acc": [],
                    "size": [],
                    "expl": [],
                    "time": [],
                    "axp_valid": [],
                    "axp_minimal": [],
                    "axp_backend": [],
                    "theorem_certified": [],
                    "path_certificate": [],
                    "rejected_reason": [],
                    "theorem_mode_used": [],
                    "leaves": [],
                    "max_depth_reached": [],
                    "avg_leaf_depth": [],
                    "avg_predicate_arity": [],
                    "max_predicate_arity": [],
                    "total_literals": [],
                    "predict_time": [],
                    "axp_time": [],
                    "weak_axp_checks": [],
                    "opposite_paths_checked": [],
                    "sat_vars": [],
                    "sat_clauses": [],
                    "errors": [],
                }
        return metrics

    def _summarize_metric_store(self, metrics: Dict[str, Dict[str, List[Any]]]) -> Dict[str, MethodResult]:
        out: Dict[str, MethodResult] = {}
        for key, m in metrics.items():
            accs = [float(x) for x in m["acc"] if x is not None and not np.isnan(x)]
            if not accs and not m["errors"]:
                continue
            out[key] = MethodResult(
                acc=safe_mean(m["acc"]),
                acc_std=safe_std(m["acc"]),
                size=safe_mean(m["size"]),
                expl=safe_mean(m["expl"]),
                train_time=safe_mean(m["time"]),
                train_time_std=safe_std(m["time"]),
                axp_time=safe_mean(m.get("axp_time", [])),
                axp_time_std=safe_std(m.get("axp_time", [])),
                predict_time=safe_mean(m.get("predict_time", [])),
                predict_time_std=safe_std(m.get("predict_time", [])),
                leaves=safe_mean(m.get("leaves", [])),
                leaves_std=safe_std(m.get("leaves", [])),
                max_depth_reached=safe_mean(m.get("max_depth_reached", [])),
                avg_leaf_depth=safe_mean(m.get("avg_leaf_depth", [])),
                avg_predicate_arity=safe_mean(m.get("avg_predicate_arity", [])),
                max_predicate_arity=safe_mean(m.get("max_predicate_arity", [])),
                total_literals=safe_mean(m.get("total_literals", [])),
                weak_axp_checks_mean=safe_mean(m.get("weak_axp_checks", [])),
                opposite_paths_checked_mean=safe_mean(m.get("opposite_paths_checked", [])),
                sat_vars_mean=safe_mean(m.get("sat_vars", [])),
                sat_clauses_mean=safe_mean(m.get("sat_clauses", [])),
                axp_valid_rate=safe_mean(m["axp_valid"]),
                axp_minimal_rate=safe_mean(m["axp_minimal"]),
                n_success=len(accs),
                n_fail=len(m["errors"]),
                accs=accs,
                errors=list(m["errors"]),
                axp_backend=_summarize_metadata_values(m.get("axp_backend", []), default="none"),
                theorem_certified=_summarize_bool_values(m.get("theorem_certified", [])),
                path_certificate=_summarize_metadata_values(m.get("path_certificate", []), default="none"),
                rejected_reason=_summarize_metadata_values(m.get("rejected_reason", []), default=""),
                theorem_mode_used=_summarize_bool_values(m.get("theorem_mode_used", [])),
            )
        return out

    def _evaluate_one(self, name: str, X: np.ndarray, y: np.ndarray):
        ET = self.gsnh.ExpertGSNHTree
        SC = self.gsnh.StoppingCriteria

        n_bins, use_3d, top_k = self._adaptive_settings(X)
        metrics = self._new_metric_store()

        splitter = StratifiedShuffleSplit(
            n_splits=self.n_runs,
            test_size=0.2,
            random_state=self.rs,
        )

        for run_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # sklearn DT7
            try:
                t0 = time.time()
                dt = DecisionTreeClassifier(
                    max_depth=7,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.rs + run_idx,
                )
                dt.fit(X_tr, y_tr)
                elapsed = time.time() - t0

                t_pred = time.time()
                dt_pred = dt.predict(X_te)
                pred_elapsed = time.time() - t_pred
                dt_complexity = analyze_sklearn_tree(dt)
                metrics["sklearn_dt7"]["acc"].append(float((dt_pred == y_te).mean()))
                metrics["sklearn_dt7"]["size"].append(float(dt.tree_.node_count))
                metrics["sklearn_dt7"]["expl"].append(avg_sklearn_expl(dt, X_te))
                metrics["sklearn_dt7"]["time"].append(elapsed)
                metrics["sklearn_dt7"]["predict_time"].append(pred_elapsed)
                metrics["sklearn_dt7"]["axp_time"].append(math.nan)
                for _k, _v in dt_complexity.items():
                    metrics["sklearn_dt7"][_k].append(_v)
                metrics["sklearn_dt7"]["weak_axp_checks"].append(math.nan)
                metrics["sklearn_dt7"]["opposite_paths_checked"].append(math.nan)
                metrics["sklearn_dt7"]["sat_vars"].append(math.nan)
                metrics["sklearn_dt7"]["sat_clauses"].append(math.nan)
                metrics["sklearn_dt7"]["axp_valid"].append(math.nan)
                metrics["sklearn_dt7"]["axp_minimal"].append(math.nan)
                metrics["sklearn_dt7"]["axp_backend"].append("sklearn")
                metrics["sklearn_dt7"]["theorem_certified"].append(False)
                metrics["sklearn_dt7"]["path_certificate"].append("none")
                metrics["sklearn_dt7"]["rejected_reason"].append("baseline")
                metrics["sklearn_dt7"]["theorem_mode_used"].append(False)
            except Exception as e:
                msg = f"sklearn run {run_idx}: {e}"
                metrics["sklearn_dt7"]["errors"].append(msg)
                self.failures_.append({"dataset": name, "method": "sklearn_dt7", "run": run_idx, "error": repr(e)})

            # GSNH methods
            for method in self.methods:
                lang_family = self.lang_map.get(method.label)
                if lang_family is None:
                    continue

                for depth in self.depths:
                    key = f"gsnh_{method.label}_d{depth}"
                    try:
                        t0 = time.time()

                        # 3D is controlled globally by --enable-3d and the method flag.
                        method_search_3d = bool(method.search_3d and use_3d)

                        # Theorem-oriented methods must run their explanation checks
                        # in theorem-strict mode so benchmark metadata reflects the
                        # certified backend actually used by AXp extraction.
                        #
                        # BestPN remains empirical by default. Add a separate
                        # BestPN-Certified method if you want to benchmark the
                        # path-certificate-restricted variant.
                        theorem_mode_used = method.label in THEOREM_MODE_METHODS

                        tree = ET(
                            stopping_criteria=SC(
                                max_depth=depth,
                                min_samples_leaf=5,
                                min_samples_split=10,
                            ),
                            n_bins=n_bins,
                            top_k_features=top_k,
                            search_1d=method.search_1d,
                            search_2d=method.search_2d,
                            search_3d=method_search_3d,
                            mode=method.mode,
                            language=lang_family,
                            theorem_strict=theorem_mode_used,
                            random_state=self.rs + run_idx,
                        )
                        tree.fit(X_tr, y_tr)
                        elapsed = time.time() - t0

                        t_pred = time.time()
                        pred = tree.predict(X_te)
                        pred_elapsed = time.time() - t_pred
                        acc = float((pred == y_te).mean())
                        size = float(count_gsnh_nodes(tree.root_))
                        complexity = analyze_gsnh_tree(tree.root_)
                        t_axp = time.time()
                        expl = avg_gsnh_axp_length(
                            tree, X_te,
                            n_samples=self.axp_samples,
                            seed=self.rs + 1000 * run_idx + depth,
                        )
                        axp_elapsed = time.time() - t_axp

                        axp_valid = math.nan
                        axp_minimal = math.nan
                        if self.audit_axp:
                            axp_valid, axp_minimal = audit_axp_minimality(
                                tree, X_te,
                                n_samples=self.audit_samples,
                                seed=self.rs + 2000 * run_idx + depth,
                            )

                        expected_value = getattr(lang_family, "value", str(lang_family))
                        warning = audit_language_counts(tree, method, expected_value)
                        if warning:
                            metrics[key]["errors"].append(f"run {run_idx}: {warning}")

                        theorem_meta = self._extract_tree_theorem_metadata(tree, method)
                        # The benchmark method decides whether theorem-strict mode
                        # was intentionally used.  Do not rely on optional AXp
                        # metadata defaults, which may omit this flag.
                        theorem_meta["theorem_mode_used"] = bool(theorem_mode_used)

                        metrics[key]["acc"].append(acc)
                        metrics[key]["size"].append(size)
                        metrics[key]["expl"].append(expl)
                        metrics[key]["time"].append(elapsed)
                        metrics[key]["predict_time"].append(pred_elapsed)
                        metrics[key]["axp_time"].append(axp_elapsed)
                        for _k, _v in complexity.items():
                            metrics[key][_k].append(_v)
                        metadata_items = getattr(tree, "axp_metadata_", []) or []
                        metrics[key]["weak_axp_checks"].append(float(len(metadata_items)) if metadata_items else math.nan)
                        metrics[key]["opposite_paths_checked"].append(math.nan)
                        metrics[key]["sat_vars"].append(_metadata_numeric_mean(metadata_items, "sat_vars"))
                        metrics[key]["sat_clauses"].append(_metadata_numeric_mean(metadata_items, "sat_clauses"))
                        metrics[key]["axp_valid"].append(axp_valid)
                        metrics[key]["axp_minimal"].append(axp_minimal)
                        metrics[key]["axp_backend"].append(theorem_meta.get("axp_backend", "none"))
                        metrics[key]["theorem_certified"].append(theorem_meta.get("theorem_certified", False))
                        metrics[key]["path_certificate"].append(theorem_meta.get("path_certificate", "none"))
                        metrics[key]["rejected_reason"].append(theorem_meta.get("rejected_reason", ""))
                        metrics[key]["theorem_mode_used"].append(theorem_meta.get("theorem_mode_used", False))

                    except Exception as e:
                        msg = f"run {run_idx}: {type(e).__name__}: {e}"
                        metrics[key]["errors"].append(msg)
                        self.failures_.append({
                            "dataset": name,
                            "method": key,
                            "run": run_idx,
                            "error": repr(e),
                        })

            if run_idx == 1 or run_idx % 5 == 0 or run_idx == self.n_runs:
                dt_last = metrics["sklearn_dt7"]["acc"][-1] if metrics["sklearn_dt7"]["acc"] else math.nan
                best_key = f"gsnh_BestPN_d{max(self.depths)}"
                best_last = metrics.get(best_key, {}).get("acc", [math.nan])
                best_val = best_last[-1] if best_last else math.nan
                print(f"    Run {run_idx:>2}/{self.n_runs}: DT7={dt_last:.4f}  BestPN={best_val:.4f}")

        result = self._summarize_metric_store(metrics)
        self.results_[name] = result
        self._print_dataset_summary(result)

    def _print_dataset_summary(self, result: Dict[str, MethodResult]):
        print(f"\n    {'Method':<18} {'Depth':<6} {'Category':<10} {'Accuracy':<18} {'Size':<10} {'AXp':<8} {'Time(s)':<8} {'Fail':<5}")
        print(f"    {'─' * 95}")

        r = result.get("sklearn_dt7")
        if r:
            print(
                f"    {'sklearn DT':<18} {'7':<6} {'baseline':<10} "
                f"{r.acc:.4f}±{r.acc_std:.4f}  {r.size:>7.1f}  {r.expl:>6.2f}  "
                f"{r.train_time:>6.3f}  {r.n_fail:<5}"
            )

        for method in self.methods:
            for depth in self.depths:
                key = f"gsnh_{method.label}_d{depth}"
                r = result.get(key)
                if not r:
                    continue
                print(
                    f"    {('GSNH-' + method.label):<18} {depth:<6} {method.category:<10} "
                    f"{r.acc:.4f}±{r.acc_std:.4f}  {r.size:>7.1f}  {r.expl:>6.2f}  "
                    f"{r.train_time:>6.3f}  {r.n_fail:<5}"
                )

    # -------------------------------------------------------------------------
    # Aggregation / Reporting
    # -------------------------------------------------------------------------

    def _all_result_keys(self) -> List[str]:
        keys = ["sklearn_dt7"]
        for method in self.methods:
            if method.label not in self.lang_map:
                continue
            for depth in self.depths:
                keys.append(f"gsnh_{method.label}_d{depth}")
        return keys

    def _key_label(self, key: str) -> str:
        if key == "sklearn_dt7":
            return "DT7"
        return key.replace("gsnh_", "GSNH-").replace("_d", " D")

    def _collect(self, key: str, field: str) -> List[float]:
        vals = []
        for r in self.results_.values():
            if key in r:
                v = getattr(r[key], field)
                if v is not None and not np.isnan(v):
                    vals.append(float(v))
        return vals

    def print_table(self):
        if not self.results_:
            print("No results.")
            return

        print(f"\n{'=' * 160}")
        print("LANGUAGE FAMILY COMPARISON — GSNH-MDT vs sklearn DT(depth=7)")
        print(f"{'=' * 160}")

        for depth in self.depths:
            print(f"\n--- DEPTH {depth} ACCURACY ---")
            methods_at_depth = ["sklearn_dt7"] + [
                f"gsnh_{m.label}_d{depth}" for m in self.methods if m.label in getattr(self, "lang_map", {})
            ]

            method_labels = [self._key_label(k) for k in methods_at_depth]
            col_w = max(14, max(len(label) for label in method_labels) + 2)

            header = f"{'Dataset':<25}" + "".join(f"{label:>{col_w}}" for label in method_labels)
            print(header)
            print("─" * len(header))

            wins = defaultdict(int)
            losses = defaultdict(int)
            ties = defaultdict(int)

            for name, r in self.results_.items():
                dt_acc = r.get("sklearn_dt7", MethodResult()).acc
                row = f"{name:<25}"
                for key in methods_at_depth:
                    rr = r.get(key)
                    if not rr or np.isnan(rr.acc):
                        row += f"{'N/A':>{col_w}}"
                        continue
                    marker = ""
                    if key != "sklearn_dt7" and not np.isnan(dt_acc):
                        if rr.acc > dt_acc + 0.005:
                            marker = "↑"
                            wins[key] += 1
                        elif rr.acc < dt_acc - 0.005:
                            marker = "↓"
                            losses[key] += 1
                        else:
                            marker = "="
                            ties[key] += 1
                    val_str = f"{rr.acc:.4f}{marker}"
                    row += f"{val_str:>{col_w}}"
                print(row)

            avg_row = f"{'AVERAGE':<25}"
            for key in methods_at_depth:
                vals = self._collect(key, "acc")
                if vals:
                    val_str = f"{safe_mean(vals):.4f}"
                    avg_row += f"{val_str:>{col_w}}"
                else:
                    avg_row += f"{'N/A':>{col_w}}"
            print("─" * len(header))
            print(avg_row)

            print(f"\n  Win/Loss/Tie vs sklearn DT7 at depth {depth}:")
            for m in self.methods:
                key = f"gsnh_{m.label}_d{depth}"
                if key in methods_at_depth:
                    print(f"    {('GSNH-' + m.label):<18} {wins[key]:>2}/{losses[key]:>2}/{ties[key]:>2}")

        self.print_summary_statistics()

    def print_summary_statistics(self):
        print(f"\n{'=' * 120}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 120}")

        keys = self._all_result_keys()

        print("\n  ACCURACY across datasets:")
        print(f"    {'Method':<22} {'Mean ± Std':<22} {'N':<5}")
        print(f"    {'─' * 55}")
        for key in keys:
            vals = self._collect(key, "acc")
            if vals:
                print(f"    {self._key_label(key):<22} {safe_mean(vals):.4f} ± {safe_std(vals):.4f}   {len(vals):<5}")

        print("\n  TREE SIZE:")
        print(f"    {'Method':<22} {'Mean nodes':<12} {'Compression vs DT7':<18}")
        print(f"    {'─' * 60}")
        dt_size = safe_mean(self._collect("sklearn_dt7", "size"))
        for key in keys:
            vals = self._collect(key, "size")
            if not vals:
                continue
            mean_size = safe_mean(vals)
            comp = dt_size / max(mean_size, 1.0) if key != "sklearn_dt7" and not np.isnan(dt_size) else math.nan
            comp_s = f"{comp:.1f}×" if not np.isnan(comp) else "—"
            print(f"    {self._key_label(key):<22} {mean_size:<12.1f} {comp_s:<18}")

        print("\n  EXPLANATION LENGTH:")
        print(f"    {'Method':<22} {'Mean AXp/path length':<20}")
        print(f"    {'─' * 50}")
        for key in keys:
            vals = self._collect(key, "expl")
            if vals:
                print(f"    {self._key_label(key):<22} {safe_mean(vals):.2f}")

        print("\n  TRAINING TIME:")
        print(f"    {'Method':<22} {'Mean seconds':<15}")
        print(f"    {'─' * 45}")
        for key in keys:
            vals = self._collect(key, "train_time")
            if vals:
                print(f"    {self._key_label(key):<22} {safe_mean(vals):.3f}s")

        if self.audit_axp:
            print("\n  AXp AUDIT:")
            print(f"    {'Method':<22} {'Weak valid':<12} {'Minimal':<12}")
            print(f"    {'─' * 55}")
            for key in keys:
                if key == "sklearn_dt7":
                    continue
                v = self._collect(key, "axp_valid_rate")
                m = self._collect(key, "axp_minimal_rate")
                if v or m:
                    print(f"    {self._key_label(key):<22} {safe_mean(v):<12.3f} {safe_mean(m):<12.3f}")

        print("\n  STATISTICAL TESTS vs sklearn DT7:")
        dt = np.asarray(self._collect("sklearn_dt7", "acc"), dtype=float)
        for key in keys:
            if key == "sklearn_dt7":
                continue
            vals = np.asarray(self._collect(key, "acc"), dtype=float)
            if len(vals) != len(dt) or len(vals) < 3:
                continue
            diff = vals - dt
            if np.std(diff) <= 1e-12:
                continue
            _, p_t = stats.ttest_rel(vals, dt)
            try:
                _, p_w = stats.wilcoxon(diff)
            except Exception:
                p_w = math.nan
            print(
                f"    {self._key_label(key):<22} Δ={np.mean(diff):+.4f}  "
                f"t-test p={p_t:.4f}  Wilcoxon p={p_w:.4f}"
            )

        if self.failures_:
            print(f"\n  FAILURES: {len(self.failures_)}")
            for f in self.failures_[:20]:
                print(f"    {f['dataset']} / {f['method']} / run {f['run']}: {f['error']}")
            if len(self.failures_) > 20:
                print(f"    ... {len(self.failures_) - 20} more failures")

    def generate_latex(self) -> str:
        if not self.results_:
            return ""

        chunks = []
        for depth in self.depths:
            keys = ["sklearn_dt7"] + [
                f"gsnh_{m.label}_d{depth}" for m in self.methods if m.label in getattr(self, "lang_map", {})
            ]

            col_spec = "l|" + "c" * len(keys)
            lines = []
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering\scriptsize")
            lines.append(r"\resizebox{\textwidth}{!}{%")
            lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
            lines.append(r"\toprule")
            headers = ["Dataset"] + [self._key_label(k).replace("GSNH-", "") for k in keys]
            lines.append(" & ".join(headers) + r" \\ \midrule")

            for name, r in self.results_.items():
                vals = []
                for k in keys:
                    rr = r.get(k)
                    vals.append(rr.acc if rr else math.nan)
                best = np.nanmax(vals)

                row = [name.replace("_", r"\_")]
                for v in vals:
                    if np.isnan(v):
                        row.append("N/A")
                    else:
                        s = f"{v:.4f}"
                        if abs(v - best) < 1e-9:
                            s = r"\textbf{" + s + "}"
                        row.append(s)
                lines.append(" & ".join(row) + r" \\")

            lines.append(r"\midrule")
            avg = [f"Average ({len(self.results_)})"]
            for k in keys:
                vals = self._collect(k, "acc")
                avg.append(f"{safe_mean(vals):.4f}" if vals else "N/A")
            lines.append(" & ".join(avg) + r" \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}%")
            lines.append(r"}")
            lines.append(rf"\caption{{GSNH-MDT language-family comparison at depth {depth} vs sklearn DT depth 7. ConjUI denotes the old box/AND family; Square2CNF denotes the theorem-certified paper-style 2-CNF family when executed with backend=two_sat and path_certificate=2cnf.}}")
            lines.append(r"\end{table}")
            chunks.append("\n".join(lines))

        return "\n\n".join(chunks)


    def _result_rows(self) -> List[Dict[str, Any]]:
        """Flatten aggregated results into long-format rows with theorem metadata."""
        rows: List[Dict[str, Any]] = []
        category = {m.label: m.category for m in self.methods}
        baseline_size_by_dataset: Dict[str, float] = {}
        for ds, r in self.results_.items():
            base = r.get("sklearn_dt7")
            if base and base.size and not math.isnan(base.size):
                baseline_size_by_dataset[ds] = float(base.size)

        for ds, r in self.results_.items():
            meta = getattr(self, "dataset_meta_", {}).get(ds, {})
            baseline_size = baseline_size_by_dataset.get(ds, math.nan)
            for key, res in r.items():
                if key == "sklearn_dt7":
                    label, depth, cat = "sklearn DT", 7, "baseline"
                else:
                    label_part = key.replace("gsnh_", "")
                    label, depth_s = label_part.rsplit("_d", 1)
                    depth = int(depth_s)
                    cat = category.get(label, "")
                compression = math.nan
                if baseline_size and not math.isnan(baseline_size) and res.size and not math.isnan(res.size) and res.size > 0:
                    compression = float(baseline_size) / float(res.size)
                row = {
                    "dataset": ds,
                    "n_samples": meta.get("n_samples", meta.get("n", "")),
                    "n_features_original": meta.get("n_unary_original", meta.get("n_features_original", meta.get("n_features", ""))),
                    "n_features_encoded": meta.get("n_features", meta.get("n_features_encoded", "")),
                    "positive_rate": meta.get("pos_rate", meta.get("positive_rate", "")),
                    "method_key": key,
                    "method_label": label,
                    "depth": depth,
                    "category": cat,
                    "accuracy_mean": res.acc,
                    "accuracy_std": res.acc_std,
                    "acc": res.acc,
                    "acc_std": res.acc_std,
                    "tree_nodes_mean": res.size,
                    "tree_nodes_std": math.nan,
                    "size": res.size,
                    "leaves_mean": res.leaves,
                    "leaves_std": res.leaves_std,
                    "max_depth_reached": res.max_depth_reached,
                    "avg_leaf_depth": res.avg_leaf_depth,
                    "avg_predicate_arity": res.avg_predicate_arity,
                    "max_predicate_arity": res.max_predicate_arity,
                    "total_literals": res.total_literals,
                    "compression_vs_sklearn_dt7": compression,
                    "mean_axp_length": res.expl,
                    "expl": res.expl,
                    "axp_time_mean": res.axp_time,
                    "axp_time_std": res.axp_time_std,
                    "train_time_mean": res.train_time,
                    "train_time_std": res.train_time_std,
                    "predict_time_mean": res.predict_time,
                    "predict_time_std": res.predict_time_std,
                    "weak_axp_checks_mean": res.weak_axp_checks_mean,
                    "opposite_paths_checked_mean": res.opposite_paths_checked_mean,
                    "sat_vars_mean": res.sat_vars_mean,
                    "sat_clauses_mean": res.sat_clauses_mean,
                    "axp_valid_rate": res.axp_valid_rate,
                    "axp_minimal_rate": res.axp_minimal_rate,
                    "n_success": res.n_success,
                    "n_fail": res.n_fail,
                    "axp_backend": res.axp_backend,
                    "theorem_certified": bool(res.theorem_certified),
                    "path_certificate": res.path_certificate,
                    "rejected_reason": res.rejected_reason,
                    "theorem_mode_used": bool(res.theorem_mode_used),
                    "random_state": self.rs,
                    "n_runs": self.n_runs,
                    "train_test_split_protocol": "StratifiedShuffleSplit(test_size=0.2)",
                }
                rows.append(row)
        return rows

    @staticmethod
    def _numeric(row: Dict[str, Any], key: str) -> float:
        try:
            v = row.get(key, math.nan)
            if v == "" or v is None:
                return math.nan
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    @staticmethod
    def _mean_rows(rows: List[Dict[str, Any]], key: str) -> float:
        return safe_mean([LanguageComparisonBenchmark._numeric(r, key) for r in rows])

    @staticmethod
    def _std_rows(rows: List[Dict[str, Any]], key: str) -> float:
        return safe_std([LanguageComparisonBenchmark._numeric(r, key) for r in rows])

    @staticmethod
    def _group_by(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
        out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            out[tuple(row.get(k, "") for k in keys)].append(row)
        return out

    def _summary_by_method(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for (label, category, depth), group in sorted(self._group_by(rows, ["method_label", "category", "depth"]).items()):
            summary.append({
                "method_label": label,
                "category": category,
                "depth": depth,
                "n_datasets": len({r.get("dataset") for r in group}),
                "accuracy_mean": self._mean_rows(group, "accuracy_mean"),
                "accuracy_std": self._std_rows(group, "accuracy_mean"),
                "tree_nodes_mean": self._mean_rows(group, "tree_nodes_mean"),
                "tree_nodes_std": self._std_rows(group, "tree_nodes_mean"),
                "leaves_mean": self._mean_rows(group, "leaves_mean"),
                "leaves_std": self._std_rows(group, "leaves_mean"),
                "max_depth_reached": self._mean_rows(group, "max_depth_reached"),
                "avg_leaf_depth": self._mean_rows(group, "avg_leaf_depth"),
                "avg_predicate_arity": self._mean_rows(group, "avg_predicate_arity"),
                "max_predicate_arity": self._mean_rows(group, "max_predicate_arity"),
                "total_literals": self._mean_rows(group, "total_literals"),
                "compression_vs_sklearn_dt7": self._mean_rows(group, "compression_vs_sklearn_dt7"),
                "mean_axp_length": self._mean_rows(group, "mean_axp_length"),
                "axp_time_mean": self._mean_rows(group, "axp_time_mean"),
                "axp_time_std": self._std_rows(group, "axp_time_mean"),
                "train_time_mean": self._mean_rows(group, "train_time_mean"),
                "train_time_std": self._std_rows(group, "train_time_mean"),
                "predict_time_mean": self._mean_rows(group, "predict_time_mean"),
                "predict_time_std": self._std_rows(group, "predict_time_mean"),
                "weak_axp_checks_mean": self._mean_rows(group, "weak_axp_checks_mean"),
                "opposite_paths_checked_mean": self._mean_rows(group, "opposite_paths_checked_mean"),
                "sat_vars_mean": self._mean_rows(group, "sat_vars_mean"),
                "sat_clauses_mean": self._mean_rows(group, "sat_clauses_mean"),
                "n_success": int(sum(self._numeric(r, "n_success") for r in group if not math.isnan(self._numeric(r, "n_success")))),
                "n_fail": int(sum(self._numeric(r, "n_fail") for r in group if not math.isnan(self._numeric(r, "n_fail")))),
            })
        return summary

    def _complexity_by_dataset(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for (dataset,), group in sorted(self._group_by(rows, ["dataset"]).items()):
            sample = group[0]
            out.append({
                "dataset": dataset,
                "n_samples": sample.get("n_samples", ""),
                "n_features_original": sample.get("n_features_original", ""),
                "n_features_encoded": sample.get("n_features_encoded", ""),
                "positive_rate": sample.get("positive_rate", ""),
                "accuracy_mean": self._mean_rows(group, "accuracy_mean"),
                "tree_nodes_mean": self._mean_rows(group, "tree_nodes_mean"),
                "mean_axp_length": self._mean_rows(group, "mean_axp_length"),
                "train_time_mean": self._mean_rows(group, "train_time_mean"),
                "axp_time_mean": self._mean_rows(group, "axp_time_mean"),
                "sat_vars_mean": self._mean_rows(group, "sat_vars_mean"),
                "sat_clauses_mean": self._mean_rows(group, "sat_clauses_mean"),
            })
        return out

    def _pareto_front(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = [r for r in rows if not math.isnan(self._numeric(r, "accuracy_mean"))]
        out: List[Dict[str, Any]] = []
        for row in candidates:
            acc = self._numeric(row, "accuracy_mean")
            size = self._numeric(row, "tree_nodes_mean")
            axp = self._numeric(row, "mean_axp_length")
            dominated_size = False
            dominated_axp = False
            for other in candidates:
                if other is row:
                    continue
                oacc = self._numeric(other, "accuracy_mean")
                osize = self._numeric(other, "tree_nodes_mean")
                oaxp = self._numeric(other, "mean_axp_length")
                if not math.isnan(size) and not math.isnan(osize):
                    if oacc >= acc and osize <= size and (oacc > acc or osize < size):
                        dominated_size = True
                if not math.isnan(axp) and not math.isnan(oaxp):
                    if oacc >= acc and oaxp <= axp and (oacc > acc or oaxp < axp):
                        dominated_axp = True
            if not dominated_size or not dominated_axp:
                out.append({**row, "pareto_accuracy_size": not dominated_size, "pareto_accuracy_axp": not dominated_axp})
        return out

    def _dataset_win_loss(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_dataset = self._group_by(rows, ["dataset"])
        out: List[Dict[str, Any]] = []
        for (dataset,), group in sorted(by_dataset.items()):
            base = next((r for r in group if r.get("method_label") == "sklearn DT"), None)
            base_acc = self._numeric(base or {}, "accuracy_mean")
            for row in group:
                if row.get("method_label") == "sklearn DT":
                    continue
                acc = self._numeric(row, "accuracy_mean")
                delta = acc - base_acc if not math.isnan(acc) and not math.isnan(base_acc) else math.nan
                if math.isnan(delta):
                    outcome = "unknown"
                elif delta > 1e-12:
                    outcome = "win"
                elif delta < -1e-12:
                    outcome = "loss"
                else:
                    outcome = "tie"
                out.append({
                    "dataset": dataset,
                    "method_label": row.get("method_label"),
                    "category": row.get("category"),
                    "depth": row.get("depth"),
                    "baseline_accuracy": base_acc,
                    "method_accuracy": acc,
                    "accuracy_delta_vs_sklearn_dt7": delta,
                    "outcome": outcome,
                })
        return out

    @staticmethod
    def _write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = list(dict.fromkeys(REQUIRED_RESULT_COLUMNS + keys)) if rows else REQUIRED_RESULT_COLUMNS
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)

    @staticmethod
    def _write_rows_json(path: str, rows: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=json_default)

    @staticmethod
    def _latex_escape(value: Any) -> str:
        return str(value).replace("_", r"\_").replace("%", r"\%")

    @staticmethod
    def _fmt_latex(value: Any) -> str:
        try:
            f = float(value)
            if math.isnan(f):
                return "--"
            return f"{f:.4g}"
        except (TypeError, ValueError):
            return LanguageComparisonBenchmark._latex_escape(value)

    @staticmethod
    def _write_simple_latex(path: str, rows: List[Dict[str, Any]], columns: Sequence[str], caption: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(r"\begin{table}[htbp]" + "\n")
            f.write(r"\centering\scriptsize" + "\n")
            f.write(r"\begin{tabular}{" + "l" * len(columns) + "}\n")
            f.write(r"\toprule" + "\n")
            f.write(" & ".join(LanguageComparisonBenchmark._latex_escape(c) for c in columns) + r" \\ \midrule" + "\n")
            for row in rows[:80]:
                f.write(" & ".join(LanguageComparisonBenchmark._fmt_latex(row.get(c, "")) for c in columns) + r" \\" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(rf"\caption{{{LanguageComparisonBenchmark._latex_escape(caption)}}}" + "\n")
            f.write(r"\end{table}" + "\n")

    @staticmethod
    def _write_png(path: str, width: int, height: int, pixels: bytearray) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)
        raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)

    @staticmethod
    def _canvas(width: int = 1200, height: int = 800, color: Tuple[int, int, int] = (255, 255, 255)) -> bytearray:
        return bytearray(color * (width * height))

    @staticmethod
    def _rect(pixels: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        for y in range(y0, y1):
            row = y * width * 3
            for x in range(x0, x1):
                i = row + x * 3
                pixels[i:i+3] = bytes(color)

    @staticmethod
    def _bar_plot(path: str, labels: Sequence[str], values: Sequence[float]) -> None:
        width, height = 1200, 800
        px = LanguageComparisonBenchmark._canvas(width, height)
        LanguageComparisonBenchmark._rect(px, width, height, 90, 80, 95, 700, (0, 0, 0))
        LanguageComparisonBenchmark._rect(px, width, height, 90, 695, 1120, 700, (0, 0, 0))
        clean_vals = [0.0 if v is None or math.isnan(float(v)) else max(0.0, float(v)) for v in values]
        vmax = max(clean_vals) if clean_vals else 1.0
        vmax = vmax if vmax > 0 else 1.0
        n = max(1, len(clean_vals))
        slot = 1000 / n
        colors = [(47, 112, 193), (68, 157, 91), (204, 120, 50), (150, 90, 180), (200, 70, 90), (80, 160, 170), (120, 120, 120)]
        for i, v in enumerate(clean_vals):
            x0 = int(110 + i * slot)
            x1 = int(110 + (i + 0.72) * slot)
            bar_h = int((v / vmax) * 560)
            LanguageComparisonBenchmark._rect(px, width, height, x0, 695 - bar_h, x1, 695, colors[i % len(colors)])
        LanguageComparisonBenchmark._write_png(path, width, height, px)

    @staticmethod
    def _scatter_plot(path: str, xs: Sequence[float], ys: Sequence[float]) -> None:
        width, height = 1200, 800
        px = LanguageComparisonBenchmark._canvas(width, height)
        LanguageComparisonBenchmark._rect(px, width, height, 90, 80, 95, 700, (0, 0, 0))
        LanguageComparisonBenchmark._rect(px, width, height, 90, 695, 1120, 700, (0, 0, 0))
        pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and not math.isnan(float(x)) and not math.isnan(float(y))]
        if not pairs:
            LanguageComparisonBenchmark._write_png(path, width, height, px); return
        minx, maxx = min(x for x, _ in pairs), max(x for x, _ in pairs)
        miny, maxy = min(y for _, y in pairs), max(y for _, y in pairs)
        if minx == maxx: maxx = minx + 1.0
        if miny == maxy: maxy = miny + 1.0
        colors = [(47, 112, 193), (68, 157, 91), (204, 120, 50), (150, 90, 180), (200, 70, 90)]
        for i, (x, y) in enumerate(pairs):
            px_x = int(100 + (x - minx) / (maxx - minx) * 1000)
            px_y = int(690 - (y - miny) / (maxy - miny) * 590)
            LanguageComparisonBenchmark._rect(px, width, height, px_x - 6, px_y - 6, px_x + 7, px_y + 7, colors[i % len(colors)])
        LanguageComparisonBenchmark._write_png(path, width, height, px)

    @staticmethod
    def _heatmap_plot(path: str, rows: List[Dict[str, Any]]) -> None:
        width, height = 1200, 800
        px = LanguageComparisonBenchmark._canvas(width, height)
        methods = sorted({str(r.get("method_label")) for r in rows}) or ["none"]
        datasets = sorted({str(r.get("dataset")) for r in rows}) or ["none"]
        cell_w = max(1, min(90, 1000 // max(1, len(methods))))
        cell_h = max(1, min(70, 620 // max(1, len(datasets))))
        outcome_color = {"win": (72, 160, 90), "tie": (210, 185, 80), "loss": (190, 75, 75), "unknown": (170, 170, 170)}
        lookup = {(str(r.get("dataset")), str(r.get("method_label"))): str(r.get("outcome")) for r in rows}
        for iy, ds in enumerate(datasets):
            for ix, method in enumerate(methods):
                color = outcome_color.get(lookup.get((ds, method), "unknown"), (170, 170, 170))
                x0, y0 = 110 + ix * cell_w, 90 + iy * cell_h
                LanguageComparisonBenchmark._rect(px, width, height, x0, y0, x0 + cell_w - 2, y0 + cell_h - 2, color)
        LanguageComparisonBenchmark._write_png(path, width, height, px)

    def _write_figures(self, figure_dir: str, rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], win_loss: List[Dict[str, Any]]) -> None:
        os.makedirs(figure_dir, exist_ok=True)
        labels = [str(r.get("method_label")) for r in summary]
        self._bar_plot(os.path.join(figure_dir, "accuracy_by_method.png"), labels, [self._numeric(r, "accuracy_mean") for r in summary])
        self._bar_plot(os.path.join(figure_dir, "tree_size_by_method.png"), labels, [self._numeric(r, "tree_nodes_mean") for r in summary])
        self._bar_plot(os.path.join(figure_dir, "axp_length_by_method.png"), labels, [self._numeric(r, "mean_axp_length") for r in summary])
        self._bar_plot(os.path.join(figure_dir, "train_time_by_method.png"), labels, [self._numeric(r, "train_time_mean") for r in summary])
        self._bar_plot(os.path.join(figure_dir, "axp_time_by_method.png"), labels, [self._numeric(r, "axp_time_mean") for r in summary])
        self._scatter_plot(os.path.join(figure_dir, "accuracy_vs_size_pareto.png"), [self._numeric(r, "tree_nodes_mean") for r in rows], [self._numeric(r, "accuracy_mean") for r in rows])
        self._scatter_plot(os.path.join(figure_dir, "accuracy_vs_axp_pareto.png"), [self._numeric(r, "mean_axp_length") for r in rows], [self._numeric(r, "accuracy_mean") for r in rows])
        self._scatter_plot(os.path.join(figure_dir, "runtime_vs_dataset_size.png"), [self._numeric(r, "n_samples") for r in rows], [self._numeric(r, "train_time_mean") for r in rows])
        self._heatmap_plot(os.path.join(figure_dir, "win_loss_heatmap.png"), win_loss)
        status = Counter(str(r.get("category")) for r in rows)
        self._bar_plot(os.path.join(figure_dir, "theorem_vs_auxiliary_summary.png"), list(status.keys()), [float(v) for v in status.values()])

    def _write_report(self, report_dir: str, rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], theorem_rows: List[Dict[str, Any]], auxiliary_rows: List[Dict[str, Any]]) -> None:
        os.makedirs(report_dir, exist_ok=True)
        md_path = os.path.join(report_dir, "benchmark_summary.md")
        html_path = os.path.join(report_dir, "benchmark_summary.html")
        lines = [
            "# GSNH-MDT Language-Family Benchmark Summary",
            "",
            "This report is generated by `scripts/benchmark_dl8_languages_updated.py` as a scientific evidence package for comparing GSNH-MDT split-language families.",
            "",
            "## How to read the metrics",
            "",
            "- **Accuracy** is held-out test-set classification accuracy under a deterministic stratified split protocol.",
            "- **Tree size** is the number of fitted tree nodes; smaller trees are usually easier to inspect.",
            "- **AXp length** is the mean number of input features retained in extracted abductive explanations; shorter AXps are more compact explanations.",
            "- **Train time** measures model fitting time. **AXp time** measures explanation extraction time on sampled test instances.",
            "- **SAT variables/clauses** summarize the certificate/path encodings used during AXp checks when metadata is available.",
            "- **Complexity metrics are measured empirical complexity, not formal Big-O proofs.**",
            "",
            "## Theorem boundary",
            "",
            "The theorem-certified table is intentionally strict: rows require `theorem_mode_used=True` and `theorem_certified=True`; Square2CNF additionally requires `axp_backend=two_sat` and `path_certificate=2cnf`. Affine remains auxiliary until a verified Python GF(2) certificate checker exists, and BestPN remains empirical by default.",
            "",
            f"- Full result rows: {len(rows)}",
            f"- Theorem-certified rows: {len(theorem_rows)}",
            f"- Auxiliary/baseline/empirical rows: {len(auxiliary_rows)}",
            "",
            "## Method summary",
            "",
            "| Method | Category | Depth | Accuracy | Nodes | AXp length | Train time |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            lines.append(
                f"| {row.get('method_label')} | {row.get('category')} | {row.get('depth')} | "
                f"{self._fmt_latex(row.get('accuracy_mean'))} | {self._fmt_latex(row.get('tree_nodes_mean'))} | "
                f"{self._fmt_latex(row.get('mean_axp_length'))} | {self._fmt_latex(row.get('train_time_mean'))} |"
            )
        md = "\n".join(lines) + "\n"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        html = "<html><head><meta charset='utf-8'><title>GSNH-MDT Benchmark Summary</title></head><body>" + "\n".join(
            f"<p>{line}</p>" if line and not line.startswith("#") and not line.startswith("|") and not line.startswith("-") else
            (f"<h1>{line[2:]}</h1>" if line.startswith("# ") else f"<h2>{line[3:]}</h2>" if line.startswith("## ") else f"<pre>{line}</pre>")
            for line in lines
        ) + "</body></html>"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    def save_outputs(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        table_dir = os.path.join(output_dir, "tables")
        figure_dir = os.path.join(output_dir, "figures")
        report_dir = os.path.join(output_dir, "report")
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        rows = self._result_rows()
        theorem_rows = [row for row in rows if self._is_theorem_row(row)]
        auxiliary_rows = [row for row in rows if not self._is_theorem_row(row)]
        summary = self._summary_by_method(rows)
        complexity_by_method = [dict(r) for r in summary]
        complexity_by_dataset = self._complexity_by_dataset(rows)
        pareto = self._pareto_front(rows)
        win_loss = self._dataset_win_loss(rows)

        self._write_rows_csv(os.path.join(output_dir, "full_results.csv"), rows)
        self._write_rows_csv(os.path.join(output_dir, "summary_by_method.csv"), summary)
        self._write_rows_csv(os.path.join(output_dir, "theorem_certified_results.csv"), theorem_rows)
        self._write_rows_csv(os.path.join(output_dir, "auxiliary_results.csv"), auxiliary_rows)
        self._write_rows_csv(os.path.join(output_dir, "complexity_by_method.csv"), complexity_by_method)
        self._write_rows_csv(os.path.join(output_dir, "complexity_by_dataset.csv"), complexity_by_dataset)
        self._write_rows_csv(os.path.join(output_dir, "pareto_front.csv"), pareto)
        self._write_rows_csv(os.path.join(output_dir, "dataset_win_loss.csv"), win_loss)

        self._write_rows_json(os.path.join(output_dir, "full_results.json"), rows)
        self._write_rows_json(os.path.join(output_dir, "theorem_certified_results.json"), theorem_rows)
        self._write_rows_json(os.path.join(output_dir, "auxiliary_results.json"), auxiliary_rows)

        self._write_simple_latex(os.path.join(table_dir, "main_summary.tex"), summary,
                                 ["method_label", "category", "depth", "accuracy_mean", "tree_nodes_mean", "mean_axp_length", "train_time_mean"],
                                 "Main GSNH-MDT language-family summary.")
        self._write_simple_latex(os.path.join(table_dir, "theorem_certified.tex"), theorem_rows,
                                 ["dataset", "method_label", "depth", "accuracy_mean", "tree_nodes_mean", "mean_axp_length", "axp_backend", "path_certificate"],
                                 "Strict theorem-certified rows only.")
        self._write_simple_latex(os.path.join(table_dir, "auxiliary.tex"), auxiliary_rows,
                                 ["dataset", "method_label", "category", "depth", "accuracy_mean", "axp_backend", "path_certificate"],
                                 "Auxiliary, empirical, baseline, and non-certified rows.")
        self._write_simple_latex(os.path.join(table_dir, "complexity_summary.tex"), complexity_by_method,
                                 ["method_label", "depth", "tree_nodes_mean", "leaves_mean", "avg_leaf_depth", "total_literals", "sat_vars_mean", "sat_clauses_mean"],
                                 "Measured structural and SAT-encoding complexity summary.")
        self._write_simple_latex(os.path.join(table_dir, "per_dataset_accuracy.tex"), rows,
                                 ["dataset", "method_label", "depth", "accuracy_mean", "tree_nodes_mean", "mean_axp_length"],
                                 "Per-dataset accuracy and compactness.")

        # Backward-compatible filenames from the older script.
        payload = {
            "methods": [asdict(m) for m in self.methods],
            "depths": self.depths,
            "n_runs": self.n_runs,
            "random_state": self.rs,
            "dataset_meta": self.dataset_meta_,
            "results": {ds: {k: asdict(v) for k, v in r.items()} for ds, r in self.results_.items()},
            "failures": self.failures_,
        }
        with open(os.path.join(output_dir, "language_comparison_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=json_default)
        self._write_rows_csv(os.path.join(output_dir, "language_comparison_results.csv"), rows)
        with open(os.path.join(output_dir, "language_comparison_results.tex"), "w", encoding="utf-8") as f:
            f.write(self.generate_latex())
        self._write_simple_latex(os.path.join(output_dir, "theorem_certified_results.tex"), theorem_rows,
                                 ["dataset", "method_label", "depth", "accuracy_mean", "axp_backend", "path_certificate"],
                                 "Theorem-certified GSNH-MDT benchmark rows.")
        self._write_simple_latex(os.path.join(output_dir, "auxiliary_results.tex"), auxiliary_rows,
                                 ["dataset", "method_label", "category", "depth", "accuracy_mean", "axp_backend"],
                                 "Auxiliary, prototype, fallback, or non-certified GSNH-MDT benchmark rows.")

        self._write_figures(figure_dir, rows, summary, win_loss)
        self._write_report(report_dir, rows, summary, theorem_rows, auxiliary_rows)

        print(f"\n✓ Saved evidence package under: {output_dir}")
        print("  CSV: full_results, summary_by_method, theorem_certified_results, auxiliary_results, complexity, pareto, win/loss")
        print("  LaTeX: tables/main_summary.tex, theorem_certified.tex, auxiliary.tex, complexity_summary.tex, per_dataset_accuracy.tex")
        print("  Figures: figures/*.png")
        print("  Report: report/benchmark_summary.html and report/benchmark_summary.md")


# =============================================================================
# CLI
# =============================================================================

def find_data_dir(explicit: Optional[str]) -> Optional[str]:
    """Backward-compatible wrapper for resolved data directory."""
    return str(resolve_data_dir(explicit))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark GSNH-MDT language families on .dl8 datasets."
    )
    p.add_argument("--data-dir", default=None, help="Directory containing .dl8 files.")
    p.add_argument("--output-dir", default="experiment_artifacts/language_benchmark",
                   help="Directory for JSON/CSV/LaTeX outputs.")
    p.add_argument("--runs", type=int, default=10, help="Number of train/test splits.")
    p.add_argument("--depths", type=int, nargs="+", default=[5, 7], help="GSNH depths.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--skip-large", type=int, default=2000,
                   help="Skip datasets with more than this many features. Use 0 to disable.")
    p.add_argument("--max-datasets", type=int, default=None, help="Limit number of datasets.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional exact dataset names without .dl8 extension.")
    p.add_argument("--axp-samples", type=int, default=50,
                   help="Number of test instances sampled for AXp length per run.")
    p.add_argument("--audit-axp", action="store_true",
                   help="Check weak AXp validity and subset-minimality on samples.")
    p.add_argument("--audit-samples", type=int, default=20,
                   help="Number of sampled test instances for AXp audit.")
    p.add_argument("--enable-3d", action="store_true",
                   help="Allow 3D search for methods that request it.")
    p.add_argument("--no-square2cnf", action="store_true",
                   help="Disable the new Square2CNF method.")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke test: runs=1, depths=5, max-datasets=2, axp-samples=10.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    if args.quick:
        args.runs = 1
        args.depths = [5]
        args.max_datasets = 2 if args.max_datasets is None else args.max_datasets
        args.axp_samples = min(args.axp_samples, 10)
        args.audit_samples = min(args.audit_samples, 5)

    print("\n" + "#" * 90)
    print("#   GSNH-MDT Language Family Benchmark — Professor's .dl8 Data")
    print("#" * 90)

    print("\nMethods:")
    methods = default_method_configs(include_square2cnf=not args.no_square2cnf)
    for m in methods:
        print(f"  - GSNH-{m.label:<12} [{m.category:<9}] {m.description}")
    print("  - sklearn DT7     [baseline ] CART baseline at depth 7")

    print("\n[1] Importing GSNH...")
    gsnh = import_gsnh()

    datasets = None
    try:
        data_dir = resolve_data_dir(args.data_dir)
        print()
        print(f"[2] Resolved data directory: {data_dir}")
        discovered_files = discover_dl8_files(data_dir)
        print(f"[2] Discovered {len(discovered_files)} .dl8 files recursively")
        print(f"[2] Loading .dl8 files from {data_dir}/...")
        datasets = load_all_dl8(
            data_dir,
            max_datasets=args.max_datasets,
            dataset_filter=args.datasets,
        )
    except FileNotFoundError as exc:
        if not args.quick:
            print(f"✗ {exc}")
            return None
        print(f"⚠ {exc}")
        print("[2] --quick mode: using deterministic synthetic smoke dataset instead.")
        datasets = make_quick_synthetic_datasets()
    if not datasets:
        if args.quick:
            print("[2] --quick mode: no .dl8 datasets loaded; using deterministic synthetic smoke dataset instead.")
            datasets = make_quick_synthetic_datasets()
        else:
            print("No datasets loaded.")
            return None

    print(
        f"\n[3] Running benchmark: {len(datasets)} datasets × {args.runs} runs × "
        f"depths={args.depths}"
    )
    bench = LanguageComparisonBenchmark(
        gsnh_module=gsnh,
        methods=methods,
        depths=args.depths,
        n_runs=args.runs,
        random_state=args.seed,
        axp_samples=args.axp_samples,
        audit_axp=args.audit_axp,
        audit_samples=args.audit_samples,
        enable_3d=args.enable_3d,
    )
    skip_large = None if args.skip_large == 0 else args.skip_large
    bench.run_all(datasets, skip_large=skip_large)

    print("\n[4] Results")
    bench.print_table()

    print("\n[5] Saving outputs")
    bench.save_outputs(args.output_dir)

    print("\n" + "#" * 90)
    print("# COMPLETE")
    print("#" * 90 + "\n")

    return bench


if __name__ == "__main__":
    main()
