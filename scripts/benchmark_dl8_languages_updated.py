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

        forbidden = {"interval_dfs_fallback", "prototype_case_split", "rejected_non_theorem", "none"}
        if backend in forbidden:
            return False
        if any(x in backend for x in forbidden):
            return False

        if label == "Square2CNF":
            return backend == "two_sat" and cert == "2cnf"

        if label == "BestPN":
            return LanguageComparisonBenchmark._certificates_are_safe(cert)

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

                metrics["sklearn_dt7"]["acc"].append(float((dt.predict(X_te) == y_te).mean()))
                metrics["sklearn_dt7"]["size"].append(float(dt.tree_.node_count))
                metrics["sklearn_dt7"]["expl"].append(avg_sklearn_expl(dt, X_te))
                metrics["sklearn_dt7"]["time"].append(elapsed)
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

                        pred = tree.predict(X_te)
                        acc = float((pred == y_te).mean())
                        size = float(count_gsnh_nodes(tree.root_))
                        expl = avg_gsnh_axp_length(
                            tree, X_te,
                            n_samples=self.axp_samples,
                            seed=self.rs + 1000 * run_idx + depth,
                        )

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
                f"gsnh_{m.label}_d{depth}" for m in self.methods if m.label in self.lang_map
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
                f"gsnh_{m.label}_d{depth}" for m in self.methods if m.label in self.lang_map
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
        for ds, r in self.results_.items():
            for key, res in r.items():
                if key == "sklearn_dt7":
                    label, depth, cat = "sklearn DT", 7, "baseline"
                else:
                    label_part = key.replace("gsnh_", "")
                    label, depth_s = label_part.rsplit("_d", 1)
                    depth = int(depth_s)
                    cat = category.get(label, "")
                rows.append({
                    "dataset": ds,
                    "method_key": key,
                    "method_label": label,
                    "depth": depth,
                    "category": cat,
                    "acc": res.acc,
                    "acc_std": res.acc_std,
                    "size": res.size,
                    "expl": res.expl,
                    "train_time": res.train_time,
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
                })
        return rows

    @staticmethod
    def _write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        fieldnames = [
            "dataset", "method_key", "method_label", "depth", "category",
            "acc", "acc_std", "size", "expl", "train_time",
            "axp_valid_rate", "axp_minimal_rate", "n_success", "n_fail",
            "axp_backend", "theorem_certified", "path_certificate",
            "rejected_reason", "theorem_mode_used", "random_state", "n_runs",
            "train_test_split_protocol",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)

    @staticmethod
    def _write_rows_json(path: str, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=json_default)

    @staticmethod
    def _latex_escape(value: Any) -> str:
        return str(value).replace("_", r"\_")

    @staticmethod
    def _write_rows_latex(path: str, rows: List[Dict[str, Any]], caption: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(r"\begin{table}[htbp]" + "\n")
            f.write(r"\centering\scriptsize" + "\n")
            f.write(r"\begin{tabular}{llllrrrlll}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(
                r"Dataset & Method & Depth & Cert. & Acc. & Size & AXp & Backend & Path Cert. & Category \\ \midrule" + "\n"
            )
            for row in rows:
                acc = row.get("acc", math.nan)
                size = row.get("size", math.nan)
                expl = row.get("expl", math.nan)
                f.write(
                    f"{LanguageComparisonBenchmark._latex_escape(row.get('dataset',''))} & "
                    f"{LanguageComparisonBenchmark._latex_escape(row.get('method_label',''))} & "
                    f"{row.get('depth','')} & "
                    f"{row.get('theorem_certified', False)} & "
                    f"{float(acc):.4f} & "
                    f"{float(size):.1f} & "
                    f"{float(expl):.2f} & "
                    f"{LanguageComparisonBenchmark._latex_escape(row.get('axp_backend',''))} & "
                    f"{LanguageComparisonBenchmark._latex_escape(row.get('path_certificate',''))} & "
                    f"{LanguageComparisonBenchmark._latex_escape(row.get('category',''))} \\\n"
                )
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(rf"\caption{{{caption}}}" + "\n")
            f.write(r"\end{table}" + "\n")

    def save_outputs(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        rows = self._result_rows()
        theorem_rows = [row for row in rows if self._is_theorem_row(row)]
        auxiliary_rows = [row for row in rows if not self._is_theorem_row(row)]

        full_json_path = os.path.join(output_dir, "full_results.json")
        full_csv_path = os.path.join(output_dir, "full_results.csv")
        theorem_json_path = os.path.join(output_dir, "theorem_certified_results.json")
        theorem_csv_path = os.path.join(output_dir, "theorem_certified_results.csv")
        auxiliary_json_path = os.path.join(output_dir, "auxiliary_results.json")
        auxiliary_csv_path = os.path.join(output_dir, "auxiliary_results.csv")
        theorem_tex_path = os.path.join(output_dir, "theorem_certified_results.tex")
        auxiliary_tex_path = os.path.join(output_dir, "auxiliary_results.tex")

        self._write_rows_json(full_json_path, rows)
        self._write_rows_csv(full_csv_path, rows)
        self._write_rows_json(theorem_json_path, theorem_rows)
        self._write_rows_csv(theorem_csv_path, theorem_rows)
        self._write_rows_json(auxiliary_json_path, auxiliary_rows)
        self._write_rows_csv(auxiliary_csv_path, auxiliary_rows)
        self._write_rows_latex(theorem_tex_path, theorem_rows, "Theorem-certified GSNH-MDT benchmark rows.")
        self._write_rows_latex(auxiliary_tex_path, auxiliary_rows, "Auxiliary, prototype, fallback, or non-certified GSNH-MDT benchmark rows.")

        # JSON
        payload = {
            "methods": [asdict(m) for m in self.methods],
            "depths": self.depths,
            "n_runs": self.n_runs,
            "random_state": self.rs,
            "dataset_meta": self.dataset_meta_,
            "results": {
                ds: {k: asdict(v) for k, v in r.items()}
                for ds, r in self.results_.items()
            },
            "failures": self.failures_,
        }
        json_path = os.path.join(output_dir, "language_comparison_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=json_default)

        # CSV long format
        csv_path = os.path.join(output_dir, "language_comparison_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset", "method_key", "method_label", "depth", "category",
                "acc", "acc_std", "size", "expl", "train_time",
                "axp_valid_rate", "axp_minimal_rate", "n_success", "n_fail",
                "axp_backend", "theorem_certified", "path_certificate",
                "rejected_reason", "theorem_mode_used",
            ])
            category = {m.label: m.category for m in self.methods}
            for ds, r in self.results_.items():
                for key, res in r.items():
                    if key == "sklearn_dt7":
                        label, depth, cat = "sklearn DT", 7, "baseline"
                    else:
                        label_part = key.replace("gsnh_", "")
                        label, depth_s = label_part.rsplit("_d", 1)
                        depth = int(depth_s)
                        cat = category.get(label, "")
                    w.writerow([
                        ds, key, label, depth, cat,
                        res.acc, res.acc_std, res.size, res.expl, res.train_time,
                        res.axp_valid_rate, res.axp_minimal_rate,
                        res.n_success, res.n_fail,
                        res.axp_backend, res.theorem_certified, res.path_certificate,
                        res.rejected_reason, res.theorem_mode_used,
                    ])

        # LaTeX
        latex = self.generate_latex()
        tex_path = os.path.join(output_dir, "language_comparison_results.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)

        print(f"\n✓ Saved JSON:  {json_path}")
        print(f"✓ Saved CSV:   {csv_path}")
        print(f"✓ Saved LaTeX: {tex_path}")
        print(f"✓ Saved full theorem-aware JSON/CSV: {full_json_path}, {full_csv_path}")
        print(f"✓ Saved theorem-certified JSON/CSV/LaTeX: {theorem_json_path}, {theorem_csv_path}, {theorem_tex_path}")
        print(f"✓ Saved auxiliary JSON/CSV/LaTeX: {auxiliary_json_path}, {auxiliary_csv_path}, {auxiliary_tex_path}")


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

    data_dir = resolve_data_dir(args.data_dir)
    print()
    print(f"[2] Resolved data directory: {data_dir}")
    try:
        discovered_files = discover_dl8_files(data_dir)
    except FileNotFoundError as exc:
        print(f"✗ {exc}")
        return None
    print(f"[2] Discovered {len(discovered_files)} .dl8 files recursively")
    print(f"[2] Loading .dl8 files from {data_dir}/...")
    datasets = load_all_dl8(
        data_dir,
        max_datasets=args.max_datasets,
        dataset_filter=args.datasets,
    )
    if not datasets:
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
