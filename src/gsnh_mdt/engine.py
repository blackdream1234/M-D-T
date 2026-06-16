"""Optional Python/Rust engine wrapper for GSNH classifiers.

The default remains the Python reference implementation. The Rust engine is
available only when explicitly requested with ``engine="rust"`` and when the
optional ``_rust_gsnh`` extension has been installed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from gsnh_mdt.tree.builder import ExpertGSNHTree
from gsnh_mdt.tree.stopping import StoppingCriteria
from gsnh_mdt.types import LanguageFamily


_SUPPORTED_RUST_FAMILIES = {"ConjUI", "Horn", "AntiHorn", "Affine", "Square2CNF"}
_UNSUPPORTED_RUST_FAMILIES = {"Any", "BestPerNode", "SquareCNF"}

_PYTHON_FAMILY_MAP = {
    "ConjUI": LanguageFamily.CONJ_UI,
    "Horn": LanguageFamily.HORN,
    "AntiHorn": LanguageFamily.ANTI_HORN,
    "Affine": LanguageFamily.AFFINE,
    "Square2CNF": LanguageFamily.SQUARE_2CNF,
    "Any": LanguageFamily.ANY,
    "BestPerNode": LanguageFamily.BEST_PER_NODE,
    "SquareCNF": LanguageFamily.SQUARE_CNF,
}


class GSNHEngineClassifier:
    """Explicit engine-selecting wrapper around Python GSNH and optional Rust GSNH.

    Parameters are intentionally limited to the stable shared subset. Additional
    keyword arguments are forwarded only to the Python reference engine. The Rust
    path rejects extra options because the binding does not support them yet.
    """

    def __init__(
        self,
        engine: str = "python",
        family: str = "ConjUI",
        max_arity: int = 2,
        max_depth: int = 2,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        **kwargs: Any,
    ):
        if engine not in {"python", "rust"}:
            raise ValueError("engine must be either 'python' or 'rust'")
        self.engine = engine
        self.family = family
        self.max_arity = max_arity
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.kwargs = dict(kwargs)
        self.model_: Optional[Any] = None
        self._is_fitted = False

        if self.engine == "rust":
            self._validate_rust_config()

    def _validate_rust_config(self) -> None:
        if self.family in _UNSUPPORTED_RUST_FAMILIES:
            raise ValueError(f"family={self.family!r} is not supported by the Rust engine wrapper")
        if self.family not in _SUPPORTED_RUST_FAMILIES:
            raise ValueError(f"unknown Rust engine family: {self.family!r}")
        if self.kwargs:
            unsupported = ", ".join(sorted(self.kwargs))
            raise ValueError(f"Rust engine does not support extra options yet: {unsupported}")

    def _python_language(self) -> LanguageFamily:
        if isinstance(self.family, LanguageFamily):
            return self.family
        try:
            return _PYTHON_FAMILY_MAP[self.family]
        except KeyError as exc:
            raise ValueError(f"unknown Python engine family: {self.family!r}") from exc

    def _build_python_model(self) -> ExpertGSNHTree:
        stopping = StoppingCriteria(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
        )
        return ExpertGSNHTree(
            stopping_criteria=stopping,
            language=self._python_language(),
            **self.kwargs,
        )

    def _build_rust_model(self):
        try:
            from _rust_gsnh import RustGsnHClassifier
        except ImportError as exc:
            raise ImportError(
                "engine='rust' requires the optional _rust_gsnh extension. "
                "Install it with: maturin develop --manifest-path rust_gsnh/Cargo.toml --features python"
            ) from exc
        return RustGsnHClassifier(
            family=self.family,
            max_arity=self.max_arity,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
        )

    @staticmethod
    def _as_python_array(X):
        return np.asarray(X, dtype=np.float64)

    @staticmethod
    def _as_python_labels(y):
        return np.asarray(y, dtype=np.int32)

    @staticmethod
    def _as_rust_rows(X):
        return np.asarray(X, dtype=np.float64).tolist()

    @staticmethod
    def _as_rust_labels(y):
        return [int(v) for v in np.asarray(y, dtype=np.int32).tolist()]

    def fit(self, X, y):
        if self.engine == "python":
            self.model_ = self._build_python_model()
            self.model_.fit(self._as_python_array(X), self._as_python_labels(y))
        else:
            self.model_ = self._build_rust_model()
            self.model_.fit(self._as_rust_rows(X), self._as_rust_labels(y))
        self._is_fitted = True
        return self

    def _require_fitted(self):
        if not self._is_fitted or self.model_ is None:
            raise RuntimeError("GSNHEngineClassifier is not fitted")
        return self.model_

    def predict(self, X):
        model = self._require_fitted()
        if self.engine == "python":
            return model.predict(self._as_python_array(X))
        return model.predict(self._as_rust_rows(X))

    def score(self, X, y) -> float:
        model = self._require_fitted()
        if self.engine == "python":
            predictions = model.predict(self._as_python_array(X))
            labels = self._as_python_labels(y)
            if len(predictions) != len(labels):
                raise ValueError(f"X/y length mismatch: {len(predictions)} vs {len(labels)}")
            return float(np.mean(predictions == labels))
        return float(model.score(self._as_rust_rows(X), self._as_rust_labels(y)))

    def summary(self) -> Dict[str, int]:
        model = self._require_fitted()
        if self.engine == "python":
            n_nodes = int(getattr(model, "n_nodes_", 0))
            n_leaves = int(getattr(model, "n_leaves_", 0))
            return {
                "n_nodes": n_nodes,
                "n_leaves": n_leaves,
                "n_internal_nodes": n_nodes - n_leaves,
                "max_depth": int(getattr(model, "max_depth_reached_", 0)),
            }
        return model.summary()
