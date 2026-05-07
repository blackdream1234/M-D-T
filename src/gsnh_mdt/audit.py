"""
Audit utilities for GSNH-MDT trees.

Provides post-training verification of language consistency,
predicate semantics, and AXp soundness/minimality.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from gsnh_mdt.literals.predicates import GSNHPredicate, Square2CNFPredicate
from gsnh_mdt.types import LanguageFamily


@dataclass
class AuditReport:
    """Structured audit output."""
    mode: str = "unknown"
    requested_language: str = "unknown"
    n_internal_nodes: int = 0
    language_counts: dict = field(default_factory=dict)
    unsupported_literals: int = 0
    mismatched_predicates: list = field(default_factory=list)
    axp_weak_valid: int = 0
    axp_weak_total: int = 0
    axp_minimal: int = 0
    axp_minimal_total: int = 0
    is_journal_clean: bool = True

    def __str__(self):
        lines = [
            "LANGUAGE AUDIT",
            f"  Mode: {self.mode}",
            f"  Requested language: {self.requested_language}",
            f"  Internal nodes: {self.n_internal_nodes}",
        ]
        for lang, count in sorted(self.language_counts.items()):
            lines.append(f"  {lang} predicates: {count}")
        lines.append(f"  Unsupported literals: {self.unsupported_literals}")
        if self.mismatched_predicates:
            lines.append(f"  ⚠ MISMATCHED predicates: {len(self.mismatched_predicates)}")
            for m in self.mismatched_predicates[:5]:
                lines.append(f"    - {m}")
        if self.axp_weak_total > 0:
            lines.append(f"  AXp weak validity: {self.axp_weak_valid}/{self.axp_weak_total} "
                         f"{'PASS' if self.axp_weak_valid == self.axp_weak_total else 'FAIL'}")
        if self.axp_minimal_total > 0:
            lines.append(f"  AXp minimality: {self.axp_minimal}/{self.axp_minimal_total} "
                         f"{'PASS' if self.axp_minimal == self.axp_minimal_total else 'FAIL'}")
        if not self.is_journal_clean:
            lines.append("  ⚠ JOURNAL MODE VIOLATION: mixed families detected!")
        return "\n".join(lines)


def _collect_predicates(node, predicates=None):
    """Recursively collect all predicates from tree nodes."""
    if predicates is None:
        predicates = []
    if node is None:
        return predicates
    if node.get('is_leaf', True) or node.get('predicate') is None:
        return predicates
    predicates.append(node['predicate'])
    _collect_predicates(node.get('left'), predicates)
    _collect_predicates(node.get('right'), predicates)
    return predicates


def audit_tree_languages(tree):
    """Audit which language families are used in the tree.

    Returns an AuditReport with language counts and violation detection.
    """
    report = AuditReport()
    report.mode = getattr(tree, 'mode', 'unknown')
    report.requested_language = str(getattr(tree, 'language', 'unknown'))

    predicates = _collect_predicates(tree.root_)
    report.n_internal_nodes = len(predicates)

    counts = Counter()
    for pred in predicates:
        lang = pred.language_family.value if hasattr(pred.language_family, 'value') else str(pred.language_family)
        counts[lang] += 1
    report.language_counts = dict(counts)

    # Check journal-mode consistency
    if report.mode == 'journal':
        expected = getattr(tree, 'language', None)
        if expected and expected != LanguageFamily.BEST_PER_NODE:
            for pred in predicates:
                if pred.language_family != expected:
                    # CONJ_UI == old SQUARE_CNF is acceptable
                    if (expected in (LanguageFamily.CONJ_UI, LanguageFamily.SQUARE_CNF)
                            and pred.language_family in (LanguageFamily.CONJ_UI, LanguageFamily.SQUARE_CNF)):
                        continue
                    report.mismatched_predicates.append(
                        f"Expected {expected.value}, got {pred.language_family.value}: {pred}")
                    report.is_journal_clean = False

    return report


def audit_no_unencoded_literals(tree):
    """Check for unsupported literal types in tree predicates."""
    from gsnh_mdt.literals.compare import CompareLiteral
    from gsnh_mdt.literals.binary import GSNHBinaryLiteral

    predicates = _collect_predicates(tree.root_)
    unsupported = 0
    for pred in predicates:
        if isinstance(pred, Square2CNFPredicate):
            for lit in pred.iter_literals():
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    unsupported += 1
        elif hasattr(pred, 'literals'):
            for lit in pred.literals:
                if isinstance(lit, (CompareLiteral, GSNHBinaryLiteral)):
                    unsupported += 1
    return unsupported


def audit_predicate_semantics(tree):
    """Verify each predicate's family matches its construction.

    Returns list of (node_index, issue_description) tuples.
    """
    predicates = _collect_predicates(tree.root_)
    issues = []

    for i, pred in enumerate(predicates):
        if isinstance(pred, Square2CNFPredicate):
            if pred.language_family != LanguageFamily.SQUARE_2CNF:
                issues.append((i, f"Square2CNFPredicate has family={pred.language_family}"))
        elif isinstance(pred, GSNHPredicate):
            if pred.is_xor and pred.language_family != LanguageFamily.AFFINE:
                issues.append((i, f"XOR predicate has family={pred.language_family}"))
            # ConjUI: verify AND semantics would match
            if pred.language_family == LanguageFamily.CONJ_UI:
                n_pos = sum(1 for l in pred.literals if l.is_positive())
                # ConjUI allows all polarities — no violation possible
                pass
            elif pred.language_family == LanguageFamily.HORN:
                n_pos = sum(1 for l in pred.literals if l.is_positive())
                if n_pos > 1:
                    issues.append((i, f"Horn predicate has {n_pos} positive literals"))

    return issues


def audit_axp_minimality(tree, X, n_samples=100):
    """Verify AXp soundness and minimality on a sample of instances.

    For each sampled instance:
      1. Extract AXp S.
      2. Verify weak_axp_check(S) == True (soundness).
      3. For each f in S, verify weak_axp_check(S - {f}) == False (minimality).

    Returns AuditReport with results.
    """
    from gsnh_mdt.tree.explainer import extract_axp, weak_axp_check, _enumerate_paths
    from gsnh_mdt.tree.prediction import predict

    report = AuditReport()
    n = min(n_samples, X.shape[0])
    rng = np.random.RandomState(42)
    indices = rng.choice(X.shape[0], n, replace=False)

    paths = _enumerate_paths(tree)

    valid = 0
    minimal = 0
    total = 0

    for idx in indices:
        x = X[idx]
        y = predict(tree, x.reshape(1, -1))[0]
        S = extract_axp(tree, x)

        total += 1

        # Soundness check
        if weak_axp_check(tree, x, y, S, paths=paths):
            valid += 1
        else:
            continue

        # Minimality check
        is_minimal = True
        for f in list(S):
            S_minus = S - {f}
            if weak_axp_check(tree, x, y, S_minus, paths=paths):
                is_minimal = False
                break
        if is_minimal:
            minimal += 1

    report.axp_weak_valid = valid
    report.axp_weak_total = total
    report.axp_minimal = minimal
    report.axp_minimal_total = valid  # only check minimality on valid ones
    return report
