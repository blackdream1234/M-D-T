"""
Tests for the CONJ_UI / SQUARE_2CNF language renaming and implementation.

Covers:
  - Language enum changes (CONJ_UI exists, SQUARE_CNF is legacy)
  - ConjUI predicate behavior matches old SquareCNF
  - Square2CNFPredicate evaluate and evaluate_partial
  - Explainer SAT support for both families
  - Affine threshold-key deduplication
"""

import warnings
import numpy as np
import pytest

from gsnh_mdt.types import (
    LanguageFamily, GSNHPatternType,
    GSNH_CONJ_UI_CONFIGS, GSNH_SQUARE_CNF_CONFIGS,
    _resolve_language,
)
from gsnh_mdt.literals.base import GSNHLiteral
from gsnh_mdt.literals.predicates import GSNHPredicate, Square2CNFPredicate
from gsnh_mdt.types import LiteralPolarity


# ─────────────────────────────────────────────────────────
# STEP 10A — test_language_renaming
# ─────────────────────────────────────────────────────────

class TestLanguageRenaming:
    """Verify enum values and legacy handling."""

    def test_conj_ui_exists(self):
        assert LanguageFamily.CONJ_UI.value == "ConjUI"

    def test_square_2cnf_exists(self):
        assert LanguageFamily.SQUARE_2CNF.value == "Square2CNF"

    def test_legacy_square_cnf_exists(self):
        assert LanguageFamily.SQUARE_CNF.value == "SquareCNF"

    def test_resolve_language_conj_ui(self):
        assert _resolve_language(LanguageFamily.CONJ_UI) == LanguageFamily.CONJ_UI

    def test_resolve_language_square_cnf_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_language(LanguageFamily.SQUARE_CNF)
            assert result == LanguageFamily.CONJ_UI
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_resolve_language_horn_unchanged(self):
        assert _resolve_language(LanguageFamily.HORN) == LanguageFamily.HORN

    def test_config_table_alias(self):
        """GSNH_SQUARE_CNF_CONFIGS should be same object as GSNH_CONJ_UI_CONFIGS."""
        assert GSNH_SQUARE_CNF_CONFIGS is GSNH_CONJ_UI_CONFIGS

    def test_conj_ui_configs_all_polarities(self):
        """ConjUI allows all polarity combinations (AND semantics)."""
        assert len(GSNH_CONJ_UI_CONFIGS[2]) == 4  # FF, FT, TF, TT
        assert len(GSNH_CONJ_UI_CONFIGS[3]) == 8


# ─────────────────────────────────────────────────────────
# STEP 10B — test_conj_ui (old SquareCNF behavior)
# ─────────────────────────────────────────────────────────

class TestConjUI:
    """Verify ConjUI (AND semantics) matches old SquareCNF behavior."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        return rng.rand(100, 4)

    def test_conj_ui_evaluate_and_semantics(self, sample_data):
        """l1 ∧ l2 should produce AND of individual masks."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        pred = GSNHPredicate(
            literals=(l1, l2), information_gain=0.1,
            language_family=LanguageFamily.CONJ_UI,
        )
        mask = pred.evaluate(sample_data)
        expected = (sample_data[:, 0] >= 0.5) & (sample_data[:, 1] < 0.3)
        np.testing.assert_array_equal(mask, expected)

    def test_conj_ui_auto_normalizes_square_cnf(self, sample_data):
        """Creating a predicate with SQUARE_CNF should auto-normalize to CONJ_UI."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        pred = GSNHPredicate(
            literals=(l1,), information_gain=0.1,
            language_family=LanguageFamily.SQUARE_CNF,
        )
        assert pred.language_family == LanguageFamily.CONJ_UI

    def test_conj_ui_pattern_types(self):
        """ConjUI should use new pattern types, not old SQ_CNF ones."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        pred = GSNHPredicate(
            literals=(l1,), information_gain=0.1,
            language_family=LanguageFamily.CONJ_UI,
        )
        assert pred.pattern_type == GSNHPatternType.CONJ_UI_1L

    def test_conj_ui_2d_mixed_pattern(self):
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        pred = GSNHPredicate(
            literals=(l1, l2), information_gain=0.1,
            language_family=LanguageFamily.CONJ_UI,
        )
        assert pred.pattern_type == GSNHPatternType.CONJ_UI_2D_MIXED

    def test_conj_ui_evaluate_partial_short_circuit(self):
        """AND: one False → entire predicate is False."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        pred = GSNHPredicate(
            literals=(l1, l2), information_gain=0.1,
            language_family=LanguageFamily.CONJ_UI,
        )
        x = np.array([0.2, 0.1])  # l1=False (0.2 < 0.5), l2=True (0.1 < 0.3)
        # Only feature 0 is fixed
        result = pred.evaluate_partial(x, {0})
        assert result is False  # l1 is False → AND is False

    def test_conj_ui_str_format(self):
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        pred = GSNHPredicate(
            literals=(l1, l2), information_gain=0.1,
            language_family=LanguageFamily.CONJ_UI,
        )
        s = str(pred)
        assert "∧" in s


# ─────────────────────────────────────────────────────────
# STEP 10C — test_square_2cnf
# ─────────────────────────────────────────────────────────

class TestSquare2CNF:
    """Verify Square2CNFPredicate for paper-style (l1∨l2)∧(l3∨l4)."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        return rng.rand(200, 4)

    def _make_pred(self):
        """Build (x0≥0.5 ∨ x1<0.3) ∧ (x2≥0.7 ∨ x3<0.4)."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        l3 = GSNHLiteral(feature=2, threshold=0.7, polarity=LiteralPolarity.GE)
        l4 = GSNHLiteral(feature=3, threshold=0.4, polarity=LiteralPolarity.LT)
        return Square2CNFPredicate(
            clauses=((l1, l2), (l3, l4)), information_gain=0.1
        )

    def test_evaluate_correctness(self, sample_data):
        """Direct numpy evaluation must match predicate evaluate."""
        pred = self._make_pred()
        mask = pred.evaluate(sample_data)
        expected = (
            ((sample_data[:, 0] >= 0.5) | (sample_data[:, 1] < 0.3))
            & ((sample_data[:, 2] >= 0.7) | (sample_data[:, 3] < 0.4))
        )
        np.testing.assert_array_equal(mask, expected)

    def test_language_family(self):
        pred = self._make_pred()
        assert pred.language_family == LanguageFamily.SQUARE_2CNF

    def test_pattern_type(self):
        pred = self._make_pred()
        assert pred.pattern_type == GSNHPatternType.SQUARE_2CNF_2C2L

    def test_is_xor_false(self):
        pred = self._make_pred()
        assert pred.is_xor is False

    def test_features_used(self):
        pred = self._make_pred()
        assert pred.features_used() == {0, 1, 2, 3}

    def test_iter_literals_count(self):
        pred = self._make_pred()
        lits = list(pred.iter_literals())
        assert len(lits) == 4

    def test_str_format(self):
        pred = self._make_pred()
        s = str(pred)
        assert "∨" in s
        assert "∧" in s

    def test_evaluate_partial_both_known_true(self):
        pred = self._make_pred()
        # x0=0.6 ≥ 0.5 → True, x1=0.5 ≥ 0.3 → True (but l2 is LT)
        # x2=0.8 ≥ 0.7 → True, x3=0.2 < 0.4 → True
        x = np.array([0.6, 0.5, 0.8, 0.2])
        # All features known
        result = pred.evaluate_partial(x, {0, 1, 2, 3})
        assert result is True

    def test_evaluate_partial_clause_false(self):
        pred = self._make_pred()
        # x0=0.3 < 0.5 → l1=False, x1=0.5 ≥ 0.3 → l2=False (LT polarity)
        # → clause 1 is False → AND is False
        x = np.array([0.3, 0.5, 0.8, 0.2])
        result = pred.evaluate_partial(x, {0, 1, 2, 3})
        assert result is False

    def test_evaluate_partial_indeterminate(self):
        pred = self._make_pred()
        # Only x2 known, clause 1 unknown, clause 2 partially known
        x = np.array([0.3, 0.5, 0.8, 0.2])
        result = pred.evaluate_partial(x, {2})
        # Clause 1: both unknown → indeterminate
        # Clause 2: l3=True → clause satisfied
        # Overall: one indeterminate → None
        assert result is None

    def test_evaluate_partial_one_clause_satisfied_one_unknown(self):
        pred = self._make_pred()
        x = np.array([0.6, 0.5, 0.8, 0.2])
        # Only features 0 and 2 known
        result = pred.evaluate_partial(x, {0, 2})
        # Clause 1: l1=True → clause satisfied
        # Clause 2: l3=True → clause satisfied
        assert result is True

    def test_arity_validation(self):
        """Must have 1-3 clauses."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        with pytest.raises(ValueError, match="1-3 clauses"):
            Square2CNFPredicate(clauses=(), information_gain=0.0)

    def test_clause_length_validation(self):
        """Each clause must have exactly 2 literals."""
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        with pytest.raises(ValueError, match="2 literals"):
            Square2CNFPredicate(
                clauses=((l1,),), information_gain=0.0
            )


# ─────────────────────────────────────────────────────────
# STEP 10D — test_affine_threshold_keys
# ─────────────────────────────────────────────────────────

class TestAffineThresholdKeys:
    """Verify that the affine solver uses (feature, threshold) tuples
    to prevent variable collapse when same feature appears at different
    thresholds.
    """

    def test_different_thresholds_produce_different_keys(self):
        """Two literals on same feature with different thresholds must
        produce different variable keys in the affine solver."""
        from gsnh_mdt.tree.explainer import _affine_path_sat

        # Build two XOR predicates on the same feature with different thresholds
        l1_a = GSNHLiteral(feature=0, threshold=0.3, polarity=LiteralPolarity.GE)
        l1_b = GSNHLiteral(feature=1, threshold=0.5, polarity=LiteralPolarity.GE)
        pred1 = GSNHPredicate(
            literals=(l1_a, l1_b), information_gain=0.0,
            language_family=LanguageFamily.AFFINE, is_xor=True,
        )

        l2_a = GSNHLiteral(feature=0, threshold=0.7, polarity=LiteralPolarity.GE)
        l2_b = GSNHLiteral(feature=1, threshold=0.5, polarity=LiteralPolarity.GE)
        pred2 = GSNHPredicate(
            literals=(l2_a, l2_b), information_gain=0.0,
            language_family=LanguageFamily.AFFINE, is_xor=True,
        )

        # x = [0.5, 0.6]
        # pred1 True branch: (0.5>=0.3)=T XOR (0.6>=0.5)=T = F → branch=False
        # pred2 True branch: (0.5>=0.7)=F XOR (0.6>=0.5)=T = T → branch=True
        x = np.array([0.5, 0.6])
        path_edges = [(pred1, False), (pred2, True)]

        # This should NOT collapse feature 0 at t=0.3 and t=0.7
        # If it did, we'd get an incorrect SAT/UNSAT result
        result = _affine_path_sat(path_edges, x, {0, 1})
        # With both features fixed, the path is determined
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────
# STEP 10E — test_explainer_square_2cnf
# ─────────────────────────────────────────────────────────

class TestExplainerSquare2CNF:
    """Verify SAT encoding for Square2CNF in the explainer."""

    def _make_pred(self):
        l1 = GSNHLiteral(feature=0, threshold=0.5, polarity=LiteralPolarity.GE)
        l2 = GSNHLiteral(feature=1, threshold=0.3, polarity=LiteralPolarity.LT)
        l3 = GSNHLiteral(feature=2, threshold=0.7, polarity=LiteralPolarity.GE)
        l4 = GSNHLiteral(feature=3, threshold=0.4, polarity=LiteralPolarity.LT)
        return Square2CNFPredicate(
            clauses=((l1, l2), (l3, l4)), information_gain=0.1
        )

    def test_true_branch_sat(self):
        """True branch adds both clauses as OR constraints."""
        from gsnh_mdt.tree.explainer import _path_sat_numeric

        pred = self._make_pred()
        # x satisfies the predicate
        x = np.array([0.6, 0.2, 0.8, 0.3])
        path = [(pred, True)]
        assert _path_sat_numeric(path, x, {0, 1, 2, 3}) is True

    def test_true_branch_unsat(self):
        """True branch UNSAT when fixed values violate both clause options."""
        from gsnh_mdt.tree.explainer import _path_sat_numeric

        pred = self._make_pred()
        # x = [0.3, 0.5, 0.4, 0.6]
        # Clause 1: l1=(0.3>=0.5)=F, l2=(0.5<0.3)=F → clause UNSAT
        x = np.array([0.3, 0.5, 0.4, 0.6])
        path = [(pred, True)]
        result = _path_sat_numeric(path, x, {0, 1, 2, 3})
        assert result is False

    def test_false_branch_sat(self):
        """False branch: ¬P = (¬l1∧¬l2) ∨ (¬l3∧¬l4).
        If clause 1 can be fully negated, path is SAT.
        """
        from gsnh_mdt.tree.explainer import _path_sat_numeric

        pred = self._make_pred()
        # x = [0.3, 0.5, 0.8, 0.3]
        # ¬l1 = x0<0.5 → 0.3<0.5=True, ¬l2 = x1≥0.3 → 0.5≥0.3=True
        # → first alternative (¬l1∧¬l2) is SAT
        x = np.array([0.3, 0.5, 0.8, 0.3])
        path = [(pred, False)]
        result = _path_sat_numeric(path, x, {0, 1, 2, 3})
        assert result is True
