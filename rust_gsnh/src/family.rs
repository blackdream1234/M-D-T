//! Unified fixed-predicate family facade for the incremental Rust GSNH engine.
//!
//! This module dispatches already-constructed predicates to the fixed-family
//! evaluators.  It does not enumerate candidates, recurse trees, bind to
//! Python, or make theorem-certification claims.

use crate::{
    evaluate_affine_candidate_with_min_leaf, evaluate_antihorn_candidate_with_min_leaf,
    evaluate_composed_candidate_with_min_leaf, evaluate_horn_candidate_with_min_leaf,
    evaluate_square2cnf_candidate_with_min_leaf, ComposedPredicate, Dataset,
    EvaluatedComposedPredicate, EvaluatedSquare2CNFPredicate, Square2CNFPredicate,
};

/// Rust subset of Python `LanguageFamily` supported by the fixed-predicate facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageFamily {
    Horn,
    AntiHorn,
    ConjUI,
    Affine,
    Square2CNF,
}

/// Fixed predicate shape accepted by the facade.
#[derive(Clone, Debug, PartialEq)]
pub enum FixedPredicate {
    /// Generic Python `GSNHPredicate`-style mask composition for Horn,
    /// AntiHorn, ConjUI, and Affine/XOR.
    Composed(ComposedPredicate),
    /// Python `Square2CNFPredicate`-style conjunction of 2-literal OR clauses.
    Square2CNF(Square2CNFPredicate),
}

/// Family-specific evaluated fixed predicate result.
#[derive(Clone, Debug, PartialEq)]
pub enum EvaluatedFixedPredicate {
    Composed(EvaluatedComposedPredicate),
    Square2CNF(EvaluatedSquare2CNFPredicate),
}

/// Evaluate one already-constructed fixed predicate using the requested family.
///
/// Dispatch is intentionally conservative:
/// - Horn/AntiHorn/ConjUI/Affine accept only `FixedPredicate::Composed`.
/// - Square2CNF accepts only `FixedPredicate::Square2CNF`.
/// - Wrong mask operators, invalid polarity, invalid features, and empty
///   predicates are rejected by the underlying family evaluators.
pub fn evaluate_fixed_predicate_with_min_leaf(
    dataset: &Dataset,
    family: LanguageFamily,
    predicate: FixedPredicate,
    min_samples_leaf: usize,
    arity: usize,
) -> Result<Option<EvaluatedFixedPredicate>, String> {
    match (family, predicate) {
        (LanguageFamily::ConjUI, FixedPredicate::Composed(predicate)) => Ok(
            evaluate_composed_candidate_with_min_leaf(dataset, predicate, min_samples_leaf, arity)?
                .map(EvaluatedFixedPredicate::Composed),
        ),
        (LanguageFamily::Affine, FixedPredicate::Composed(predicate)) => Ok(
            evaluate_affine_candidate_with_min_leaf(dataset, predicate, min_samples_leaf, arity)?
                .map(EvaluatedFixedPredicate::Composed),
        ),
        (LanguageFamily::Horn, FixedPredicate::Composed(predicate)) => Ok(
            evaluate_horn_candidate_with_min_leaf(dataset, predicate, min_samples_leaf, arity)?
                .map(EvaluatedFixedPredicate::Composed),
        ),
        (LanguageFamily::AntiHorn, FixedPredicate::Composed(predicate)) => Ok(
            evaluate_antihorn_candidate_with_min_leaf(dataset, predicate, min_samples_leaf, arity)?
                .map(EvaluatedFixedPredicate::Composed),
        ),
        (LanguageFamily::Square2CNF, FixedPredicate::Square2CNF(predicate)) => {
            Ok(evaluate_square2cnf_candidate_with_min_leaf(
                dataset,
                predicate,
                min_samples_leaf,
                arity,
            )?
            .map(EvaluatedFixedPredicate::Square2CNF))
        }
        (LanguageFamily::Square2CNF, FixedPredicate::Composed(_)) => {
            Err("Square2CNF family requires FixedPredicate::Square2CNF".to_string())
        }
        (family, FixedPredicate::Square2CNF(_)) => Err(format!(
            "{family:?} family requires FixedPredicate::Composed, not Square2CNF"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComparisonOp, MaskOp, Square2Clause, ThresholdPredicate};

    fn dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![2.0, 2.0, 2.0, 2.0],
                vec![3.0, 3.0, 3.0, 3.0],
                vec![4.0, 4.0, 4.0, 4.0],
                vec![5.0, 5.0, 5.0, 5.0],
            ],
            vec![1, 0, 0, 0, 1, 1],
        )
        .unwrap()
    }

    fn ge(feature_index: usize, threshold: f64) -> ThresholdPredicate {
        ThresholdPredicate {
            feature_index,
            threshold,
            op: ComparisonOp::GreaterEqual,
        }
    }

    fn lt(feature_index: usize, threshold: f64) -> ThresholdPredicate {
        ThresholdPredicate {
            feature_index,
            threshold,
            op: ComparisonOp::LessThan,
        }
    }

    fn false_lit() -> ThresholdPredicate {
        lt(2, -1.0)
    }

    fn conjui_predicate() -> ComposedPredicate {
        ComposedPredicate {
            op: MaskOp::And,
            literals: vec![ge(0, 1.0), lt(1, 4.0)],
        }
    }

    fn horn_antihorn_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 1.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ],
            vec![0, 1, 1, 1],
        )
        .unwrap()
    }

    fn horn_predicate() -> ComposedPredicate {
        ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![ge(0, 0.5), lt(1, 0.5)],
        }
    }

    fn antihorn_predicate() -> ComposedPredicate {
        ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![ge(0, 0.5), ge(1, 0.5)],
        }
    }

    fn square2cnf_predicate() -> Square2CNFPredicate {
        Square2CNFPredicate {
            clauses: vec![
                Square2Clause {
                    left: ge(0, 1.0),
                    right: false_lit(),
                },
                Square2Clause {
                    left: lt(1, 4.0),
                    right: false_lit(),
                },
            ],
        }
    }

    #[test]
    fn facade_dispatches_conjui_composed_predicate() {
        let ds = dataset();
        let evaluated = evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::ConjUI,
            FixedPredicate::Composed(conjui_predicate()),
            1,
            2,
        )
        .unwrap()
        .unwrap();
        let EvaluatedFixedPredicate::Composed(evaluated) = evaluated else {
            panic!("expected composed result");
        };
        assert_eq!(evaluated.inside_mask.indices(), vec![1, 2, 3]);
    }

    #[test]
    fn facade_dispatches_affine_xor_composed_predicate() {
        let ds = Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0, 1, 1, 0],
        )
        .unwrap();
        let pred = ComposedPredicate {
            op: MaskOp::Xor,
            literals: vec![ge(0, 0.5), ge(1, 0.5)],
        };
        let evaluated = evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::Affine,
            FixedPredicate::Composed(pred),
            1,
            2,
        )
        .unwrap()
        .unwrap();
        let EvaluatedFixedPredicate::Composed(evaluated) = evaluated else {
            panic!("expected composed result");
        };
        assert_eq!(evaluated.inside_mask.indices(), vec![1, 2]);
    }

    #[test]
    fn facade_dispatches_horn_and_antihorn_or_predicates() {
        let horn_ds = horn_antihorn_dataset();
        let horn = evaluate_fixed_predicate_with_min_leaf(
            &horn_ds,
            LanguageFamily::Horn,
            FixedPredicate::Composed(horn_predicate()),
            1,
            2,
        )
        .unwrap()
        .unwrap();

        let antihorn_ds = Dataset::from_rows(
            vec![
                vec![0.0, 1.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ],
            vec![1, 1, 0, 1],
        )
        .unwrap();
        let antihorn = evaluate_fixed_predicate_with_min_leaf(
            &antihorn_ds,
            LanguageFamily::AntiHorn,
            FixedPredicate::Composed(antihorn_predicate()),
            1,
            2,
        )
        .unwrap()
        .unwrap();
        let EvaluatedFixedPredicate::Composed(horn) = horn else {
            panic!("expected composed horn result");
        };
        let EvaluatedFixedPredicate::Composed(antihorn) = antihorn else {
            panic!("expected composed antihorn result");
        };
        assert_eq!(horn.inside_mask.indices(), vec![1, 2, 3]);
        assert_eq!(antihorn.inside_mask.indices(), vec![0, 1, 3]);
    }

    #[test]
    fn facade_dispatches_square2cnf_predicate() {
        let ds = dataset();
        let evaluated = evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::Square2CNF,
            FixedPredicate::Square2CNF(square2cnf_predicate()),
            1,
            2,
        )
        .unwrap()
        .unwrap();
        let EvaluatedFixedPredicate::Square2CNF(evaluated) = evaluated else {
            panic!("expected Square2CNF result");
        };
        assert_eq!(evaluated.inside_mask.indices(), vec![1, 2, 3]);
    }

    #[test]
    fn square2cnf_family_rejects_composed_shape() {
        let ds = dataset();
        let err = evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::Square2CNF,
            FixedPredicate::Composed(conjui_predicate()),
            1,
            2,
        )
        .unwrap_err();
        assert!(err.contains("Square2CNF family requires"));
    }

    #[test]
    fn composed_families_reject_square2cnf_shape() {
        let ds = dataset();
        for family in [
            LanguageFamily::ConjUI,
            LanguageFamily::Affine,
            LanguageFamily::Horn,
            LanguageFamily::AntiHorn,
        ] {
            let err = evaluate_fixed_predicate_with_min_leaf(
                &ds,
                family,
                FixedPredicate::Square2CNF(square2cnf_predicate()),
                1,
                2,
            )
            .unwrap_err();
            assert!(err.contains("requires FixedPredicate::Composed"));
        }
    }

    #[test]
    fn wrong_mask_operator_rejection_is_preserved() {
        let ds = dataset();
        let err = evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::ConjUI,
            FixedPredicate::Composed(horn_predicate()),
            1,
            2,
        )
        .unwrap_err();
        assert!(err.contains("ConjUI"));
    }

    #[test]
    fn horn_and_antihorn_polarity_rejections_are_preserved() {
        let ds = dataset();
        let invalid_horn = ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![ge(0, 1.0), ge(1, 1.0)],
        };
        let invalid_antihorn = ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![lt(0, 4.0), lt(1, 4.0)],
        };
        assert!(evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::Horn,
            FixedPredicate::Composed(invalid_horn),
            1,
            2,
        )
        .unwrap_err()
        .contains("Horn polarity violation"));
        assert!(evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::AntiHorn,
            FixedPredicate::Composed(invalid_antihorn),
            1,
            2,
        )
        .unwrap_err()
        .contains("AntiHorn polarity violation"));
    }

    #[test]
    fn min_leaf_rejection_returns_none_through_facade() {
        let ds = dataset();
        assert!(evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::ConjUI,
            FixedPredicate::Composed(conjui_predicate()),
            4,
            2,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn nonpositive_gain_returns_none_through_facade() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 1, 0, 1],
        )
        .unwrap();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: vec![lt(0, 2.0)],
        };
        assert!(evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::ConjUI,
            FixedPredicate::Composed(pred),
            1,
            1,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn invalid_feature_errors_propagate_through_facade() {
        let ds = dataset();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: vec![ge(99, 1.0)],
        };
        assert!(evaluate_fixed_predicate_with_min_leaf(
            &ds,
            LanguageFamily::ConjUI,
            FixedPredicate::Composed(pred),
            1,
            1,
        )
        .is_err());
    }
}
