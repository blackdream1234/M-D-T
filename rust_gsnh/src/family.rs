//! Unified family facades for the incremental Rust GSNH engine.
//!
//! This module contains two conservative dispatch layers: one for evaluating
//! already-constructed fixed predicates, and one for searching exactly one
//! selected family. It does not recurse trees, bind to Python, compare families,
//! or make theorem-certification claims.

use crate::{
    best_affine_split_with_min_leaf, best_antihorn_split_with_min_leaf,
    best_conjui_split_with_min_leaf, best_horn_split_with_min_leaf,
    best_square2cnf_split_with_min_leaf, evaluate_affine_candidate_with_min_leaf,
    evaluate_antihorn_candidate_with_min_leaf, evaluate_composed_candidate_with_min_leaf,
    evaluate_horn_candidate_with_min_leaf, evaluate_square2cnf_candidate_with_min_leaf,
    ComposedPredicate, Dataset, EvaluatedComposedPredicate, EvaluatedSquare2CNFPredicate,
    Square2CNFPredicate,
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

/// Unified best-split result returned by one selected family search.
#[derive(Clone, Debug, PartialEq)]
pub enum BestFamilySplit {
    /// Search result for ConjUI, Horn, AntiHorn, and Affine/XOR.
    Composed(EvaluatedComposedPredicate),
    /// Search result for Square2CNF clause predicates.
    Square2CNF(EvaluatedSquare2CNFPredicate),
}

/// Configuration for searching exactly one language family.
///
/// For ConjUI, Horn, AntiHorn, and Affine, `max_arity` means maximum number of
/// threshold literals. For Square2CNF, `max_arity` means maximum number of
/// two-literal OR clauses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilySearchConfig {
    pub family: LanguageFamily,
    pub max_arity: usize,
    pub min_samples_leaf: usize,
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

/// Search exactly one selected family and return its best valid split.
///
/// This facade deliberately dispatches to one family-specific search function
/// and does not compare across families. It does not implement Python `Any`,
/// `BestPerNode`, legacy `SquareCNF`, tree recursion, theorem certificates,
/// benchmark integration, or bindings.
pub fn best_family_split(
    dataset: &Dataset,
    config: FamilySearchConfig,
) -> Result<Option<BestFamilySplit>, String> {
    match config.family {
        LanguageFamily::ConjUI => Ok(best_conjui_split_with_min_leaf(
            dataset,
            config.max_arity,
            config.min_samples_leaf,
        )?
        .map(BestFamilySplit::Composed)),
        LanguageFamily::Horn => {
            Ok(
                best_horn_split_with_min_leaf(dataset, config.max_arity, config.min_samples_leaf)?
                    .map(BestFamilySplit::Composed),
            )
        }
        LanguageFamily::AntiHorn => Ok(best_antihorn_split_with_min_leaf(
            dataset,
            config.max_arity,
            config.min_samples_leaf,
        )?
        .map(BestFamilySplit::Composed)),
        LanguageFamily::Affine => Ok(best_affine_split_with_min_leaf(
            dataset,
            config.max_arity,
            config.min_samples_leaf,
        )?
        .map(BestFamilySplit::Composed)),
        LanguageFamily::Square2CNF => Ok(best_square2cnf_split_with_min_leaf(
            dataset,
            config.max_arity,
            config.min_samples_leaf,
        )?
        .map(BestFamilySplit::Square2CNF)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        best_affine_split_with_min_leaf, best_antihorn_split_with_min_leaf,
        best_conjui_split_with_min_leaf, best_horn_split_with_min_leaf,
        best_square2cnf_split_with_min_leaf, ComparisonOp, MaskOp, Square2Clause,
        ThresholdPredicate,
    };

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

    fn tie_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![1.0, 1.0],
            ],
            vec![0, 0, 1, 1],
        )
        .unwrap()
    }

    fn horn_search_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
                vec![2.0, 0.0],
                vec![2.0, 1.0],
            ],
            vec![1, 0, 1, 1, 0, 1],
        )
        .unwrap()
    }

    fn antihorn_search_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
                vec![2.0, 0.0],
                vec![2.0, 1.0],
            ],
            vec![1, 1, 1, 0, 1, 1],
        )
        .unwrap()
    }

    fn xor_search_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![2.0, 0.0],
                vec![2.0, 1.0],
            ],
            vec![0, 1, 1, 0, 1, 0],
        )
        .unwrap()
    }

    fn square_search_dataset() -> Dataset {
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for a in 0..=1 {
            for b in 0..=1 {
                for c in 0..=1 {
                    for d in 0..=1 {
                        rows.push(vec![a as f64, b as f64, c as f64, d as f64]);
                        labels.push(((a == 1 || b == 1) && (c == 1 || d == 1)) as u8);
                    }
                }
            }
        }
        Dataset::from_rows(rows, labels).unwrap()
    }

    fn no_gain_dataset() -> Dataset {
        Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 0, 0, 0],
        )
        .unwrap()
    }

    fn assert_composed_score_matches(
        family: LanguageFamily,
        facade: Option<BestFamilySplit>,
        direct: Option<EvaluatedComposedPredicate>,
    ) {
        let facade = facade.unwrap_or_else(|| panic!("{family:?} facade returned None"));
        let direct = direct.unwrap_or_else(|| panic!("{family:?} direct search returned None"));
        let BestFamilySplit::Composed(facade) = facade else {
            panic!("{family:?} should return a composed split");
        };
        assert_eq!(facade, direct);
        assert_eq!(facade.candidate.score, direct.candidate.score);
    }

    #[test]
    fn best_family_split_dispatches_conjui_and_matches_direct_search() {
        let ds = tie_dataset();
        let config = FamilySearchConfig {
            family: LanguageFamily::ConjUI,
            max_arity: 2,
            min_samples_leaf: 1,
        };
        assert_composed_score_matches(
            LanguageFamily::ConjUI,
            best_family_split(&ds, config).unwrap(),
            best_conjui_split_with_min_leaf(&ds, 2, 1).unwrap(),
        );
    }

    #[test]
    fn best_family_split_dispatches_horn_and_matches_direct_search() {
        let ds = horn_search_dataset();
        let config = FamilySearchConfig {
            family: LanguageFamily::Horn,
            max_arity: 2,
            min_samples_leaf: 1,
        };
        assert_composed_score_matches(
            LanguageFamily::Horn,
            best_family_split(&ds, config).unwrap(),
            best_horn_split_with_min_leaf(&ds, 2, 1).unwrap(),
        );
    }

    #[test]
    fn best_family_split_dispatches_antihorn_and_matches_direct_search() {
        let ds = antihorn_search_dataset();
        let config = FamilySearchConfig {
            family: LanguageFamily::AntiHorn,
            max_arity: 2,
            min_samples_leaf: 1,
        };
        assert_composed_score_matches(
            LanguageFamily::AntiHorn,
            best_family_split(&ds, config).unwrap(),
            best_antihorn_split_with_min_leaf(&ds, 2, 1).unwrap(),
        );
    }

    #[test]
    fn best_family_split_dispatches_affine_and_matches_direct_search() {
        let ds = xor_search_dataset();
        let config = FamilySearchConfig {
            family: LanguageFamily::Affine,
            max_arity: 2,
            min_samples_leaf: 1,
        };
        assert_composed_score_matches(
            LanguageFamily::Affine,
            best_family_split(&ds, config).unwrap(),
            best_affine_split_with_min_leaf(&ds, 2, 1).unwrap(),
        );
    }

    #[test]
    fn best_family_split_dispatches_square2cnf_and_matches_direct_search() {
        let ds = square_search_dataset();
        let config = FamilySearchConfig {
            family: LanguageFamily::Square2CNF,
            max_arity: 2,
            min_samples_leaf: 1,
        };
        let facade = best_family_split(&ds, config).unwrap().unwrap();
        let direct = best_square2cnf_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .unwrap();
        let BestFamilySplit::Square2CNF(facade) = facade else {
            panic!("Square2CNF should return a Square2CNF split");
        };
        assert_eq!(facade, direct);
        assert_eq!(facade.candidate.score, direct.candidate.score);
    }

    #[test]
    fn best_family_split_propagates_none_and_errors() {
        let ds = no_gain_dataset();
        let config = FamilySearchConfig {
            family: LanguageFamily::ConjUI,
            max_arity: 1,
            min_samples_leaf: 1,
        };
        assert!(best_family_split(&ds, config).unwrap().is_none());

        let strict_config = FamilySearchConfig {
            family: LanguageFamily::ConjUI,
            max_arity: 1,
            min_samples_leaf: 3,
        };
        assert!(best_family_split(&tie_dataset(), strict_config)
            .unwrap()
            .is_none());

        let invalid_config = FamilySearchConfig {
            family: LanguageFamily::Square2CNF,
            max_arity: 0,
            min_samples_leaf: 1,
        };
        assert!(best_family_split(&square_search_dataset(), invalid_config).is_err());
    }
}
