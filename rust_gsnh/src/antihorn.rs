//! Deterministic AntiHorn candidate enumeration for the incremental Rust engine.
//!
//! This module searches only supplied-threshold AntiHorn OR clauses up to arity
//! 2. It is intentionally not full tree search and does not enumerate Affine,
//! Square2CNF, Horn, or ConjUI candidates.

use crate::{
    evaluate_antihorn_candidate_with_min_leaf, generate_1d_thresholds, ComparisonOp,
    ComposedPredicate, Dataset, EvaluatedComposedPredicate, MaskOp, ThresholdPredicate,
};
use std::cmp::Ordering;

/// Return the best deterministic AntiHorn split up to `max_arity`.
///
/// Supported arities are 1 and 2. `max_arity = 1` enumerates one-literal OR
/// predicates, which are semantically equivalent to 1D threshold splits.
/// `max_arity = 2` additionally enumerates canonical feature pairs `(i, j)`
/// with `i < j`, midpoint thresholds, and Python's three AntiHorn polarity
/// configurations in deterministic order: GE/GE, LT/GE, GE/LT. The LT/LT case
/// is not generated because AntiHorn permits at most one negative literal.
///
/// Candidates are evaluated by the fixed AntiHorn evaluator, so polarity
/// validation, branch-size filtering, information gain, and BIC-style penalized
/// gain are shared with the fixed-family parity layer. If no valid positive
/// candidate exists, returns `Ok(None)`.
pub fn best_antihorn_split_with_min_leaf(
    dataset: &Dataset,
    max_arity: usize,
    min_samples_leaf: usize,
) -> Result<Option<EvaluatedComposedPredicate>, String> {
    if max_arity == 0 {
        return Err("AntiHorn search requires max_arity >= 1".to_string());
    }
    if max_arity > 2 {
        return Err("Rust AntiHorn search currently supports max_arity <= 2".to_string());
    }

    let mut best: Option<EvaluatedComposedPredicate> = None;

    for feature_index in 0..dataset.n_features() {
        for threshold in generate_1d_thresholds(dataset, feature_index)? {
            for op in [ComparisonOp::LessThan, ComparisonOp::GreaterEqual] {
                let predicate = ComposedPredicate {
                    literals: vec![ThresholdPredicate {
                        feature_index,
                        threshold,
                        op,
                    }],
                    op: MaskOp::Or,
                };
                consider_candidate(dataset, predicate, min_samples_leaf, &mut best)?;
            }
        }
    }

    if max_arity >= 2 {
        for left_feature in 0..dataset.n_features() {
            let left_thresholds = generate_1d_thresholds(dataset, left_feature)?;
            if left_thresholds.is_empty() {
                continue;
            }
            for right_feature in (left_feature + 1)..dataset.n_features() {
                let right_thresholds = generate_1d_thresholds(dataset, right_feature)?;
                if right_thresholds.is_empty() {
                    continue;
                }
                for &left_threshold in &left_thresholds {
                    for &right_threshold in &right_thresholds {
                        for (left_op, right_op) in [
                            (ComparisonOp::GreaterEqual, ComparisonOp::GreaterEqual),
                            (ComparisonOp::LessThan, ComparisonOp::GreaterEqual),
                            (ComparisonOp::GreaterEqual, ComparisonOp::LessThan),
                        ] {
                            let predicate = ComposedPredicate {
                                literals: vec![
                                    ThresholdPredicate {
                                        feature_index: left_feature,
                                        threshold: left_threshold,
                                        op: left_op,
                                    },
                                    ThresholdPredicate {
                                        feature_index: right_feature,
                                        threshold: right_threshold,
                                        op: right_op,
                                    },
                                ],
                                op: MaskOp::Or,
                            };
                            consider_candidate(dataset, predicate, min_samples_leaf, &mut best)?;
                        }
                    }
                }
            }
        }
    }

    Ok(best)
}

fn consider_candidate(
    dataset: &Dataset,
    predicate: ComposedPredicate,
    min_samples_leaf: usize,
    best: &mut Option<EvaluatedComposedPredicate>,
) -> Result<(), String> {
    let arity = predicate.literals.len();
    if let Some(candidate) =
        evaluate_antihorn_candidate_with_min_leaf(dataset, predicate, min_samples_leaf, arity)?
    {
        if should_replace(best.as_ref(), &candidate) {
            *best = Some(candidate);
        }
    }
    Ok(())
}

fn should_replace(
    current: Option<&EvaluatedComposedPredicate>,
    candidate: &EvaluatedComposedPredicate,
) -> bool {
    let Some(current) = current else {
        return true;
    };
    const EPS: f64 = 1e-12;
    let candidate_score = candidate.candidate.score;
    let current_score = current.candidate.score;
    if candidate_score > current_score + EPS {
        return true;
    }
    if (candidate_score - current_score).abs() > EPS {
        return false;
    }

    let candidate_lits = &candidate.candidate.predicate.literals;
    let current_lits = &current.candidate.predicate.literals;
    if candidate_lits.len() != current_lits.len() {
        return candidate_lits.len() < current_lits.len();
    }

    compare_literal_sequences(candidate_lits, current_lits) == Ordering::Less
}

fn compare_literal_sequences(
    left: &[ThresholdPredicate],
    right: &[ThresholdPredicate],
) -> Ordering {
    for (a, b) in left.iter().zip(right.iter()) {
        match a.feature_index.cmp(&b.feature_index) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
        match a.threshold.total_cmp(&b.threshold) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
        match op_rank(a.op).cmp(&op_rank(b.op)) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
    }
    left.len().cmp(&right.len())
}

fn op_rank(op: ComparisonOp) -> usize {
    match op {
        ComparisonOp::LessThan => 0,
        ComparisonOp::GreaterEqual => 1,
        ComparisonOp::LessEqual => 2,
        ComparisonOp::GreaterThan => 3,
        ComparisonOp::Equal => 4,
    }
}

#[cfg(test)]
fn is_negative(op: ComparisonOp) -> bool {
    matches!(op, ComparisonOp::LessThan | ComparisonOp::LessEqual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{best_1d_split_with_min_leaf, entropy, penalized_gain, ClassCounts};

    const EPS: f64 = 1e-12;

    fn antihorn_dataset() -> Dataset {
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

    fn no_gain_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![2.0, 2.0],
                vec![3.0, 3.0],
            ],
            vec![0, 1, 0, 1],
        )
        .unwrap()
    }

    #[test]
    fn max_arity_one_matches_existing_1d_best_split() {
        let ds = tie_dataset();
        let one_d = best_1d_split_with_min_leaf(&ds, 1).unwrap().unwrap();
        let antihorn = best_antihorn_split_with_min_leaf(&ds, 1, 1)
            .unwrap()
            .unwrap();
        let lit = antihorn.candidate.predicate.literals[0];

        assert_eq!(antihorn.candidate.predicate.literals.len(), 1);
        assert_eq!(lit.feature_index, one_d.candidate.feature_index);
        assert_eq!(lit.threshold, one_d.candidate.threshold);
        assert_eq!(lit.op, one_d.candidate.op);
        assert!((antihorn.candidate.score - one_d.candidate.score).abs() < EPS);
        assert_eq!(antihorn.inside_mask.indices(), one_d.inside_mask.indices());
        assert_eq!(
            antihorn.outside_mask.indices(),
            one_d.outside_mask.indices()
        );
    }

    #[test]
    fn max_arity_two_finds_known_two_literal_antihorn_or_split() {
        let ds = antihorn_dataset();
        let best = best_antihorn_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .unwrap();
        let lits = &best.candidate.predicate.literals;

        assert_eq!(lits.len(), 2);
        assert_eq!(lits[0].feature_index, 0);
        assert_eq!(lits[0].threshold, 0.5);
        assert_eq!(lits[0].op, ComparisonOp::GreaterEqual);
        assert_eq!(lits[1].feature_index, 1);
        assert_eq!(lits[1].threshold, 0.5);
        assert_eq!(lits[1].op, ComparisonOp::LessThan);
        assert_eq!(best.inside_mask.indices(), vec![0, 1, 2, 4, 5]);
        assert_eq!(best.outside_mask.indices(), vec![3]);
        assert_eq!(
            best.candidate.inside_counts,
            ClassCounts {
                positive: 5,
                negative: 0
            }
        );
        assert_eq!(
            best.candidate.outside_counts,
            ClassCounts {
                positive: 0,
                negative: 1
            }
        );
        let expected_raw = entropy(ClassCounts {
            positive: 5,
            negative: 1,
        });
        assert!(
            (best.candidate.score - penalized_gain(expected_raw, 2, ds.n_samples())).abs() < EPS
        );
    }

    #[test]
    fn antihorn_search_accepts_one_negative_and_all_positive_candidates() {
        let ds = antihorn_dataset();
        let best = best_antihorn_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .unwrap();
        let negative_count = best
            .candidate
            .predicate
            .literals
            .iter()
            .filter(|literal| is_negative(literal.op))
            .count();
        assert!(negative_count <= 1);

        let all_positive_ds = Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0, 1, 1, 1],
        )
        .unwrap();
        let all_positive = ComposedPredicate {
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 0.5,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 0.5,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
            op: MaskOp::Or,
        };
        assert!(
            evaluate_antihorn_candidate_with_min_leaf(&all_positive_ds, all_positive, 1, 2)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn two_negative_literal_tuple_is_not_selected() {
        let ds = Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![1, 1, 1, 0],
        )
        .unwrap();
        if let Some(best) = best_antihorn_split_with_min_leaf(&ds, 2, 1).unwrap() {
            let negatives = best
                .candidate
                .predicate
                .literals
                .iter()
                .filter(|literal| is_negative(literal.op))
                .count();
            assert!(negatives <= 1);
            assert_ne!(best.inside_mask.indices(), vec![0, 1, 2]);
        }
    }

    #[test]
    fn min_samples_leaf_rejects_too_small_branches() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 0, 0, 1],
        )
        .unwrap();
        assert!(best_antihorn_split_with_min_leaf(&ds, 1, 2)
            .unwrap()
            .is_none());
    }

    #[test]
    fn min_samples_leaf_zero_allows_branch_size_filter_to_be_disabled() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 0, 0, 1],
        )
        .unwrap();
        assert!(best_antihorn_split_with_min_leaf(&ds, 1, 2)
            .unwrap()
            .is_none());
        let best = best_antihorn_split_with_min_leaf(&ds, 1, 0)
            .unwrap()
            .unwrap();
        assert!(best.inside_mask.count_ones() < 2 || best.outside_mask.count_ones() < 2);
    }

    #[test]
    fn no_positive_candidate_returns_none() {
        let ds = no_gain_dataset();
        assert!(best_antihorn_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .is_none());
    }

    #[test]
    fn constant_features_are_harmless() {
        let ds = Dataset::from_rows(
            vec![
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![1.0, 2.0],
                vec![1.0, 3.0],
            ],
            vec![0, 0, 1, 1],
        )
        .unwrap();
        let best = best_antihorn_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .unwrap();
        assert_eq!(best.candidate.predicate.literals.len(), 1);
        assert_eq!(best.candidate.predicate.literals[0].feature_index, 1);
    }

    #[test]
    fn deterministic_tie_breaking_prefers_smaller_feature_sequence() {
        let ds = tie_dataset();
        let best = best_antihorn_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .unwrap();
        let lits = &best.candidate.predicate.literals;
        assert_eq!(lits.len(), 1);
        assert_eq!(lits[0].feature_index, 0);
        assert_eq!(lits[0].threshold, 0.5);
        assert_eq!(lits[0].op, ComparisonOp::LessThan);
    }

    #[test]
    fn duplicate_feature_pairs_are_not_generated() {
        let ds = antihorn_dataset();
        let best = best_antihorn_split_with_min_leaf(&ds, 2, 1)
            .unwrap()
            .unwrap();
        let lits = &best.candidate.predicate.literals;
        if lits.len() == 2 {
            assert!(lits[0].feature_index < lits[1].feature_index);
        }
    }

    #[test]
    fn invalid_max_arity_is_rejected() {
        let ds = antihorn_dataset();
        assert!(best_antihorn_split_with_min_leaf(&ds, 0, 1)
            .unwrap_err()
            .contains("max_arity >= 1"));
        assert!(best_antihorn_split_with_min_leaf(&ds, 3, 1)
            .unwrap_err()
            .contains("max_arity <= 2"));
    }

    #[test]
    fn nan_threshold_generation_error_propagates() {
        let ds =
            Dataset::from_rows(vec![vec![0.0], vec![f64::NAN], vec![1.0]], vec![0, 1, 1]).unwrap();
        assert!(best_antihorn_split_with_min_leaf(&ds, 1, 1)
            .unwrap_err()
            .contains("NaN"));
    }
}
