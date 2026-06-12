//! Deterministic 1D threshold candidate generation.
//!
//! This is the smallest safe search milestone: single-feature scalar threshold
//! predicates scored with the Rust formulas that mirror Python.  It is not full
//! GSNH tree search and does not implement Horn/AntiHorn/SquareCNF/Affine
//! composition.

use crate::{
    class_counts, information_gain, negative_label_mask, penalized_gain, positive_label_mask,
    BitSet, ClassCounts, ComparisonOp, Dataset, ThresholdPredicate,
};

/// A scored 1D threshold split candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitCandidate {
    pub feature_index: usize,
    pub threshold: f64,
    pub op: ComparisonOp,
    pub score: f64,
    pub inside_counts: ClassCounts,
    pub outside_counts: ClassCounts,
}

/// Best split plus inside/outside row masks.
#[derive(Clone, Debug, PartialEq)]
pub struct BestSplit {
    pub candidate: SplitCandidate,
    pub inside_mask: BitSet,
    pub outside_mask: BitSet,
}

/// Generate deterministic 1D threshold candidates for a feature.
///
/// Matches the exact-value Python convention for low-cardinality node-local
/// features: sorted unique values produce midpoints between adjacent unique
/// values. Constant features produce no thresholds.
pub fn generate_1d_thresholds(dataset: &Dataset, feature_index: usize) -> Result<Vec<f64>, String> {
    if feature_index >= dataset.n_features() {
        return Err(format!(
            "feature index {} out of range for dataset with {} features",
            feature_index,
            dataset.n_features()
        ));
    }

    let mut values = dataset.column(feature_index);
    if values.iter().any(|value| value.is_nan()) {
        return Err("cannot generate deterministic thresholds for NaN feature values".to_string());
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| *a == *b);

    if values.len() <= 1 {
        return Ok(Vec::new());
    }

    Ok(values
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect())
}

/// Evaluate and score a single 1D threshold predicate.
///
/// The stored score is the Python default tree-search objective for 1D splits:
/// information gain followed by BIC-style `penalized_gain` with arity 1.
pub fn evaluate_1d_candidate(
    dataset: &Dataset,
    predicate: ThresholdPredicate,
) -> Result<SplitCandidate, String> {
    let inside_mask = predicate.evaluate_mask(dataset)?;
    let outside_mask = inside_mask.complement();
    let positive_mask = positive_label_mask(dataset);
    let negative_mask = negative_label_mask(dataset);
    let parent_counts = class_counts(
        &BitSet::with_all(dataset.n_samples()),
        &positive_mask,
        &negative_mask,
    )?;
    let inside_counts = class_counts(&inside_mask, &positive_mask, &negative_mask)?;
    let outside_counts = class_counts(&outside_mask, &positive_mask, &negative_mask)?;

    let raw_gain = information_gain(parent_counts, inside_counts)?;
    let score = if raw_gain > 0.0 {
        penalized_gain(raw_gain, 1, dataset.n_samples())
    } else {
        raw_gain
    };

    Ok(SplitCandidate {
        feature_index: predicate.feature_index,
        threshold: predicate.threshold,
        op: predicate.op,
        score,
        inside_counts,
        outside_counts,
    })
}

/// Return the best deterministic 1D split for one feature, if any positive
/// penalized-gain split exists.
pub fn best_1d_split_for_feature(
    dataset: &Dataset,
    feature_index: usize,
) -> Result<Option<BestSplit>, String> {
    let thresholds = generate_1d_thresholds(dataset, feature_index)?;
    let mut best: Option<BestSplit> = None;

    for threshold in thresholds {
        for op in [ComparisonOp::LessThan, ComparisonOp::GreaterEqual] {
            let predicate = ThresholdPredicate {
                feature_index,
                threshold,
                op,
            };
            let candidate = evaluate_1d_candidate(dataset, predicate)?;
            if candidate.score <= 0.0 {
                continue;
            }
            let inside_mask = predicate.evaluate_mask(dataset)?;
            let outside_mask = inside_mask.complement();
            let split = BestSplit {
                candidate,
                inside_mask,
                outside_mask,
            };
            if should_replace(best.as_ref().map(|s| &s.candidate), &split.candidate) {
                best = Some(split);
            }
        }
    }

    Ok(best)
}

/// Return the best deterministic 1D split across all features, if any positive
/// penalized-gain split exists.
pub fn best_1d_split(dataset: &Dataset) -> Result<Option<BestSplit>, String> {
    let mut best: Option<BestSplit> = None;
    for feature_index in 0..dataset.n_features() {
        if let Some(split) = best_1d_split_for_feature(dataset, feature_index)? {
            if should_replace(best.as_ref().map(|s| &s.candidate), &split.candidate) {
                best = Some(split);
            }
        }
    }
    Ok(best)
}

fn should_replace(current: Option<&SplitCandidate>, candidate: &SplitCandidate) -> bool {
    let Some(current) = current else {
        return true;
    };
    const EPS: f64 = 1e-12;
    if candidate.score > current.score + EPS {
        return true;
    }
    if (candidate.score - current.score).abs() > EPS {
        return false;
    }
    if candidate.feature_index != current.feature_index {
        return candidate.feature_index < current.feature_index;
    }
    if candidate.threshold != current.threshold {
        return candidate.threshold < current.threshold;
    }
    op_rank(candidate.op) < op_rank(current.op)
}

fn op_rank(op: ComparisonOp) -> usize {
    match op {
        // Python exhaustive 1D scans LT/low-anchor before GE/high-anchor.
        ComparisonOp::LessThan => 0,
        ComparisonOp::GreaterEqual => 1,
        ComparisonOp::LessEqual => 2,
        ComparisonOp::GreaterThan => 3,
        ComparisonOp::Equal => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn one_feature_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
            ],
            vec![0, 0, 0, 1, 1, 1],
        )
        .unwrap()
    }

    #[test]
    fn generates_sorted_midpoint_thresholds() {
        let ds = Dataset::from_rows(
            vec![vec![3.0], vec![1.0], vec![1.0], vec![2.0], vec![5.0]],
            vec![0, 1, 0, 1, 0],
        )
        .unwrap();
        assert_eq!(generate_1d_thresholds(&ds, 0).unwrap(), vec![1.5, 2.5, 4.0]);
    }

    #[test]
    fn constant_feature_generates_no_thresholds() {
        let ds = Dataset::from_rows(vec![vec![7.0], vec![7.0]], vec![0, 1]).unwrap();
        assert!(generate_1d_thresholds(&ds, 0).unwrap().is_empty());
        assert!(best_1d_split_for_feature(&ds, 0).unwrap().is_none());
    }

    #[test]
    fn invalid_feature_index_errors() {
        let ds = one_feature_dataset();
        assert!(generate_1d_thresholds(&ds, 1).is_err());
        assert!(best_1d_split_for_feature(&ds, 1).is_err());
    }

    #[test]
    fn candidate_evaluation_counts_and_score() {
        let ds = one_feature_dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 2.5,
            op: ComparisonOp::LessThan,
        };
        let candidate = evaluate_1d_candidate(&ds, pred).unwrap();

        assert_eq!(
            candidate.inside_counts,
            ClassCounts {
                positive: 0,
                negative: 3
            }
        );
        assert_eq!(
            candidate.outside_counts,
            ClassCounts {
                positive: 3,
                negative: 0
            }
        );
        let expected_raw = 1.0_f64;
        let expected_score = penalized_gain(expected_raw, 1, ds.n_samples());
        assert!((candidate.score - expected_score).abs() < EPS);
    }

    #[test]
    fn invalid_empty_split_scores_minus_one() {
        let ds = one_feature_dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: -1.0,
            op: ComparisonOp::LessThan,
        };
        let candidate = evaluate_1d_candidate(&ds, pred).unwrap();
        assert_eq!(
            candidate.inside_counts,
            ClassCounts {
                positive: 0,
                negative: 0
            }
        );
        assert_eq!(candidate.score, -1.0);
    }

    #[test]
    fn best_split_for_one_feature_is_manually_known() {
        let ds = one_feature_dataset();
        let best = best_1d_split_for_feature(&ds, 0).unwrap().unwrap();
        assert_eq!(best.candidate.feature_index, 0);
        assert_eq!(best.candidate.threshold, 2.5);
        assert_eq!(best.candidate.op, ComparisonOp::LessThan);
        assert_eq!(best.inside_mask.indices(), vec![0, 1, 2]);
        assert_eq!(best.outside_mask.indices(), vec![3, 4, 5]);
        assert_eq!(
            best.candidate.inside_counts,
            ClassCounts {
                positive: 0,
                negative: 3
            }
        );
        assert_eq!(
            best.candidate.outside_counts,
            ClassCounts {
                positive: 3,
                negative: 0
            }
        );
    }

    #[test]
    fn best_split_across_multiple_features_uses_feature_order_tie_break() {
        let ds = Dataset::from_rows(
            vec![
                vec![0.0, 10.0],
                vec![1.0, 11.0],
                vec![2.0, 12.0],
                vec![3.0, 13.0],
            ],
            vec![0, 0, 1, 1],
        )
        .unwrap();
        let best = best_1d_split(&ds).unwrap().unwrap();
        assert_eq!(best.candidate.feature_index, 0);
        assert_eq!(best.candidate.threshold, 1.5);
    }

    #[test]
    fn deterministic_tie_break_prefers_smaller_threshold_then_operator_order() {
        let ds = Dataset::from_rows(
            (0..20).map(|value| vec![value as f64]).collect(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        .unwrap();
        let best = best_1d_split_for_feature(&ds, 0).unwrap().unwrap();
        // thresholds 4.5 and 14.5 have equal positive penalized score; choose smaller threshold.
        assert_eq!(best.candidate.threshold, 4.5);
        // LT and GE tie at threshold 4.5; LT matches Python low-anchor order.
        assert_eq!(best.candidate.op, ComparisonOp::LessThan);
    }

    #[test]
    fn no_positive_gain_returns_none() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![1, 1, 1, 1],
        )
        .unwrap();
        assert!(best_1d_split(&ds).unwrap().is_none());
    }

    #[test]
    fn rejects_nan_threshold_generation_for_now() {
        let ds = Dataset::from_rows(vec![vec![f64::NAN], vec![1.0]], vec![0, 1]).unwrap();
        assert!(generate_1d_thresholds(&ds, 0).unwrap_err().contains("NaN"));
    }
}
