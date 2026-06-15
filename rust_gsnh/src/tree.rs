//! Minimal one-node stump builder for the incremental Rust GSNH engine.
//!
//! Python remains the reference/oracle. This module only builds a depth-1 stump
//! for one selected family through `best_family_split`; it does not implement
//! recursive tree learning, pruning, bindings, theorem certificates, or
//! benchmark integration.

use crate::{
    best_family_split, class_counts, negative_label_mask, positive_label_mask, BestFamilySplit,
    BitSet, ClassCounts, Dataset, FamilySearchConfig,
};

/// Binary prediction label returned by the Rust stump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionLabel {
    Negative,
    Positive,
}

/// Leaf node with majority prediction and class counts for its training mask.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode {
    pub prediction: PredictionLabel,
    pub counts: ClassCounts,
}

/// One internal root split plus true/false leaves.
#[derive(Clone, Debug, PartialEq)]
pub struct StumpNode {
    pub split: BestFamilySplit,
    pub true_leaf: LeafNode,
    pub false_leaf: LeafNode,
}

/// Depth-1 tree: either a single leaf or one split with two leaves.
#[derive(Clone, Debug, PartialEq)]
pub enum StumpTree {
    Leaf(LeafNode),
    Split(StumpNode),
}

/// Build a majority leaf from a row mask.
///
/// Python stores leaf probabilities with Laplace smoothing and predicts class 1
/// when `proba >= 0.5`; therefore exact count ties map to `Positive` here.
pub fn majority_leaf_from_mask(dataset: &Dataset, mask: &BitSet) -> Result<LeafNode, String> {
    if mask.len() != dataset.n_samples() {
        return Err(format!(
            "mask length {} does not match dataset samples {}",
            mask.len(),
            dataset.n_samples()
        ));
    }
    let positive_mask = positive_label_mask(dataset);
    let negative_mask = negative_label_mask(dataset);
    let counts = class_counts(mask, &positive_mask, &negative_mask)?;
    let prediction = if counts.positive >= counts.negative {
        PredictionLabel::Positive
    } else {
        PredictionLabel::Negative
    };
    Ok(LeafNode { prediction, counts })
}

/// Build a non-recursive depth-1 stump for exactly one selected family.
pub fn build_stump_with_family(
    dataset: &Dataset,
    config: FamilySearchConfig,
) -> Result<StumpTree, String> {
    let Some(split) = best_family_split(dataset, config)? else {
        let all_rows = BitSet::with_all(dataset.n_samples());
        return Ok(StumpTree::Leaf(majority_leaf_from_mask(
            dataset, &all_rows,
        )?));
    };

    let (inside_mask, outside_mask) = split_masks(&split);
    let true_leaf = majority_leaf_from_mask(dataset, &inside_mask)?;
    let false_leaf = majority_leaf_from_mask(dataset, &outside_mask)?;
    Ok(StumpTree::Split(StumpNode {
        split,
        true_leaf,
        false_leaf,
    }))
}

/// Predict one row with a leaf or one-node stump.
pub fn predict_stump_row(
    tree: &StumpTree,
    dataset: &Dataset,
    row_index: usize,
) -> Result<PredictionLabel, String> {
    if row_index >= dataset.n_samples() {
        return Err(format!(
            "row index {} out of range for dataset with {} samples",
            row_index,
            dataset.n_samples()
        ));
    }

    match tree {
        StumpTree::Leaf(leaf) => Ok(leaf.prediction),
        StumpTree::Split(node) => {
            let mask = evaluate_split_mask(&node.split, dataset)?;
            if mask.contains(row_index)? {
                Ok(node.true_leaf.prediction)
            } else {
                Ok(node.false_leaf.prediction)
            }
        }
    }
}

/// Predict every row in `dataset` with a leaf or one-node stump.
pub fn predict_stump(tree: &StumpTree, dataset: &Dataset) -> Result<Vec<PredictionLabel>, String> {
    let mut out = Vec::with_capacity(dataset.n_samples());
    for row_index in 0..dataset.n_samples() {
        out.push(predict_stump_row(tree, dataset, row_index)?);
    }
    Ok(out)
}

fn split_masks(split: &BestFamilySplit) -> (BitSet, BitSet) {
    match split {
        BestFamilySplit::Composed(evaluated) => (
            evaluated.inside_mask.clone(),
            evaluated.outside_mask.clone(),
        ),
        BestFamilySplit::Square2CNF(evaluated) => (
            evaluated.inside_mask.clone(),
            evaluated.outside_mask.clone(),
        ),
    }
}

fn evaluate_split_mask(split: &BestFamilySplit, dataset: &Dataset) -> Result<BitSet, String> {
    match split {
        BestFamilySplit::Composed(evaluated) => {
            evaluated.candidate.predicate.evaluate_mask(dataset)
        }
        BestFamilySplit::Square2CNF(evaluated) => {
            evaluated.candidate.predicate.evaluate_mask(dataset)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LanguageFamily;

    fn labels_to_predictions(labels: &[u8]) -> Vec<PredictionLabel> {
        labels
            .iter()
            .map(|label| {
                if *label == 1 {
                    PredictionLabel::Positive
                } else {
                    PredictionLabel::Negative
                }
            })
            .collect()
    }

    fn ge_config(family: LanguageFamily) -> FamilySearchConfig {
        FamilySearchConfig {
            family,
            max_arity: 2,
            min_samples_leaf: 1,
        }
    }

    fn horn_dataset() -> Dataset {
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

    fn and_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0, 0, 0, 1],
        )
        .unwrap()
    }

    fn xor_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0, 1, 1, 0],
        )
        .unwrap()
    }

    fn square_dataset() -> Dataset {
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

    #[test]
    fn majority_leaf_counts_and_tie_breaking_match_python_prediction_rule() {
        let positive_ds =
            Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0]], vec![1, 1, 0]).unwrap();
        let positive = majority_leaf_from_mask(&positive_ds, &BitSet::with_all(3)).unwrap();
        assert_eq!(positive.prediction, PredictionLabel::Positive);
        assert_eq!(
            positive.counts,
            ClassCounts {
                positive: 2,
                negative: 1
            }
        );

        let negative_ds =
            Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0]], vec![0, 0, 1]).unwrap();
        let negative = majority_leaf_from_mask(&negative_ds, &BitSet::with_all(3)).unwrap();
        assert_eq!(negative.prediction, PredictionLabel::Negative);
        assert_eq!(
            negative.counts,
            ClassCounts {
                positive: 1,
                negative: 2
            }
        );

        let tie_ds = Dataset::from_rows(vec![vec![0.0], vec![1.0]], vec![0, 1]).unwrap();
        let tie = majority_leaf_from_mask(&tie_ds, &BitSet::with_all(2)).unwrap();
        assert_eq!(tie.prediction, PredictionLabel::Positive);
        assert_eq!(
            tie.counts,
            ClassCounts {
                positive: 1,
                negative: 1
            }
        );

        let empty = BitSet::new(2);
        let empty_leaf = majority_leaf_from_mask(&tie_ds, &empty).unwrap();
        assert_eq!(empty_leaf.prediction, PredictionLabel::Positive);
        assert_eq!(
            empty_leaf.counts,
            ClassCounts {
                positive: 0,
                negative: 0
            }
        );
    }

    #[test]
    fn majority_leaf_rejects_length_mismatch() {
        let ds = Dataset::from_rows(vec![vec![0.0], vec![1.0]], vec![0, 1]).unwrap();
        assert!(majority_leaf_from_mask(&ds, &BitSet::with_all(1)).is_err());
    }

    #[test]
    fn stump_returns_leaf_when_no_valid_split_exists() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 0, 0, 0],
        )
        .unwrap();
        let tree = build_stump_with_family(&ds, ge_config(LanguageFamily::ConjUI)).unwrap();
        let StumpTree::Leaf(leaf) = tree else {
            panic!("expected leaf stump");
        };
        assert_eq!(leaf.prediction, PredictionLabel::Negative);
        assert_eq!(
            leaf.counts,
            ClassCounts {
                positive: 0,
                negative: 4
            }
        );
    }

    #[test]
    fn stump_builds_conjui_split_with_expected_leaf_counts_and_predictions() {
        let ds = and_dataset();
        let tree = build_stump_with_family(&ds, ge_config(LanguageFamily::ConjUI)).unwrap();
        let StumpTree::Split(node) = &tree else {
            panic!("expected split stump");
        };
        assert_eq!(
            node.true_leaf.counts,
            ClassCounts {
                positive: 1,
                negative: 0
            }
        );
        assert_eq!(
            node.false_leaf.counts,
            ClassCounts {
                positive: 0,
                negative: 3
            }
        );
        assert_eq!(
            predict_stump_row(&tree, &ds, 3).unwrap(),
            PredictionLabel::Positive
        );
        assert_eq!(
            predict_stump_row(&tree, &ds, 0).unwrap(),
            PredictionLabel::Negative
        );
        assert_eq!(
            predict_stump(&tree, &ds).unwrap(),
            labels_to_predictions(&[0, 0, 0, 1])
        );
    }

    #[test]
    fn predict_stump_row_rejects_invalid_row_index() {
        let ds = and_dataset();
        let tree = build_stump_with_family(&ds, ge_config(LanguageFamily::ConjUI)).unwrap();
        assert!(predict_stump_row(&tree, &ds, ds.n_samples()).is_err());
    }

    fn assert_family_stump_predicts_all_rows(
        dataset: Dataset,
        family: LanguageFamily,
    ) -> StumpTree {
        let tree = build_stump_with_family(&dataset, ge_config(family)).unwrap();
        assert!(matches!(tree, StumpTree::Split(_)));
        let predictions = predict_stump(&tree, &dataset).unwrap();
        assert_eq!(predictions.len(), dataset.n_samples());
        assert!(predict_stump_row(&tree, &dataset, 0).is_ok());
        tree
    }

    #[test]
    fn stump_prediction_works_for_horn_antihorn_affine_and_square2cnf() {
        let horn_ds = horn_dataset();
        let horn_tree =
            assert_family_stump_predicts_all_rows(horn_ds.clone(), LanguageFamily::Horn);
        assert_eq!(
            predict_stump(&horn_tree, &horn_ds).unwrap(),
            labels_to_predictions(&[1, 0, 1, 1, 0, 1])
        );

        let antihorn_ds = antihorn_dataset();
        let antihorn_tree =
            assert_family_stump_predicts_all_rows(antihorn_ds.clone(), LanguageFamily::AntiHorn);
        assert_eq!(
            predict_stump(&antihorn_tree, &antihorn_ds).unwrap(),
            labels_to_predictions(&[1, 1, 1, 0, 1, 1])
        );

        let affine_ds = xor_dataset();
        let affine_tree =
            assert_family_stump_predicts_all_rows(affine_ds.clone(), LanguageFamily::Affine);
        assert_eq!(
            predict_stump(&affine_tree, &affine_ds).unwrap(),
            labels_to_predictions(&[0, 1, 1, 0])
        );

        let square_ds = square_dataset();
        let square_tree =
            assert_family_stump_predicts_all_rows(square_ds.clone(), LanguageFamily::Square2CNF);
        let expected: Vec<u8> = square_ds.labels().to_vec();
        assert_eq!(
            predict_stump(&square_tree, &square_ds).unwrap(),
            labels_to_predictions(&expected)
        );
    }

    #[test]
    fn prediction_reevaluates_split_mask_for_current_dataset() {
        let ds = and_dataset();
        let tree = build_stump_with_family(&ds, ge_config(LanguageFamily::ConjUI)).unwrap();
        let shifted = Dataset::from_rows(
            vec![
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
            vec![1, 0, 0, 0],
        )
        .unwrap();
        assert_eq!(
            predict_stump(&tree, &shifted).unwrap(),
            labels_to_predictions(&[1, 0, 0, 0])
        );
    }
}
