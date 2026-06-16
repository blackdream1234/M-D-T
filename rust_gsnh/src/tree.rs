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

/// Minimal depth-limited recursive decision tree.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionTree {
    Leaf(LeafNode),
    Split(DecisionNode),
}

/// Recursive split node for one selected family.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionNode {
    pub split: BestFamilySplit,
    pub true_child: Box<DecisionTree>,
    pub false_child: Box<DecisionTree>,
    pub counts: ClassCounts,
    pub depth: usize,
}

/// Configuration for the shallow recursive Rust tree skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeBuildConfig {
    pub family_config: FamilySearchConfig,
    pub max_depth: usize,
    pub min_samples_split: usize,
}

/// Read-only structural summary for a Rust decision tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeSummary {
    pub n_nodes: usize,
    pub n_leaves: usize,
    pub n_internal_nodes: usize,
    pub max_depth: usize,
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

/// Convert a Rust prediction label to the binary label convention used by Python.
pub fn prediction_label_to_u8(label: PredictionLabel) -> u8 {
    match label {
        PredictionLabel::Negative => 0,
        PredictionLabel::Positive => 1,
    }
}

/// Convert a slice of Rust prediction labels to binary `u8` labels.
pub fn prediction_labels_to_u8(labels: &[PredictionLabel]) -> Vec<u8> {
    labels.iter().copied().map(prediction_label_to_u8).collect()
}

/// Compute classification accuracy from Rust prediction labels.
pub fn accuracy_from_predictions(
    predictions: &[PredictionLabel],
    dataset: &Dataset,
) -> Result<f64, String> {
    let as_u8 = prediction_labels_to_u8(predictions);
    accuracy_from_u8_predictions(&as_u8, dataset)
}

/// Compute classification accuracy from binary `u8` predictions.
///
/// The formula mirrors Python benchmark usage: `(pred == y).mean()`.
pub fn accuracy_from_u8_predictions(predictions: &[u8], dataset: &Dataset) -> Result<f64, String> {
    if predictions.len() != dataset.n_samples() {
        return Err(format!(
            "prediction length {} does not match dataset samples {}",
            predictions.len(),
            dataset.n_samples()
        ));
    }

    let mut correct = 0usize;
    for (idx, (&pred, &actual)) in predictions.iter().zip(dataset.labels()).enumerate() {
        match pred {
            0 | 1 => {
                if pred == actual {
                    correct += 1;
                }
            }
            other => {
                return Err(format!(
                    "invalid prediction label {} at row {}; expected 0 or 1",
                    other, idx
                ));
            }
        }
    }

    Ok(correct as f64 / dataset.n_samples() as f64)
}

/// Predict every row with the stump and compute classification accuracy.
pub fn stump_accuracy(tree: &StumpTree, dataset: &Dataset) -> Result<f64, String> {
    let predictions = predict_stump(tree, dataset)?;
    accuracy_from_predictions(&predictions, dataset)
}

/// Build a depth-limited recursive tree for exactly one selected family.
pub fn build_tree_with_family(
    dataset: &Dataset,
    config: TreeBuildConfig,
) -> Result<DecisionTree, String> {
    let all_rows = BitSet::with_all(dataset.n_samples());
    build_tree_on_mask(dataset, &all_rows, config, 0)
}

fn build_tree_on_mask(
    dataset: &Dataset,
    active_mask: &BitSet,
    config: TreeBuildConfig,
    depth: usize,
) -> Result<DecisionTree, String> {
    if active_mask.len() != dataset.n_samples() {
        return Err(format!(
            "active mask length {} does not match dataset samples {}",
            active_mask.len(),
            dataset.n_samples()
        ));
    }

    let leaf = majority_leaf_from_mask(dataset, active_mask)?;
    let active_count = active_mask.count_ones();
    if active_count == 0
        || depth >= config.max_depth
        || active_count < config.min_samples_split
        || leaf.counts.positive == 0
        || leaf.counts.negative == 0
    {
        return Ok(DecisionTree::Leaf(leaf));
    }

    let local_dataset = dataset_from_mask(dataset, active_mask)?;
    let Some(split) = best_family_split(&local_dataset, config.family_config)? else {
        return Ok(DecisionTree::Leaf(leaf));
    };

    let (true_mask, false_mask) = active_split_masks_from_predicate(dataset, active_mask, &split)?;

    if true_mask.count_ones() == 0 || false_mask.count_ones() == 0 {
        return Ok(DecisionTree::Leaf(leaf));
    }

    let true_child = build_tree_on_mask(dataset, &true_mask, config, depth + 1)?;
    let false_child = build_tree_on_mask(dataset, &false_mask, config, depth + 1)?;

    Ok(DecisionTree::Split(DecisionNode {
        split,
        true_child: Box::new(true_child),
        false_child: Box::new(false_child),
        counts: leaf.counts,
        depth,
    }))
}

/// Predict one row with a depth-limited recursive tree.
pub fn predict_tree_row(
    tree: &DecisionTree,
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
        DecisionTree::Leaf(leaf) => Ok(leaf.prediction),
        DecisionTree::Split(node) => {
            let mask = evaluate_split_mask(&node.split, dataset)?;
            if mask.contains(row_index)? {
                predict_tree_row(&node.true_child, dataset, row_index)
            } else {
                predict_tree_row(&node.false_child, dataset, row_index)
            }
        }
    }
}

/// Predict every row in `dataset` with a depth-limited recursive tree.
pub fn predict_tree(
    tree: &DecisionTree,
    dataset: &Dataset,
) -> Result<Vec<PredictionLabel>, String> {
    let mut out = Vec::with_capacity(dataset.n_samples());
    for row_index in 0..dataset.n_samples() {
        out.push(predict_tree_row(tree, dataset, row_index)?);
    }
    Ok(out)
}

/// Predict every row with the recursive tree and compute classification accuracy.
pub fn tree_accuracy(tree: &DecisionTree, dataset: &Dataset) -> Result<f64, String> {
    let predictions = predict_tree(tree, dataset)?;
    accuracy_from_predictions(&predictions, dataset)
}

/// Count all nodes in a decision tree.
pub fn count_tree_nodes(tree: &DecisionTree) -> usize {
    match tree {
        DecisionTree::Leaf(_) => 1,
        DecisionTree::Split(node) => {
            1 + count_tree_nodes(&node.true_child) + count_tree_nodes(&node.false_child)
        }
    }
}

/// Count leaf nodes in a decision tree.
pub fn count_tree_leaves(tree: &DecisionTree) -> usize {
    match tree {
        DecisionTree::Leaf(_) => 1,
        DecisionTree::Split(node) => {
            count_tree_leaves(&node.true_child) + count_tree_leaves(&node.false_child)
        }
    }
}

/// Count internal split nodes in a decision tree.
pub fn count_tree_internal_nodes(tree: &DecisionTree) -> usize {
    match tree {
        DecisionTree::Leaf(_) => 0,
        DecisionTree::Split(node) => {
            1 + count_tree_internal_nodes(&node.true_child)
                + count_tree_internal_nodes(&node.false_child)
        }
    }
}

/// Return the maximum observed depth in a decision tree.
pub fn observed_tree_depth(tree: &DecisionTree) -> usize {
    match tree {
        DecisionTree::Leaf(_) => 0,
        DecisionTree::Split(node) => {
            1 + observed_tree_depth(&node.true_child).max(observed_tree_depth(&node.false_child))
        }
    }
}

/// Summarize read-only tree structure counts.
pub fn summarize_tree(tree: &DecisionTree) -> TreeSummary {
    let n_leaves = count_tree_leaves(tree);
    let n_internal_nodes = count_tree_internal_nodes(tree);
    TreeSummary {
        n_nodes: n_leaves + n_internal_nodes,
        n_leaves,
        n_internal_nodes,
        max_depth: observed_tree_depth(tree),
    }
}

/// Convenience wrapper for training-set accuracy on an already-built tree.
pub fn training_accuracy(tree: &DecisionTree, dataset: &Dataset) -> Result<f64, String> {
    tree_accuracy(tree, dataset)
}

fn dataset_from_mask(dataset: &Dataset, mask: &BitSet) -> Result<Dataset, String> {
    if mask.len() != dataset.n_samples() {
        return Err(format!(
            "mask length {} does not match dataset samples {}",
            mask.len(),
            dataset.n_samples()
        ));
    }

    let mut rows = Vec::with_capacity(mask.count_ones());
    let mut labels = Vec::with_capacity(mask.count_ones());
    for row_index in mask.indices() {
        rows.push(dataset.row(row_index).to_vec());
        labels.push(dataset.labels()[row_index]);
    }
    Dataset::from_rows(rows, labels).map_err(|err| err.to_string())
}

fn active_split_masks_from_predicate(
    dataset: &Dataset,
    active_mask: &BitSet,
    split: &BestFamilySplit,
) -> Result<(BitSet, BitSet), String> {
    if active_mask.len() != dataset.n_samples() {
        return Err(format!(
            "active mask length {} does not match dataset samples {}",
            active_mask.len(),
            dataset.n_samples()
        ));
    }
    let full_inside = evaluate_split_mask(split, dataset)?;
    let true_mask = active_mask.intersection(&full_inside)?;
    let false_mask = active_mask.difference(&true_mask)?;
    Ok((true_mask, false_mask))
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
    fn prediction_label_conversions_match_binary_label_convention() {
        assert_eq!(prediction_label_to_u8(PredictionLabel::Negative), 0);
        assert_eq!(prediction_label_to_u8(PredictionLabel::Positive), 1);
        assert_eq!(
            prediction_labels_to_u8(&[
                PredictionLabel::Negative,
                PredictionLabel::Positive,
                PredictionLabel::Positive,
            ]),
            vec![0, 1, 1]
        );
    }

    #[test]
    fn accuracy_helpers_cover_perfect_partial_and_zero_accuracy() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 1, 1, 0],
        )
        .unwrap();

        assert_eq!(
            accuracy_from_u8_predictions(&[0, 1, 1, 0], &ds).unwrap(),
            1.0
        );
        assert_eq!(
            accuracy_from_u8_predictions(&[0, 0, 1, 1], &ds).unwrap(),
            0.5
        );
        assert_eq!(
            accuracy_from_u8_predictions(&[1, 0, 0, 1], &ds).unwrap(),
            0.0
        );

        let rust_predictions = [
            PredictionLabel::Negative,
            PredictionLabel::Positive,
            PredictionLabel::Negative,
            PredictionLabel::Negative,
        ];
        assert_eq!(
            accuracy_from_predictions(&rust_predictions, &ds).unwrap(),
            0.75
        );
    }

    #[test]
    fn accuracy_helpers_reject_length_mismatch_and_invalid_u8_labels() {
        let ds = Dataset::from_rows(vec![vec![0.0], vec![1.0]], vec![0, 1]).unwrap();
        assert!(accuracy_from_u8_predictions(&[0], &ds).is_err());
        assert!(accuracy_from_predictions(&[PredictionLabel::Negative], &ds).is_err());
        assert!(accuracy_from_u8_predictions(&[0, 2], &ds).is_err());
    }

    #[test]
    fn stump_accuracy_handles_single_leaf_and_split_stumps() {
        let leaf_ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 0, 1, 1],
        )
        .unwrap();
        let leaf_tree =
            StumpTree::Leaf(majority_leaf_from_mask(&leaf_ds, &BitSet::with_all(4)).unwrap());
        assert_eq!(stump_accuracy(&leaf_tree, &leaf_ds).unwrap(), 0.5);

        let split_ds = and_dataset();
        let split_tree =
            build_stump_with_family(&split_ds, ge_config(LanguageFamily::ConjUI)).unwrap();
        assert_eq!(stump_accuracy(&split_tree, &split_ds).unwrap(), 1.0);
        assert_eq!(
            accuracy_from_predictions(&predict_stump(&split_tree, &split_ds).unwrap(), &split_ds)
                .unwrap(),
            1.0
        );
    }

    fn tree_config(max_depth: usize, min_samples_split: usize) -> TreeBuildConfig {
        TreeBuildConfig {
            family_config: FamilySearchConfig {
                family: LanguageFamily::ConjUI,
                max_arity: 1,
                min_samples_leaf: 1,
            },
            max_depth,
            min_samples_split,
        }
    }

    fn recursive_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0],
                vec![0.0],
                vec![0.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                vec![2.0],
                vec![2.0],
                vec![2.0],
                vec![2.0],
                vec![3.0],
                vec![3.0],
                vec![3.0],
                vec![3.0],
            ],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        )
        .unwrap()
    }

    #[test]
    fn dataset_from_mask_preserves_active_row_order_features_and_labels() {
        let ds = Dataset::from_rows(
            vec![
                vec![10.0, 0.0],
                vec![11.0, 1.0],
                vec![12.0, 0.0],
                vec![13.0, 1.0],
                vec![14.0, 0.0],
            ],
            vec![0, 1, 0, 1, 1],
        )
        .unwrap();
        let mask = BitSet::from_indices(ds.n_samples(), &[1, 3, 4]).unwrap();
        let local = dataset_from_mask(&ds, &mask).unwrap();
        assert_eq!(local.n_samples(), 3);
        assert_eq!(local.row(0), &[11.0, 1.0]);
        assert_eq!(local.row(1), &[13.0, 1.0]);
        assert_eq!(local.row(2), &[14.0, 0.0]);
        assert_eq!(local.labels(), &[1, 1, 1]);
    }

    #[test]
    fn active_split_masks_partition_only_the_active_mask() {
        let ds = recursive_dataset();
        let active_mask =
            BitSet::from_indices(ds.n_samples(), &[4, 5, 6, 7, 8, 9, 10, 11]).unwrap();
        let local = dataset_from_mask(&ds, &active_mask).unwrap();
        let split = best_family_split(
            &local,
            FamilySearchConfig {
                family: LanguageFamily::ConjUI,
                max_arity: 2,
                min_samples_leaf: 1,
            },
        )
        .unwrap()
        .unwrap();
        let (true_mask, false_mask) =
            active_split_masks_from_predicate(&ds, &active_mask, &split).unwrap();

        assert!(true_mask
            .difference(&active_mask)
            .unwrap()
            .indices()
            .is_empty());
        assert!(false_mask
            .difference(&active_mask)
            .unwrap()
            .indices()
            .is_empty());
        assert!(true_mask
            .intersection(&false_mask)
            .unwrap()
            .indices()
            .is_empty());
        assert_eq!(
            true_mask.union(&false_mask).unwrap().indices(),
            active_mask.indices()
        );
        assert_eq!(
            true_mask.count_ones() + false_mask.count_ones(),
            active_mask.count_ones()
        );
    }

    #[test]
    fn active_subset_helpers_reject_incompatible_mask_lengths() {
        let ds = recursive_dataset();
        let bad_mask = BitSet::with_all(ds.n_samples() - 1);
        assert!(dataset_from_mask(&ds, &bad_mask).is_err());
        let split = best_family_split(&ds, tree_config(1, 2).family_config)
            .unwrap()
            .unwrap();
        assert!(active_split_masks_from_predicate(&ds, &bad_mask, &split).is_err());
    }

    #[test]
    fn recursive_tree_stopping_rules_build_leaves() {
        let ds = recursive_dataset();
        assert!(matches!(
            build_tree_with_family(&ds, tree_config(0, 2)).unwrap(),
            DecisionTree::Leaf(_)
        ));

        let pure = Dataset::from_rows(vec![vec![0.0], vec![1.0]], vec![1, 1]).unwrap();
        assert!(matches!(
            build_tree_with_family(&pure, tree_config(3, 2)).unwrap(),
            DecisionTree::Leaf(_)
        ));

        let no_split = Dataset::from_rows(vec![vec![0.0], vec![1.0]], vec![0, 0]).unwrap();
        assert!(matches!(
            build_tree_with_family(&no_split, tree_config(3, 2)).unwrap(),
            DecisionTree::Leaf(_)
        ));

        assert!(matches!(
            build_tree_with_family(&ds, tree_config(3, 17)).unwrap(),
            DecisionTree::Leaf(_)
        ));
    }

    #[test]
    fn max_depth_one_matches_existing_stump_predictions() {
        let ds = and_dataset();
        let family_config = ge_config(LanguageFamily::ConjUI);
        let stump = build_stump_with_family(&ds, family_config).unwrap();
        let tree = build_tree_with_family(
            &ds,
            TreeBuildConfig {
                family_config,
                max_depth: 1,
                min_samples_split: 2,
            },
        )
        .unwrap();
        assert_eq!(
            predict_tree(&tree, &ds).unwrap(),
            predict_stump(&stump, &ds).unwrap()
        );
        assert_eq!(
            tree_accuracy(&tree, &ds).unwrap(),
            stump_accuracy(&stump, &ds).unwrap()
        );
    }

    #[test]
    fn max_depth_two_uses_active_subsets_for_recursive_splits() {
        let ds = recursive_dataset();
        let tree = build_tree_with_family(&ds, tree_config(2, 2)).unwrap();
        let DecisionTree::Split(root) = &tree else {
            panic!("expected recursive root split");
        };
        assert_eq!(root.depth, 0);
        assert_eq!(
            root.counts,
            ClassCounts {
                positive: 4,
                negative: 12
            }
        );

        let true_child = root.true_child.as_ref();
        let false_child = root.false_child.as_ref();
        let recursive_child = match (true_child, false_child) {
            (DecisionTree::Split(child), DecisionTree::Leaf(_)) => child,
            (DecisionTree::Leaf(_), DecisionTree::Split(child)) => child,
            _ => panic!("expected exactly one recursive child split"),
        };
        assert_eq!(recursive_child.depth, 1);
        assert!(recursive_child.counts.positive > 0);
        assert!(recursive_child.counts.negative > 0);
        assert!(recursive_child.counts.positive + recursive_child.counts.negative < 16);

        assert_eq!(
            predict_tree(&tree, &ds).unwrap(),
            labels_to_predictions(&[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        );
        assert_eq!(tree_accuracy(&tree, &ds).unwrap(), 1.0);
        let stump = build_stump_with_family(&ds, tree_config(1, 2).family_config).unwrap();
        assert_ne!(
            predict_tree(&tree, &ds).unwrap(),
            predict_stump(&stump, &ds).unwrap()
        );
        assert!(tree_accuracy(&tree, &ds).unwrap() > stump_accuracy(&stump, &ds).unwrap());
        assert_eq!(
            predict_tree_row(&tree, &ds, 8).unwrap(),
            PredictionLabel::Positive
        );
        assert!(predict_tree_row(&tree, &ds, ds.n_samples()).is_err());
    }

    #[test]
    fn mixed_child_with_no_valid_local_split_becomes_leaf() {
        let ds = Dataset::from_rows(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
            ],
            vec![0, 0, 0, 0, 0, 1, 0, 1],
        )
        .unwrap();
        let tree = build_tree_with_family(
            &ds,
            TreeBuildConfig {
                family_config: FamilySearchConfig {
                    family: LanguageFamily::ConjUI,
                    max_arity: 2,
                    min_samples_leaf: 1,
                },
                max_depth: 3,
                min_samples_split: 2,
            },
        )
        .unwrap();
        let DecisionTree::Split(root) = tree else {
            panic!("expected root split");
        };
        let has_mixed_leaf_child = [&root.true_child, &root.false_child].iter().any(|child| {
            matches!(
                child.as_ref(),
                DecisionTree::Leaf(LeafNode {
                    counts: ClassCounts {
                        positive: 2,
                        negative: 2
                    },
                    ..
                })
            )
        });
        assert!(has_mixed_leaf_child);
    }

    #[test]
    fn tree_summary_counts_single_leaf_tree() {
        let ds = Dataset::from_rows(vec![vec![0.0], vec![1.0]], vec![0, 1]).unwrap();
        let tree = DecisionTree::Leaf(majority_leaf_from_mask(&ds, &BitSet::with_all(2)).unwrap());
        let summary = summarize_tree(&tree);
        assert_eq!(
            summary,
            TreeSummary {
                n_nodes: 1,
                n_leaves: 1,
                n_internal_nodes: 0,
                max_depth: 0
            }
        );
        assert_eq!(count_tree_nodes(&tree), 1);
        assert_eq!(count_tree_leaves(&tree), 1);
        assert_eq!(count_tree_internal_nodes(&tree), 0);
        assert_eq!(observed_tree_depth(&tree), 0);
    }

    #[test]
    fn tree_summary_counts_stump_shape() {
        let ds = and_dataset();
        let tree = build_tree_with_family(
            &ds,
            TreeBuildConfig {
                family_config: ge_config(LanguageFamily::ConjUI),
                max_depth: 1,
                min_samples_split: 2,
            },
        )
        .unwrap();
        let summary = summarize_tree(&tree);
        assert_eq!(summary.n_nodes, 3);
        assert_eq!(summary.n_leaves, 2);
        assert_eq!(summary.n_internal_nodes, 1);
        assert_eq!(summary.max_depth, 1);
        assert_eq!(summary.n_nodes, summary.n_internal_nodes + summary.n_leaves);
    }

    #[test]
    fn tree_summary_counts_depth_two_tree_and_training_accuracy() {
        let ds = recursive_dataset();
        let tree = build_tree_with_family(&ds, tree_config(2, 2)).unwrap();
        let summary = summarize_tree(&tree);
        assert_eq!(summary.max_depth, 2);
        assert_eq!(summary.n_nodes, summary.n_internal_nodes + summary.n_leaves);
        assert!(summary.n_nodes > 3);
        assert!(summary.n_internal_nodes >= 2);
        assert!(summary.n_leaves >= 3);
        assert_eq!(
            training_accuracy(&tree, &ds).unwrap(),
            tree_accuracy(&tree, &ds).unwrap()
        );
        assert_eq!(training_accuracy(&tree, &ds).unwrap(), 1.0);
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
