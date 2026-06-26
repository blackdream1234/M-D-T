//! Stable Rust-facing train/predict API shape for future bindings.
//!
//! This module is deliberately a thin wrapper around the existing tree, family,
//! prediction, and summary helpers. It does not add PyO3, benchmark integration,
//! pruning, theorem certificates, BestPerNode, or new learning semantics.

use crate::{
    build_tree_with_family, predict_tree, prediction_labels_to_u8, summarize_tree,
    training_accuracy, tree_accuracy, Dataset, DecisionTree, FamilySearchConfig, LanguageFamily,
    TreeBuildConfig, TreeSummary,
};

/// Public Rust configuration for training a selected-family GSNH tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RustGsnHConfig {
    pub family: LanguageFamily,
    pub max_arity: usize,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
}

/// Trained Rust GSNH model plus immutable training configuration and summary.
#[derive(Clone, Debug, PartialEq)]
pub struct RustGsnHModel {
    pub tree: DecisionTree,
    pub config: RustGsnHConfig,
    pub summary: TreeSummary,
}

/// Fit result with model and training-set accuracy.
#[derive(Clone, Debug, PartialEq)]
pub struct RustGsnHFitResult {
    pub model: RustGsnHModel,
    pub training_accuracy: f64,
}

/// Fit a selected-family Rust GSNH tree using existing tree-building semantics.
pub fn fit_rust_gsnh(
    dataset: &Dataset,
    config: RustGsnHConfig,
) -> Result<RustGsnHFitResult, String> {
    validate_config(config)?;
    let family_config = FamilySearchConfig {
        family: config.family,
        max_arity: config.max_arity,
        min_samples_leaf: config.min_samples_leaf,
    };
    let tree_config = TreeBuildConfig {
        family_config,
        max_depth: config.max_depth,
        min_samples_split: config.min_samples_split,
    };
    let tree = build_tree_with_family(dataset, tree_config)?;
    let summary = summarize_tree(&tree);
    let training_accuracy = training_accuracy(&tree, dataset)?;
    Ok(RustGsnHFitResult {
        model: RustGsnHModel {
            tree,
            config,
            summary,
        },
        training_accuracy,
    })
}

/// Predict binary labels with a fitted Rust GSNH model.
pub fn predict_rust_gsnh(model: &RustGsnHModel, dataset: &Dataset) -> Result<Vec<u8>, String> {
    let predictions = predict_tree(&model.tree, dataset)?;
    Ok(prediction_labels_to_u8(&predictions))
}

/// Score a fitted Rust GSNH model with classification accuracy.
pub fn score_rust_gsnh(model: &RustGsnHModel, dataset: &Dataset) -> Result<f64, String> {
    tree_accuracy(&model.tree, dataset)
}

/// Return the stored model summary.
pub fn summarize_rust_gsnh(model: &RustGsnHModel) -> TreeSummary {
    model.summary
}

fn validate_config(config: RustGsnHConfig) -> Result<(), String> {
    if config.max_arity == 0 {
        return Err("max_arity must be >= 1".to_string());
    }
    if config.min_samples_split == 0 {
        return Err("min_samples_split must be >= 1".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{summarize_tree, training_accuracy};

    fn config(family: LanguageFamily, max_depth: usize) -> RustGsnHConfig {
        RustGsnHConfig {
            family,
            max_arity: 2,
            max_depth,
            min_samples_leaf: 1,
            min_samples_split: 2,
        }
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
    fn fit_trains_leaf_split_and_deeper_tree() {
        let ds = and_dataset();
        let leaf_fit = fit_rust_gsnh(&ds, config(LanguageFamily::ConjUI, 0)).unwrap();
        assert_eq!(leaf_fit.model.summary.n_nodes, 1);
        assert_eq!(leaf_fit.model.summary.max_depth, 0);

        let split_fit = fit_rust_gsnh(&ds, config(LanguageFamily::ConjUI, 1)).unwrap();
        assert_eq!(split_fit.model.summary.n_nodes, 3);
        assert_eq!(split_fit.model.summary.max_depth, 1);

        let recursive = recursive_dataset();
        let deep_fit = fit_rust_gsnh(&recursive, config(LanguageFamily::ConjUI, 2)).unwrap();
        assert!(deep_fit.model.summary.n_nodes > 3);
        assert_eq!(deep_fit.model.summary.max_depth, 2);
    }

    #[test]
    fn fit_result_summary_and_training_accuracy_match_tree_helpers() {
        let ds = recursive_dataset();
        let fit = fit_rust_gsnh(&ds, config(LanguageFamily::ConjUI, 2)).unwrap();
        assert_eq!(fit.model.summary, summarize_tree(&fit.model.tree));
        assert_eq!(
            fit.training_accuracy,
            training_accuracy(&fit.model.tree, &ds).unwrap()
        );
        assert_eq!(summarize_rust_gsnh(&fit.model), fit.model.summary);
    }

    #[test]
    fn predict_and_score_return_binary_predictions_and_accuracy() {
        let ds = and_dataset();
        let fit = fit_rust_gsnh(&ds, config(LanguageFamily::ConjUI, 1)).unwrap();
        let predictions = predict_rust_gsnh(&fit.model, &ds).unwrap();
        assert_eq!(predictions, vec![0, 0, 0, 1]);
        assert_eq!(score_rust_gsnh(&fit.model, &ds).unwrap(), 1.0);
    }

    #[test]
    fn invalid_config_values_return_errors() {
        let ds = and_dataset();
        let mut bad = config(LanguageFamily::ConjUI, 1);
        bad.max_arity = 0;
        assert!(fit_rust_gsnh(&ds, bad).is_err());

        bad = config(LanguageFamily::ConjUI, 1);
        bad.min_samples_split = 0;
        assert!(fit_rust_gsnh(&ds, bad).is_err());
    }

    #[test]
    fn selected_family_configs_fit_through_stable_api() {
        assert!(fit_rust_gsnh(&and_dataset(), config(LanguageFamily::ConjUI, 1)).is_ok());
        assert!(fit_rust_gsnh(&horn_dataset(), config(LanguageFamily::Horn, 1)).is_ok());
        assert!(fit_rust_gsnh(&antihorn_dataset(), config(LanguageFamily::AntiHorn, 1)).is_ok());
        assert!(fit_rust_gsnh(&xor_dataset(), config(LanguageFamily::Affine, 1)).is_ok());
        assert!(fit_rust_gsnh(&square_dataset(), config(LanguageFamily::Square2CNF, 1)).is_ok());
    }
}
