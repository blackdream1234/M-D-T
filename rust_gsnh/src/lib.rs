//! Incremental Rust engine for GSNH-MDT.
//!
//! This crate is intentionally introduced beside the Python implementation.
//! Python remains the reference/oracle while Rust modules are added and tested
//! one layer at a time.

pub mod affine;
pub mod antihorn;
pub mod bitset;
pub mod cache;
pub mod conjui;
pub mod data;
pub mod family;
pub mod horn;
pub mod predicates;
pub mod scoring;
pub mod search;
pub mod square_cnf;
pub mod tree;

pub use affine::best_affine_split_with_min_leaf;
pub use antihorn::best_antihorn_split_with_min_leaf;
pub use bitset::BitSet;
pub use conjui::best_conjui_split_with_min_leaf;
pub use data::{Dataset, DatasetError, FeatureSummary};
pub use family::{
    best_family_split, evaluate_fixed_predicate_with_min_leaf, BestFamilySplit,
    EvaluatedFixedPredicate, FamilySearchConfig, FixedPredicate, LanguageFamily,
};
pub use horn::best_horn_split_with_min_leaf;
pub use predicates::{ComparisonOp, ComposedPredicate, MaskOp, ThresholdPredicate};
pub use scoring::{
    class_counts, count_negative, count_positive, entropy, gain_ratio, information_gain,
    negative_label_mask, penalized_gain, positive_label_mask, ClassCounts,
};
pub use search::{
    best_1d_split, best_1d_split_for_feature, best_1d_split_for_feature_with_min_leaf,
    best_1d_split_with_min_leaf, evaluate_1d_candidate, evaluate_1d_candidate_with_min_leaf,
    evaluate_affine_candidate_with_min_leaf, evaluate_antihorn_candidate_with_min_leaf,
    evaluate_composed_candidate_with_min_leaf, evaluate_horn_candidate_with_min_leaf,
    generate_1d_thresholds, BestSplit, ComposedCandidate, EvaluatedComposedPredicate,
    SplitCandidate,
};

pub use square_cnf::{
    best_square2cnf_split_with_min_leaf, evaluate_square2cnf_candidate_with_min_leaf,
    EvaluatedSquare2CNFPredicate, Square2CNFCandidate, Square2CNFPredicate, Square2Clause,
};

pub use tree::{
    accuracy_from_predictions, accuracy_from_u8_predictions, build_stump_with_family,
    build_tree_with_family, count_tree_internal_nodes, count_tree_leaves, count_tree_nodes,
    majority_leaf_from_mask, observed_tree_depth, predict_stump, predict_stump_row, predict_tree,
    predict_tree_row, prediction_label_to_u8, prediction_labels_to_u8, stump_accuracy,
    summarize_tree, training_accuracy, tree_accuracy, DecisionNode, DecisionTree, LeafNode,
    PredictionLabel, StumpNode, StumpTree, TreeBuildConfig, TreeSummary,
};
