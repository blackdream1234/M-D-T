//! Incremental Rust engine for GSNH-MDT.
//!
//! This crate is intentionally introduced beside the Python implementation.
//! Python remains the reference/oracle while Rust modules are added and tested
//! one layer at a time.

pub mod affine;
pub mod antihorn;
pub mod bitset;
pub mod cache;
pub mod data;
pub mod horn;
pub mod predicates;
pub mod scoring;
pub mod search;
pub mod square_cnf;
pub mod tree;

pub use bitset::BitSet;
pub use data::{Dataset, DatasetError, FeatureSummary};
pub use predicates::{ComparisonOp, ThresholdPredicate};
pub use scoring::{
    class_counts, count_negative, count_positive, entropy, gain_ratio, information_gain,
    negative_label_mask, penalized_gain, positive_label_mask, ClassCounts,
};
