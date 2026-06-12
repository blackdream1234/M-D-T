//! Threshold predicate mask evaluation for the incremental Rust GSNH engine.
//!
//! Python remains the reference/oracle.  This module mirrors the deterministic
//! finite-value comparison behavior used by NumPy-backed Python literals and
//! returns `BitSet` row masks for future split search layers.

use crate::{BitSet, Dataset};

/// Scalar comparison operator for threshold predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    LessEqual,
    LessThan,
    GreaterEqual,
    GreaterThan,
    /// Present for API completeness, but not evaluated yet because the Python
    /// GSNH threshold literal family does not define equality literals.
    Equal,
}

/// Single-feature threshold predicate: `x[feature_index] op threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdPredicate {
    pub feature_index: usize,
    pub threshold: f64,
    pub op: ComparisonOp,
}

impl ThresholdPredicate {
    /// Evaluate the predicate for every row and return the true row indices as
    /// a `BitSet` mask.
    ///
    /// Invalid feature indices and unsupported operators return `Err` rather
    /// than panicking.  This function assumes ordinary finite `f64` values; for
    /// NaN, Rust comparisons match Python/NumPy by evaluating ordered
    /// comparisons to false.
    pub fn evaluate_mask(&self, dataset: &Dataset) -> Result<BitSet, String> {
        if self.feature_index >= dataset.n_features() {
            return Err(format!(
                "feature index {} out of range for dataset with {} features",
                self.feature_index,
                dataset.n_features()
            ));
        }
        if self.op == ComparisonOp::Equal {
            return Err(
                "ComparisonOp::Equal is not implemented because Python GSNH threshold literals do not define equality semantics"
                    .to_string(),
            );
        }

        let mut mask = BitSet::new(dataset.n_samples());
        for row in 0..dataset.n_samples() {
            let value = dataset.value(row, self.feature_index);
            if self.evaluate_value(value)? {
                // Safe because row is constructed in 0..n_samples.
                mask.set(row)?;
            }
        }
        Ok(mask)
    }

    fn evaluate_value(&self, value: f64) -> Result<bool, String> {
        match self.op {
            ComparisonOp::LessEqual => Ok(value <= self.threshold),
            ComparisonOp::LessThan => Ok(value < self.threshold),
            ComparisonOp::GreaterEqual => Ok(value >= self.threshold),
            ComparisonOp::GreaterThan => Ok(value > self.threshold),
            ComparisonOp::Equal => Err(
                "ComparisonOp::Equal is not implemented because Python GSNH threshold literals do not define equality semantics"
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 10.0],
                vec![1.0, 9.0],
                vec![2.0, 8.0],
                vec![3.0, 7.0],
                vec![4.0, 6.0],
            ],
            vec![0, 1, 0, 1, 1],
        )
        .unwrap()
    }

    #[test]
    fn evaluates_less_equal_threshold() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 2.0,
            op: ComparisonOp::LessEqual,
        };
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![0, 1, 2]);
    }

    #[test]
    fn evaluates_less_than_threshold() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 2.0,
            op: ComparisonOp::LessThan,
        };
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![0, 1]);
    }

    #[test]
    fn evaluates_greater_equal_threshold() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 1,
            threshold: 8.0,
            op: ComparisonOp::GreaterEqual,
        };
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![0, 1, 2]);
    }

    #[test]
    fn evaluates_greater_than_threshold() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 1,
            threshold: 8.0,
            op: ComparisonOp::GreaterThan,
        };
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![0, 1]);
    }

    #[test]
    fn rejects_equal_until_python_semantics_exist() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 2.0,
            op: ComparisonOp::Equal,
        };
        let err = pred.evaluate_mask(&ds).unwrap_err();
        assert!(err.contains("equality semantics"));
    }

    #[test]
    fn rejects_invalid_feature_index() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 2,
            threshold: 0.0,
            op: ComparisonOp::LessThan,
        };
        let err = pred.evaluate_mask(&ds).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn deterministic_mask_indices_across_rows_and_features() {
        let ds = tiny_dataset();
        let low_first = ThresholdPredicate {
            feature_index: 0,
            threshold: 3.5,
            op: ComparisonOp::LessThan,
        };
        let high_second = ThresholdPredicate {
            feature_index: 1,
            threshold: 7.0,
            op: ComparisonOp::GreaterEqual,
        };

        assert_eq!(
            low_first.evaluate_mask(&ds).unwrap().indices(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            high_second.evaluate_mask(&ds).unwrap().indices(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn nan_ordered_comparisons_match_numpy_false_behavior() {
        let ds = Dataset::from_rows(vec![vec![f64::NAN], vec![1.0]], vec![0, 1]).unwrap();
        let ge = ThresholdPredicate {
            feature_index: 0,
            threshold: 0.0,
            op: ComparisonOp::GreaterEqual,
        };
        let lt = ThresholdPredicate {
            feature_index: 0,
            threshold: 2.0,
            op: ComparisonOp::LessThan,
        };

        assert_eq!(ge.evaluate_mask(&ds).unwrap().indices(), vec![1]);
        assert_eq!(lt.evaluate_mask(&ds).unwrap().indices(), vec![1]);
    }

    #[test]
    fn equivalence_style_test_against_documented_python_literal_semantics() {
        let ds = tiny_dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 2.0,
            op: ComparisonOp::GreaterEqual,
        };

        // Mirrors Python GSNHLiteral.evaluate for LiteralPolarity.GE:
        // X[:, feature] >= threshold.
        let python_reference_indices: Vec<usize> = (0..ds.n_samples())
            .filter(|&row| ds.value(row, 0) >= 2.0)
            .collect();

        assert_eq!(
            pred.evaluate_mask(&ds).unwrap().indices(),
            python_reference_indices
        );
    }
}
