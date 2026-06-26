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

/// Boolean mask-composition operator for Python `GSNHPredicate` semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// ConjUI / box semantics: all literal masks must be true.
    And,
    /// Horn and AntiHorn clause semantics: at least one literal mask is true.
    Or,
    /// Affine/XOR auxiliary semantics: odd parity of literal masks is true.
    Xor,
}

/// Predicate made by composing threshold literals with one Python-matched mask operator.
///
/// This is a generic mask layer only.  It does not validate Horn/AntiHorn
/// polarity restrictions, does not implement Square2CNF clauses, and does not
/// make any theorem-certification claim.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposedPredicate {
    pub literals: Vec<ThresholdPredicate>,
    pub op: MaskOp,
}

impl ComposedPredicate {
    /// Evaluate a composed predicate and return the true row indices as a `BitSet`.
    ///
    /// Empty predicates return `Err`, matching Python `GSNHPredicate`, which
    /// rejects arity 0 during construction.  Invalid literal feature indices and
    /// unsupported literal operators are propagated from `ThresholdPredicate`.
    pub fn evaluate_mask(&self, dataset: &Dataset) -> Result<BitSet, String> {
        let Some((first, rest)) = self.literals.split_first() else {
            return Err("composed predicate must contain at least one literal".to_string());
        };

        let mut result = first.evaluate_mask(dataset)?;
        for literal in rest {
            let mask = literal.evaluate_mask(dataset)?;
            result = match self.op {
                MaskOp::And => result.intersection(&mask)?,
                MaskOp::Or => result.union(&mask)?,
                MaskOp::Xor => xor_masks(&result, &mask)?,
            };
        }
        Ok(result)
    }
}

fn xor_masks(left: &BitSet, right: &BitSet) -> Result<BitSet, String> {
    let left_only = left.difference(right)?;
    let right_only = right.difference(left)?;
    left_only.union(&right_only)
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

    #[test]
    fn composed_and_matches_python_conjui_semantics_for_two_literals() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 8.0,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
        };

        // Python ConjUI uses literal_mask_0 & literal_mask_1.
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![1, 2]);
    }

    #[test]
    fn composed_and_over_three_literals_is_deterministic() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 4.0,
                    op: ComparisonOp::LessThan,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 7.0,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
        };

        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![1, 2, 3]);
    }

    #[test]
    fn composed_or_matches_python_horn_antihorn_clause_semantics() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::LessThan,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 7.0,
                    op: ComparisonOp::LessThan,
                },
            ],
        };

        // Python Horn/AntiHorn `GSNHPredicate.evaluate` ORs literal masks.
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![0, 4]);
    }

    #[test]
    fn composed_or_over_three_literals_is_sorted_and_deterministic() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::LessThan,
                },
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 3.0,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 8.0,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
        };

        assert_eq!(
            pred.evaluate_mask(&ds).unwrap().indices(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn composed_xor_matches_python_affine_auxiliary_parity_semantics() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::Xor,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 2.0,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 8.0,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
        };

        // Odd parity of [rows 2,3,4] XOR [rows 0,1,2] is [0,1,3,4].
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![0, 1, 3, 4]);
    }

    #[test]
    fn composed_xor_over_three_literals_uses_odd_parity() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::Xor,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 4.0,
                    op: ComparisonOp::LessThan,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 8.0,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
        };

        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![1, 2, 4]);
    }

    #[test]
    fn composed_predicate_rejects_invalid_literal_feature_index() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: vec![ThresholdPredicate {
                feature_index: 99,
                threshold: 0.0,
                op: ComparisonOp::LessThan,
            }],
        };

        assert!(pred
            .evaluate_mask(&ds)
            .unwrap_err()
            .contains("out of range"));
    }

    #[test]
    fn composed_predicate_rejects_empty_predicate_like_python_arity_check() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: Vec::new(),
        };

        assert!(pred
            .evaluate_mask(&ds)
            .unwrap_err()
            .contains("at least one"));
    }

    #[test]
    fn composed_mask_feeds_class_counts() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::And,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::GreaterEqual,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 8.0,
                    op: ComparisonOp::GreaterEqual,
                },
            ],
        };
        let mask = pred.evaluate_mask(&ds).unwrap();
        let counts = crate::class_counts(
            &mask,
            &crate::positive_label_mask(&ds),
            &crate::negative_label_mask(&ds),
        )
        .unwrap();

        assert_eq!(mask.indices(), vec![1, 2]);
        assert_eq!(counts.positive, 1);
        assert_eq!(counts.negative, 1);
    }

    #[test]
    fn composed_mask_supports_min_leaf_checks_without_extra_api() {
        let ds = tiny_dataset();
        let pred = ComposedPredicate {
            op: MaskOp::Or,
            literals: vec![
                ThresholdPredicate {
                    feature_index: 0,
                    threshold: 1.0,
                    op: ComparisonOp::LessThan,
                },
                ThresholdPredicate {
                    feature_index: 1,
                    threshold: 7.0,
                    op: ComparisonOp::LessThan,
                },
            ],
        };
        let inside = pred.evaluate_mask(&ds).unwrap();
        let outside = inside.complement();

        assert_eq!(inside.count_ones(), 2);
        assert_eq!(outside.count_ones(), 3);
        assert!(inside.count_ones() >= 2 && outside.count_ones() >= 2);
        assert!(!(inside.count_ones() >= 3 && outside.count_ones() >= 3));
    }
}
