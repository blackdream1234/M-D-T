//! Label masks and scoring formulas for the incremental Rust GSNH engine.
//!
//! This module mirrors the Python scoring code in `src/gsnh_mdt/scoring/`:
//! binary entropy, information gain, gain ratio, and BIC-style penalized gain.
//! Python remains the reference/oracle; Rust scoring is kept small and tested on
//! deterministic examples before any candidate search is implemented.

use crate::{BitSet, Dataset};

/// Binary class counts for a mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassCounts {
    pub positive: usize,
    pub negative: usize,
}

impl ClassCounts {
    #[inline]
    pub fn total(self) -> usize {
        self.positive + self.negative
    }
}

/// Mask containing all rows with label `1`.
pub fn positive_label_mask(dataset: &Dataset) -> BitSet {
    let mut mask = BitSet::new(dataset.n_samples());
    for (idx, &label) in dataset.labels().iter().enumerate() {
        if label == 1 {
            // idx is constructed from the dataset length, so this cannot fail.
            mask.set(idx).expect("dataset row index must be in range");
        }
    }
    mask
}

/// Mask containing all rows with label `0`.
pub fn negative_label_mask(dataset: &Dataset) -> BitSet {
    let mut mask = BitSet::new(dataset.n_samples());
    for (idx, &label) in dataset.labels().iter().enumerate() {
        if label == 0 {
            // idx is constructed from the dataset length, so this cannot fail.
            mask.set(idx).expect("dataset row index must be in range");
        }
    }
    mask
}

/// Count positive labels in `mask` using a precomputed positive-label mask.
pub fn count_positive(mask: &BitSet, positive_mask: &BitSet) -> Result<usize, String> {
    Ok(mask.intersection(positive_mask)?.count_ones())
}

/// Count negative labels in `mask` using a precomputed negative-label mask.
pub fn count_negative(mask: &BitSet, negative_mask: &BitSet) -> Result<usize, String> {
    Ok(mask.intersection(negative_mask)?.count_ones())
}

/// Count positive and negative labels in `mask`.
pub fn class_counts(
    mask: &BitSet,
    positive_mask: &BitSet,
    negative_mask: &BitSet,
) -> Result<ClassCounts, String> {
    if positive_mask.len() != negative_mask.len() {
        return Err(format!(
            "label mask length mismatch: positive len {} != negative len {}",
            positive_mask.len(),
            negative_mask.len()
        ));
    }
    Ok(ClassCounts {
        positive: count_positive(mask, positive_mask)?,
        negative: count_negative(mask, negative_mask)?,
    })
}

/// Binary entropy `H(pos, neg)`.
///
/// Matches Python `entropy(pos, neg)`: empty and pure counts return `0.0`, and
/// the logarithm base is 2.
pub fn entropy(counts: ClassCounts) -> f64 {
    let total = counts.total() as f64;
    if total <= 0.0 {
        return 0.0;
    }
    let p = counts.positive as f64 / total;
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

/// Information gain for a binary split.
///
/// Matches Python `information_gain(total_pos, total_neg, in_pos, in_neg)`:
/// invalid empty parent/inside/outside splits return `-1.0`, while valid gains
/// are clamped at zero from below.
pub fn information_gain(parent: ClassCounts, inside: ClassCounts) -> Result<f64, String> {
    if inside.positive > parent.positive || inside.negative > parent.negative {
        return Err(format!(
            "inside counts {:?} exceed parent counts {:?}",
            inside, parent
        ));
    }

    let total = parent.total();
    let in_total = inside.total();
    let out_total = total.saturating_sub(in_total);

    if in_total == 0 || out_total == 0 || total == 0 {
        return Ok(-1.0);
    }

    let outside = ClassCounts {
        positive: parent.positive - inside.positive,
        negative: parent.negative - inside.negative,
    };

    let total_f = total as f64;
    let gain = entropy(parent)
        - (in_total as f64 / total_f) * entropy(inside)
        - (out_total as f64 / total_f) * entropy(outside);
    Ok(gain.max(0.0))
}

/// Gain ratio: information gain normalized by split information.
///
/// Matches Python `gain_ratio`: if information gain is not positive, return
/// `-1.0`; if split information is tiny, return raw information gain.
pub fn gain_ratio(parent: ClassCounts, inside: ClassCounts) -> Result<f64, String> {
    let ig = information_gain(parent, inside)?;
    if ig <= 0.0 {
        return Ok(-1.0);
    }

    let total = parent.total();
    let in_total = inside.total();
    let out_total = total.saturating_sub(in_total);
    if total == 0 || in_total == 0 || out_total == 0 {
        return Ok(-1.0);
    }

    let p_in = in_total as f64 / total as f64;
    let p_out = out_total as f64 / total as f64;
    let split_info = -(p_in * p_in.log2() + p_out * p_out.log2());

    if split_info <= 1e-10 {
        Ok(ig)
    } else {
        Ok(ig / split_info)
    }
}

/// BIC-style penalized gain.
///
/// Matches Python `penalized_gain`: invalid/nonpositive raw gains or sample
/// counts return `-1.0`; otherwise subtract `k * ln(N) / (2N)` where
/// `k = arity + 1`, and return `-1.0` if the penalized value is nonpositive.
pub fn penalized_gain(raw_gain: f64, arity: usize, n_samples: usize) -> f64 {
    if raw_gain <= 0.0 || n_samples == 0 {
        return -1.0;
    }
    let k = arity as f64 + 1.0;
    let n = (n_samples as f64).max(2.0);
    let bic_penalty = k * n.ln() / (2.0 * n_samples as f64);
    let penalized = raw_gain - bic_penalty;
    if penalized <= 0.0 {
        -1.0
    } else {
        penalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComparisonOp, ThresholdPredicate};

    const EPS: f64 = 1e-12;

    fn dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
            ],
            vec![0, 1, 1, 0, 1, 0],
        )
        .unwrap()
    }

    #[test]
    fn builds_positive_and_negative_label_masks() {
        let ds = dataset();
        assert_eq!(positive_label_mask(&ds).indices(), vec![1, 2, 4]);
        assert_eq!(negative_label_mask(&ds).indices(), vec![0, 3, 5]);
    }

    #[test]
    fn class_counts_on_all_rows() {
        let ds = dataset();
        let all = BitSet::with_all(ds.n_samples());
        let pos = positive_label_mask(&ds);
        let neg = negative_label_mask(&ds);
        assert_eq!(
            class_counts(&all, &pos, &neg).unwrap(),
            ClassCounts {
                positive: 3,
                negative: 3
            }
        );
    }

    #[test]
    fn class_counts_on_predicate_subset() {
        let ds = dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 3.0,
            op: ComparisonOp::LessThan,
        };
        let mask = pred.evaluate_mask(&ds).unwrap();
        let pos = positive_label_mask(&ds);
        let neg = negative_label_mask(&ds);
        assert_eq!(mask.indices(), vec![0, 1, 2]);
        assert_eq!(
            class_counts(&mask, &pos, &neg).unwrap(),
            ClassCounts {
                positive: 2,
                negative: 1
            }
        );
        assert_eq!(count_positive(&mask, &pos).unwrap(), 2);
        assert_eq!(count_negative(&mask, &neg).unwrap(), 1);
    }

    #[test]
    fn class_counts_length_mismatch_errors() {
        let mask = BitSet::new(3);
        let pos = BitSet::new(3);
        let neg = BitSet::new(4);
        assert!(class_counts(&mask, &pos, &neg).is_err());
        assert!(count_positive(&mask, &neg).is_err());
    }

    #[test]
    fn pure_and_empty_subsets_count_correctly() {
        let ds = dataset();
        let pos = positive_label_mask(&ds);
        let neg = negative_label_mask(&ds);
        let pure_pos = BitSet::from_indices(ds.n_samples(), &[1, 2, 4]).unwrap();
        let pure_neg = BitSet::from_indices(ds.n_samples(), &[0, 3, 5]).unwrap();
        let empty = BitSet::new(ds.n_samples());

        assert_eq!(
            class_counts(&pure_pos, &pos, &neg).unwrap(),
            ClassCounts {
                positive: 3,
                negative: 0
            }
        );
        assert_eq!(
            class_counts(&pure_neg, &pos, &neg).unwrap(),
            ClassCounts {
                positive: 0,
                negative: 3
            }
        );
        assert_eq!(
            class_counts(&empty, &pos, &neg).unwrap(),
            ClassCounts {
                positive: 0,
                negative: 0
            }
        );
    }

    #[test]
    fn entropy_matches_python_formula() {
        assert!(
            (entropy(ClassCounts {
                positive: 1,
                negative: 1
            }) - 1.0)
                .abs()
                < EPS
        );
        assert_eq!(
            entropy(ClassCounts {
                positive: 0,
                negative: 0
            }),
            0.0
        );
        assert_eq!(
            entropy(ClassCounts {
                positive: 4,
                negative: 0
            }),
            0.0
        );
        assert_eq!(
            entropy(ClassCounts {
                positive: 0,
                negative: 4
            }),
            0.0
        );

        let expected = 0.9182958340544896_f64;
        assert!(
            (entropy(ClassCounts {
                positive: 1,
                negative: 2
            }) - expected)
                .abs()
                < EPS
        );
    }

    #[test]
    fn information_gain_matches_python_formula() {
        let parent = ClassCounts {
            positive: 3,
            negative: 3,
        };
        let inside = ClassCounts {
            positive: 2,
            negative: 0,
        };
        let expected = 0.4591479170272448_f64;
        assert!((information_gain(parent, inside).unwrap() - expected).abs() < EPS);
    }

    #[test]
    fn information_gain_invalid_splits_match_python_minus_one() {
        let parent = ClassCounts {
            positive: 3,
            negative: 3,
        };
        assert_eq!(
            information_gain(
                parent,
                ClassCounts {
                    positive: 0,
                    negative: 0
                }
            )
            .unwrap(),
            -1.0
        );
        assert_eq!(information_gain(parent, parent).unwrap(), -1.0);
        assert_eq!(
            information_gain(
                ClassCounts {
                    positive: 0,
                    negative: 0
                },
                ClassCounts {
                    positive: 0,
                    negative: 0
                }
            )
            .unwrap(),
            -1.0
        );
        assert!(information_gain(
            parent,
            ClassCounts {
                positive: 4,
                negative: 0
            }
        )
        .is_err());
    }

    #[test]
    fn gain_ratio_matches_python_formula() {
        let parent = ClassCounts {
            positive: 3,
            negative: 3,
        };
        let inside = ClassCounts {
            positive: 2,
            negative: 0,
        };
        assert!((gain_ratio(parent, inside).unwrap() - 0.5).abs() < EPS);
        assert_eq!(
            gain_ratio(
                parent,
                ClassCounts {
                    positive: 0,
                    negative: 0
                }
            )
            .unwrap(),
            -1.0
        );
    }

    #[test]
    fn penalized_gain_matches_python_formula() {
        let raw_gain = 0.5;
        let expected = raw_gain - (3.0_f64 * 100.0_f64.ln() / 200.0_f64);
        assert!((penalized_gain(raw_gain, 2, 100) - expected).abs() < EPS);
        assert_eq!(penalized_gain(0.0, 2, 100), -1.0);
        assert_eq!(penalized_gain(0.5, 2, 0), -1.0);
        assert_eq!(penalized_gain(0.001, 2, 10), -1.0);
    }

    #[test]
    fn deterministic_scoring_from_predicate_mask() {
        let ds = dataset();
        let pred = ThresholdPredicate {
            feature_index: 0,
            threshold: 3.0,
            op: ComparisonOp::LessThan,
        };
        let mask = pred.evaluate_mask(&ds).unwrap();
        let pos = positive_label_mask(&ds);
        let neg = negative_label_mask(&ds);
        let parent = class_counts(&BitSet::with_all(ds.n_samples()), &pos, &neg).unwrap();
        let inside = class_counts(&mask, &pos, &neg).unwrap();

        assert_eq!(
            parent,
            ClassCounts {
                positive: 3,
                negative: 3
            }
        );
        assert_eq!(
            inside,
            ClassCounts {
                positive: 2,
                negative: 1
            }
        );
        assert!((information_gain(parent, inside).unwrap() - 0.08170416594551044_f64).abs() < EPS);
    }
}
