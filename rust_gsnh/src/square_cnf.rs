//! Square2CNF fixed-predicate mask evaluation for the incremental Rust engine.
//!
//! Python remains the reference/oracle.  This module mirrors Python
//! `Square2CNFPredicate` mask semantics only: a predicate is a conjunction of
//! one to three two-literal disjunctive clauses, `(a OR b) AND ...`.  It does
//! not enumerate Square2CNF candidates, build theorem certificates, or perform
//! tree recursion.

use crate::{
    class_counts, information_gain, negative_label_mask, penalized_gain, positive_label_mask,
    BitSet, ClassCounts, Dataset, ThresholdPredicate,
};

/// One Square2CNF clause `(left OR right)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Square2Clause {
    pub left: ThresholdPredicate,
    pub right: ThresholdPredicate,
}

impl Square2Clause {
    /// Evaluate this clause as the union of its two literal masks.
    pub fn evaluate_mask(&self, dataset: &Dataset) -> Result<BitSet, String> {
        let left = self.left.evaluate_mask(dataset)?;
        let right = self.right.evaluate_mask(dataset)?;
        left.union(&right)
    }
}

/// Python-compatible Square2CNF predicate: conjunction of 2-literal OR clauses.
#[derive(Clone, Debug, PartialEq)]
pub struct Square2CNFPredicate {
    pub clauses: Vec<Square2Clause>,
}

impl Square2CNFPredicate {
    /// Evaluate the predicate as `AND` over all clause masks.
    ///
    /// Empty predicates return an error, matching Python `Square2CNFPredicate`,
    /// which rejects clause counts outside `1..=3` at construction time.
    pub fn evaluate_mask(&self, dataset: &Dataset) -> Result<BitSet, String> {
        validate_square2cnf_clause_count(self.clauses.len())?;
        let Some((first, rest)) = self.clauses.split_first() else {
            return Err("Square2CNF predicate must contain 1-3 clauses".to_string());
        };

        let mut result = first.evaluate_mask(dataset)?;
        for clause in rest {
            let clause_mask = clause.evaluate_mask(dataset)?;
            result = result.intersection(&clause_mask)?;
        }
        Ok(result)
    }
}

/// A scored fixed Square2CNF candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Square2CNFCandidate {
    pub predicate: Square2CNFPredicate,
    pub score: f64,
    pub inside_counts: ClassCounts,
    pub outside_counts: ClassCounts,
}

/// Evaluated Square2CNF predicate plus inside/outside row masks.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedSquare2CNFPredicate {
    pub candidate: Square2CNFCandidate,
    pub inside_mask: BitSet,
    pub outside_mask: BitSet,
}

/// Evaluate and score one supplied Square2CNF predicate.
///
/// This function mirrors Python fixed-candidate behavior: evaluate the mask,
/// form the outside branch as the complement, enforce `min_samples_leaf` on
/// both branches, compute information gain, then apply BIC-style
/// `penalized_gain` with the caller-provided arity.  Invalid branch sizes and
/// nonpositive raw/penalized gains return `Ok(None)`.
pub fn evaluate_square2cnf_candidate_with_min_leaf(
    dataset: &Dataset,
    predicate: Square2CNFPredicate,
    min_samples_leaf: usize,
    arity: usize,
) -> Result<Option<EvaluatedSquare2CNFPredicate>, String> {
    validate_square2cnf_clause_count(predicate.clauses.len())?;

    let inside_mask = predicate.evaluate_mask(dataset)?;
    let outside_mask = inside_mask.complement();
    if inside_mask.count_ones() < min_samples_leaf || outside_mask.count_ones() < min_samples_leaf {
        return Ok(None);
    }

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
    if raw_gain <= 0.0 {
        return Ok(None);
    }
    let score = penalized_gain(raw_gain, arity, dataset.n_samples());
    if score <= 0.0 {
        return Ok(None);
    }

    Ok(Some(EvaluatedSquare2CNFPredicate {
        candidate: Square2CNFCandidate {
            predicate,
            score,
            inside_counts,
            outside_counts,
        },
        inside_mask,
        outside_mask,
    }))
}

fn validate_square2cnf_clause_count(n_clauses: usize) -> Result<(), String> {
    if (1..=3).contains(&n_clauses) {
        Ok(())
    } else {
        Err(format!(
            "Square2CNF predicate must contain 1-3 clauses, got {n_clauses}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComparisonOp;

    const EPS: f64 = 1e-12;

    fn dataset() -> Dataset {
        Dataset::from_rows(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![2.0, 2.0, 2.0, 2.0],
                vec![3.0, 3.0, 3.0, 3.0],
                vec![4.0, 4.0, 4.0, 4.0],
                vec![5.0, 5.0, 5.0, 5.0],
            ],
            vec![1, 0, 0, 0, 1, 1],
        )
        .unwrap()
    }

    fn ge(feature_index: usize, threshold: f64) -> ThresholdPredicate {
        ThresholdPredicate {
            feature_index,
            threshold,
            op: ComparisonOp::GreaterEqual,
        }
    }

    fn lt(feature_index: usize, threshold: f64) -> ThresholdPredicate {
        ThresholdPredicate {
            feature_index,
            threshold,
            op: ComparisonOp::LessThan,
        }
    }

    fn false_lit() -> ThresholdPredicate {
        lt(2, -1.0)
    }

    fn true_lit() -> ThresholdPredicate {
        ge(2, 0.0)
    }

    fn one_clause_predicate() -> Square2CNFPredicate {
        Square2CNFPredicate {
            clauses: vec![Square2Clause {
                left: ge(0, 3.0),
                right: lt(1, 1.0),
            }],
        }
    }

    fn two_clause_predicate() -> Square2CNFPredicate {
        Square2CNFPredicate {
            clauses: vec![
                Square2Clause {
                    left: ge(0, 1.0),
                    right: false_lit(),
                },
                Square2Clause {
                    left: lt(1, 4.0),
                    right: false_lit(),
                },
            ],
        }
    }

    fn three_clause_predicate() -> Square2CNFPredicate {
        let mut pred = two_clause_predicate();
        pred.clauses.push(Square2Clause {
            left: true_lit(),
            right: false_lit(),
        });
        pred
    }

    #[test]
    fn one_clause_evaluates_or_mask() {
        let ds = dataset();
        let clause_mask = one_clause_predicate().clauses[0]
            .evaluate_mask(&ds)
            .unwrap();
        assert_eq!(clause_mask.indices(), vec![0, 3, 4, 5]);
        assert_eq!(
            one_clause_predicate().evaluate_mask(&ds).unwrap().indices(),
            vec![0, 3, 4, 5]
        );
    }

    #[test]
    fn two_clauses_evaluate_clause_or_then_final_and_mask() {
        let ds = dataset();
        let pred = two_clause_predicate();
        assert_eq!(
            pred.clauses[0].evaluate_mask(&ds).unwrap().indices(),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(
            pred.clauses[1].evaluate_mask(&ds).unwrap().indices(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![1, 2, 3]);
    }

    #[test]
    fn three_clauses_are_supported_like_python() {
        let ds = dataset();
        let pred = three_clause_predicate();
        assert_eq!(pred.evaluate_mask(&ds).unwrap().indices(), vec![1, 2, 3]);
    }

    #[test]
    fn square2cnf_candidate_returns_masks_counts_and_positive_score() {
        let ds = dataset();
        let pred = two_clause_predicate();
        let evaluated = evaluate_square2cnf_candidate_with_min_leaf(&ds, pred.clone(), 1, 2)
            .unwrap()
            .unwrap();

        assert_eq!(evaluated.inside_mask.indices(), vec![1, 2, 3]);
        assert_eq!(evaluated.outside_mask.indices(), vec![0, 4, 5]);
        assert_eq!(evaluated.candidate.predicate, pred);
        assert_eq!(
            evaluated.candidate.inside_counts,
            ClassCounts {
                positive: 0,
                negative: 3
            }
        );
        assert_eq!(
            evaluated.candidate.outside_counts,
            ClassCounts {
                positive: 3,
                negative: 0
            }
        );
        let expected_score = penalized_gain(1.0, 2, ds.n_samples());
        assert!((evaluated.candidate.score - expected_score).abs() < EPS);
    }

    #[test]
    fn square2cnf_candidate_rejects_nonpositive_gain() {
        let ds = Dataset::from_rows(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![0, 1, 0, 1],
        )
        .unwrap();
        let pred = Square2CNFPredicate {
            clauses: vec![Square2Clause {
                left: lt(0, 2.0),
                right: lt(0, -1.0),
            }],
        };
        assert!(evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 1, 1)
            .unwrap()
            .is_none());
    }

    #[test]
    fn square2cnf_candidate_rejects_inside_branch_too_small() {
        let ds = dataset();
        let pred = Square2CNFPredicate {
            clauses: vec![Square2Clause {
                left: lt(0, 1.0),
                right: false_lit(),
            }],
        };
        assert!(evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 2, 1)
            .unwrap()
            .is_none());
    }

    #[test]
    fn square2cnf_candidate_rejects_outside_branch_too_small() {
        let ds = dataset();
        let pred = Square2CNFPredicate {
            clauses: vec![Square2Clause {
                left: lt(0, 5.0),
                right: false_lit(),
            }],
        };
        assert!(evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 2, 1)
            .unwrap()
            .is_none());
    }

    #[test]
    fn min_samples_leaf_zero_disables_branch_size_rejection() {
        let ds = Dataset::from_rows(
            (0..10).map(|value| vec![value as f64]).collect(),
            vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        .unwrap();
        let pred = Square2CNFPredicate {
            clauses: vec![Square2Clause {
                left: lt(0, 0.5),
                right: lt(0, -1.0),
            }],
        };
        let evaluated = evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 0, 1)
            .unwrap()
            .unwrap();
        assert_eq!(evaluated.inside_mask.indices(), vec![0]);
        assert_eq!(evaluated.outside_mask.count_ones(), 9);
    }

    #[test]
    fn invalid_feature_index_propagates_error() {
        let ds = dataset();
        let pred = Square2CNFPredicate {
            clauses: vec![Square2Clause {
                left: ge(99, 0.0),
                right: false_lit(),
            }],
        };
        assert!(pred.evaluate_mask(&ds).is_err());
        assert!(evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 1, 1).is_err());
    }

    #[test]
    fn empty_clause_list_errors_like_python_constructor_rejection() {
        let ds = dataset();
        let pred = Square2CNFPredicate { clauses: vec![] };
        assert!(pred.evaluate_mask(&ds).is_err());
        assert!(evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 1, 0).is_err());
    }

    #[test]
    fn more_than_three_clauses_errors_like_python_constructor_rejection() {
        let ds = dataset();
        let pred = Square2CNFPredicate {
            clauses: vec![
                Square2Clause {
                    left: true_lit(),
                    right: false_lit(),
                },
                Square2Clause {
                    left: true_lit(),
                    right: false_lit(),
                },
                Square2Clause {
                    left: true_lit(),
                    right: false_lit(),
                },
                Square2Clause {
                    left: true_lit(),
                    right: false_lit(),
                },
            ],
        };
        assert!(pred.evaluate_mask(&ds).is_err());
        assert!(evaluate_square2cnf_candidate_with_min_leaf(&ds, pred, 1, 4).is_err());
    }
}
