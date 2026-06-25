//! Dataset representation for the Rust GSNH engine.
//!
//! Phase 1 keeps this deliberately small and auditable: a dense row-major
//! `f64` matrix plus binary `u8` labels.  Python is still responsible for
//! production data loading and preprocessing; this module mirrors the contract
//! needed by the Python `.dl8` loader so later PyO3 bindings can compare both
//! engines on exactly the same arrays.

use std::error::Error;
use std::fmt;

/// Per-feature descriptive statistics used by search/pruning layers.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSummary {
    pub min: f64,
    pub max: f64,
    pub is_constant: bool,
    pub is_binary: bool,
}

/// Dense, row-major, binary-labelled dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    n_samples: usize,
    n_features: usize,
    features: Vec<f64>,
    labels: Vec<u8>,
    positive_count: usize,
    feature_summaries: Vec<FeatureSummary>,
}

/// Validation and parsing errors for [`Dataset`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    Empty,
    LabelLengthMismatch {
        rows: usize,
        labels: usize,
    },
    RaggedRows {
        expected: usize,
        found: usize,
        row: usize,
    },
    NonBinaryLabel {
        label: u8,
        row: usize,
    },
    Dl8LineTooShort {
        line: usize,
    },
    Dl8Parse {
        line: usize,
        token: String,
    },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::Empty => {
                write!(f, "dataset must contain at least one row and one feature")
            }
            DatasetError::LabelLengthMismatch { rows, labels } => {
                write!(
                    f,
                    "label length mismatch: {rows} feature rows but {labels} labels"
                )
            }
            DatasetError::RaggedRows {
                expected,
                found,
                row,
            } => {
                write!(
                    f,
                    "ragged row {row}: expected {expected} features, found {found}"
                )
            }
            DatasetError::NonBinaryLabel { label, row } => {
                write!(f, "non-binary label {label} at row {row}; expected 0 or 1")
            }
            DatasetError::Dl8LineTooShort { line } => {
                write!(f, "line {line}: expected label plus at least one feature")
            }
            DatasetError::Dl8Parse { line, token } => {
                write!(
                    f,
                    "line {line}: could not parse token '{token}' as a number"
                )
            }
        }
    }
}

impl Error for DatasetError {}

impl Dataset {
    /// Build a dataset from dense rows and binary labels.
    pub fn from_rows(rows: Vec<Vec<f64>>, labels: Vec<u8>) -> Result<Self, DatasetError> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(DatasetError::Empty);
        }
        if rows.len() != labels.len() {
            return Err(DatasetError::LabelLengthMismatch {
                rows: rows.len(),
                labels: labels.len(),
            });
        }

        let n_samples = rows.len();
        let n_features = rows[0].len();
        let mut features = Vec::with_capacity(n_samples * n_features);
        let mut positive_count = 0usize;

        for (row_idx, label) in labels.iter().copied().enumerate() {
            match label {
                0 => {}
                1 => positive_count += 1,
                other => {
                    return Err(DatasetError::NonBinaryLabel {
                        label: other,
                        row: row_idx,
                    })
                }
            }
        }

        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != n_features {
                return Err(DatasetError::RaggedRows {
                    expected: n_features,
                    found: row.len(),
                    row: row_idx,
                });
            }
            features.extend_from_slice(row);
        }

        let feature_summaries = summarize_features(n_samples, n_features, &features);

        Ok(Self {
            n_samples,
            n_features,
            features,
            labels,
            positive_count,
            feature_summaries,
        })
    }

    /// Parse `.dl8` text using the Python benchmark convention:
    /// first column is the label, remaining columns are features.
    pub fn from_dl8_text(text: &str) -> Result<Self, DatasetError> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<u8> = Vec::new();

        for (zero_line, raw) in text.lines().enumerate() {
            let line_number = zero_line + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() < 2 {
                return Err(DatasetError::Dl8LineTooShort { line: line_number });
            }

            let label = parse_u8_token(tokens[0], line_number)?;
            let mut row = Vec::with_capacity(tokens.len() - 1);
            for token in &tokens[1..] {
                let value = token.parse::<f64>().map_err(|_| DatasetError::Dl8Parse {
                    line: line_number,
                    token: (*token).to_string(),
                })?;
                row.push(value);
            }
            labels.push(label);
            rows.push(row);
        }

        Self::from_rows(rows, labels)
    }

    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[inline]
    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    #[inline]
    pub fn features_row_major(&self) -> &[f64] {
        &self.features
    }

    #[inline]
    pub fn value(&self, row: usize, feature: usize) -> f64 {
        assert!(row < self.n_samples, "row index out of bounds");
        assert!(feature < self.n_features, "feature index out of bounds");
        self.features[row * self.n_features + feature]
    }

    pub fn row(&self, row: usize) -> &[f64] {
        assert!(row < self.n_samples, "row index out of bounds");
        let start = row * self.n_features;
        &self.features[start..start + self.n_features]
    }

    pub fn column(&self, feature: usize) -> Vec<f64> {
        assert!(feature < self.n_features, "feature index out of bounds");
        (0..self.n_samples)
            .map(|row| self.value(row, feature))
            .collect()
    }

    #[inline]
    pub fn positive_count(&self) -> usize {
        self.positive_count
    }

    #[inline]
    pub fn negative_count(&self) -> usize {
        self.n_samples - self.positive_count
    }

    #[inline]
    pub fn positive_rate(&self) -> f64 {
        self.positive_count as f64 / self.n_samples as f64
    }

    #[inline]
    pub fn feature_summaries(&self) -> &[FeatureSummary] {
        &self.feature_summaries
    }
}

fn parse_u8_token(token: &str, line: usize) -> Result<u8, DatasetError> {
    token.parse::<u8>().map_err(|_| DatasetError::Dl8Parse {
        line,
        token: token.to_string(),
    })
}

fn summarize_features(
    n_samples: usize,
    n_features: usize,
    features: &[f64],
) -> Vec<FeatureSummary> {
    (0..n_features)
        .map(|feature| {
            let first = features[feature];
            let mut min = first;
            let mut max = first;
            let mut is_binary = true;
            for row in 0..n_samples {
                let value = features[row * n_features + feature];
                min = min.min(value);
                max = max.max(value);
                if value != 0.0 && value != 1.0 {
                    is_binary = false;
                }
            }
            FeatureSummary {
                min,
                max,
                is_constant: min == max,
                is_binary,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_dense_row_major_dataset() {
        let ds = Dataset::from_rows(
            vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![0, 1, 1],
        )
        .unwrap();

        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.labels(), &[0, 1, 1]);
        assert_eq!(ds.features_row_major(), &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(ds.value(1, 0), 1.0);
        assert_eq!(ds.row(2), &[1.0, 1.0]);
        assert_eq!(ds.column(1), vec![1.0, 0.0, 1.0]);
        assert_eq!(ds.positive_count(), 2);
        assert_eq!(ds.negative_count(), 1);
        assert!((ds.positive_rate() - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn summarizes_features() {
        let ds = Dataset::from_rows(
            vec![
                vec![0.0, 7.0, 2.0],
                vec![1.0, 7.0, 3.0],
                vec![0.0, 7.0, 4.0],
            ],
            vec![0, 1, 0],
        )
        .unwrap();
        let summaries = ds.feature_summaries();

        assert_eq!(
            summaries[0],
            FeatureSummary {
                min: 0.0,
                max: 1.0,
                is_constant: false,
                is_binary: true
            }
        );
        assert_eq!(
            summaries[1],
            FeatureSummary {
                min: 7.0,
                max: 7.0,
                is_constant: true,
                is_binary: false
            }
        );
        assert_eq!(
            summaries[2],
            FeatureSummary {
                min: 2.0,
                max: 4.0,
                is_constant: false,
                is_binary: false
            }
        );
    }

    #[test]
    fn rejects_invalid_shapes_and_labels() {
        assert_eq!(
            Dataset::from_rows(vec![], vec![]).unwrap_err(),
            DatasetError::Empty
        );
        assert!(matches!(
            Dataset::from_rows(vec![vec![1.0], vec![2.0]], vec![1]).unwrap_err(),
            DatasetError::LabelLengthMismatch { rows: 2, labels: 1 }
        ));
        assert!(matches!(
            Dataset::from_rows(vec![vec![1.0], vec![2.0, 3.0]], vec![0, 1]).unwrap_err(),
            DatasetError::RaggedRows {
                expected: 1,
                found: 2,
                row: 1
            }
        ));
        assert!(matches!(
            Dataset::from_rows(vec![vec![1.0]], vec![2]).unwrap_err(),
            DatasetError::NonBinaryLabel { label: 2, row: 0 }
        ));
    }

    #[test]
    fn parses_dl8_like_python_loader_contract() {
        let text = "# label f0 f1\n0 1 2\n1 3 4\n\n0 5 6\n";
        let ds = Dataset::from_dl8_text(text).unwrap();

        assert_eq!(ds.labels(), &[0, 1, 0]);
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.features_row_major(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
