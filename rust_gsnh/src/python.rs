//! Minimal PyO3 binding skeleton for the stable Rust GSNH API.
//!
//! The real PyO3 class is kept behind `pyo3-extension` so the repository can
//! still verify `--features python` in environments where PyO3 cannot be
//! downloaded. The helper semantics below are ordinary Rust and remain testable.

use crate::{Dataset, LanguageFamily};

pub fn parse_family_name(family: &str) -> Result<LanguageFamily, String> {
    match family {
        "ConjUI" => Ok(LanguageFamily::ConjUI),
        "Horn" => Ok(LanguageFamily::Horn),
        "AntiHorn" => Ok(LanguageFamily::AntiHorn),
        "Affine" => Ok(LanguageFamily::Affine),
        "Square2CNF" => Ok(LanguageFamily::Square2CNF),
        "Any" | "BestPerNode" | "SquareCNF" => {
            Err(format!("unsupported Rust GSNH family '{family}'"))
        }
        other => Err(format!("unknown Rust GSNH family '{other}'")),
    }
}

pub fn prediction_dataset_from_rows(x: Vec<Vec<f64>>) -> Result<Dataset, String> {
    let labels = vec![0; x.len()];
    Dataset::from_rows(x, labels).map_err(|err| err.to_string())
}

#[cfg(feature = "pyo3-extension")]
mod pyo3_binding {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use crate::{
        fit_rust_gsnh, predict_rust_gsnh, score_rust_gsnh, summarize_rust_gsnh, Dataset,
        RustGsnHConfig, RustGsnHModel,
    };

    use super::parse_family_name;

    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct RustGsnHClassifier {
        config: RustGsnHConfig,
        model: Option<RustGsnHModel>,
    }

    #[pymethods]
    impl RustGsnHClassifier {
        #[new]
        #[pyo3(signature = (family="ConjUI", max_arity=2, max_depth=1, min_samples_leaf=1, min_samples_split=2))]
        fn new(
            family: &str,
            max_arity: usize,
            max_depth: usize,
            min_samples_leaf: usize,
            min_samples_split: usize,
        ) -> PyResult<Self> {
            Ok(Self {
                config: RustGsnHConfig {
                    family: parse_family_name(family).map_err(PyValueError::new_err)?,
                    max_arity,
                    max_depth,
                    min_samples_leaf,
                    min_samples_split,
                },
                model: None,
            })
        }

        fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<u8>) -> PyResult<Self> {
            let dataset = dataset_from_xy(x, y)?;
            let fit = fit_rust_gsnh(&dataset, self.config).map_err(PyValueError::new_err)?;
            self.model = Some(fit.model);
            Ok(self.clone())
        }

        fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<u8>> {
            let model = self.fitted_model()?;
            let dataset = super::prediction_dataset_from_rows(x).map_err(PyValueError::new_err)?;
            predict_rust_gsnh(model, &dataset).map_err(PyValueError::new_err)
        }

        fn score(&self, x: Vec<Vec<f64>>, y: Vec<u8>) -> PyResult<f64> {
            let model = self.fitted_model()?;
            let dataset = dataset_from_xy(x, y)?;
            score_rust_gsnh(model, &dataset).map_err(PyValueError::new_err)
        }

        fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let model = self.fitted_model()?;
            let summary = summarize_rust_gsnh(model);
            let dict = PyDict::new_bound(py);
            dict.set_item("n_nodes", summary.n_nodes)?;
            dict.set_item("n_leaves", summary.n_leaves)?;
            dict.set_item("n_internal_nodes", summary.n_internal_nodes)?;
            dict.set_item("max_depth", summary.max_depth)?;
            Ok(dict)
        }
    }

    impl RustGsnHClassifier {
        fn fitted_model(&self) -> PyResult<&RustGsnHModel> {
            self.model
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("RustGsnHClassifier is not fitted"))
        }
    }

    fn dataset_from_xy(x: Vec<Vec<f64>>, y: Vec<u8>) -> PyResult<Dataset> {
        Dataset::from_rows(x, y).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[pymodule]
    fn _rust_gsnh(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<RustGsnHClassifier>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_family_accepts_supported_and_rejects_unsupported_names() {
        assert_eq!(parse_family_name("ConjUI").unwrap(), LanguageFamily::ConjUI);
        assert_eq!(parse_family_name("Horn").unwrap(), LanguageFamily::Horn);
        assert_eq!(
            parse_family_name("AntiHorn").unwrap(),
            LanguageFamily::AntiHorn
        );
        assert_eq!(parse_family_name("Affine").unwrap(), LanguageFamily::Affine);
        assert_eq!(
            parse_family_name("Square2CNF").unwrap(),
            LanguageFamily::Square2CNF
        );
        assert!(parse_family_name("Any").is_err());
        assert!(parse_family_name("BestPerNode").is_err());
        assert!(parse_family_name("SquareCNF").is_err());
        assert!(parse_family_name("Nope").is_err());
    }

    #[test]
    fn prediction_dataset_uses_dummy_binary_labels() {
        let dataset = prediction_dataset_from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
        assert_eq!(dataset.n_samples(), 2);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.labels(), &[0, 0]);
    }
}
