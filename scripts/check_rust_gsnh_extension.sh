#!/usr/bin/env bash
set -euo pipefail

cargo fmt --manifest-path rust_gsnh/Cargo.toml --check
cargo test --manifest-path rust_gsnh/Cargo.toml
cargo test --manifest-path rust_gsnh/Cargo.toml --features python
maturin develop --manifest-path rust_gsnh/Cargo.toml --features python,pyo3-extension
pytest tests/test_rust_gsnh_binding.py -q
pytest tests/test_engine_wrapper.py -q
pytest tests/test_engine_wrapper_parity.py -q
