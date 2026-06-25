use rust_gsnh::Dataset;

#[test]
fn dl8_parser_matches_python_benchmark_column_contract() {
    let dl8 = "1 0 1 0\n0 1 0 1\n";
    let ds = Dataset::from_dl8_text(dl8).unwrap();

    assert_eq!(ds.labels(), &[1, 0]);
    assert_eq!(ds.n_samples(), 2);
    assert_eq!(ds.n_features(), 3);
    assert_eq!(ds.row(0), &[0.0, 1.0, 0.0]);
    assert_eq!(ds.row(1), &[1.0, 0.0, 1.0]);
}
