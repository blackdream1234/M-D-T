from collections import OrderedDict
from pathlib import Path
import csv
import importlib.util
import sys


REQUIRED_CSV = [
    "full_results.csv",
    "summary_by_method.csv",
    "theorem_certified_results.csv",
    "auxiliary_results.csv",
    "complexity_by_method.csv",
    "complexity_by_dataset.csv",
    "pareto_front.csv",
    "dataset_win_loss.csv",
]

REQUIRED_TEX = [
    "tables/main_summary.tex",
    "tables/theorem_certified.tex",
    "tables/auxiliary.tex",
    "tables/complexity_summary.tex",
    "tables/per_dataset_accuracy.tex",
]

REQUIRED_PNG = [
    "figures/accuracy_by_method.png",
    "figures/tree_size_by_method.png",
    "figures/axp_length_by_method.png",
    "figures/train_time_by_method.png",
    "figures/axp_time_by_method.png",
    "figures/accuracy_vs_size_pareto.png",
    "figures/accuracy_vs_axp_pareto.png",
    "figures/runtime_vs_dataset_size.png",
    "figures/win_loss_heatmap.png",
    "figures/theorem_vs_auxiliary_summary.png",
]


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_dl8_languages_updated.py"
    spec = importlib.util.spec_from_file_location("benchmark_dl8_languages_updated_evidence", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_benchmark():
    m = _load_benchmark_module()
    bench = m.LanguageComparisonBenchmark.__new__(m.LanguageComparisonBenchmark)
    bench.methods = m.default_method_configs(include_square2cnf=True)
    bench.lang_map = {cfg.label: cfg.enum_name for cfg in bench.methods}
    bench.depths = [5]
    bench.rs = 42
    bench.n_runs = 1
    bench.dataset_meta_ = {
        "toy": {
            "n_samples": 24,
            "n_features": 6,
            "n_unary_original": 6,
            "pos_rate": 0.5,
        }
    }
    bench.failures_ = []
    bench.results_ = OrderedDict({
        "toy": OrderedDict({
            "sklearn_dt7": m.MethodResult(acc=0.80, acc_std=0.0, size=11, expl=3.0, train_time=0.01, n_success=1),
            "gsnh_1D_d5": m.MethodResult(acc=0.83, acc_std=0.0, size=7, expl=2.0, train_time=0.02, n_success=1, axp_backend="structural_horn", theorem_certified=True, path_certificate="horn", theorem_mode_used=True),
            "gsnh_Horn_d5": m.MethodResult(acc=0.85, acc_std=0.0, size=9, expl=2.5, train_time=0.02, n_success=1, axp_backend="structural_horn", theorem_certified=True, path_certificate="horn", theorem_mode_used=True),
            "gsnh_AntiHorn_d5": m.MethodResult(acc=0.84, acc_std=0.0, size=9, expl=2.4, train_time=0.02, n_success=1, axp_backend="structural_antihorn", theorem_certified=True, path_certificate="antihorn", theorem_mode_used=True),
            "gsnh_Square2CNF_d5": m.MethodResult(acc=0.86, acc_std=0.0, size=8, expl=2.2, train_time=0.03, n_success=1, axp_backend="two_sat", theorem_certified=True, path_certificate="2cnf", theorem_mode_used=True),
            "gsnh_Affine_d5": m.MethodResult(acc=0.82, acc_std=0.0, size=8, expl=2.1, train_time=0.03, n_success=1, axp_backend="affine", theorem_certified=False, path_certificate="affine_unverified", theorem_mode_used=False),
            "gsnh_BestPN_d5": m.MethodResult(acc=0.87, acc_std=0.0, size=6, expl=2.0, train_time=0.04, n_success=1, axp_backend="mixed:structural_horn=1,two_sat=1", theorem_certified=True, path_certificate="mixed:horn=1,2cnf=1", theorem_mode_used=True),
        })
    })
    return m, bench


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_benchmark_evidence_output_files_created(tmp_path):
    _, bench = _fake_benchmark()
    bench.save_outputs(str(tmp_path))

    for rel in REQUIRED_CSV + REQUIRED_TEX + REQUIRED_PNG + [
        "report/benchmark_summary.html",
        "report/benchmark_summary.md",
    ]:
        path = tmp_path / rel
        assert path.exists(), rel
        assert path.stat().st_size > 0, rel


def test_theorem_outputs_exclude_affine_and_bestpn(tmp_path):
    _, bench = _fake_benchmark()
    bench.save_outputs(str(tmp_path))

    rows = _read_csv(tmp_path / "theorem_certified_results.csv")
    labels = {row["method_label"] for row in rows}
    assert "Affine" not in labels
    assert "BestPN" not in labels
    assert {"1D", "Horn", "AntiHorn", "Square2CNF"}.issubset(labels)


def test_complexity_csv_contains_required_columns(tmp_path):
    _, bench = _fake_benchmark()
    bench.save_outputs(str(tmp_path))

    with open(tmp_path / "complexity_by_method.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
    for col in [
        "method_label",
        "category",
        "accuracy_mean",
        "tree_nodes_mean",
        "leaves_mean",
        "avg_leaf_depth",
        "total_literals",
        "mean_axp_length",
        "train_time_mean",
        "axp_time_mean",
        "sat_vars_mean",
        "sat_clauses_mean",
        "n_success",
        "n_fail",
    ]:
        assert col in columns


def test_theorem_table_still_requires_theorem_mode_used_true(tmp_path):
    m, bench = _fake_benchmark()
    bench.results_["toy"]["gsnh_Horn_d5"].theorem_mode_used = False
    bench.save_outputs(str(tmp_path))

    rows = _read_csv(tmp_path / "theorem_certified_results.csv")
    assert not any(row["method_label"] == "Horn" for row in rows)
    assert not m.LanguageComparisonBenchmark._is_theorem_row({
        "method_label": "Horn",
        "axp_backend": "structural_horn",
        "path_certificate": "horn",
        "theorem_certified": True,
        "theorem_mode_used": False,
    })


def test_quick_mode_creates_evidence_outputs_without_dl8_data(tmp_path, monkeypatch):
    m = _load_benchmark_module()
    monkeypatch.setattr(m, "resolve_data_dir", lambda _explicit=None: tmp_path / "missing_data")

    bench = m.main(["--quick", "--output-dir", str(tmp_path / "out")])

    assert bench is not None
    for rel in REQUIRED_CSV + ["report/benchmark_summary.html", "report/benchmark_summary.md"]:
        path = tmp_path / "out" / rel
        assert path.exists(), rel
        assert path.stat().st_size > 0, rel
