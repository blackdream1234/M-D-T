from collections import OrderedDict
from pathlib import Path
import importlib.util
import sys


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_dl8_languages_updated.py"
    # Fallback for older layouts.
    if not script.exists():
        script = root / "benchmark_dl8_languages_updated.py"
    spec = importlib.util.spec_from_file_location("benchmark_dl8_languages_updated", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_records_axp_backend_metadata():
    m = _load_benchmark_module()

    bench = m.LanguageComparisonBenchmark.__new__(m.LanguageComparisonBenchmark)
    bench.methods = [
        m.MethodConfig(
            label="Horn",
            enum_name="HORN",
            search_1d=True,
            search_2d=True,
            search_3d=False,
            mode="journal",
            category="main",
            description="Horn test",
        )
    ]
    bench.depths = [5]
    bench.rs = 123
    bench.n_runs = 1
    bench.results_ = OrderedDict({
        "toy": {
            "gsnh_Horn_d5": m.MethodResult(
                acc=1.0,
                acc_std=0.0,
                size=3.0,
                expl=1.0,
                train_time=0.01,
                n_success=1,
                axp_backend="structural_horn",
                theorem_certified=True,
                path_certificate="horn",
                rejected_reason="",
                theorem_mode_used=True,
            )
        }
    })

    rows = bench._result_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["axp_backend"] == "structural_horn"
    assert row["theorem_certified"] is True
    assert row["path_certificate"] == "horn"
    assert row["rejected_reason"] == ""
    assert row["theorem_mode_used"] is True
    assert row["random_state"] == 123
    assert row["n_runs"] == 1


def test_benchmark_excludes_non_theorem_fallback_from_theorem_table():
    m = _load_benchmark_module()

    fallback_row = {
        "method_label": "BestPN",
        "axp_backend": "interval_dfs_fallback",
        "theorem_certified": True,
        "path_certificate": "horn",
        "theorem_mode_used": True,
    }
    prototype_row = {
        "method_label": "Square2CNF",
        "axp_backend": "prototype_case_split",
        "theorem_certified": True,
        "path_certificate": "2cnf",
        "theorem_mode_used": True,
    }
    square_wrong_backend = {
        "method_label": "Square2CNF",
        "axp_backend": "structural_horn",
        "theorem_certified": True,
        "path_certificate": "2cnf",
        "theorem_mode_used": True,
    }
    square_certified = {
        "method_label": "Square2CNF",
        "axp_backend": "two_sat",
        "theorem_certified": True,
        "path_certificate": "2cnf",
        "theorem_mode_used": True,
    }
    bestpn_certified_mixed_safe = {
        "method_label": "BestPN",
        "axp_backend": "mixed:structural_horn=1,two_sat=1",
        "theorem_certified": True,
        "path_certificate": "mixed:horn=1,2cnf=1",
        "theorem_mode_used": True,
    }

    assert not m.LanguageComparisonBenchmark._is_theorem_row(fallback_row)
    assert not m.LanguageComparisonBenchmark._is_theorem_row(prototype_row)
    assert not m.LanguageComparisonBenchmark._is_theorem_row(square_wrong_backend)
    assert m.LanguageComparisonBenchmark._is_theorem_row(square_certified)
    assert m.LanguageComparisonBenchmark._is_theorem_row(bestpn_certified_mixed_safe)


def test_theorem_row_requires_theorem_mode_used_true():
    m = _load_benchmark_module()

    row = {
        "method_label": "Square2CNF",
        "axp_backend": "two_sat",
        "theorem_certified": True,
        "path_certificate": "2cnf",
        "theorem_mode_used": False,
    }
    assert not m.LanguageComparisonBenchmark._is_theorem_row(row)

    row["theorem_mode_used"] = True
    assert m.LanguageComparisonBenchmark._is_theorem_row(row)
