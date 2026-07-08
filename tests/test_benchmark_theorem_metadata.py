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
    bench.results_ = OrderedDict(
        {
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
        }
    )

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
    assert not m.LanguageComparisonBenchmark._is_theorem_row(bestpn_certified_mixed_safe)


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


def _row(method_label, backend, certificate, theorem_mode_used=True):
    return {
        "method_label": method_label,
        "axp_backend": backend,
        "theorem_certified": True,
        "path_certificate": certificate,
        "theorem_mode_used": theorem_mode_used,
    }


def test_main_certified_methods_use_theorem_mode():
    m = _load_benchmark_module()

    rows = [
        _row("1D", "structural_horn", "horn", True),
        _row("Horn", "structural_horn", "horn", True),
        _row("AntiHorn", "structural_antihorn", "antihorn", True),
        _row("Square2CNF", "two_sat", "2cnf", True),
    ]

    for row in rows:
        assert m.LanguageComparisonBenchmark._is_theorem_row(row), row


def test_safe_backend_rejected_without_theorem_mode():
    m = _load_benchmark_module()

    rows = [
        _row("1D", "structural_horn", "horn", False),
        _row("Horn", "structural_horn", "horn", False),
        _row("AntiHorn", "structural_antihorn", "antihorn", False),
        _row("Square2CNF", "two_sat", "2cnf", False),
    ]

    for row in rows:
        assert not m.LanguageComparisonBenchmark._is_theorem_row(row), row


def test_square2cnf_requires_two_sat_2cnf_theorem_mode():
    m = _load_benchmark_module()

    assert m.LanguageComparisonBenchmark._is_theorem_row(
        _row("Square2CNF", "two_sat", "2cnf", True)
    )
    assert not m.LanguageComparisonBenchmark._is_theorem_row(
        _row("Square2CNF", "two_sat", "2cnf", False)
    )
    assert not m.LanguageComparisonBenchmark._is_theorem_row(
        _row("Square2CNF", "structural_horn", "2cnf", True)
    )
    assert not m.LanguageComparisonBenchmark._is_theorem_row(
        _row("Square2CNF", "two_sat", "horn", True)
    )


def test_theorem_table_excludes_affine_without_verified_certificate():
    m = _load_benchmark_module()

    affine_row = {
        "method_label": "Affine",
        "axp_backend": "affine",
        "theorem_certified": True,
        "path_certificate": "affine_unverified",
        "theorem_mode_used": True,
    }
    mixed_affine_bestpn = {
        "method_label": "BestPN",
        "axp_backend": "mixed:affine=1,structural_horn=1",
        "theorem_certified": True,
        "path_certificate": "mixed:affine_unverified=1,horn=1",
        "theorem_mode_used": True,
    }

    assert not m.LanguageComparisonBenchmark._is_theorem_row(affine_row)
    assert not m.LanguageComparisonBenchmark._is_theorem_row(mixed_affine_bestpn)


def test_theorem_table_requires_theorem_mode_used_true():
    m = _load_benchmark_module()

    row = {
        "method_label": "Horn",
        "axp_backend": "structural_horn",
        "theorem_certified": True,
        "path_certificate": "horn",
        "theorem_mode_used": False,
    }
    assert not m.LanguageComparisonBenchmark._is_theorem_row(row)


def test_bestpn_remains_empirical_in_benchmark_theorem_table():
    m = _load_benchmark_module()

    empirical_bestpn = {
        "method_label": "BestPN",
        "axp_backend": "interval_dfs_fallback",
        "theorem_certified": False,
        "path_certificate": "none",
        "theorem_mode_used": False,
    }
    certified_bestpn = {
        "method_label": "BestPN",
        "axp_backend": "mixed:structural_horn=1,two_sat=1",
        "theorem_certified": True,
        "path_certificate": "mixed:horn=1,2cnf=1",
        "theorem_mode_used": True,
    }
    unsafe_bestpn = {
        "method_label": "BestPN",
        "axp_backend": "mixed:structural_horn=1,affine=1",
        "theorem_certified": True,
        "path_certificate": "mixed:horn=1,affine_unverified=1",
        "theorem_mode_used": True,
    }

    assert not m.LanguageComparisonBenchmark._is_theorem_row(empirical_bestpn)
    assert not m.LanguageComparisonBenchmark._is_theorem_row(certified_bestpn)
    assert not m.LanguageComparisonBenchmark._is_theorem_row(unsafe_bestpn)


def test_theorem_table_allows_only_certified_method_labels_and_backends():
    m = _load_benchmark_module()

    forbidden_backends = [
        "interval_dfs_fallback",
        "prototype_case_split",
        "rejected_non_theorem",
        "affine",
        "none",
    ]
    for backend in forbidden_backends:
        assert not m.LanguageComparisonBenchmark._is_theorem_row(
            _row("Horn", backend, "horn", True)
        )

    for method_label in ["Affine", "BestPN", "ConjUI"]:
        assert not m.LanguageComparisonBenchmark._is_theorem_row(
            _row(method_label, "structural_horn", "horn", True)
        )

    assert m.LanguageComparisonBenchmark._is_theorem_row(
        _row("Horn", "structural_horn", "horn", True)
    )
    assert m.LanguageComparisonBenchmark._is_theorem_row(
        _row("AntiHorn", "structural_antihorn", "antihorn", True)
    )
    assert m.LanguageComparisonBenchmark._is_theorem_row(
        _row("Square2CNF", "two_sat", "2cnf", True)
    )
