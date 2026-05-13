#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"

MMD_FILES = [
    "training_pipeline.mmd",
    "language_families.mmd",
    "axp_extraction_pipeline.mmd",
    "theorem_strict_checker.mmd",
    "square2cnf_two_sat_encoding.mmd",
    "benchmark_reporting_pipeline.mmd",
]

DOT_FILES = [f.replace(".mmd", ".dot") for f in MMD_FILES]


def run(cmd):
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def main():
    print(f"[info] figure directory: {FIG_DIR}")

    mmdc = shutil.which("mmdc")
    dot = shutil.which("dot")

    if mmdc:
        print(f"[info] found mermaid-cli: {mmdc}")
        for name in MMD_FILES:
            inp = FIG_DIR / name
            out = FIG_DIR / name.replace(".mmd", ".svg")
            ok = run([mmdc, "-i", str(inp), "-o", str(out)])
            print(f"[{'ok' if ok else 'warn'}] mermaid {name} -> {out.name}")
    else:
        print("[warn] mermaid-cli (mmdc) not found. Install: npm i -g @mermaid-js/mermaid-cli")

    if dot:
        print(f"[info] found graphviz dot: {dot}")
        for name in DOT_FILES:
            inp = FIG_DIR / name
            out = FIG_DIR / name.replace(".dot", ".dot.svg")
            ok = run([dot, "-Tsvg", str(inp), "-o", str(out)])
            print(f"[{'ok' if ok else 'warn'}] dot {name} -> {out.name}")
    else:
        print("[warn] graphviz 'dot' not found. Install graphviz to render .dot files.")

    print("\n[info] manual examples:")
    print("  mmdc -i docs/figures/training_pipeline.mmd -o docs/figures/training_pipeline.svg")
    print("  dot -Tsvg docs/figures/training_pipeline.dot -o docs/figures/training_pipeline.dot.svg")


if __name__ == "__main__":
    main()
