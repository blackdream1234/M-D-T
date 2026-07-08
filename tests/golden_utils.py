"""Helpers for post-theorem-boundary golden regression data.

The checked-in golden JSON files correspond to deterministic behavior after the
 theorem-boundary safety patch.  Dataset files may live directly under DATA_DIR
or under a nested directory such as data/data1, so tests and capture scripts use
recursive lookup instead of hard-coded direct paths.
"""

import json
from pathlib import Path

from data_path import DATA_DIR


def load_golden(path):
    """Load a golden JSON file, ignoring optional underscore metadata keys."""
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def find_dataset_file(name: str) -> str:
    """Resolve a named .dl8 dataset under DATA_DIR, searching recursively."""
    data_dir = Path(DATA_DIR)
    direct = data_dir / f"{name}.dl8"
    if direct.exists():
        return str(direct)
    matches = sorted(data_dir.rglob(f"{name}.dl8")) if data_dir.exists() else []
    if matches:
        return str(matches[0])
    return str(direct)
