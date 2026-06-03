"""Shared data-directory constants for dataset-driven tests."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RS_ROOT = str(REPO_ROOT)
_DEFAULT_DATA_DIR = REPO_ROOT / "data"


def find_data_dir() -> str:
    """Return the benchmark .dl8 data directory.

    The canonical repository layout stores benchmark data in ``<repo>/data``.
    ``GSNH_MDT_DATA_DIR`` or ``DATA_DIR`` may override this for local runs.
    """
    for env_name in ("GSNH_MDT_DATA_DIR", "DATA_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            return str(Path(env_value))
    return str(_DEFAULT_DATA_DIR)


DATA_DIR = find_data_dir()


def has_dl8_data() -> bool:
    return Path(DATA_DIR).is_dir() and any(Path(DATA_DIR).glob("*.dl8"))


HAS_DL8_DATA = has_dl8_data()
