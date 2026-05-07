"""
DEPRECATED: Use conj_ui.py instead.

These functions are backward-compatible wrappers that emit deprecation
warnings and delegate to the canonical conj_ui implementations.
"""

import warnings

from gsnh_mdt.search.conj_ui import search_2d_conj_ui, search_3d_conj_ui


def search_2d_square_cnf(*args, **kwargs):
    """Deprecated: use search_2d_conj_ui instead."""
    warnings.warn(
        "search_2d_square_cnf is deprecated; use search_2d_conj_ui",
        DeprecationWarning, stacklevel=2,
    )
    return search_2d_conj_ui(*args, **kwargs)


def search_3d_square_cnf(*args, **kwargs):
    """Deprecated: use search_3d_conj_ui instead."""
    warnings.warn(
        "search_3d_square_cnf is deprecated; use search_3d_conj_ui",
        DeprecationWarning, stacklevel=2,
    )
    return search_3d_conj_ui(*args, **kwargs)
