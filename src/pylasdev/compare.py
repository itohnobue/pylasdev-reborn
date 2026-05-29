"""LAS data comparison utilities.

Replaces las_compare.py with Python 3 syntax and proper logging
instead of print statements.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compare_las_dicts(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> bool:
    """Compare two LAS data dictionaries for equality.

    Args:
        dict1: First LAS data dictionary.
        dict2: Second LAS data dictionary.
        rtol: Relative tolerance for numpy array comparison.
        atol: Absolute tolerance for numpy array comparison.

    Returns:
        True if the dictionaries are equivalent, False otherwise.
    """
    # Check for key equality using set operations
    if set(dict1.keys()) != set(dict2.keys()):
        only_in_first = set(dict1.keys()) - set(dict2.keys())
        only_in_second = set(dict2.keys()) - set(dict1.keys())
        if only_in_first:
            logger.warning("Keys only in first dict: %s", only_in_first)
        if only_in_second:
            logger.warning("Keys only in second dict: %s", only_in_second)
        return False

    for key in dict2:
        val1, val2 = dict1[key], dict2[key]

        if isinstance(val2, dict):
            # Check for key equality in nested dicts using set operations
            if isinstance(val1, dict):
                if set(val1.keys()) != set(val2.keys()):
                    only_in_first = set(val1.keys()) - set(val2.keys())
                    only_in_second = set(val2.keys()) - set(val1.keys())
                    if only_in_first:
                        logger.warning("Keys '%s'.%s only in first dict", key, only_in_first)
                    if only_in_second:
                        logger.warning("Keys '%s'.%s only in second dict", key, only_in_second)
                    return False

            for in_key in val2:
                if isinstance(val1, dict) and in_key not in val1:
                    logger.warning("Key '%s.%s' not found in first dict", key, in_key)
                    return False

                if isinstance(val2[in_key], np.ndarray):
                    if not _compare_arrays(val1[in_key], val2[in_key], key, in_key, rtol, atol):
                        return False
                elif isinstance(val1[in_key], np.ndarray):
                    # val1 has array but val2 doesn't — type mismatch
                    logger.warning(
                        "Type mismatch at '%s.%s': %s vs %s",
                        key, in_key,
                        type(val1[in_key]).__name__, type(val2[in_key]).__name__,
                    )
                    return False
                elif val1[in_key] != val2[in_key]:
                    logger.warning(
                        "Mismatch at '%s.%s': %r vs %r", key, in_key, val1[in_key], val2[in_key]
                    )
                    return False

        elif isinstance(val2, np.ndarray):
            if not _compare_arrays(val1, val2, key, None, rtol, atol):
                return False

        elif isinstance(val2, list):
            if val1 != val2:
                logger.warning("List mismatch at '%s': %r vs %r", key, val1, val2)
                return False
        else:
            if val1 != val2:
                logger.warning("Mismatch at '%s': %r vs %r", key, val1, val2)
                return False

    return True


def _compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    key: str,
    in_key: str | None,
    rtol: float,
    atol: float,
) -> bool:
    """Compare two numpy arrays with tolerance."""
    label = f"{key}.{in_key}" if in_key else key

    if arr1.size != arr2.size:
        logger.warning("Array size mismatch at '%s': %d vs %d", label, arr1.size, arr2.size)
        return False

    if not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
        logger.warning("Array values mismatch at '%s'", label)
        return False

    return True
