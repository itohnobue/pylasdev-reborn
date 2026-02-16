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
    # Check for keys in dict1 not present in dict2
    for key in dict1:
        if key not in dict2:
            logger.warning("Key '%s' not found in second dict", key)
            return False

    for key in dict2:
        if key not in dict1:
            logger.warning("Key '%s' not found in first dict", key)
            return False

        val1, val2 = dict1[key], dict2[key]

        if isinstance(val2, dict):
            # Check for keys in val1 not present in val2
            if isinstance(val1, dict):
                for in_key in val1:
                    if in_key not in val2:
                        logger.warning("Key '%s.%s' not found in second dict", key, in_key)
                        return False

            for in_key in val2:
                if in_key not in val1:
                    logger.warning("Key '%s.%s' not found in first dict", key, in_key)
                    return False

                if isinstance(val2[in_key], np.ndarray):
                    if not _compare_arrays(val1[in_key], val2[in_key], key, in_key, rtol, atol):
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
