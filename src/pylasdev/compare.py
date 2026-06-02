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

    The high branch count (23) is inherent to the function's purpose:
    it must handle every type that appears in LAS dicts — numpy arrays
    (with tolerance comparison), nested dicts, lists (with special
    handling for data_sections containing numpy arrays), and scalars —
    each requiring distinct comparison logic.  Breaking this into
    separate functions for each type would scatter the comparison
    protocol and make it harder to follow the control flow.

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
            # F-10: data_sections contains numpy arrays that can't
            # be compared with generic list equality (ambiguity error).
            if key == "data_sections":
                if not _compare_data_sections(val1, val2, rtol, atol):
                    return False
            elif val1 != val2:
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

    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        logger.warning(
            "Type mismatch at '%s': %s vs %s",
            label,
            type(arr1).__name__,
            type(arr2).__name__,
        )
        return False

    if arr1.size != arr2.size:
        logger.warning("Array size mismatch at '%s': %d vs %d", label, arr1.size, arr2.size)
        return False

    # F-4: np.allclose() fails on string/object arrays (e.g. string_data).
    # Use np.array_equal for non-numeric dtypes.
    if arr1.dtype.kind in ("U", "S", "O") or arr2.dtype.kind in ("U", "S", "O"):
        if not np.array_equal(arr1, arr2):
            logger.warning("Array values mismatch at '%s'", label)
            return False
    elif not np.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True):
        logger.warning("Array values mismatch at '%s'", label)
        return False

    return True


def _compare_data_sections(
    sections1: list[dict[str, Any]],
    sections2: list[dict[str, Any]],
    rtol: float,
    atol: float,
) -> bool:
    """Compare data_sections lists element-wise.

    Handles numpy arrays inside nested dicts where generic list
    equality (val1 != val2) would raise "truth value of an array
    is ambiguous" (F-10).
    """
    if len(sections1) != len(sections2):
        logger.warning(
            "data_sections length mismatch: %d vs %d",
            len(sections1), len(sections2),
        )
        return False

    for i, (ds1, ds2) in enumerate(zip(sections1, sections2, strict=False)):
        if set(ds1.keys()) != set(ds2.keys()):
            only_in_first = set(ds1.keys()) - set(ds2.keys())
            only_in_second = set(ds2.keys()) - set(ds1.keys())
            if only_in_first:
                logger.warning(
                    "Keys 'data_sections[%d]' only in first: %s", i, only_in_first,
                )
            if only_in_second:
                logger.warning(
                    "Keys 'data_sections[%d]' only in second: %s", i, only_in_second,
                )
            return False

        for k in ds2:
            v1, v2 = ds1[k], ds2[k]
            if isinstance(v2, np.ndarray):
                if not _compare_arrays(
                    v1, v2, f"data_sections[{i}].{k}", None, rtol, atol,
                ):
                    return False
            elif isinstance(v2, list):
                if v1 != v2:
                    logger.warning(
                        "List mismatch at 'data_sections[%d].%s': %r vs %r",
                        i, k, v1, v2,
                    )
                    return False
            else:
                if v1 != v2:
                    logger.warning(
                        "Mismatch at 'data_sections[%d].%s': %r vs %r",
                        i, k, v1, v2,
                    )
                    return False

    return True
